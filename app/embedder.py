"""Image embeddings, L2-normalised so inner product == cosine similarity.

- OnnxClipEmbedder (default): CLIP ViT-B/32 run locally with ONNX Runtime. Smaller and
  faster on CPU than PyTorch, and its DLLs are signed (Windows Smart App Control
  blocks torch's).
- NVClipEmbedder (optional): NVIDIA NV-CLIP through build.nvidia.com, enabled with
  VS_EMBEDDER=nvclip and NVIDIA_API_KEY. The key stays on the server.

The two produce different vector spaces, so each backend has its own index folder.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import time
from typing import Protocol

import httpx
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image

from app.config import Settings
from app.errors import APIError

logger = logging.getLogger("visual_search")


class EmbeddingServiceError(APIError):
    status_code = 503


class BaseEmbedder(Protocol):
    backend: str
    model_id: str
    dim: int

    def embed(self, images: list[Image.Image]) -> np.ndarray: ...


def _normalise(features: np.ndarray) -> np.ndarray:
    features = features.astype("float32")
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    return features


def _download(repo: str, filename: str) -> str:
    """Use the local cache first, so the server starts even if the Hub is unreachable."""
    try:
        return hf_hub_download(repo, filename, local_files_only=True)
    except Exception:
        return hf_hub_download(repo, filename)


def _edge(value, key: str) -> int:
    return value[key] if isinstance(value, dict) else int(value)


class OnnxClipEmbedder:
    backend = "onnx"

    def __init__(self, model_repo: str, model_file: str, threads: int = 0):
        model_path = _download(model_repo, model_file)
        with open(_download(model_repo, "preprocessor_config.json"), encoding="utf-8") as f:
            pre = json.load(f)
        self.resize_to = _edge(pre["size"], "shortest_edge")
        self.crop = _edge(pre["crop_size"], "height")
        self.mean = np.array(pre["image_mean"], dtype=np.float32)
        self.std = np.array(pre["image_std"], dtype=np.float32)

        options = ort.SessionOptions()
        if threads > 0:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        output = next(o for o in self.session.get_outputs() if o.name == "image_embeds")
        self.output_name = output.name
        self.dim: int = output.shape[-1]
        self.model_id = f"{model_repo}/{model_file}"

    def _preprocess(self, img: Image.Image) -> np.ndarray:
        """Same steps as CLIPImageProcessor: resize short side (bicubic), centre crop, normalise."""
        w, h = img.size
        scale = self.resize_to / min(w, h)
        img = img.resize(
            (max(self.resize_to, round(w * scale)), max(self.resize_to, round(h * scale))),
            Image.Resampling.BICUBIC,
        )
        w, h = img.size
        left, top = (w - self.crop) // 2, (h - self.crop) // 2
        img = img.crop((left, top, left + self.crop, top + self.crop))
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - self.mean) / self.std
        return arr.transpose(2, 0, 1)

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        batch = np.stack([self._preprocess(img) for img in images])
        return _normalise(self.session.run([self.output_name], {self.input_name: batch})[0])


class NVClipEmbedder:
    backend = "nvclip"
    MAX_SIDE = 448  # the model sees 224px; smaller payloads stay under the API's inline-image limit
    BATCH = 16
    RETRIES = 3

    def __init__(self, api_key: str, model: str, url: str):
        self._client = httpx.Client(
            timeout=30, headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
        )
        self.model = model
        self.url = url
        self.model_id = f"nvidia-api/{model}"
        # One call up front: checks the key and learns the vector size.
        self.dim = self.embed([Image.new("RGB", (64, 64), (128, 128, 128))]).shape[1]

    @classmethod
    def _data_url(cls, img: Image.Image) -> str:
        img = img.copy()
        img.thumbnail((cls.MAX_SIDE, cls.MAX_SIDE))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=90)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    def _request(self, inputs: list[str]) -> np.ndarray:
        body = {"input": inputs, "model": self.model, "encoding_format": "float"}
        for attempt in range(1, self.RETRIES + 1):
            try:
                resp = self._client.post(self.url, json=body)
            except httpx.HTTPError as exc:
                error = f"{exc.__class__.__name__}"
            else:
                if resp.status_code == 200:
                    data = sorted(resp.json()["data"], key=lambda d: d["index"])
                    return np.array([d["embedding"] for d in data], dtype="float32")
                error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if resp.status_code not in (429, 500, 502, 503, 504):
                    break
            time.sleep(attempt)
        raise EmbeddingServiceError(f"NV-CLIP embedding service failed ({error}).")

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        chunks = [
            self._request([self._data_url(img) for img in images[i:i + self.BATCH]])
            for i in range(0, len(images), self.BATCH)
        ]
        return _normalise(np.vstack(chunks))


def create_embedder(settings: Settings) -> BaseEmbedder:
    """NV-CLIP when configured and reachable, otherwise the local ONNX model."""
    if settings.embedder == "nvclip":
        if not settings.nvidia_api_key:
            logger.warning("VS_EMBEDDER=nvclip but NVIDIA_API_KEY is not set; using local ONNX model")
        else:
            try:
                return NVClipEmbedder(settings.nvidia_api_key, settings.nvclip_model, settings.nvclip_url)
            except EmbeddingServiceError as exc:
                logger.warning("NV-CLIP unavailable (%s); using local ONNX model", exc.message)
    return OnnxClipEmbedder(settings.model_repo, settings.model_file, settings.threads)
