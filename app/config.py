"""Settings, read from the environment (and `.env` in the project root)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent


def _flag(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    data_dir: Path = ROOT_DIR / "data"
    # "onnx" (local CLIP, default) or "nvclip" (NVIDIA API; falls back to onnx without a key)
    embedder: str = "onnx"
    model_repo: str = "Xenova/clip-vit-base-patch32"
    model_file: str = "onnx/vision_model.onnx"
    nvidia_api_key: str = field(default="", repr=False)
    nvclip_model: str = "nvidia/nvclip"
    nvclip_url: str = "https://integrate.api.nvidia.com/v1/embeddings"
    max_upload_mb: int = 10
    default_top_k: int = 5
    max_top_k: int = 50
    # Below this many vectors exact search is fast enough; above it we switch to HNSW.
    hnsw_threshold: int = 50_000
    rate_limit_per_minute: int = 120  # full/search keys, per key
    web_rate_limit_per_minute: int = 20  # web keys and the built-in page, per client IP
    threads: int = 0  # inference threads, 0 = all cores
    # Extra API keys as "label:scope:sha256;..." (for hosts where data/ is not writable, e.g. HF secrets)
    api_keys_env: str = field(default="", repr=False)
    cors_origins: tuple[str, ...] = ()
    cors_origin_regex: str = (
        r"^https://([a-z0-9-]+\.)*(lovable\.app|lovableproject\.com|lovable\.dev|hf\.space)$"
        r"|^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
    )
    # Set when the proxy does not pass the public host/scheme; otherwise taken from the request.
    public_base_url: str = ""
    ingest_on_startup: bool = True
    # The built-in page searches through /web/search without a key: same-origin only,
    # limited per visitor IP (web_rate_limit_per_minute). Disable to require keys everywhere.
    web_search: bool = True

    @property
    def active_embedder(self) -> str:
        return "nvclip" if self.embedder == "nvclip" and self.nvidia_api_key else "onnx"

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def uploads_dir(self) -> Path:
        return self.images_dir / "uploads"

    @property
    def catalog_path(self) -> Path:
        return self.data_dir / "products.csv"

    @property
    def index_dir(self) -> Path:
        # Each embedder has its own vector space, so its own index, DB and thumbnails.
        return self.data_dir / "index" / self.active_embedder

    @property
    def thumbs_dir(self) -> Path:
        return self.index_dir / "thumbs"

    @property
    def index_path(self) -> Path:
        return self.index_dir / "index.faiss"

    @property
    def db_path(self) -> Path:
        return self.index_dir / "meta.sqlite"

    @property
    def keys_path(self) -> Path:
        return self.data_dir / "api_keys.json"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def resolve_stored_path(self, stored: str) -> Path:
        """Paths in the DB are relative to data_dir when the file lives inside it."""
        path = Path(stored)
        return path if path.is_absolute() else self.data_dir / path

    def to_stored_path(self, path: Path) -> str:
        path = path.resolve()
        try:
            return path.relative_to(self.data_dir.resolve()).as_posix()
        except ValueError:
            return str(path)

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv(ROOT_DIR / ".env")
        env = os.getenv
        return cls(
            data_dir=Path(env("VS_DATA_DIR", str(cls.data_dir))).resolve(),
            embedder=env("VS_EMBEDDER", cls.embedder).strip().lower(),
            model_repo=env("VS_MODEL_REPO", cls.model_repo),
            model_file=env("VS_MODEL_FILE", cls.model_file),
            nvidia_api_key=env("NVIDIA_API_KEY", "").strip(),
            nvclip_model=env("VS_NVCLIP_MODEL", cls.nvclip_model),
            nvclip_url=env("VS_NVCLIP_URL", cls.nvclip_url),
            max_upload_mb=int(env("VS_MAX_UPLOAD_MB", cls.max_upload_mb)),
            default_top_k=int(env("VS_DEFAULT_TOP_K", cls.default_top_k)),
            max_top_k=int(env("VS_MAX_TOP_K", cls.max_top_k)),
            hnsw_threshold=int(env("VS_HNSW_THRESHOLD", cls.hnsw_threshold)),
            rate_limit_per_minute=int(env("VS_RATE_LIMIT_PER_MINUTE", cls.rate_limit_per_minute)),
            web_rate_limit_per_minute=int(env("VS_WEB_RATE_LIMIT_PER_MINUTE", cls.web_rate_limit_per_minute)),
            threads=int(env("VS_THREADS", cls.threads)),
            api_keys_env=env("VS_API_KEYS", ""),
            cors_origins=tuple(o.strip() for o in env("VS_CORS_ORIGINS", "").split(",") if o.strip()),
            public_base_url=env("VS_PUBLIC_BASE_URL", "").rstrip("/"),
            ingest_on_startup=_flag(env("VS_INGEST_ON_STARTUP", "1")),
            web_search=_flag(env("VS_WEB_SEARCH", "1")),
        )
