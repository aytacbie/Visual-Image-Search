"""Decoding, validation and thumbnails for images."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from app.errors import InvalidImageError

# Refuse decompression bombs (a tiny PNG that expands to gigabytes of pixels).
Image.MAX_IMAGE_PIXELS = 60_000_000
THUMB_SIZE = (320, 320)
_EXTENSIONS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif", "BMP": ".bmp", "TIFF": ".tif"}


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decode_image(data: bytes) -> Image.Image:
    """Bytes -> RGB image, with EXIF rotation applied and transparency flattened on white."""
    if not data:
        raise InvalidImageError("Empty file.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        img = Image.open(io.BytesIO(data))
        img.load()
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA", "P"):
            rgba = img.convert("RGBA")
            img = Image.new("RGB", rgba.size, (255, 255, 255))
            img.paste(rgba, mask=rgba.getchannel("A"))
        elif img.mode != "RGB":
            img = img.convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError, ValueError) as exc:
        raise InvalidImageError(
            "File is not a valid image (supported: JPEG, PNG, WebP, GIF, BMP, TIFF)."
        ) from exc
    return img


def detect_extension(data: bytes) -> str:
    with Image.open(io.BytesIO(data)) as img:
        return _EXTENSIONS.get(img.format or "", ".img")


def save_thumbnail(img: Image.Image, path: Path) -> None:
    thumb = img.copy()
    thumb.thumbnail(THUMB_SIZE, Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    thumb.save(path, "JPEG", quality=85)
