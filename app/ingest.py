"""Bulk ingest: folder of images -> CLIP embeddings -> FAISS + SQLite.

Idempotent and resumable: files already in the DB are skipped by path and the index
is checkpointed every N images. Every file is indexed, even if another file has the
same bytes: in a catalogue those are different products that share a photo.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from app.catalog import Product, load_catalog
from app.config import Settings
from app.embedder import BaseEmbedder
from app.errors import InvalidImageError
from app.images import decode_image, save_thumbnail, sha256_hex
from app.index import NewImage, VectorStore

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


@dataclass
class IngestStats:
    found: int = 0
    already_indexed: int = 0
    added: int = 0
    identical_content: int = 0  # indexed, but same bytes as another file
    with_metadata: int = 0  # files that have a row in products.csv
    errors: int = 0


def iter_image_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def has_images(root: Path) -> bool:
    return root.is_dir() and next(iter_image_files(root), None) is not None


def _load(path: Path) -> tuple[Path, str | None, Image.Image | None, str | None]:
    try:
        data = path.read_bytes()
        return path, sha256_hex(data), decode_image(data), None
    except (OSError, InvalidImageError) as exc:
        return path, None, None, str(exc)


def ingest_folder(
    settings: Settings,
    embedder: BaseEmbedder,
    store: VectorStore,
    images_dir: Path | None = None,
    batch_size: int = 32,
    workers: int = 4,
    checkpoint_every: int = 5000,
    progress: bool = True,
) -> IngestStats:
    images_dir = images_dir or settings.images_dir
    stats = IngestStats()
    catalog = load_catalog(settings)
    known = store.known_paths()
    all_files = list(iter_image_files(images_dir))
    files = [p for p in all_files if settings.to_stored_path(p) not in known]
    stats.found = len(all_files)
    stats.already_indexed = len(all_files) - len(files)
    error_log = settings.data_dir / "ingest_errors.log"
    since_checkpoint = 0

    with ThreadPoolExecutor(workers) as pool, tqdm(total=len(files), unit="img", disable=not progress) as bar:
        for start in range(0, len(files), batch_size):
            batch = files[start:start + batch_size]
            items, images, batch_shas = [], [], set()
            for path, sha, img, err in pool.map(_load, batch):
                if err:
                    stats.errors += 1
                    with error_log.open("a", encoding="utf-8") as log:
                        log.write(f"{path}\t{err}\n")
                else:
                    if sha in batch_shas or store.find_by_sha(sha):
                        stats.identical_content += 1
                    batch_shas.add(sha)
                    stored = settings.to_stored_path(path)
                    product = catalog.get(stored, Product())
                    items.append(NewImage(stored, sha, img.width, img.height, "ingest", product))
                    images.append(img)

            if items:
                ids = store.add(items, embedder.embed(images))
                for image_id, img in zip(ids, images):
                    save_thumbnail(img, settings.thumbs_dir / f"{image_id}.jpg")
                stats.added += len(ids)
                since_checkpoint += len(ids)
                if since_checkpoint >= checkpoint_every:
                    store.save()
                    since_checkpoint = 0
            bar.update(len(batch))

    # Picks up edits to products.csv for images indexed on earlier runs.
    store.update_products(catalog)
    stats.with_metadata = sum(1 for p in map(settings.to_stored_path, all_files) if p in catalog)
    if store.ntotal >= settings.hnsw_threshold and store.index_type != "hnsw":
        store.rebuild("hnsw")
    store.save()
    return stats
