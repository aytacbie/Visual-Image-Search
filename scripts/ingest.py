"""Index every image under data/images (or --images-dir). Safe to re-run: indexed files are skipped.

    python scripts/ingest.py
    python scripts/ingest.py --images-dir D:/dataset --batch-size 64 --workers 8
"""
import argparse
import dataclasses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.embedder import create_embedder  # noqa: E402
from app.index import VectorStore  # noqa: E402
from app.ingest import ingest_folder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", type=Path, help="default: data/images")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4, help="threads for reading/decoding files")
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--rebuild", choices=["flat", "hnsw"], help="force the index type after ingest")
    args = parser.parse_args()

    settings = Settings.from_env()
    images_dir = (args.images_dir or settings.images_dir).resolve()
    if not images_dir.is_dir():
        sys.exit(f"Images folder not found: {images_dir}")

    embedder = create_embedder(settings)
    settings = dataclasses.replace(settings, embedder=embedder.backend)
    print(f"Embedder: {embedder.model_id} -> {settings.index_dir}")
    store = VectorStore(settings.index_path, settings.db_path, embedder.dim, embedder.model_id)

    started = time.perf_counter()
    stats = ingest_folder(
        settings, embedder, store, images_dir,
        batch_size=args.batch_size, workers=args.workers, checkpoint_every=args.checkpoint_every,
    )
    if args.rebuild and args.rebuild != store.index_type:
        store.rebuild(args.rebuild)
    store.close()

    elapsed = time.perf_counter() - started
    print(
        f"\nDone in {elapsed:.1f}s | files found: {stats.found} | added: {stats.added} | "
        f"already indexed: {stats.already_indexed} | identical content: {stats.identical_content} | "
        f"errors: {stats.errors} | with products.csv metadata: {stats.with_metadata}"
    )
    print(f"Index now holds {store.ntotal} images ({store.index_type}).")
    if stats.errors:
        print(f"Unreadable files are listed in {settings.data_dir / 'ingest_errors.log'}")


if __name__ == "__main__":
    main()
