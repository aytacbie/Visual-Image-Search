"""Vector index (FAISS) + image/product metadata (SQLite), kept consistent with each other.

Image ids are SQLite rowids and are also the FAISS ids. Rows are committed before
the index file is written, and on startup any row whose vector never reached the
saved index is dropped, so a crash mid-ingest only means those files get
re-ingested on the next run. AUTOINCREMENT guarantees dropped ids are never reused.

Filters (category, price, ...) are resolved to an id set in SQLite and applied
inside FAISS with an IDSelector, so a filtered query still returns the true top-k
among matching items instead of post-filtering a short list.
"""
from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import astuple, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

from app.catalog import Product

HNSW_M = 32
HNSW_EF_CONSTRUCTION = 80
HNSW_EF_SEARCH = 96

_SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""
_PRODUCT_COLUMNS = {
    "product_id": "INTEGER", "name": "TEXT", "brand": "TEXT",
    "category": "TEXT", "subcategory": "TEXT", "price_usd": "REAL",
}
_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_images_path ON images(path);
CREATE INDEX IF NOT EXISTS idx_images_sha256 ON images(sha256);
CREATE INDEX IF NOT EXISTS idx_images_category ON images(category COLLATE NOCASE, subcategory COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_images_price ON images(price_usd);
"""
_SELECT = (
    "SELECT id, path, sha256, width, height, source, "
    "product_id, name, brand, category, subcategory, price_usd FROM images"
)


@dataclass(frozen=True)
class NewImage:
    path: str
    sha256: str
    width: int
    height: int
    source: str
    product: Product = field(default_factory=Product)


@dataclass(frozen=True)
class ImageRecord:
    id: int
    path: str
    sha256: str
    width: int
    height: int
    source: str
    product: Product


def _record(row) -> ImageRecord:
    return ImageRecord(*row[:6], Product(*row[6:]))


@dataclass(frozen=True)
class SearchFilter:
    category: str | None = None
    subcategory: str | None = None
    brand: str | None = None
    min_price: float | None = None
    max_price: float | None = None

    def is_empty(self) -> bool:
        return all(v is None for v in astuple(self))

    def where(self) -> tuple[str, list]:
        clauses, params = [], []
        for column in ("category", "subcategory", "brand"):
            value = getattr(self, column)
            if value is not None:
                clauses.append(f"{column} = ? COLLATE NOCASE")
                params.append(value)
        if self.min_price is not None:
            clauses.append("price_usd >= ?")
            params.append(self.min_price)
        if self.max_price is not None:
            clauses.append("price_usd <= ?")
            params.append(self.max_price)
        return " AND ".join(clauses) or "1", params


def _new_index(kind: str, dim: int) -> faiss.IndexIDMap2:
    if kind == "hnsw":
        base = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
        base.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        base.hnsw.efSearch = HNSW_EF_SEARCH
    elif kind == "flat":
        base = faiss.IndexFlatIP(dim)
    else:
        raise ValueError(f"Unknown index kind: {kind}")
    return faiss.IndexIDMap2(base)


class VectorStore:
    def __init__(self, index_path: Path, db_path: Path, dim: int, model_id: str):
        self.index_path = index_path
        self.dim = dim
        self._lock = threading.RLock()
        self._dirty = False
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        self._migrate()
        self._check_model(model_id)
        self._index = self._load_index()
        self._drop_orphan_rows()

    # --- setup -------------------------------------------------------------

    def _migrate(self) -> None:
        existing = {r[1] for r in self._db.execute("PRAGMA table_info(images)")}
        for column, sql_type in _PRODUCT_COLUMNS.items():
            if column not in existing:
                self._db.execute(f"ALTER TABLE images ADD COLUMN {column} {sql_type}")
        self._db.executescript(_INDEXES)
        self._db.commit()

    def _check_model(self, model_id: str) -> None:
        row = self._db.execute("SELECT value FROM meta WHERE key = 'model'").fetchone()
        if row is None:
            self._db.execute("INSERT INTO meta (key, value) VALUES ('model', ?)", (model_id,))
            self._db.commit()
        elif row[0] != model_id:
            raise RuntimeError(
                f"Index in {self.index_path.parent} was built with {row[0]} but the embedder is {model_id}. "
                "Delete that folder and run scripts/ingest.py again."
            )

    def _load_index(self) -> faiss.IndexIDMap2:
        if not self.index_path.exists():
            return _new_index("flat", self.dim)
        index = faiss.read_index(str(self.index_path))
        if index.d != self.dim:
            raise RuntimeError(f"{self.index_path} has dim {index.d}, model has dim {self.dim}.")
        base = faiss.downcast_index(index.index)
        if isinstance(base, faiss.IndexHNSWFlat):
            base.hnsw.efSearch = HNSW_EF_SEARCH
        return index

    def _drop_orphan_rows(self) -> None:
        indexed = set(faiss.vector_to_array(self._index.id_map).tolist())
        rows = [r[0] for r in self._db.execute("SELECT id FROM images")]
        orphans = [(i,) for i in rows if i not in indexed]
        if orphans:
            self._db.executemany("DELETE FROM images WHERE id = ?", orphans)
            self._db.commit()

    # --- reads -------------------------------------------------------------

    @property
    def ntotal(self) -> int:
        return int(self._index.ntotal)

    @property
    def index_type(self) -> str:
        base = faiss.downcast_index(self._index.index)
        return "hnsw" if isinstance(base, faiss.IndexHNSWFlat) else "flat"

    def known_paths(self) -> set[str]:
        return {r[0] for r in self._db.execute("SELECT path FROM images")}

    def find_by_sha(self, sha256: str) -> ImageRecord | None:
        row = self._db.execute(f"{_SELECT} WHERE sha256 = ? ORDER BY id LIMIT 1", (sha256,)).fetchone()
        return _record(row) if row else None

    def get(self, image_id: int) -> ImageRecord | None:
        return self.get_many([image_id]).get(image_id)

    def get_many(self, ids: list[int]) -> dict[int, ImageRecord]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self._db.execute(f"{_SELECT} WHERE id IN ({marks})", ids).fetchall()
        return {row[0]: _record(row) for row in rows}

    def matching_ids(self, flt: SearchFilter) -> np.ndarray:
        where, params = flt.where()
        rows = self._db.execute(f"SELECT id FROM images WHERE {where}", params).fetchall()
        return np.fromiter((r[0] for r in rows), dtype="int64", count=len(rows))

    def categories(self) -> list[dict]:
        rows = self._db.execute(
            "SELECT category, subcategory, COUNT(*) FROM images WHERE category IS NOT NULL "
            "GROUP BY category, subcategory ORDER BY category, subcategory"
        ).fetchall()
        result: dict[str, dict] = {}
        for category, subcategory, count in rows:
            entry = result.setdefault(category, {"name": category, "count": 0, "subcategories": []})
            entry["count"] += count
            if subcategory:
                entry["subcategories"].append({"name": subcategory, "count": count})
        return list(result.values())

    def price_range(self) -> tuple[float | None, float | None]:
        return self._db.execute("SELECT MIN(price_usd), MAX(price_usd) FROM images").fetchone()

    def all_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            ids = faiss.vector_to_array(self._index.id_map).astype("int64")
            if not len(ids):
                return ids, np.zeros((0, self.dim), dtype="float32")
            return ids, np.vstack([self._index.reconstruct(int(i)) for i in ids])

    def search(
        self, vector: np.ndarray, k: int, flt: SearchFilter | None = None
    ) -> list[tuple[ImageRecord, float]]:
        params = None
        limit = self.ntotal
        if flt is not None and not flt.is_empty():
            allowed = self.matching_ids(flt)
            if not len(allowed):
                return []
            selector = faiss.IDSelectorBatch(allowed)
            params = (
                faiss.SearchParametersHNSW(sel=selector, efSearch=max(HNSW_EF_SEARCH, k))
                if self.index_type == "hnsw"
                else faiss.SearchParameters(sel=selector)
            )
            limit = len(allowed)
        with self._lock:
            if self.ntotal == 0:
                return []
            scores, ids = self._index.search(
                vector.reshape(1, -1).astype("float32"), min(k, limit, self.ntotal), params=params
            )
        pairs = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
        records = self.get_many([i for i, _ in pairs])
        return [(records[i], s) for i, s in pairs if i in records]

    # --- writes ------------------------------------------------------------

    def add(self, items: list[NewImage], vectors: np.ndarray) -> list[int]:
        """Insert rows and vectors; returns the new ids in the same order."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock:
            cur = self._db.cursor()
            try:
                ids = []
                for it in items:
                    cur.execute(
                        "INSERT INTO images (path, sha256, width, height, source, created_at, "
                        "product_id, name, brand, category, subcategory, price_usd) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (it.path, it.sha256, it.width, it.height, it.source, now, *astuple(it.product)),
                    )
                    ids.append(cur.lastrowid)
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
            self._index.add_with_ids(vectors.astype("float32"), np.asarray(ids, dtype="int64"))
            self._dirty = True
        return ids

    def update_products(self, catalog: dict[str, Product]) -> int:
        """Apply catalogue metadata to already-indexed images (keyed by stored path)."""
        with self._lock:
            cur = self._db.executemany(
                "UPDATE images SET product_id = ?, name = ?, brand = ?, category = ?, "
                "subcategory = ?, price_usd = ? WHERE path = ?",
                [(*astuple(p), path) for path, p in catalog.items()],
            )
            self._db.commit()
            return cur.rowcount

    def rebuild(self, kind: str) -> None:
        """Rebuild the index as `kind` ("flat" or "hnsw") from the vectors it already holds."""
        with self._lock:
            ids, vectors = self.all_vectors()
            index = _new_index(kind, self.dim)
            if len(ids):
                index.add_with_ids(vectors, ids)
            self._index = index
            self._dirty = True

    def save(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            tmp = self.index_path.with_suffix(".faiss.tmp")
            faiss.write_index(self._index, str(tmp))
            os.replace(tmp, self.index_path)
            self._dirty = False

    def close(self) -> None:
        self.save()
        self._db.close()
