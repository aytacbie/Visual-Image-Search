"""Product catalogue (data/products.csv): metadata attached to each image."""
from __future__ import annotations

import csv
from dataclasses import dataclass

from app.config import Settings


@dataclass(frozen=True)
class Product:
    product_id: int | None = None
    name: str | None = None
    brand: str | None = None
    category: str | None = None
    subcategory: str | None = None
    price_usd: float | None = None


def _num(value: str, cast):
    value = (value or "").strip()
    return cast(value) if value else None


def load_catalog(settings: Settings) -> dict[str, Product]:
    """Map stored image path -> Product. Empty if there is no products.csv."""
    if not settings.catalog_path.exists():
        return {}
    catalog = {}
    with settings.catalog_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            path = settings.to_stored_path(settings.data_dir / row["image_file"])
            catalog[path] = Product(
                product_id=_num(row.get("id"), int),
                name=(row.get("name") or "").strip() or None,
                brand=(row.get("brand") or "").strip() or None,
                category=(row.get("category") or "").strip() or None,
                subcategory=(row.get("subcategory") or "").strip() or None,
                price_usd=_num(row.get("price_usd"), float),
            )
    return catalog
