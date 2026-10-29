from __future__ import annotations

from pydantic import BaseModel, Field


class SearchJSONBody(BaseModel):
    image_base64: str | None = Field(None, description="Base64 image; a `data:image/...;base64,` prefix is allowed.")
    image_url: str | None = Field(None, description="Public http(s) URL of an image.")
    top_k: int | None = None
    category: str | None = None
    subcategory: str | None = None
    brand: str | None = None
    min_price: float | None = Field(None, ge=0)
    max_price: float | None = Field(None, ge=0)


class SearchResult(BaseModel):
    rank: int
    id: int
    score: float = Field(description="Cosine similarity, 1.0 = identical.")
    product_id: int | None
    name: str | None
    brand: str | None
    category: str | None
    subcategory: str | None
    price_usd: float | None
    filename: str
    image_url: str
    thumbnail_url: str


class AppliedFilters(BaseModel):
    category: str | None = None
    subcategory: str | None = None
    brand: str | None = None
    min_price: float | None = None
    max_price: float | None = None


class SearchResponse(BaseModel):
    query_id: str
    took_ms: float
    model: str
    index_type: str
    total_indexed: int
    top_k: int
    filters: AppliedFilters
    results: list[SearchResult]


class AddImageResponse(BaseModel):
    id: int
    filename: str
    duplicate: bool
    total_indexed: int
    image_url: str


class Subcategory(BaseModel):
    name: str
    count: int


class Category(BaseModel):
    name: str
    count: int
    subcategories: list[Subcategory]


class PriceRange(BaseModel):
    min: float | None
    max: float | None


class CategoriesResponse(BaseModel):
    categories: list[Category]
    price_range: PriceRange


class HealthResponse(BaseModel):
    status: str
    indexed: int
    index_type: str
    embedder: str
    model: str
    dim: int
