"""Visual Search API.

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000 --proxy-headers
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import base64
import binascii
import dataclasses
import logging
import math
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request, Response, Security
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from app.auth import KeyStore, Principal, RateLimiter
from app.catalog import load_catalog
from app.config import Settings
from app.embedder import BaseEmbedder, create_embedder
from app.errors import APIError, BadRequestError, PayloadTooLargeError, UnsupportedMediaTypeError
from app.fetch import fetch_image
from app.images import decode_image, detect_extension, save_thumbnail, sha256_hex
from app.index import NewImage, SearchFilter, VectorStore
from app.ingest import has_images, ingest_folder
from app.schemas import (
    AddImageResponse, AppliedFilters, CategoriesResponse, HealthResponse, SearchJSONBody,
    SearchResponse, SearchResult,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("visual_search")

STATIC_DIR = Path(__file__).parent / "static"
FILTER_FIELDS = ("category", "subcategory", "brand", "min_price", "max_price")

api_key_header = APIKeyHeader(
    name="X-API-Key", auto_error=False, description="Your API key, e.g. `vs_...`."
)

_ERROR_RESPONSES = {
    400: {"description": "Not a valid image / bad input", "content": {"application/json": {"example": {"detail": "File is not a valid image."}}}},
    401: {"description": "Missing or invalid API key", "content": {"application/json": {"example": {"detail": "Invalid API key."}}}},
    403: {"description": "Key is not allowed to do this (web keys are search-only)"},
    413: {"description": "Image too large"},
    415: {"description": "Unsupported Content-Type"},
    422: {"description": "Invalid parameter (e.g. top_k out of range)"},
    429: {"description": "Rate limit exceeded"},
    503: {"description": "Embedding service unavailable (NV-CLIP mode only)"},
}


def _image_body_openapi(with_options: bool) -> dict:
    multipart = {"file": {"type": "string", "format": "binary", "description": "Image file"}}
    json_body = {
        "image_base64": {"type": "string", "description": "Base64 image (data: URL prefix allowed)"},
        "image_url": {"type": "string", "format": "uri", "description": "Public http(s) image URL"},
    }
    if with_options:
        options = {
            "top_k": {"type": "integer"},
            "category": {"type": "string"}, "subcategory": {"type": "string"}, "brand": {"type": "string"},
            "min_price": {"type": "number"}, "max_price": {"type": "number"},
        }
        multipart |= options
        json_body |= options
    return {"requestBody": {"required": True, "content": {
        "multipart/form-data": {"schema": {"type": "object", "required": ["file"], "properties": multipart}},
        "application/json": {
            "schema": {"type": "object", "properties": json_body},
            "example": {"image_url": "https://example.com/red-lipstick.jpg", "top_k": 5, "category": "Beauty & Personal Care"},
        },
        "image/*": {"schema": {"type": "string", "format": "binary"}},
    }}}


def client_ip(request: Request) -> str:
    # Behind Cloudflare the real client address is in CF-Connecting-IP; behind HF Spaces,
    # uvicorn --proxy-headers has already applied X-Forwarded-For to request.client.
    return request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "-")


def require(permission: str):
    """Dependency: valid X-API-Key whose scope allows `permission`, within its rate limit."""

    def dependency(request: Request, api_key: str | None = Security(api_key_header)) -> Principal:
        challenge = {"WWW-Authenticate": "ApiKey"}
        if not api_key:
            raise HTTPException(401, "Missing API key. Send it in the X-API-Key header.", headers=challenge)
        principal = request.app.state.keys.verify(api_key)
        if principal is None:
            logger.warning("rejected invalid API key from %s", client_ip(request))
            raise HTTPException(401, "Invalid API key.", headers=challenge)
        if not principal.can(permission):
            raise HTTPException(403, f"This API key ({principal.scope}) is not allowed to {permission}.")
        if principal.scope == "web":
            retry_after = request.app.state.web_limiter.hit(f"{principal.label}|{client_ip(request)}")
        else:
            retry_after = request.app.state.limiter.hit(principal.label)
        if retry_after is not None:
            raise HTTPException(429, "Rate limit exceeded, slow down.", headers={"Retry-After": str(math.ceil(retry_after))})
        return principal

    return dependency


def create_app(settings: Settings | None = None, embedder: BaseEmbedder | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    max_bytes = settings.max_upload_bytes
    too_large = f"Image is larger than {settings.max_upload_mb} MB."

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        emb = embedder or await run_in_threadpool(create_embedder, settings)
        # The embedder may have fallen back to ONNX; index paths follow the one in use.
        cfg = dataclasses.replace(settings, embedder=emb.backend)
        store = VectorStore(cfg.index_path, cfg.db_path, emb.dim, emb.model_id)
        if cfg.ingest_on_startup and store.ntotal == 0 and has_images(cfg.images_dir):
            logger.info("index is empty, ingesting %s ...", cfg.images_dir)
            await run_in_threadpool(ingest_folder, cfg, emb, store, progress=False)
        else:
            store.update_products(load_catalog(cfg))
        app.state.settings = cfg
        app.state.embedder = emb
        app.state.store = store
        app.state.keys = KeyStore(cfg.keys_path, cfg.api_keys_env)
        app.state.limiter = RateLimiter(cfg.rate_limit_per_minute)
        app.state.web_limiter = RateLimiter(cfg.web_rate_limit_per_minute)
        logger.info("ready: %d images indexed (%s, %s)", store.ntotal, store.index_type, emb.model_id)
        yield
        store.close()

    app = FastAPI(
        title="Visual Search API",
        version="1.1.0",
        description=(
            "Find the most visually similar products in the catalogue.\n\n"
            "Search endpoints need an `X-API-Key` header; requests without a valid key get **401**. "
            "`web` keys can only search; `full` keys can also add images."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_origin_regex=settings.cors_origin_regex,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["X-API-Key", "Content-Type"],
        expose_headers=["X-Request-ID", "Retry-After"],
        max_age=600,
    )

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request.state.request_id
        if request.url.path.startswith("/api/"):
            logger.info("%s %s -> %d %.0fms ip=%s", request.method, request.url.path, response.status_code, ms, client_ip(request))
        return response

    @app.exception_handler(APIError)
    async def api_error_handler(_: Request, exc: APIError):
        return JSONResponse({"detail": exc.message}, status_code=exc.status_code)

    def check_size(data: bytes) -> bytes:
        if len(data) > max_bytes:
            raise PayloadTooLargeError(too_large)
        return data

    def form_options(form) -> dict:
        options = {}
        for name, cast in (("top_k", int), ("min_price", float), ("max_price", float)):
            value = form.get(name)
            if value not in (None, ""):
                try:
                    options[name] = cast(value)
                except (TypeError, ValueError):
                    raise HTTPException(422, f"{name} must be a number.")
        for name in ("category", "subcategory", "brand"):
            if form.get(name):
                options[name] = str(form.get(name))
        return options

    async def read_image_payload(request: Request) -> tuple[bytes, dict]:
        """Accept multipart (`file`), JSON (`image_base64` or `image_url`) or a raw image/* body.
        Returns the image bytes and any search options sent in the body."""
        declared = request.headers.get("content-length", "")
        # base64 inflates by 4/3; allow some room for multipart/JSON framing.
        if declared.isdigit() and int(declared) > max_bytes * 4 // 3 + 64 * 1024:
            raise PayloadTooLargeError(too_large)
        content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()

        if content_type == "multipart/form-data":
            async with request.form(max_files=1) as form:
                upload = form.get("file")
                if upload is None or isinstance(upload, str):
                    raise BadRequestError("Multipart body must contain the image in a 'file' field.")
                return check_size(await upload.read(max_bytes + 1)), form_options(form)

        if content_type == "application/json":
            try:
                body = SearchJSONBody.model_validate_json(await request.body())
            except ValidationError as exc:
                raise HTTPException(422, exc.errors(include_url=False, include_context=False))
            if bool(body.image_base64) == bool(body.image_url):
                raise BadRequestError("Send exactly one of 'image_base64' or 'image_url'.")
            options = body.model_dump(exclude={"image_base64", "image_url"}, exclude_none=True)
            if body.image_url:
                return await fetch_image(body.image_url, max_bytes), options
            encoded = body.image_base64
            if encoded.startswith("data:") and "," in encoded:
                encoded = encoded.split(",", 1)[1]
            encoded = "".join(encoded.split())
            try:
                return check_size(base64.b64decode(encoded, validate=True)), options
            except (binascii.Error, ValueError):
                raise BadRequestError("image_base64 is not valid base64.")

        if content_type.startswith("image/"):
            return check_size(await request.body()), {}

        raise UnsupportedMediaTypeError(
            "Send the image as multipart/form-data ('file' field), application/json "
            "('image_base64' or 'image_url'), or a raw image/* body."
        )

    def image_urls(request: Request, image_id: int) -> tuple[str, str]:
        root = request.app.state.settings.public_base_url or str(request.base_url).rstrip("/")
        base = f"{root}/api/v1/images/{image_id}"
        return f"{base}?size=full", f"{base}?size=thumb"

    # --- routes ----------------------------------------------------------------

    @app.get("/api/v1/health", response_model=HealthResponse, tags=["system"])
    def health(request: Request):
        store: VectorStore = request.app.state.store
        emb: BaseEmbedder = request.app.state.embedder
        return HealthResponse(
            status="ok", indexed=store.ntotal, index_type=store.index_type,
            embedder=emb.backend, model=emb.model_id, dim=store.dim,
        )

    @app.get(
        "/api/v1/categories", response_model=CategoriesResponse, tags=["search"],
        summary="Categories, subcategories and price range (for filter dropdowns)",
        responses={401: _ERROR_RESPONSES[401]},
    )
    def categories(request: Request, _: Principal = Depends(require("search"))):
        store: VectorStore = request.app.state.store
        low, high = store.price_range()
        return {"categories": store.categories(), "price_range": {"min": low, "max": high}}

    @app.post(
        "/api/v1/search", response_model=SearchResponse, tags=["search"],
        summary="Find products that look like the uploaded image",
        openapi_extra=_image_body_openapi(with_options=True), responses=_ERROR_RESPONSES,
    )
    async def search(
        request: Request,
        top_k: int | None = Query(None, ge=1, le=settings.max_top_k, description="Number of results"),
        category: str | None = Query(None, description="Exact category, case-insensitive"),
        subcategory: str | None = Query(None),
        brand: str | None = Query(None),
        min_price: float | None = Query(None, ge=0),
        max_price: float | None = Query(None, ge=0),
        principal: Principal = Depends(require("search")),
    ):
        started = time.perf_counter()
        data, body = await read_image_payload(request)
        query = {"category": category, "subcategory": subcategory, "brand": brand,
                 "min_price": min_price, "max_price": max_price}
        flt = SearchFilter(**{f: query[f] if query[f] is not None else body.get(f) for f in FILTER_FIELDS})
        k = top_k or body.get("top_k") or settings.default_top_k
        if not 1 <= k <= settings.max_top_k:
            raise HTTPException(422, f"top_k must be between 1 and {settings.max_top_k}.")
        if flt.min_price is not None and flt.max_price is not None and flt.min_price > flt.max_price:
            raise HTTPException(422, "min_price must not be greater than max_price.")

        store: VectorStore = request.app.state.store
        emb: BaseEmbedder = request.app.state.embedder
        img = await run_in_threadpool(decode_image, data)
        vector = (await run_in_threadpool(emb.embed, [img]))[0]
        hits = await run_in_threadpool(store.search, vector, k, flt)

        results = []
        for rank, (record, score) in enumerate(hits, start=1):
            full, thumb = image_urls(request, record.id)
            results.append(SearchResult(
                rank=rank, id=record.id, score=round(score, 4), **dataclasses.asdict(record.product),
                filename=Path(record.path).name, image_url=full, thumbnail_url=thumb,
            ))
        took_ms = round((time.perf_counter() - started) * 1000, 1)
        if results:
            logger.info("search by %s: top1=%s score=%.3f (%.0fms)", principal.label, results[0].filename, results[0].score, took_ms)
        return SearchResponse(
            query_id=request.state.request_id, took_ms=took_ms, model=emb.model_id,
            index_type=store.index_type, total_indexed=store.ntotal, top_k=k,
            filters=AppliedFilters(**dataclasses.asdict(flt)), results=results,
        )

    @app.post(
        "/api/v1/images", response_model=AddImageResponse, status_code=201, tags=["images"],
        summary="Add an image to the collection (200 if it is already there). Full keys only.",
        openapi_extra=_image_body_openapi(with_options=False), responses=_ERROR_RESPONSES,
    )
    async def add_image(
        request: Request, response: Response, background: BackgroundTasks,
        principal: Principal = Depends(require("write")),
    ):
        cfg: Settings = request.app.state.settings
        store: VectorStore = request.app.state.store
        data, _ = await read_image_payload(request)
        img = await run_in_threadpool(decode_image, data)
        sha = sha256_hex(data)

        if existing := store.find_by_sha(sha):
            response.status_code = 200
            return AddImageResponse(
                id=existing.id, filename=Path(existing.path).name, duplicate=True,
                total_indexed=store.ntotal, image_url=image_urls(request, existing.id)[0],
            )

        path = cfg.uploads_dir / f"{sha[:16]}{detect_extension(data)}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        vector = await run_in_threadpool(request.app.state.embedder.embed, [img])
        item = NewImage(cfg.to_stored_path(path), sha, img.width, img.height, f"api:{principal.label}")
        [image_id] = await run_in_threadpool(store.add, [item], vector)
        await run_in_threadpool(save_thumbnail, img, cfg.thumbs_dir / f"{image_id}.jpg")
        # Persisting the index file is slow on large collections, so it happens after the response.
        background.add_task(store.save)
        logger.info("image %d added by %s", image_id, principal.label)
        return AddImageResponse(
            id=image_id, filename=path.name, duplicate=False,
            total_indexed=store.ntotal, image_url=image_urls(request, image_id)[0],
        )

    @app.get(
        "/api/v1/images/{image_id}", tags=["images"], response_class=FileResponse,
        summary="Image file (thumbnail by default). Public, so it works in <img> tags.",
        responses={404: {"description": "No such image"}},
    )
    def get_image(request: Request, image_id: int, size: Literal["thumb", "full"] = "thumb"):
        cfg: Settings = request.app.state.settings
        record = request.app.state.store.get(image_id)
        if record is None:
            raise HTTPException(404, "Image not found.")
        full_path = cfg.resolve_stored_path(record.path)
        cache = {"Cache-Control": "public, max-age=86400"}
        if size == "thumb":
            thumb = cfg.thumbs_dir / f"{image_id}.jpg"
            if not thumb.exists() and full_path.exists():
                save_thumbnail(decode_image(full_path.read_bytes()), thumb)
            if thumb.exists():
                return FileResponse(thumb, media_type="image/jpeg", headers=cache)
        if not full_path.exists():
            raise HTTPException(404, "Image file is missing on the server.")
        return FileResponse(full_path, headers=cache)

    @app.get("/", include_in_schema=False)
    def index_page():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
