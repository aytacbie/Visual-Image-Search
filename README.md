# Visual Search

Upload a product photo, get the most visually similar products from the catalogue,
with name, brand, category and price. REST API with API-key auth (plus a minimal
fallback web page), built to scale from 194 to 500 000+ images.

## How it works

```
Lovable frontend / curl
   │  HTTPS, X-API-Key
FastAPI (Docker, on Render)
   ├─ auth: hashed API keys, scopes (full / web), per-key and per-visitor rate limits
   ├─ embedder: CLIP ViT-B/32 via ONNX Runtime on CPU (default)
   │            or NVIDIA NV-CLIP via build.nvidia.com (optional, key stays server-side)
   ├─ FAISS index, cosine similarity
   │    < 50k images: exact (IndexFlatIP)   ≥ 50k: HNSW (approximate, ~ms)
   │    filters (category, price, ...) applied inside FAISS via an id selector
   └─ SQLite: image + product metadata from data/products.csv
```

Every image becomes a 512-number "fingerprint"; similar images have nearby fingerprints.
A query is embedded the same way and FAISS returns the nearest ones.

**Scaling to 500k:** ingest runs in batches with a thread pool for decoding, skips files
already indexed (resumable after a crash) and checkpoints the index every 5 000 images.
Past 50 000 vectors the index is rebuilt as HNSW. Filters narrow the candidate set
inside FAISS, so a filtered query still returns the true top-k of the matching products.
`python scripts/benchmark.py scale --n 500000` (clustered synthetic vectors, laptop CPU):

| 500 000 × 512-d | query p50 | query p95 | recall@10 vs exact | original item #1 |
|---|---|---|---|---|
| exact (Flat) | 76 ms | 122 ms | 1.000 | 100% |
| HNSW | 0.6 ms | 1.1 ms | 0.994 | 99.5% |

HNSW build takes ~7 minutes once at 500k; embedding 500k images on CPU is the real cost
(~10 images/s with the int8 model, so hours, which is why ingest is resumable).

**Accuracy on the 194-product catalogue** (`python scripts/benchmark.py accuracy`, leave-one-out):
nearest other product is in the same subcategory 84% of the time, same category 92%.
Most misses are near-synonyms (men's vs women's watches, "Women's Tops" vs "Women's Dresses").

## API

Base URL: `https://<service>.onrender.com/api/v1` · interactive docs at `/docs`

| Method | Path | Key | |
|---|---|---|---|
| GET | `/health` | – | status, number of indexed images, embedder |
| GET | `/categories` | web / full | categories, subcategories, counts, price range |
| POST | `/search` | web / full | find similar products |
| POST | `/images` | full | add an image (201 new, 200 duplicate) |
| GET | `/images/{id}?size=thumb\|full` | – | image file (public, for `<img>` tags) |

`/search` takes the image as multipart `file`, JSON `{"image_base64"}` / `{"image_url"}`,
or a raw `image/*` body. Options (query string, form fields or JSON): `top_k` (1-50, default 5),
`category`, `subcategory`, `brand` (exact, case-insensitive), `min_price`, `max_price`.

```bash
# search (on Windows PowerShell write curl.exe)
curl -X POST "https://<service>.onrender.com/api/v1/search?top_k=5" \
  -H "X-API-Key: vs_..." -F "file=@photo.jpg"

# only Beauty products under $20
curl -X POST "https://<service>.onrender.com/api/v1/search?category=Beauty%20%26%20Personal%20Care&max_price=20" \
  -H "X-API-Key: vs_..." -F "file=@photo.jpg"

# without a key -> 401
curl -i -X POST "https://<service>.onrender.com/api/v1/search" -F "file=@photo.jpg"
```

Response:

```json
{
  "query_id": "3f9c1a2b7e41",
  "took_ms": 48.2,
  "model": "Xenova/clip-vit-base-patch32/onnx/vision_model.onnx",
  "index_type": "flat",
  "total_indexed": 194,
  "top_k": 5,
  "filters": {"category": null, "subcategory": null, "brand": null, "min_price": null, "max_price": 20.0},
  "results": [
    {"rank": 1, "id": 4, "score": 0.9412, "product_id": 4, "name": "Red Lipstick",
     "brand": "Chic Cosmetics", "category": "Beauty & Personal Care", "subcategory": "Makeup",
     "price_usd": 12.99, "filename": "004_red-lipstick.jpg",
     "image_url": "https://<service>.onrender.com/api/v1/images/4?size=full",
     "thumbnail_url": "https://<service>.onrender.com/api/v1/images/4?size=thumb"}
  ]
}
```

Errors are `{"detail": "..."}`: 400 bad image/input · 401 missing or invalid key ·
403 key scope not allowed · 413 over 10 MB · 415 wrong Content-Type · 422 bad parameter ·
429 rate limit (`Retry-After` header) · 503 NV-CLIP unavailable.

## API keys

| Scope | Can | Rate limit | Use |
|---|---|---|---|
| `full` | search, add images | 120/min per key | mentor's curl, admin |
| `search` | search only | 120/min per key | server-side proxy (Lovable edge function) |
| `web` | search only | 20/min per key **and** visitor IP | a key that has to live in browser code |

The Lovable frontend never sees a key: it calls a Lovable Cloud edge function, which holds
a `search` key as the secret `VISUAL_SEARCH_API_KEY` and forwards the upload to
`POST /api/v1/search`.

The built-in page at `/` also needs no key: it posts to `/web/search`, which only accepts
requests whose `Origin`/`Referer` is the same host and is limited to 20/min per visitor IP
(`VS_WEB_SEARCH=0` turns it off). `/api/v1/*` always requires a key.

```powershell
python scripts\create_api_key.py mentor                        # full
python scripts\create_api_key.py lovable-proxy --scope search  # for the edge function
python scripts\create_api_key.py --list
python scripts\create_api_key.py --revoke lovable-proxy
```

Locally keys live in `data/api_keys.json` (hashes only). When deployed, set the
`VS_API_KEYS` secret to the printed `label:scope:hash` entries joined with `;`.

CORS allows `*.lovable.app`, `*.lovableproject.com`, `*.lovable.dev`, `*.hf.space` and
localhost; add more with `VS_CORS_ORIGINS`.

## Embedders

Default is the local ONNX CLIP model (no external calls). To use NVIDIA NV-CLIP set
`VS_EMBEDDER=nvclip` and `NVIDIA_API_KEY=nvapi-...` (in `.env` locally, as secret environment variables
when deployed). If the key is missing or the API is unreachable at startup the server
falls back to ONNX. Each embedder keeps its own index under `data/index/<embedder>/`.

## Run locally (VS Code terminal)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\ingest.py            # data\images + data\products.csv -> data\index\onnx\
uvicorn app.main:app --port 8000 --reload
```

## Deploy (Render, free)

`Dockerfile` builds a self-contained image: dependencies, the int8 CLIP model (~200 MB RAM,
fits the 512 MB free instance) and the index are baked in, so the service starts ready.
`render.yaml` describes the service.

1. Push this repo to GitHub.
2. Render dashboard → **New → Blueprint** → pick the repo.
3. When asked, set `VS_API_KEYS` (the `label:scope:hash` lines from `create_api_key.py`, joined with `;`).
   Optional: `VS_EMBEDDER=nvclip` + `NVIDIA_API_KEY`.

Free instances sleep after 15 minutes without traffic; the first request after that takes
about a minute. The same image runs on any Docker host (it listens on `$PORT`, default 7860).

## Tests

```powershell
pytest -q
python scripts\benchmark.py accuracy --show-misses
python scripts\benchmark.py api --url http://localhost:8000 --key vs_... --image data\images\004_red-lipstick.jpg
python scripts\benchmark.py scale --n 500000
```
