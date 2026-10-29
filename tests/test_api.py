"""End-to-end API tests on a throwaway data dir with a handful of generated images."""
import base64
import csv
import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.auth import KeyStore, RateLimiter, hash_key  # noqa: E402
from app.config import Settings  # noqa: E402
from app.embedder import OnnxClipEmbedder  # noqa: E402
from app.index import VectorStore  # noqa: E402
from app.ingest import ingest_folder  # noqa: E402
from app.main import create_app  # noqa: E402

ENV_KEY = "vs_from_environment"
PRODUCTS = [
    # name, category, subcategory, price
    ("red_circle", "Shapes", "Round", 5.0),
    ("blue_circle", "Shapes", "Round", 50.0),
    ("red_square", "Shapes", "Angular", 15.0),
    ("blue_square", "Shapes", "Angular", 150.0),
    ("red_stripes", "Patterns", "Lines", 25.0),
    ("blue_stripes", "Patterns", "Lines", 250.0),
]


def make_image(kind: str, color: tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGB", (256, 256), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    if kind == "circle":
        draw.ellipse((40, 40, 216, 216), fill=color)
    elif kind == "square":
        draw.rectangle((40, 40, 216, 216), fill=color)
    elif kind == "stripes":
        for x in range(0, 256, 32):
            draw.rectangle((x, 0, x + 15, 255), fill=color)
    return img


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    settings = Settings(
        data_dir=data_dir, max_upload_mb=1, rate_limit_per_minute=1000, web_rate_limit_per_minute=1000,
        api_keys_env=f"ci:web:{hash_key(ENV_KEY)}",
    )
    settings.images_dir.mkdir(parents=True)
    colors = {"red": (220, 30, 30), "blue": (30, 60, 220)}
    with settings.catalog_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name", "brand", "category", "subcategory", "price_usd", "image_file"])
        for i, (name, category, subcategory, price) in enumerate(PRODUCTS, start=1):
            color, kind = name.split("_")
            make_image(kind, colors[color]).save(settings.images_dir / f"{name}.png")
            writer.writerow([i, name.replace("_", " ").title(), "Acme", category, subcategory, price, f"images/{name}.png"])
    (settings.images_dir / "broken.jpg").write_bytes(b"not an image")

    embedder = OnnxClipEmbedder(settings.model_repo, settings.model_file)
    store = VectorStore(settings.index_path, settings.db_path, embedder.dim, embedder.model_id)
    stats = ingest_folder(settings, embedder, store, progress=False)
    store.close()
    assert (stats.added, stats.errors, stats.with_metadata) == (6, 1, 6)

    keys = KeyStore(settings.keys_path)
    full, web = keys.create("tests"), keys.create("site", "web")
    with TestClient(create_app(settings, embedder)) as client:
        yield settings, client, full, web


def search(client, key, image: bytes, params: str = ""):
    return client.post(f"/api/v1/search{params}", files={"file": ("q.png", image)}, headers={"X-API-Key": key})


def test_health_is_public(env):
    _, client, _, _ = env
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok" and body["indexed"] == 6 and body["embedder"] == "onnx"


def test_search_without_or_with_wrong_key_is_401(env):
    settings, client, _, _ = env
    data = (settings.images_dir / "red_circle.png").read_bytes()
    resp = client.post("/api/v1/search", files={"file": ("q.png", data)})
    assert resp.status_code == 401 and "Missing API key" in resp.json()["detail"]
    assert search(client, "vs_wrong", data).status_code == 401


def test_exact_image_ranks_first_with_product_metadata(env):
    settings, client, full, _ = env
    resp = search(client, full, (settings.images_dir / "blue_square.png").read_bytes(), "?top_k=3")
    assert resp.status_code == 200
    top = resp.json()["results"][0]
    assert top["filename"] == "blue_square.png" and top["score"] > 0.99
    assert (top["name"], top["brand"], top["category"], top["subcategory"], top["price_usd"]) == (
        "Blue Square", "Acme", "Shapes", "Angular", 150.0,
    )
    assert top["image_url"].startswith("http") and top["thumbnail_url"].startswith("http")


def test_filters(env):
    settings, client, full, _ = env
    query = (settings.images_dir / "red_circle.png").read_bytes()

    body = search(client, full, query, "?top_k=10&category=patterns").json()
    assert {r["category"] for r in body["results"]} == {"Patterns"} and len(body["results"]) == 2
    assert body["filters"]["category"] == "patterns"

    body = search(client, full, query, "?top_k=10&max_price=20").json()
    assert {r["filename"] for r in body["results"]} == {"red_circle.png", "red_square.png"}

    body = search(client, full, query, "?top_k=10&category=Shapes&min_price=40&max_price=200").json()
    assert {r["filename"] for r in body["results"]} == {"blue_circle.png", "blue_square.png"}

    assert search(client, full, query, "?category=Nope").json()["results"] == []
    assert search(client, full, query, "?min_price=10&max_price=5").status_code == 422

    # Filters in the multipart form and JSON body work too.
    resp = client.post("/api/v1/search", files={"file": ("q.png", query)}, data={"subcategory": "Lines"}, headers={"X-API-Key": full})
    assert {r["subcategory"] for r in resp.json()["results"]} == {"Lines"}
    encoded = base64.b64encode(query).decode()
    resp = client.post("/api/v1/search", json={"image_base64": encoded, "max_price": 5}, headers={"X-API-Key": full})
    assert [r["filename"] for r in resp.json()["results"]] == ["red_circle.png"]


def test_categories(env):
    _, client, _, web = env
    assert client.get("/api/v1/categories").status_code == 401
    body = client.get("/api/v1/categories", headers={"X-API-Key": web}).json()
    assert {c["name"]: c["count"] for c in body["categories"]} == {"Patterns": 2, "Shapes": 4}
    assert body["price_range"] == {"min": 5.0, "max": 250.0}


def test_json_base64_and_raw_body(env):
    settings, client, full, _ = env
    data = (settings.images_dir / "red_stripes.png").read_bytes()
    encoded = "data:image/png;base64," + base64.b64encode(data).decode()
    resp = client.post("/api/v1/search", json={"image_base64": encoded, "top_k": 1}, headers={"X-API-Key": full})
    assert resp.status_code == 200 and resp.json()["results"][0]["filename"] == "red_stripes.png"

    resp = client.post("/api/v1/search", content=data, headers={"X-API-Key": full, "Content-Type": "image/png"})
    assert resp.status_code == 200 and resp.json()["results"][0]["filename"] == "red_stripes.png"


def test_bad_inputs(env):
    _, client, full, _ = env
    auth = {"X-API-Key": full}
    assert client.post("/api/v1/search", files={"file": ("a.txt", b"hello")}, headers=auth).status_code == 400
    assert client.post("/api/v1/search", json={"image_base64": "!!!"}, headers=auth).status_code == 400
    assert client.post("/api/v1/search", json={}, headers=auth).status_code == 400
    assert client.post("/api/v1/search", content=b"x", headers={**auth, "Content-Type": "text/plain"}).status_code == 415
    big = b"\xff" * (2 * 1024 * 1024)
    assert client.post("/api/v1/search", files={"file": ("big.jpg", big)}, headers=auth).status_code == 413
    png = png_bytes(make_image("circle", (0, 0, 0)))
    assert search(client, full, png, "?top_k=0").status_code == 422
    assert search(client, full, png, "?top_k=999").status_code == 422


def test_image_url_cannot_reach_internal_hosts(env):
    _, client, full, _ = env
    for url in ("http://127.0.0.1:8000/x.jpg", "http://localhost/x.jpg", "file:///etc/passwd"):
        resp = client.post("/api/v1/search", json={"image_url": url}, headers={"X-API-Key": full})
        assert resp.status_code == 400, url


def test_web_key_is_search_only(env):
    settings, client, _, web = env
    data = (settings.images_dir / "red_circle.png").read_bytes()
    assert search(client, web, data).status_code == 200
    resp = client.post("/api/v1/images", files={"file": ("g.png", data)}, headers={"X-API-Key": web})
    assert resp.status_code == 403


def test_search_scope_key_is_search_only(env):
    settings, client, _, _ = env
    key = KeyStore(settings.keys_path).create("proxy", "search")
    data = (settings.images_dir / "red_circle.png").read_bytes()
    assert search(client, key, data).status_code == 200
    assert client.post("/api/v1/images", files={"file": ("g.png", data)}, headers={"X-API-Key": key}).status_code == 403


def test_builtin_page_searches_without_key_same_origin_only(env):
    settings, client, _, _ = env
    data = (settings.images_dir / "red_circle.png").read_bytes()
    files = {"file": ("q.png", data)}
    resp = client.post("/web/search?top_k=2", files=files, headers={"Origin": "http://testserver"})
    assert resp.status_code == 200 and resp.json()["results"][0]["filename"] == "red_circle.png"
    assert client.post("/web/search", files=files, headers={"Referer": "http://testserver/"}).status_code == 200
    assert client.post("/web/search", files=files).status_code == 403
    assert client.post("/web/search", files=files, headers={"Origin": "https://evil.example.com"}).status_code == 403
    assert "/web/search" not in client.get("/openapi.json").json()["paths"]


def test_keys_from_environment(env):
    settings, client, _, _ = env
    data = (settings.images_dir / "red_circle.png").read_bytes()
    assert search(client, ENV_KEY, data).status_code == 200
    assert client.post("/api/v1/images", files={"file": ("g.png", data)}, headers={"X-API-Key": ENV_KEY}).status_code == 403


def test_rate_limiter():
    limiter = RateLimiter(per_minute=2)
    assert limiter.hit("a") is None and limiter.hit("a") is None
    assert limiter.hit("a") > 0
    assert limiter.hit("b") is None


def test_cors_allows_lovable_only(env):
    _, client, _, _ = env
    preflight = {"Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "x-api-key"}
    for origin in ("https://my-shop.lovable.app", "https://id-preview--abc123.lovableproject.com", "http://localhost:5173"):
        resp = client.options("/api/v1/search", headers={**preflight, "Origin": origin})
        assert resp.status_code == 200 and resp.headers["access-control-allow-origin"] == origin, origin
    resp = client.options("/api/v1/search", headers={**preflight, "Origin": "https://evil.example.com"})
    assert "access-control-allow-origin" not in resp.headers
    resp = client.options("/api/v1/search", headers={**preflight, "Origin": "https://lovable.app.evil.com"})
    assert "access-control-allow-origin" not in resp.headers


def test_add_image_then_find_it(env):
    _, client, full, _ = env
    auth = {"X-API-Key": full}
    data = png_bytes(make_image("circle", (30, 180, 60)))
    assert client.post("/api/v1/images", files={"file": ("g.png", data)}).status_code == 401

    resp = client.post("/api/v1/images", files={"file": ("g.png", data)}, headers=auth)
    assert resp.status_code == 201
    new_id = resp.json()["id"]

    again = client.post("/api/v1/images", files={"file": ("g.png", data)}, headers=auth)
    assert again.status_code == 200 and again.json()["duplicate"] and again.json()["id"] == new_id

    top = search(client, full, data, "?top_k=1").json()
    assert top["results"][0]["id"] == new_id and top["results"][0]["name"] is None

    img = client.get(f"/api/v1/images/{new_id}")
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    assert client.get("/api/v1/images/999999").status_code == 404


def test_concurrent_image_requests_all_succeed(env):
    """Regression: a shared sqlite connection used without a lock returned spurious 404s."""
    from concurrent.futures import ThreadPoolExecutor

    _, client, _, _ = env
    ids = list(range(1, 7)) * 15
    with ThreadPoolExecutor(12) as pool:
        codes = list(pool.map(lambda i: client.get(f"/api/v1/images/{i}?size=thumb").status_code, ids))
    assert codes == [200] * len(ids)


def test_website_is_served(env):
    _, client, _, _ = env
    resp = client.get("/")
    assert resp.status_code == 200 and "Visual Search" in resp.text
