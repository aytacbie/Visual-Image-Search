"""Benchmarks.

  api    latency of the running API end to end:
         python scripts/benchmark.py api --url http://localhost:8000 --key vs_... --image data/images/x.jpg
  scale  what the same index code does at 500k vectors (synthetic, no images needed):
         python scripts/benchmark.py scale --n 500000
  accuracy  leave-one-out on the indexed catalogue: does the nearest *other* product
         share the query's subcategory / category? (needs products.csv)
         python scripts/benchmark.py accuracy
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p))


def bench_api(args) -> None:
    import httpx

    data = Path(args.image).read_bytes()
    times, server_times = [], []
    with httpx.Client(timeout=30) as client:
        for i in range(args.n + 1):
            started = time.perf_counter()
            resp = client.post(
                f"{args.url.rstrip('/')}/api/v1/search",
                headers={"X-API-Key": args.key},
                files={"file": (Path(args.image).name, data)},
            )
            resp.raise_for_status()
            if i == 0:
                continue  # warm-up
            times.append((time.perf_counter() - started) * 1000)
            server_times.append(resp.json()["took_ms"])
    print(f"{args.n} requests to {args.url}")
    print(f"  end-to-end  p50 {percentile(times, 50):.0f} ms   p95 {percentile(times, 95):.0f} ms")
    print(f"  server-side p50 {percentile(server_times, 50):.0f} ms   p95 {percentile(server_times, 95):.0f} ms")


def bench_scale(args) -> None:
    import faiss

    from app.index import _new_index

    rng = np.random.default_rng(0)
    dim, n, q = 512, args.n, args.queries

    def unit(x: np.ndarray, norm: float) -> np.ndarray:
        """Random directions scaled to a given length (noise of length `norm` in total)."""
        return x / np.linalg.norm(x, axis=1, keepdims=True) * norm

    # Real embeddings are clustered (product types); uniform random vectors are not,
    # and would make every neighbour equally far. So: product-type centres + variation.
    clusters = max(1, n // 250)
    print(f"Generating {n:,} clustered {dim}-d unit vectors in {clusters:,} clusters ({n * dim * 4 / 1e9:.2f} GB) ...")
    centres = unit(rng.standard_normal((clusters, dim), dtype=np.float32), 1.0)
    base = centres[rng.integers(0, clusters, n)] + unit(rng.standard_normal((n, dim), dtype=np.float32), 0.8)
    base = unit(base, 1.0)
    # Queries are slightly changed copies of stored vectors, like a re-photographed product.
    sources = rng.choice(n, q, replace=False)
    queries = unit(base[sources] + unit(rng.standard_normal((q, dim), dtype=np.float32), 0.25), 1.0)
    ids = np.arange(n, dtype=np.int64)

    results = {}
    for kind in ("flat", "hnsw"):
        index = _new_index(kind, dim)
        started = time.perf_counter()
        index.add_with_ids(base, ids)
        build_s = time.perf_counter() - started
        faiss.omp_set_num_threads(1)  # measure single-query latency, as served
        lat = []
        found = []
        for vec in queries:
            started = time.perf_counter()
            _, idx = index.search(vec.reshape(1, -1), 10)
            lat.append((time.perf_counter() - started) * 1000)
            found.append(idx[0])
        faiss.omp_set_num_threads(0)
        results[kind] = np.array(found)
        print(f"{kind:>5}: build {build_s:6.1f}s | query p50 {statistics.median(lat):6.2f} ms, p95 {percentile(lat, 95):6.2f} ms")

    recall = np.mean([len(set(h) & set(f)) / 10 for h, f in zip(results["hnsw"], results["flat"])])
    print(f"HNSW recall@10 vs exact search: {recall:.3f}")
    for kind in ("flat", "hnsw"):
        print(f"{kind:>5}: original item ranked #1 for {np.mean(results[kind][:, 0] == sources):.1%} of queries")


def bench_accuracy(args) -> None:
    import dataclasses

    from app.config import Settings
    from app.embedder import create_embedder
    from app.index import VectorStore

    settings = Settings.from_env()
    emb = create_embedder(settings)
    settings = dataclasses.replace(settings, embedder=emb.backend)
    if not settings.index_path.exists():
        sys.exit(f"No index at {settings.index_path}; run scripts/ingest.py first.")
    store = VectorStore(settings.index_path, settings.db_path, emb.dim, emb.model_id)
    ids, vectors = store.all_vectors()
    records = store.get_many(ids.tolist())
    labelled = [i for i, rid in enumerate(ids) if records[int(rid)].product.subcategory]
    if not labelled:
        sys.exit("No products.csv metadata in the index.")

    sims = vectors @ vectors.T
    k = args.k
    hits = {"sub@1": 0, "cat@1": 0, f"sub@{k}": 0.0}
    misses = []
    for i in labelled:
        me = records[int(ids[i])]
        # Exclude the query itself and byte-identical copies of it.
        order = [j for j in np.argsort(-sims[i]) if records[int(ids[j])].sha256 != me.sha256][:k]
        neighbours = [records[int(ids[j])].product for j in order]
        hits["sub@1"] += neighbours[0].subcategory == me.product.subcategory
        hits["cat@1"] += neighbours[0].category == me.product.category
        hits[f"sub@{k}"] += sum(n.subcategory == me.product.subcategory for n in neighbours) / k
        if neighbours[0].subcategory != me.product.subcategory:
            misses.append(f"{me.product.name} ({me.product.subcategory}) -> {neighbours[0].name} ({neighbours[0].subcategory})")

    n = len(labelled)
    print(f"{n} products, embedder {emb.model_id}, leave-one-out")
    print(f"  nearest neighbour same subcategory : {hits['sub@1'] / n:.1%}")
    print(f"  nearest neighbour same category    : {hits['cat@1'] / n:.1%}")
    print(f"  precision@{k} (subcategory)          : {hits[f'sub@{k}'] / n:.1%}")
    if misses and args.show_misses:
        print("\nTop-1 subcategory misses:")
        print("\n".join(f"  {m}" for m in misses))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    api = sub.add_parser("api")
    api.add_argument("--url", default="http://localhost:8000")
    api.add_argument("--key", required=True)
    api.add_argument("--image", required=True)
    api.add_argument("-n", type=int, default=30)
    scale = sub.add_parser("scale")
    scale.add_argument("--n", type=int, default=500_000)
    scale.add_argument("--queries", type=int, default=200)
    accuracy = sub.add_parser("accuracy")
    accuracy.add_argument("-k", type=int, default=5)
    accuracy.add_argument("--show-misses", action="store_true")
    args = parser.parse_args()
    {"api": bench_api, "scale": bench_scale, "accuracy": bench_accuracy}[args.cmd](args)


if __name__ == "__main__":
    main()
