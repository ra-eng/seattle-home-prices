"""Compare /predict latency before and after preloading, and load-test a running API.

Usage:
    uv run python scripts/benchmark.py
    uv run python scripts/benchmark.py --load http://localhost:8000 --requests 500 --concurrency 20
"""
import argparse
import asyncio
import json
import pickle
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from api.resources import load_resources

HOUSE_FEATURES = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
                  "sqft_above", "sqft_basement"]


def load_examples(missing_rate: float) -> list[dict]:
    """Load the unseen examples, blanking a fraction of house features at random."""
    examples = pd.read_csv(SRC_DIR / "data" / "future_unseen_examples.csv", dtype={"zipcode": str})
    examples = examples[HOUSE_FEATURES + ["zipcode"]].astype(object)
    mask = np.random.default_rng(42).random((len(examples), len(HOUSE_FEATURES))) < missing_rate
    examples[HOUSE_FEATURES] = examples[HOUSE_FEATURES].mask(mask, None)
    return examples.to_dict(orient="records")


def predict_reloading(example: dict) -> float:
    """Previous implementation: reload every artifact on each request."""
    with open(SRC_DIR / "model" / "model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open(SRC_DIR / "model" / "model_features.json") as features_file:
        features = json.load(features_file)
    demographics = pd.read_csv(SRC_DIR / "data" / "zipcode_demographics.csv", dtype={"zipcode": str})
    demographic_info = demographics[demographics["zipcode"] == example["zipcode"]]
    input_data = pd.concat(
        [pd.DataFrame([example]), demographic_info.reset_index(drop=True)], axis=1
    )[features].astype(float)
    return model.predict(input_data)[0]


def report(name: str, latencies_ms: list[float]) -> None:
    p50, p95 = np.percentile(latencies_ms, [50, 95])
    print(f"{name:<32} p50={p50:7.2f} ms  p95={p95:7.2f} ms")


def benchmark_in_process() -> None:
    """Time each prediction path with perf_counter on complete and incomplete inputs."""
    resources = load_resources(SRC_DIR / "model", SRC_DIR / "data")

    def predict_preloaded(example: dict) -> float:
        house = {name: example[name] for name in HOUSE_FEATURES}
        return resources.model.predict(resources.build_row(example["zipcode"], house))[0]

    paths = (("reload per request", predict_reloading, None),
             ("preloaded, 1 thread", predict_preloaded, 1))
    for missing_rate in (0.0, 0.3):
        examples = load_examples(missing_rate)
        for name, predict, thread_limit in paths:
            latencies = []
            with threadpool_limits(limits=thread_limit):
                for example in examples:
                    start = time.perf_counter()
                    predict(example)
                    latencies.append((time.perf_counter() - start) * 1000)
            report(f"{name} ({missing_rate:.0%} missing)", latencies)


async def load_test(base_url: str, total: int, concurrency: int) -> None:
    """Send concurrent requests to a running API and report throughput and latency."""
    examples = load_examples(0.3)
    semaphore = asyncio.Semaphore(concurrency)
    latencies = []

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        async def send(example: dict) -> None:
            async with semaphore:
                start = time.perf_counter()
                response = await client.post("/predict", json=example)
                response.raise_for_status()
                latencies.append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        await asyncio.gather(*(send(examples[i % len(examples)]) for i in range(total)))
        elapsed = time.perf_counter() - start

    report(f"HTTP x{concurrency} concurrent", latencies)
    print(f"throughput: {total / elapsed:.1f} req/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--load", metavar="URL", help="load-test a running API instead")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()

    if args.load:
        asyncio.run(load_test(args.load, args.requests, args.concurrency))
    else:
        benchmark_in_process()
