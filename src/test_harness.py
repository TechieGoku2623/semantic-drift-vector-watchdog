"""Deterministic latency and edge-case harness for the drift watchdog."""

from __future__ import annotations

import asyncio
import math
import random
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SEED = 20261001
ITERATIONS = 5000


def _percentile_index(count: int) -> int:
    index = (99 * count + 99) // 100 - 1
    if index < 0:
        return 0
    if index >= count:
        return count - 1
    return index


def _unit(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


def _edge_zero_vector(module: object) -> str:
    engine_cls = module.SemanticDriftWatchdog
    error_cls = module.EngineKernelException

    async def _run() -> None:
        engine = engine_cls()
        try:
            await engine.score_window([[0.0] * engine_cls.DIM])
        except error_cls as exc:
            if "zero vector" not in str(exc):
                raise AssertionError("zero-vector fault was mislabeled") from exc
        else:
            raise AssertionError("zero vector was accepted")
        snapshot = await engine.stats()
        if snapshot["centroid_n"] != 0:
            raise AssertionError("zero vector updated the centroid")

    try:
        asyncio.run(_run())
    except Exception as exc:
        print(f"edge_zero_vector failed: {exc}", file=sys.stderr)
        return "FAIL"
    return "PASS"


def _edge_vocabulary_shift(module: object) -> str:
    engine_cls = module.SemanticDriftWatchdog

    async def _run() -> None:
        engine = engine_cls(cosine_distance_threshold=0.35, window=8)
        base = [0.0] * engine_cls.DIM
        base[0] = 1.0
        primed = await engine.score_window([base, list(base)])
        if primed["alert"] is not False:
            raise AssertionError("baseline window raised an alert")
        orthogonal = [0.0] * engine_cls.DIM
        orthogonal[1] = 1.0
        shifted = await engine.score_window([orthogonal])
        if shifted["alert"] is not True:
            raise AssertionError("orthogonal vocabulary shift was not alerted")
        if float(shifted["max_cosine_distance"]) <= 0.9:
            raise AssertionError("cosine distance did not show the planted shift")

    try:
        asyncio.run(_run())
    except Exception as exc:
        print(f"edge_vocabulary_shift failed: {exc}", file=sys.stderr)
        return "FAIL"
    return "PASS"


def _benchmark(module: object) -> tuple[int, float, float, int]:
    engine_cls = module.SemanticDriftWatchdog
    rng = random.Random(SEED)
    windows: list[list[list[float]]] = []
    for _ in range(ITERATIONS):
        raw = [1.0]
        raw.extend((rng.random() - 0.5) * 0.02 for _ in range(engine_cls.DIM - 1))
        windows.append([_unit(raw)])
    engine = engine_cls()

    async def _run() -> list[float]:
        samples: list[float] = []
        for window in windows:
            started = time.perf_counter_ns()
            await engine.score_window(window)
            elapsed_us = (time.perf_counter_ns() - started) / 1000.0
            samples.append(elapsed_us)
        return samples

    tracemalloc.start()
    try:
        samples = asyncio.run(_run())
    finally:
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    average = sum(samples) / len(samples)
    ordered = sorted(samples)
    p99 = ordered[_percentile_index(len(ordered))]
    return len(samples), average, p99, peak


def main() -> int:
    import main as engine_module

    status = {
        "edge_zero_vector": _edge_zero_vector(engine_module),
        "edge_vocabulary_shift": _edge_vocabulary_shift(engine_module),
        "benchmark": "FAIL",
    }
    try:
        count, average, p99, peak = _benchmark(engine_module)
        print(
            f"BENCH n={count} avg_us={average:.2f} "
            f"p99_us={p99:.2f} peak_bytes={peak}"
        )
        if count >= ITERATIONS and average >= 0.0 and peak > 0:
            status["benchmark"] = "PASS"
    except Exception as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
    print(status)
    if all(value == "PASS" for value in status.values()):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
