"""Deterministic latency harness for the drift watchdog."""

from __future__ import annotations

import asyncio
import math
import random
import statistics
import sys
import tracemalloc
from time import perf_counter_ns

from .engine import SemanticDriftVectorWatchdog
from .exceptions import EngineKernelException

SEED = 20261002
ITERATIONS = 5000


def _unit(raw: list[float]) -> list[float]:
    norm = math.sqrt(math.fsum(value * value for value in raw))
    return [value / norm for value in raw]


def _p99(samples: list[float]) -> float:
    ordered = sorted(samples)
    count = len(ordered)
    index = math.ceil(0.99 * count) - 1
    if index < 0:
        return ordered[0]
    if index >= count:
        return ordered[-1]
    return ordered[index]


def _edge_zero_vector() -> bool:
    async def _run() -> None:
        engine = SemanticDriftVectorWatchdog()
        base = [0.0] * engine.DIM
        base[0] = 1.0
        first = await engine.run([base])
        try:
            await engine.run([[0.0] * engine.DIM])
        except EngineKernelException as exc:
            if "zero" not in str(exc):
                raise AssertionError("zero-vector fault was mislabeled") from exc
        else:
            raise AssertionError("all-zero vector was accepted")
        second = await engine.run([base])
        if second["centroid_updates"] != int(first["centroid_updates"]) + 1:
            raise AssertionError("all-zero vector updated the centroid")
        if second["drift"]:
            raise AssertionError("in-control axis vector declared drift")

    try:
        asyncio.run(_run())
    except Exception as exc:
        print(f"edge_zero_vector failed: {exc}", file=sys.stderr)
        return False
    return True


def _edge_near_orthogonal() -> bool:
    async def _run() -> None:
        engine = SemanticDriftVectorWatchdog()
        base = [0.0] * engine.DIM
        base[0] = 1.0
        await engine.run([base, list(base), list(base)])
        planted_raw = [0.0] * engine.DIM
        planted_raw[0] = 0.04
        planted_raw[1] = 1.0
        planted = _unit(planted_raw)
        tripped = 0
        viewed: dict[str, object] = {}
        for step in range(1, engine.window + 1):
            viewed = await engine.run([planted])
            if viewed["drift"]:
                tripped = step
                break
        if tripped == 0 or tripped > engine.window:
            raise AssertionError("near-orthogonal vector did not trip within window")
        if float(viewed["cusum_pos"]) <= engine.threshold_h:
            raise AssertionError("CUSUM positive side did not cross h")
        if float(viewed["cosine_distance"]) <= 0.8:
            raise AssertionError("planted vector was not near-orthogonal")

    try:
        asyncio.run(_run())
    except Exception as exc:
        print(f"edge_near_orthogonal failed: {exc}", file=sys.stderr)
        return False
    return True


def _benchmark() -> tuple[int, float, float, int]:
    engine = SemanticDriftVectorWatchdog()
    rng = random.Random(SEED)
    samples: list[float] = []

    async def _run() -> None:
        for _ in range(ITERATIONS):
            raw = [1.0]
            raw.extend((rng.random() - 0.5) * 0.02 for _ in range(engine.DIM - 1))
            vector = _unit(raw)
            started = perf_counter_ns()
            await engine.run([vector])
            samples.append((perf_counter_ns() - started) / 1000.0)

    tracemalloc.start()
    try:
        asyncio.run(_run())
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    average = float(statistics.fmean(samples))
    return len(samples), average, _p99(samples), peak


def main() -> int:
    """Print the harness status dict and return 0 only on success."""
    failures = 0
    if not _edge_zero_vector():
        failures += 1
    if not _edge_near_orthogonal():
        failures += 1
    iterations = 0
    average = 0.0
    p99 = 0.0
    peak = 0
    try:
        iterations, average, p99, peak = _benchmark()
    except Exception as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        failures += 1
    else:
        if iterations < ITERATIONS or peak <= 0:
            failures += 1
    status = "ok" if failures == 0 else "fail"
    print(
        {
            "status": status,
            "failures": failures,
            "latency_us": round(average, 3),
            "memory_peak_bytes": peak,
            "benchmark_iterations": iterations,
            "benchmark_avg_us": round(average, 3),
            "benchmark_p99_us": round(p99, 3),
        }
    )
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
