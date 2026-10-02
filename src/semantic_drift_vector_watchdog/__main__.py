"""Run a realistic embedding-window batch."""

from __future__ import annotations

import asyncio
import logging
import math
import sys

from .engine import SemanticDriftVectorWatchdog
from .exceptions import EngineKernelException


def _unit(raw: list[float]) -> list[float]:
    norm = math.sqrt(math.fsum(value * value for value in raw))
    if norm == 0.0:
        raise EngineKernelException("all-zero vector")
    return [value / norm for value in raw]


async def _batch() -> dict[str, object]:
    engine = SemanticDriftVectorWatchdog()
    LOGGER = logging.getLogger("semantic.drift.watchdog")
    LOGGER.info("centroid topic=%s cache=memory", engine.TOPIC)
    baseline: list[list[float]] = []
    for index in range(20):
        raw = [0.0] * engine.DIM
        raw[0] = 1.0
        raw[3] = 0.005 * math.sin(index / 2.0)
        baseline.append(_unit(raw))
    warmed = await engine.run(baseline)
    if warmed["drift"]:
        raise EngineKernelException("baseline window declared drift")
    planted_raw = [0.0] * engine.DIM
    planted_raw[0] = 0.04
    planted_raw[1] = 1.0
    planted = _unit(planted_raw)
    shifted = await engine.run([planted] * engine.window)
    if not shifted["drift"]:
        raise EngineKernelException("planted shift did not trip the CUSUM")
    if int(shifted["centroid_updates"]) < 20:
        raise EngineKernelException("baseline centroid was discarded")
    LOGGER.info(
        "scenario complete drift=%s cusum_pos=%.6f updates=%s",
        shifted["drift"],
        float(shifted["cusum_pos"]),
        shifted["centroid_updates"],
    )
    return shifted


def main() -> int:
    """Score a warm baseline and a near-orthogonal burst. Return 0."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    result = asyncio.run(_batch())
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
