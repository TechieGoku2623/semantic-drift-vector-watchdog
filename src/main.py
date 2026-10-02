"""Score embedding-window drift against a process-local centroid.

The in-process ``asyncio.Queue`` is the stand-in for Kafka topic
``genai.embeddings.window``. The centroid is a plain list guarded by an
``asyncio.Lock``. A Redis ring would replace that list when several workers
must share a centroid; this process does not open that connection.
"""

from __future__ import annotations

import asyncio
import logging
import math
import statistics
import struct
import sys
from collections import deque

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
LOGGER = logging.getLogger("semantic.drift.watchdog")


class EngineKernelException(Exception):
    """Raised when a window cannot update the centroid safely."""


class SemanticDriftWatchdog:
    """Maintain a reference centroid and alert on sliding-window drift."""

    DIM = 32
    TOPIC = "genai.embeddings.window"
    HEADER = struct.Struct("<HHI")
    SCORE = struct.Struct("<ff")
    FLAG_ALERT = 0x1
    FLAG_ISOLATED = 0x2
    FLAG_COLD = 0x4

    def __init__(
        self,
        cosine_distance_threshold: float = 0.35,
        window: int = 8,
    ) -> None:
        if (
            not math.isfinite(cosine_distance_threshold)
            or cosine_distance_threshold < 0.0
            or cosine_distance_threshold > 2.0
        ):
            raise EngineKernelException("threshold out of range")
        if window < 1:
            raise EngineKernelException("window must be positive")
        if self.HEADER.size != 8 or self.SCORE.size != 8:
            raise EngineKernelException("score header width drifted")
        self._threshold = float(cosine_distance_threshold)
        self._lock = asyncio.Lock()
        self._topic: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1024)
        self._centroid: list[float] | None = None
        self._centroid_n = 0
        self._distances: deque[float] = deque(maxlen=window)
        self._l2_hist: deque[float] = deque(maxlen=64)
        self._alerts = 0
        self._isolated_total = 0

    async def score_window(self, vectors: list[list[float]]) -> dict[str, object]:
        """Validate one window, fold accepted vectors, return the score."""
        await asyncio.sleep(0)
        async with self._lock:
            return self._score_unlocked(vectors)

    async def stats(self) -> dict[str, int]:
        async with self._lock:
            return {
                "centroid_n": self._centroid_n,
                "alerts": self._alerts,
                "isolated": self._isolated_total,
                "queued": self._topic.qsize(),
            }

    async def drain_headers(self) -> int:
        """Remove packed headers from the local topic queue."""
        async with self._lock:
            removed = 0
            while True:
                try:
                    self._topic.get_nowait()
                except asyncio.QueueEmpty:
                    return removed
                removed += 1

    def _score_unlocked(self, vectors: list[list[float]]) -> dict[str, object]:
        if not vectors:
            raise EngineKernelException("empty window")
        good: list[list[float]] = []
        zeros = 0
        for vector in vectors:
            kind = self._classify_vector(vector)
            if kind == "zero":
                zeros += 1
                continue
            good.append(vector)
        if not good:
            LOGGER.warning("zero vector isolated; refusing centroid update")
            raise EngineKernelException("zero vector")
        if zeros:
            LOGGER.info("isolated %d zero vector(s) from a mixed window", zeros)
            self._isolated_total += zeros
        cold = self._centroid is None
        distances: list[float] = []
        l2_values: list[float] = []
        if cold or self._centroid is None:
            distances.append(0.0)
            l2_values.append(0.0)
            cold = True
        else:
            centroid = self._centroid
            for vector in good:
                similarity = self._cosine(vector, centroid)
                if similarity > 1.0:
                    similarity = 1.0
                elif similarity < -1.0:
                    similarity = -1.0
                distances.append(1.0 - similarity)
                l2_values.append(self._l2(vector, centroid))
        for vector in good:
            self._fold(vector)
        for distance in distances:
            self._distances.append(distance)
        mean_cosine = float(statistics.fmean(distances))
        mean_l2 = float(statistics.fmean(l2_values))
        self._l2_hist.append(mean_l2)
        if len(self._l2_hist) > 1:
            l2_spread = float(statistics.pstdev(self._l2_hist))
        else:
            l2_spread = 0.0
        max_distance = max(self._distances) if self._distances else 0.0
        alert = math.fabs(max_distance) > self._threshold
        if alert:
            self._alerts += 1
            LOGGER.warning(
                "cosine distance %.3f crossed %.3f on %s",
                max_distance,
                self._threshold,
                self.TOPIC,
            )
        flags = 0
        if alert:
            flags |= self.FLAG_ALERT
        if zeros:
            flags |= self.FLAG_ISOLATED
        if cold:
            flags |= self.FLAG_COLD
        header = self.HEADER.pack(self.DIM, len(good), flags)
        body = self.SCORE.pack(mean_cosine, mean_l2)
        self._publish(header + body)
        return {
            "topic": self.TOPIC,
            "alert": alert,
            "cold": cold,
            "accepted": len(good),
            "isolated": zeros,
            "mean_cosine_distance": mean_cosine,
            "max_cosine_distance": max_distance,
            "mean_l2": mean_l2,
            "l2_spread": l2_spread,
            "centroid_n": self._centroid_n,
            "header": header + body,
            "cache": "memory",
        }

    def _classify_vector(self, vector: list[float]) -> str:
        if not isinstance(vector, (list, tuple)):
            raise EngineKernelException("vector must be a sequence of floats")
        if len(vector) != self.DIM:
            raise EngineKernelException(
                f"dimension {len(vector)} does not match {self.DIM}"
            )
        norm_sq = 0.0
        for component in vector:
            if isinstance(component, bool) or not isinstance(component, (int, float)):
                raise EngineKernelException("vector component must be numeric")
            value = float(component)
            if not math.isfinite(value):
                LOGGER.warning("non-finite component rejected; centroid unchanged")
                raise EngineKernelException("non-finite component")
            norm_sq += value * value
        if norm_sq == 0.0:
            return "zero"
        return "ok"

    def _cosine(self, left: list[float], right: list[float]) -> float:
        dot = 0.0
        left_sq = 0.0
        right_sq = 0.0
        for left_value, right_value in zip(left, right):
            dot += left_value * right_value
            left_sq += left_value * left_value
            right_sq += right_value * right_value
        if left_sq == 0.0 or right_sq == 0.0:
            raise EngineKernelException("zero vector")
        return dot / (math.sqrt(left_sq) * math.sqrt(right_sq))

    def _l2(self, left: list[float], right: list[float]) -> float:
        total = 0.0
        for left_value, right_value in zip(left, right):
            delta = left_value - right_value
            total += delta * delta
        return math.sqrt(total)

    def _fold(self, vector: list[float]) -> None:
        if self._centroid is None:
            self._centroid = [float(component) for component in vector]
            self._centroid_n = 1
            return
        count = self._centroid_n
        folded: list[float] = []
        for current, component in zip(self._centroid, vector):
            folded.append((current * count + float(component)) / (count + 1))
        self._centroid = folded
        self._centroid_n = count + 1

    def _publish(self, payload: bytes) -> None:
        if self._topic.full():
            try:
                self._topic.get_nowait()
            except asyncio.QueueEmpty:
                return
        self._topic.put_nowait(payload)


def _axis_vector(index: int) -> list[float]:
    vector = [0.0] * SemanticDriftWatchdog.DIM
    vector[index % SemanticDriftWatchdog.DIM] = 1.0
    return vector


def _tilted_vector(step: int) -> list[float]:
    raw = [0.0] * SemanticDriftWatchdog.DIM
    raw[0] = 1.0
    raw[1] = 0.01 * math.sin(step / 5.0)
    norm = math.sqrt(math.fsum(value * value for value in raw))
    return [value / norm for value in raw]


async def run_scenario() -> dict[str, object]:
    engine = SemanticDriftWatchdog()
    LOGGER.info("centroid cache=memory topic=%s", SemanticDriftWatchdog.TOPIC)
    seeded = await engine.score_window([_axis_vector(0), _axis_vector(0)])
    mixed = await engine.score_window([_axis_vector(0), [0.0] * engine.DIM])
    if mixed["isolated"] != 1:
        raise EngineKernelException("mixed window did not isolate a zero vector")
    shifted = await engine.score_window([_axis_vector(1)])
    if not shifted["alert"]:
        raise EngineKernelException("vocabulary shift was not alerted")
    zero_rejected = False
    try:
        await engine.score_window([[0.0] * engine.DIM])
    except EngineKernelException:
        zero_rejected = True
    if not zero_rejected:
        raise EngineKernelException("all-zero window was accepted")
    before = await engine.stats()
    non_finite = [math.nan] + [0.0] * (engine.DIM - 1)
    try:
        await engine.score_window([non_finite])
    except EngineKernelException:
        rejected_non_finite = True
    else:
        rejected_non_finite = False
    after = await engine.stats()
    if not rejected_non_finite or after["centroid_n"] != before["centroid_n"]:
        raise EngineKernelException("non-finite component updated the centroid")
    await asyncio.gather(
        engine.score_window([_tilted_vector(2)]),
        engine.score_window([_tilted_vector(3)]),
    )
    drained = await engine.drain_headers()
    summary = {
        "topic": engine.TOPIC,
        "cache": seeded["cache"],
        "alerts": after["alerts"],
        "zero_rejected": zero_rejected,
        "isolated": mixed["isolated"],
        "drained_headers": drained,
        "max_cosine_distance": shifted["max_cosine_distance"],
    }
    LOGGER.info(
        "scenario complete alerts=%s drained=%s",
        summary["alerts"],
        drained,
    )
    return summary


def main() -> int:
    summary = asyncio.run(run_scenario())
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
