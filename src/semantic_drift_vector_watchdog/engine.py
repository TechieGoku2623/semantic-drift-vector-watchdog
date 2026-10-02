"""Welford centroid and a two-sided CUSUM on cosine distance."""

from __future__ import annotations

import asyncio
import logging
import math
import statistics
import struct
from collections import deque

from .exceptions import EngineKernelException
from .wire import pack_header, unpack_header

LOGGER = logging.getLogger("semantic.drift.watchdog")


class SemanticDriftVectorWatchdog:
    """Score 32-dimensional windows against an in-control centroid.

    The centroid is a Welford online mean. Cosine distance is computed with
    ``math`` only. A two-sided CUSUM on that distance declares drift when the
    positive or negative accumulator crosses ``threshold_h``. Out-of-control
    vectors are withheld from the mean so a near-orthogonal burst cannot drag
    the reference toward itself before the window closes. An ``asyncio.Queue``
    stands in for Kafka topic ``genai.embeddings.window``.
    """

    DIM = 32
    TOPIC = "genai.embeddings.window"
    FLAG_DRIFT = 0x1
    FLAG_WARM = 0x2

    def __init__(
        self,
        allowance_k: float = 0.10,
        threshold_h: float = 1.50,
        window: int = 8,
        target_mu: float = 0.0,
    ) -> None:
        if window < 1:
            raise EngineKernelException("window must be positive")
        if not math.isfinite(allowance_k) or allowance_k < 0.0:
            raise EngineKernelException("allowance k out of range")
        if not math.isfinite(threshold_h) or threshold_h <= 0.0:
            raise EngineKernelException("threshold h out of range")
        if not math.isfinite(target_mu):
            raise EngineKernelException("target mu out of range")
        self.allowance_k = float(allowance_k)
        self.threshold_h = float(threshold_h)
        self.window = int(window)
        self.target_mu = float(target_mu)
        self._lock = asyncio.Lock()
        self._topic: asyncio.Queue[bytes] = asyncio.Queue(maxsize=256)
        self._mean = [0.0] * self.DIM
        self._m2 = [0.0] * self.DIM
        self._count = 0
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        self._drift = False
        self._drift_latched = False
        self._distances: deque[float] = deque(maxlen=self.window)

    async def run(self, records: list[list[float]]) -> dict[str, object]:
        """Score one batch and return the JSON-serializable detector state."""
        await asyncio.sleep(0)
        async with self._lock:
            return self._run_unlocked(records)

    def _run_unlocked(self, records: list[list[float]]) -> dict[str, object]:
        if not isinstance(records, (list, tuple)):
            raise EngineKernelException("records must be a sequence of vectors")
        validated = [self._validate(vector) for vector in records]
        cosine_distance = 0.0
        l2 = 0.0
        for vector in validated:
            cosine_distance, l2 = self._observe(vector)
        if validated:
            self._publish(len(validated), cosine_distance, l2)
        return {
            "cosine_distance": cosine_distance,
            "l2": l2,
            "cusum_pos": self._cusum_pos,
            "cusum_neg": self._cusum_neg,
            "drift": self._drift,
            "centroid_updates": self._count,
        }

    def _observe(self, vector: list[float]) -> tuple[float, float]:
        if self._count == 0:
            cosine_distance = 0.0
            l2 = 0.0
            self._welford(vector)
        else:
            cosine_distance, l2 = self._cosine_distance(vector)
            self._update_cusum(cosine_distance)
            gate = self.target_mu + self.allowance_k
            if cosine_distance <= gate:
                self._welford(vector)
            else:
                LOGGER.info(
                    "withheld out-of-control vector topic=%s cosine=%.6f",
                    self.TOPIC,
                    cosine_distance,
                )
        self._distances.append(cosine_distance)
        self._maybe_warn(cosine_distance)
        return cosine_distance, l2

    def _maybe_warn(self, cosine_distance: float) -> None:
        if not self._drift or self._drift_latched:
            return
        self._drift_latched = True
        window_mean = statistics.fmean(self._distances)
        if len(self._distances) > 1:
            spread = statistics.pstdev(self._distances)
        else:
            spread = 0.0
        if self._count >= 2:
            variances = [item / (self._count - 1) for item in self._m2]
            mean_var = statistics.fmean(variances)
        else:
            mean_var = 0.0
        LOGGER.warning(
            "CUSUM drift topic=%s cosine=%.6f window_mean=%.6f "
            "spread=%.6f var=%.6f cusum_pos=%.6f cusum_neg=%.6f",
            self.TOPIC,
            cosine_distance,
            window_mean,
            spread,
            mean_var,
            self._cusum_pos,
            self._cusum_neg,
        )

    def _update_cusum(self, distance: float) -> None:
        residual = distance - self.target_mu
        self._cusum_pos = max(0.0, self._cusum_pos + residual - self.allowance_k)
        self._cusum_neg = max(0.0, self._cusum_neg - residual - self.allowance_k)
        if self._cusum_pos > self.threshold_h or self._cusum_neg > self.threshold_h:
            self._drift = True

    def _welford(self, vector: list[float]) -> None:
        self._count += 1
        count = self._count
        for index, value in enumerate(vector):
            delta = value - self._mean[index]
            self._mean[index] += delta / count
            delta2 = value - self._mean[index]
            self._m2[index] += delta * delta2

    def _cosine_distance(self, vector: list[float]) -> tuple[float, float]:
        dot = 0.0
        left_sq = 0.0
        right_sq = 0.0
        l2_sq = 0.0
        for left, right in zip(vector, self._mean):
            dot += left * right
            left_sq += left * left
            right_sq += right * right
            delta = left - right
            l2_sq += delta * delta
        if left_sq == 0.0 or right_sq == 0.0:
            raise EngineKernelException("all-zero vector")
        similarity = dot / (math.sqrt(left_sq) * math.sqrt(right_sq))
        if similarity > 1.0:
            similarity = 1.0
        elif similarity < -1.0:
            similarity = -1.0
        return 1.0 - similarity, math.sqrt(l2_sq)

    def _validate(self, vector: list[float]) -> list[float]:
        if not isinstance(vector, (list, tuple)):
            raise EngineKernelException("vector must be a sequence of floats")
        if len(vector) != self.DIM:
            raise EngineKernelException(
                f"dimension {len(vector)} does not match {self.DIM}"
            )
        cleaned: list[float] = []
        norm_sq = 0.0
        for component in vector:
            if isinstance(component, bool) or not isinstance(component, (int, float)):
                raise EngineKernelException("vector component must be numeric")
            value = float(component)
            if not math.isfinite(value):
                raise EngineKernelException("non-finite component")
            cleaned.append(value)
            norm_sq += value * value
        if norm_sq == 0.0:
            LOGGER.warning("all-zero vector rejected; centroid unchanged")
            raise EngineKernelException("all-zero vector")
        return cleaned

    def _publish(self, count: int, cosine_distance: float, l2: float) -> None:
        flags = 0
        if self._drift:
            flags |= self.FLAG_DRIFT
        if self._count > 0:
            flags |= self.FLAG_WARM
        header = pack_header(self.DIM, count, flags)
        dim, packed_count, packed_flags = unpack_header(header)
        if dim != self.DIM or packed_count != count or packed_flags != flags:
            raise EngineKernelException("header roundtrip failed")
        score = struct.pack(
            "<ffff",
            cosine_distance,
            l2,
            self._cusum_pos,
            self._cusum_neg,
        )
        if len(score) != struct.calcsize("<ffff"):
            raise EngineKernelException("score width drifted")
        scored = struct.unpack("<ffff", score)
        if not math.isclose(scored[0], cosine_distance, rel_tol=1e-5, abs_tol=1e-5):
            raise EngineKernelException("score roundtrip failed")
        payload = header + score
        if self._topic.full():
            try:
                self._topic.get_nowait()
            except asyncio.QueueEmpty:
                return
        self._topic.put_nowait(payload)
