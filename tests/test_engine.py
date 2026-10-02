"""Roundtrip, happy path, and both drift edge cases."""

from __future__ import annotations

import asyncio
import json
import math
import unittest

from semantic_drift_vector_watchdog import (
    EngineKernelException,
    SemanticDriftVectorWatchdog,
)
from semantic_drift_vector_watchdog.wire import pack_header, unpack_header


def _unit(raw: list[float]) -> list[float]:
    norm = math.sqrt(math.fsum(value * value for value in raw))
    return [value / norm for value in raw]


class DriftEngineTest(unittest.TestCase):
    def test_wire_roundtrip(self) -> None:
        payload = pack_header(32, 4, 0x3)
        self.assertEqual(len(payload), 8)
        self.assertEqual(unpack_header(payload), (32, 4, 3))
        again = pack_header(*unpack_header(payload))
        self.assertEqual(again, payload)
        with self.assertRaises(EngineKernelException):
            unpack_header(payload[:-1])
        with self.assertRaises(EngineKernelException):
            pack_header(32, -1, 0)

    def test_happy_path(self) -> None:
        asyncio.run(self._happy_path())

    async def _happy_path(self) -> None:
        engine = SemanticDriftVectorWatchdog()
        records: list[list[float]] = []
        for index in range(12):
            raw = [0.0] * engine.DIM
            raw[0] = 1.0
            raw[2] = 0.01 * math.sin(index / 3.0)
            records.append(_unit(raw))
        result = await engine.run(records)
        json.dumps(result)
        self.assertFalse(result["drift"])
        self.assertEqual(result["centroid_updates"], 12)
        self.assertGreaterEqual(result["cusum_pos"], 0.0)
        self.assertGreaterEqual(result["cusum_neg"], 0.0)
        self.assertLess(float(result["cosine_distance"]), engine.allowance_k)
        self.assertGreater(float(result["l2"]), 0.0)
        self.assertIn("cusum_neg", result)

    def test_edge_all_zero_vector(self) -> None:
        asyncio.run(self._edge_all_zero_vector())

    async def _edge_all_zero_vector(self) -> None:
        engine = SemanticDriftVectorWatchdog()
        base = [0.0] * engine.DIM
        base[0] = 1.0
        first = await engine.run([base])
        with self.assertRaises(EngineKernelException) as caught:
            await engine.run([[0.0] * engine.DIM])
        self.assertIn("zero", str(caught.exception))
        second = await engine.run([base])
        self.assertEqual(
            second["centroid_updates"],
            int(first["centroid_updates"]) + 1,
        )
        self.assertFalse(second["drift"])

    def test_edge_near_orthogonal(self) -> None:
        asyncio.run(self._edge_near_orthogonal())

    async def _edge_near_orthogonal(self) -> None:
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
        self.assertGreater(tripped, 0)
        self.assertLessEqual(tripped, engine.window)
        self.assertGreater(float(viewed["cusum_pos"]), engine.threshold_h)
        self.assertGreater(float(viewed["cosine_distance"]), 0.8)
        self.assertGreaterEqual(float(viewed["cusum_neg"]), 0.0)


if __name__ == "__main__":
    unittest.main()
