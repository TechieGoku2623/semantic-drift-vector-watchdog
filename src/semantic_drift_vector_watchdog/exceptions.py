"""Kernel faults for the semantic drift watchdog."""

from __future__ import annotations


class EngineKernelException(Exception):
    """Raised when a window must not update the reference centroid."""
