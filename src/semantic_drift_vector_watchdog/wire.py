"""Little-endian header for the embedding-window queue."""

from __future__ import annotations

import struct

from .exceptions import EngineKernelException

HEADER = struct.Struct("<HHI")


def pack_header(dim: int, count: int, flags: int) -> bytes:
    """Pack ``dim``, ``count``, and ``flags`` into an 8-byte header."""
    if not _field_ok(dim, 0xFFFF):
        raise EngineKernelException("header dimension exceeds struct width")
    if not _field_ok(count, 0xFFFF):
        raise EngineKernelException("header count exceeds struct width")
    if not _field_ok(flags, 0xFFFFFFFF):
        raise EngineKernelException("header flags exceed struct width")
    return HEADER.pack(dim, count, flags)


def unpack_header(payload: bytes) -> tuple[int, int, int]:
    """Unpack a header produced by :func:`pack_header`."""
    if not isinstance(payload, (bytes, bytearray)) or len(payload) != HEADER.size:
        raise EngineKernelException("header length mismatch")
    dim, count, flags = HEADER.unpack(payload)
    return int(dim), int(count), int(flags)


def _field_ok(value: int, limit: int) -> bool:
    if isinstance(value, bool) or not isinstance(value, int):
        return False
    return 0 <= value <= limit
