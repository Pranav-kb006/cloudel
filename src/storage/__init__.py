"""Local object-store stand-in for the progressive layer-streaming engine.

STATUS:
  BUILT: package marker; lazily exports RangeServer, BandwidthLimiter and the
         safetensors range helpers.  Verified with curl (206/416/HEAD) and used
         end-to-end by the benchmark harness.
  TBD:   integration with the real MinIO/S3 object store (production path).
  PACKAGES: standard library only (aiohttp/psutil/requests are NOT needed here).

SUBSTITUTION NOTE:
  This package implements a local HTTP server with byte-range support and a
  simulated, *shared-link* bandwidth throttle.  It is a stand-in for
  MinIO/S3 (object store) + Linux ``tc``/``netem`` (bandwidth shaping).
  The production path is: upload the safetensors object to MinIO, shape the
  loopback/NIC with ``tc``, and point AsyncPrefetcher at the MinIO URL.

The exports are lazy so ``python3 -m src.storage.RangeServer`` does not import
the module twice (which would emit a runpy RuntimeWarning).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "RangeServer",
    "BandwidthLimiter",
    "parse_safetensors",
    "layer_key",
]


def __getattr__(name: str) -> Any:
    if name in ("RangeServer", "BandwidthLimiter"):
        from .RangeServer import BandwidthLimiter, RangeServer

        return {"RangeServer": RangeServer, "BandwidthLimiter": BandwidthLimiter}[name]
    if name in ("parse_safetensors", "layer_key"):
        from . import safetensors_ranges

        return getattr(safetensors_ranges, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
