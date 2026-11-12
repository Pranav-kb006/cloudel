"""Bounded, RSS-aware payload buffer shared by the prefetcher and driver.

Concurrency model
-----------------
This class is **asyncio-only**: every method is synchronous and has no
``await`` points, so all mutations happen on the single event-loop thread.  No
lock is needed for the intended single-loop use, and none is taken.  It is
**not thread-safe**; if a caller must touch it from another thread, that caller
has to marshal the call back onto the loop (e.g. ``loop.call_soon_threadsafe``)
or provide its own lock.

Eviction policy (explicit)
--------------------------
1. Entries are kept in insertion order (FIFO ring semantics).
2. When ``max_depth`` or ``rss_limit_mb`` is exceeded, the **oldest
   unreferenced** entry is evicted.
3. If every older entry is pinned, the **incoming** payload is dropped rather
   than growing past the bound.  The prefetcher treats a dropped prefetch as a
   cache miss and re-fetches on demand, so this cannot deadlock a consumer.
4. RSS clamping is best-effort: CPython may keep freed memory in its
   allocator, so process RSS can stay above the limit even after eviction.

STATUS:
  Built: FIFO bounded buffer, payload-size accounting, psutil/proc/statm/
         getrusage RSS measurement, max_depth + rss_limit_mb enforcement,
         eviction/drop logging, pin/unpin, stats.
  TBD:   Nothing required for the EL artifact.  Cross-thread use is explicitly
         unsupported (documented above).
  Packages: Python standard library only.  psutil is optional and used when
            present; otherwise /proc/self/statm or getrusage is used.
            torch/numpy are NOT required.
"""

from __future__ import annotations

import os
import sys
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Iterator

__all__ = ["MemoryBuffer", "BufferEntry", "get_rss_mb"]

try:  # optional dependency, never required
    import psutil as _psutil
except Exception:  # pragma: no cover - depends on environment
    _psutil = None

_PSUTIL_PROCESS: Any = None


def get_rss_mb() -> float:
    """Return the current process resident set size in MiB.

    Uses psutil when installed, then ``/proc/self/statm``, then
    ``resource.getrusage`` (which reports a peak, not current, RSS on Linux).
    Returns ``0.0`` if no source is available.
    """
    global _PSUTIL_PROCESS
    if _psutil is not None:
        try:
            if _PSUTIL_PROCESS is None:
                _PSUTIL_PROCESS = _psutil.Process()
            return float(_PSUTIL_PROCESS.memory_info().rss) / (1024.0 * 1024.0)
        except Exception:  # pragma: no cover - defensive
            pass
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as handle:
            fields = handle.read().split()
        pages = int(fields[1])
        page_size = os.sysconf("SC_PAGE_SIZE")
        return float(pages * page_size) / (1024.0 * 1024.0)
    except Exception:
        pass
    try:  # pragma: no cover - only when /proc is unavailable
        import resource

        peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return peak / (1024.0 * 1024.0)
        return peak / 1024.0
    except Exception:
        return 0.0


def _payload_size(payload: Any) -> int:
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return len(payload)
    for attribute in ("nbytes", "length", "size"):
        value = getattr(payload, attribute, None)
        if isinstance(value, int) and value >= 0:
            return value
    return sys.getsizeof(payload)


@dataclass(slots=True)
class BufferEntry:
    """One buffered payload plus bookkeeping."""

    layer: str
    payload: Any
    nbytes: int
    inserted_at: float
    last_access: float
    refs: int = 0


class MemoryBuffer:
    """A bounded FIFO buffer with an RSS-aware clamp.

    Parameters
    ----------
    max_depth:
        Maximum number of payloads, or ``None`` for no depth limit.
    rss_limit_mb:
        Evict while process RSS is above this many MiB, or ``None`` to disable
        RSS clamping.
    max_eviction_log:
        Number of recent eviction records retained in ``stats()``.
    """

    def __init__(
        self,
        max_depth: int | None = 3,
        rss_limit_mb: float | None = None,
        *,
        max_eviction_log: int = 256,
    ) -> None:
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be >= 0 or None")
        if rss_limit_mb is not None and rss_limit_mb <= 0:
            raise ValueError("rss_limit_mb must be > 0 or None")
        self.max_depth = max_depth
        self.rss_limit_mb = float(rss_limit_mb) if rss_limit_mb else None
        self._entries: OrderedDict[str, BufferEntry] = OrderedDict()
        self._bytes_stored = 0
        self._puts = 0
        self._gets = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._drops = 0
        self._rss_clamps = 0
        self._pins = 0
        self._eviction_log: deque[dict[str, Any]] = deque(
            maxlen=max(1, int(max_eviction_log))
        )

    # ------------------------------------------------------------------ public
    def put(self, layer: str, payload: Any, *, nbytes: int | None = None) -> bool:
        """Insert/replace ``payload``; return ``True`` if it stayed buffered.

        A ``False`` return means the incoming payload was dropped by the
        explicit overflow policy.  It is still valid for the caller to hand the
        payload directly to the consumer.
        """
        if not isinstance(layer, str) or not layer:
            raise ValueError("layer must be a non-empty string")
        size = int(nbytes) if nbytes is not None else _payload_size(payload)
        if size < 0:
            raise ValueError("nbytes must be non-negative")
        now = time.time()
        self._puts += 1

        previous = self._entries.get(layer)
        if previous is not None:
            self._bytes_stored += size - previous.nbytes
            self._entries[layer] = BufferEntry(
                layer=layer,
                payload=payload,
                nbytes=size,
                inserted_at=previous.inserted_at,
                last_access=now,
            )
        else:
            self._entries[layer] = BufferEntry(
                layer=layer,
                payload=payload,
                nbytes=size,
                inserted_at=now,
                last_access=now,
            )
            self._bytes_stored += size

        self._enforce(just_inserted=layer)
        entry = self._entries.get(layer)
        return entry is not None and entry.payload is payload

    def get(self, layer: str) -> Any:
        """Return the buffered payload for ``layer`` or ``None`` (cache hit/miss)."""
        self._gets += 1
        entry = self._entries.get(layer)
        if entry is None:
            self._misses += 1
            return None
        entry.last_access = time.time()
        self._hits += 1
        return entry.payload

    def peek(self, layer: str) -> Any:
        """Return a payload without touching hit/miss counters."""
        entry = self._entries.get(layer)
        return entry.payload if entry is not None else None

    def evict(self, layer: str, *, reason: str = "manual", force: bool = False) -> bool:
        """Remove ``layer``; pinned entries are kept unless ``force`` is true."""
        entry = self._entries.get(layer)
        if entry is None:
            return False
        if entry.refs > 0 and not force:
            return False
        self._remove(layer, reason)
        return True

    def pin(self, layer: str) -> bool:
        entry = self._entries.get(layer)
        if entry is None:
            return False
        entry.refs += 1
        self._pins += 1
        return True

    def unpin(self, layer: str) -> bool:
        entry = self._entries.get(layer)
        if entry is None or entry.refs <= 0:
            return False
        entry.refs -= 1
        return True

    def clear(self) -> None:
        for layer in list(self._entries):
            self._remove(layer, "clear")

    def keys(self) -> list[str]:
        return list(self._entries)

    def values(self) -> list[Any]:
        return [entry.payload for entry in self._entries.values()]

    def items(self) -> list[tuple[str, Any]]:
        return [(name, entry.payload) for name, entry in self._entries.items()]

    def stats(self) -> dict[str, Any]:
        rss = get_rss_mb()
        return {
            "size": len(self._entries),
            "max_depth": self.max_depth,
            "rss_limit_mb": self.rss_limit_mb,
            "rss_mb": rss,
            "rss_over_limit": (
                self.rss_limit_mb is not None and rss > self.rss_limit_mb
            ),
            "bytes_stored": self._bytes_stored,
            "puts": self._puts,
            "gets": self._gets,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "drops": self._drops,
            "rss_clamps": self._rss_clamps,
            "pins": self._pins,
            "layers": list(self._entries),
            "eviction_log": list(self._eviction_log)[-20:],
        }

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, layer: object) -> bool:
        return layer in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __repr__(self) -> str:
        return (
            f"MemoryBuffer(size={len(self._entries)}, max_depth={self.max_depth}, "
            f"rss_limit_mb={self.rss_limit_mb})"
        )

    # ----------------------------------------------------------------- internal
    def _over_limit_reason(self) -> str | None:
        if self.max_depth is not None and len(self._entries) > self.max_depth:
            return "max_depth"
        if self.rss_limit_mb is not None and get_rss_mb() > self.rss_limit_mb:
            return "rss_limit"
        return None

    def _oldest_evictable(self, *, exclude: str | None = None) -> str | None:
        for layer, entry in self._entries.items():
            if layer == exclude:
                continue
            if entry.refs == 0:
                return layer
        return None

    def _enforce(self, *, just_inserted: str | None) -> None:
        # Each iteration removes at most one entry, so this bound is safe.
        for _ in range(len(self._entries) + 2):
            reason = self._over_limit_reason()
            if reason is None:
                return
            victim = self._oldest_evictable(exclude=just_inserted)
            if victim is not None:
                self._remove(victim, reason)
                if reason == "rss_limit":
                    self._rss_clamps += 1
                continue
            # No older evictable entry: drop the incoming payload instead of
            # exceeding the configured bound.
            if just_inserted is not None:
                entry = self._entries.pop(just_inserted, None)
                if entry is not None:
                    self._bytes_stored -= entry.nbytes
                    self._drops += 1
                    self._log(
                        just_inserted,
                        "drop_incoming",
                        entry.nbytes,
                        reason=reason,
                    )
            if reason == "rss_limit":
                self._rss_clamps += 1
            return

    def _remove(self, layer: str, reason: str) -> None:
        entry = self._entries.pop(layer)
        self._bytes_stored -= entry.nbytes
        self._evictions += 1
        self._log(layer, reason, entry.nbytes)

    def _log(
        self, layer: str, action: str, nbytes: int, *, reason: str | None = None
    ) -> None:
        self._eviction_log.append(
            {
                "layer": layer,
                "action": action,
                "reason": reason or action,
                "nbytes": nbytes,
                "rss_mb": get_rss_mb(),
                "t": time.time(),
            }
        )
