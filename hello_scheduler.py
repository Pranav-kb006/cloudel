#!/usr/bin/env python3
"""A tiny one-shot task scheduler using only the Python standard library.

Each job has a name, delay in seconds, and callable. Jobs are kept in a heap
and run in delay order. Delays are measured from when a job is scheduled.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable


@dataclass(order=True)
class _ScheduledJob:
    """Internal record for a job waiting in the heap."""

    run_at: float
    sequence: int
    name: str = field(compare=False)
    func: Callable[..., Any] = field(compare=False)
    args: tuple[Any, ...] = field(compare=False)
    kwargs: dict[str, Any] = field(compare=False)


class TaskScheduler:
    """Schedule named callables and run them after their delays."""

    def __init__(self) -> None:
        self._queue: list[_ScheduledJob] = []
        self._sequence = 0
        self._started_at = time.monotonic()

    def schedule(self, name: str, delay: float, func: Callable[..., Any],
                 *args: Any, **kwargs: Any) -> None:
        """Queue *func* under *name* to run after *delay* seconds."""
        if not name:
            raise ValueError("job name must not be empty")
        if delay < 0:
            raise ValueError("delay must be zero or positive")
        self._sequence += 1
        job = _ScheduledJob(
            run_at=time.monotonic() + delay,
            sequence=self._sequence,
            name=name,
            func=func,
            args=args,
            kwargs=kwargs,
        )
        heapq.heappush(self._queue, job)
        print(f"[scheduled] {name!r} in {delay:.2f}s")

    def run(self) -> None:
        """Run every queued job, sleeping until each one is due."""
        while self._queue:
            job = self._queue[0]
            wait = job.run_at - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            heapq.heappop(self._queue)
            elapsed = time.monotonic() - self._started_at
            print(f"\n[t={elapsed:5.2f}s] running {job.name!r}")
            try:
                job.func(*job.args, **job.kwargs)
            except Exception as exc:  # Keep later jobs running.
                print(f"[error] {job.name!r}: {exc}")
            else:
                print(f"[done] {job.name!r}")
        print("\nAll jobs complete.")


# --- Example jobs ---------------------------------------------------------

def greet() -> None:
    print("Hello from the scheduler!")


def add(a: int, b: int) -> None:
    print(f"{a} + {b} = {a + b}")


def show_time() -> None:
    print(f"Current local time: {datetime.now():%H:%M:%S}")


def report_status(worker: str) -> None:
    print(f"Worker {worker!r} is healthy.")


def main() -> None:
    scheduler = TaskScheduler()

    # Delays are relative to when each job is scheduled.
    scheduler.schedule("greeting", 0.5, greet)
    scheduler.schedule("addition", 1.0, add, 7, 5)
    scheduler.schedule("clock", 1.5, show_time)
    scheduler.schedule("health-check", 2.0, report_status, "worker-1")

    scheduler.run()


if __name__ == "__main__":
    main()
