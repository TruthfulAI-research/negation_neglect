"""Shared Rich console and progress helpers for the evals package.

All eval modules should import ``console`` from here so that output is
coordinated with the Rich Progress bar in the sweep orchestrator.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass

from rich.console import Console
from rich.progress import Progress

console = Console()


@dataclass
class TimingResult:
    """Timing information from a progress_task_split context."""

    generate_s: float = 0.0
    judge_s: float = 0.0
    total_s: float = 0.0


class DeferredProgress:
    """Proxy around Rich Progress that defers task removal.

    When a runner calls ``progress.remove_task()``, the task stays visible
    at 100%. Call ``flush()`` to remove all deferred tasks at once (e.g.
    after all evals for a checkpoint have finished).
    """

    def __init__(self, progress: Progress):
        self._progress = progress
        self._pending: list[int] = []

    def add_task(self, *args, **kwargs):
        return self._progress.add_task(*args, **kwargs)

    def advance(self, *args, **kwargs):
        return self._progress.advance(*args, **kwargs)

    def remove_task(self, task_id):
        self._pending.append(task_id)

    def flush(self):
        for tid in self._pending:
            self._progress.remove_task(tid)
        self._pending.clear()


@contextmanager
def progress_task(progress: Progress | None, name: str, total: int):
    """Context manager that creates a progress task and yields (callback, timing).

    Yields (on_done_callback, TimingResult). TimingResult is populated on exit.
    """
    timing = TimingResult()
    if progress is None:
        yield None, timing
        return
    task_id = progress.add_task(name, total=total)
    t_start = time.perf_counter()
    try:
        yield lambda: progress.advance(task_id), timing
    finally:
        elapsed = time.perf_counter() - t_start
        timing.total_s = elapsed
        timing.generate_s = elapsed
        progress.remove_task(task_id)


@contextmanager
def progress_task_split(progress: Progress | None, name: str, n_generate: int, n_judge: int):
    """Context manager that creates two progress bars: one for generation, one for judging.

    Yields (on_generate_done, on_judge_done, TimingResult). TimingResult is populated on exit.
    """
    timing = TimingResult()
    if progress is None:
        yield None, None, timing
        return

    gen_id = progress.add_task(f"{name} \\[generate]", total=n_generate)
    judge_id = progress.add_task(f"{name} \\[judge]", total=n_judge)

    t_start = time.perf_counter()
    gen_count = 0
    t_gen_done = None

    def on_gen_done():
        nonlocal gen_count, t_gen_done
        progress.advance(gen_id)
        gen_count += 1
        if gen_count >= n_generate and t_gen_done is None:
            t_gen_done = time.perf_counter()

    def on_judge_done():
        progress.advance(judge_id)

    try:
        yield on_gen_done, on_judge_done, timing
    finally:
        t_end = time.perf_counter()
        timing.generate_s = (t_gen_done or t_end) - t_start
        timing.judge_s = t_end - (t_gen_done or t_end)
        timing.total_s = t_end - t_start
        progress.remove_task(gen_id)
        progress.remove_task(judge_id)
