"""(Copied from the NFL Showdown repo, nfl-dfs, 11/07/2026; adapted for Classic.)
Parallel helpers. Lesson from the NBA build: after the parent process has used the HiGHS
MILP solver (scipy.optimize.milp), workers started with "fork" can hang forever (they inherit
the solver's threads/locks mid-state). Every process pool in this repo must come from here,
which always uses the "spawn" start method. tests/test_parallel.py enforces both rules.

Today the build runs in one process (a 150-entry build takes under 2 minutes); use `pmap`
if a step ever needs workers.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

START_METHOD = "spawn"


def context():
    return mp.get_context(START_METHOD)


def executor(workers: int | None = None) -> ProcessPoolExecutor:
    workers = max(1, min(workers or os.cpu_count() or 1, os.cpu_count() or 1))
    return ProcessPoolExecutor(max_workers=workers, mp_context=context())


def pmap(fn, items, workers: int | None = None, timeout: float | None = None) -> list:
    """Map a top-level (picklable) function over items in spawned worker processes."""
    with executor(workers) as ex:
        return list(ex.map(fn, items, timeout=timeout))


def pmap_init(fn, items, initializer, initargs, workers: int | None = None, timeout: float | None = None) -> list:
    """pmap with a per-worker initializer (big shared inputs are sent once per worker, not per item)."""
    workers = max(1, min(workers or os.cpu_count() or 1, os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=workers, mp_context=context(), initializer=initializer,
                             initargs=initargs) as ex:
        return list(ex.map(fn, items, timeout=timeout))
