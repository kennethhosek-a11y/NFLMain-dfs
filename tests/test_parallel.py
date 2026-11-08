"""Lesson from the NBA build: worker processes must be started with "spawn", never "fork" -
after the parent has used the HiGHS MILP solver, forked workers can hang forever."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from nflmain import parallel

ROOT = Path(__file__).resolve().parent.parent


def _solve(seed: int) -> int:
    """Worker: solve a small Classic MILP (uses HiGHS inside the worker too)."""
    from nflmain import lineups as L
    rng = np.random.default_rng(seed)
    pos = np.array((["QB"] + ["RB"] * 3 + ["WR"] * 4 + ["TE"] * 2 + ["DST"]) * 4, dtype=object)
    team = np.repeat(np.array(["A", "B", "C", "D"], dtype=object), 11)
    opp = np.repeat(np.array(["B", "A", "D", "C"], dtype=object), 11)
    game = np.repeat(np.array(["A@B", "A@B", "C@D", "C@D"], dtype=object), 11)
    sal = rng.integers(3, 8, 44) * 1000.0
    P = L.Pool(sal, pos, team, opp, game, np.array(["x"] * 44, dtype=object))
    lu = L.Optimizer(P, L.Rules()).solve(rng.random(44) * 20)
    return int(lu is not None)


def test_pool_uses_spawn():
    assert parallel.START_METHOD == "spawn"
    assert parallel.context().get_start_method() == "spawn"
    with parallel.executor(2) as ex:
        assert ex._mp_context.get_start_method() == "spawn"


def test_workers_finish_after_parent_used_milp():
    # the parent uses the MILP solver first - the situation that hangs forked workers
    assert _solve(0) == 1
    out = parallel.pmap(_solve, [1, 2, 3, 4], workers=2, timeout=120)
    assert out == [1, 1, 1, 1]


def test_no_fork_or_unmanaged_pools_in_code():
    bad = []
    for f in list((ROOT / "nflmain").rglob("*.py")) + list((ROOT / "research").rglob("*.py")):
        if f.name == "parallel.py":
            continue
        src = f.read_text()
        if re.search(r"""["']fork["']|forkserver|os\.fork\(""", src):
            bad.append(f"{f.name}: fork start method")
        if re.search(r"\b(ProcessPoolExecutor|multiprocessing\.Pool|mp\.Pool|Pool\()", src):
            bad.append(f"{f.name}: process pool not from nflmain.parallel")
    assert not bad, bad
