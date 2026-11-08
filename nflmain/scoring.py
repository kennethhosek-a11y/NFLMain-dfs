"""(Copied from the NFL Showdown repo, nfl-dfs, 11/07/2026; adapted for Classic.)
DraftKings NFL scoring (Classic and Showdown use the same scoring; Showdown CPT = 1.5x)."""
from __future__ import annotations

import numpy as np

CPT_MULT = 1.5


def offense(pass_yd=0, pass_td=0, ints=0, rush_yd=0, rush_td=0, rec=0, rec_yd=0, rec_td=0,
            fum_lost=0, two_pt=0, ret_td=0):
    """Works on scalars or numpy arrays."""
    pts = (0.04 * pass_yd + 4 * pass_td - 1 * ints
           + 0.1 * rush_yd + 6 * rush_td
           + 1 * rec + 0.1 * rec_yd + 6 * rec_td
           - 1 * fum_lost + 2 * two_pt + 6 * ret_td)
    pts = pts + 3 * (np.asarray(pass_yd) >= 300) + 3 * (np.asarray(rush_yd) >= 100) \
        + 3 * (np.asarray(rec_yd) >= 100)
    return pts


def points_allowed(pa):
    pa = np.asarray(pa)
    return np.select([pa == 0, pa <= 6, pa <= 13, pa <= 20, pa <= 27, pa <= 34],
                     [10, 7, 4, 1, 0, -1], -4)


def dst(sacks=0, ints=0, fum_rec=0, td=0, safeties=0, blocks=0, pa=0):
    return (1 * np.asarray(sacks) + 2 * np.asarray(ints) + 2 * np.asarray(fum_rec)
            + 6 * np.asarray(td) + 2 * np.asarray(safeties) + 2 * np.asarray(blocks)
            + points_allowed(pa))


def kicker(fg_0_39=0, fg_40_49=0, fg_50=0, xp=0):
    return 3 * np.asarray(fg_0_39) + 4 * np.asarray(fg_40_49) + 5 * np.asarray(fg_50) + np.asarray(xp)
