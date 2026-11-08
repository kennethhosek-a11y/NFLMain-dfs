"""(Copied from the NFL Showdown repo, nfl-dfs, 11/07/2026; adapted for Classic.)
Team abbreviations (DK / ESPN / nflverse spell some differently) + stadium info."""
from __future__ import annotations

ALIASES = {"JAC": "JAX", "WSH": "WAS", "LA": "LAR", "STL": "LAR", "SD": "LAC", "OAK": "LV",
           "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "SL": "LAR"}

# home stadium: (lat, lon, indoor/retractable roof usually closed)
STADIUMS = {
    "ARI": (33.528, -112.263, True), "ATL": (33.755, -84.401, True), "BAL": (39.278, -76.623, False),
    "BUF": (42.774, -78.787, False), "CAR": (35.226, -80.853, False), "CHI": (41.862, -87.617, False),
    "CIN": (39.095, -84.516, False), "CLE": (41.506, -81.700, False), "DAL": (32.748, -97.093, True),
    "DEN": (39.744, -105.020, False), "DET": (42.340, -83.046, True), "GB": (44.501, -88.062, False),
    "HOU": (29.685, -95.411, True), "IND": (39.760, -86.164, True), "JAX": (30.324, -81.637, False),
    "KC": (39.049, -94.484, False), "LAC": (33.953, -118.339, True), "LAR": (33.953, -118.339, True),
    "LV": (36.091, -115.184, True), "MIA": (25.958, -80.239, False), "MIN": (44.974, -93.258, True),
    "NE": (42.091, -71.264, False), "NO": (29.951, -90.081, True), "NYG": (40.814, -74.074, False),
    "NYJ": (40.814, -74.074, False), "PHI": (39.901, -75.168, False), "PIT": (40.447, -80.016, False),
    "SEA": (47.595, -122.332, False), "SF": (37.403, -121.970, False), "TB": (27.976, -82.503, False),
    "TEN": (36.166, -86.771, False), "WAS": (38.908, -76.865, False),
}


def canon(t: str) -> str:
    t = str(t).strip().upper()
    return ALIASES.get(t, t)
