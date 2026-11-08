"""(Copied from the NFL Showdown repo, nfl-dfs, 11/07/2026; adapted for Classic.)
Name matching across DK / ESPN / nflverse."""
from __future__ import annotations

import re
import unicodedata

SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'`\-]", "", s)
    s = SUFFIX.sub("", s)
    return re.sub(r"\s+", " ", s).strip()
