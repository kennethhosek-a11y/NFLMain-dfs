"""(Copied from the NFL Showdown repo, nfl-dfs, 11/07/2026; adapted for Classic.)
Build settings: defaults and safe ranges in config/settings.yaml, Kenny's changes on top.

Layers (later wins): default -> data/settings/standing.yaml ("from now on")
-> data/settings/<slate date>.yaml ("today only"). Values out of range, of the wrong type, or for
unknown keys are rejected and reported; the lower layer's value stays. Code reads a setting when
it uses it (`get("portfolio.max_exposure")`); one command run uses one consistent set.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_FILE = ROOT / "config" / "settings.yaml"
SETTINGS_DIR = ROOT / "data" / "settings"
STANDING = "standing.yaml"
ET = ZoneInfo("America/New_York")
DAY_STARTS_HOUR = 6               # before 6 AM ET a run still belongs to the previous day's slate
DATE_ENV = "NFLMAIN_SLATE_DATE"    # YYYY-MM-DD, overrides the slate date (testing)
DIR_ENV = "NFLMAIN_SETTINGS_DIR"   # overrides data/settings (testing)

_cache: dict | None = None
_overrides: dict = {}             # command-line flags for this run (win over every layer)


def schema() -> dict[str, dict]:
    """Flat {"group.key": spec} from config/settings.yaml, in file order."""
    raw = yaml.safe_load(SCHEMA_FILE.read_text())
    return {f"{g}.{k}": spec for g, keys in raw.items() for k, spec in keys.items()}


def slate_date(now: datetime | None = None) -> date:
    if os.environ.get(DATE_ENV):
        return date.fromisoformat(os.environ[DATE_ENV])
    now = now or datetime.now(ET)
    return (now - timedelta(hours=DAY_STARTS_HOUR)).date()


def _settings_dir() -> Path:
    return Path(os.environ[DIR_ENV]) if os.environ.get(DIR_ENV) else SETTINGS_DIR


def check(key: str, value, spec: dict | None):
    """(coerced value, None) or (None, reason)."""
    if spec is None:
        return None, "unknown setting"
    if isinstance(value, bool) or value is None or isinstance(value, (list, dict)):
        return None, f"not a number: {value!r}"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None, f"not a number: {value!r}"
    if num != num:                                  # NaN
        return None, f"not a number: {value!r}"
    if spec["type"] == "int":
        if num != int(num):
            return None, f"must be a whole number: {value!r}"
        num = int(num)
    if not spec["min"] <= num <= spec["max"]:
        return None, f"{value} is outside the safe range {spec['min']}-{spec['max']}"
    return num, None


def _read_layer(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        d = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        return {"__error__": f"unreadable file: {e}"}
    return d if isinstance(d, dict) else {"__error__": "file is not a key: value list"}


def load(day: date | None = None, settings_dir: Path | None = None) -> dict:
    """Effective settings with provenance:
    {"date", "values": {key: value}, "source": {key: "default"|"from now on"|"today only"},
     "rejected": [{"key", "value", "layer", "reason"}]}."""
    spec = schema()
    day = day or slate_date()
    sdir = settings_dir or _settings_dir()
    values = {k: s["default"] for k, s in spec.items()}
    source = {k: "default" for k in spec}
    rejected = []
    for layer, path in (("from now on", sdir / STANDING), ("today only", sdir / f"{day.isoformat()}.yaml")):
        for key, val in _read_layer(path).items():
            if key == "__error__":
                rejected.append({"key": path.name, "value": None, "layer": layer, "reason": val})
                continue
            v, why = check(key, val, spec.get(key))
            if why:
                rejected.append({"key": key, "value": val, "layer": layer, "reason": why})
            else:
                values[key], source[key] = v, layer
    return {"date": day.isoformat(), "values": values, "source": source, "rejected": rejected}


def use(day: date | str | None = None) -> dict:
    """Load the settings for a slate date and make them current for this run."""
    global _cache
    if isinstance(day, str):
        day = date.fromisoformat(day)
    _cache = load(day)
    return _cache


def current() -> dict:
    global _cache
    if _cache is None:
        _cache = load()
    return _cache


def get(key: str):
    if key in _overrides:
        return _overrides[key]
    return current()["values"][key]


SLATE_TYPES = ("early", "main", "afternoon", "thanksgiving")


def for_type(slate_type: str, key: str):
    """A per-slate-type setting, e.g. for_type("afternoon", "gate_pct")."""
    t = slate_type if slate_type in SLATE_TYPES else "main"
    return get(f"{t}.{key}")


def override(key: str, value) -> None:
    """A command-line flag for this run only (checked against the safe range like any layer)."""
    v, why = check(key, value, schema().get(key))
    if why:
        raise ValueError(f"--{key}: {why}")
    _overrides[key] = v


def reset_cache() -> None:
    global _cache
    _cache = None
    _overrides.clear()


def changed(eff: dict | None = None) -> list[dict]:
    """Settings not at their default, in schema order."""
    eff = eff or current()
    spec = schema()
    out = [{"key": k, "label": spec[k]["label"], "value": eff["values"][k], "default": spec[k]["default"],
            "source": eff["source"][k]} for k in spec if eff["source"][k] != "default"]
    for k, v in _overrides.items():
        out = [c for c in out if c["key"] != k] + [{"key": k, "label": spec[k]["label"], "value": v,
                                                     "default": spec[k]["default"], "source": "this run only"}]
    return out


def summary_line(eff: dict | None = None) -> str:
    eff = eff or current()
    ch = changed(eff)
    head = (f"[settings] slate {eff['date']}: all defaults" if not ch else
            f"[settings] slate {eff['date']}: {len(ch)} changed - "
            + "; ".join(f"{c['key']} {c['value']} ({c['source']}, default {c['default']})" for c in ch))
    if eff["rejected"]:
        head += f" | {len(eff['rejected'])} rejected: " + "; ".join(
            f"{r['key']}={r['value']!r} ({r['reason']})" for r in eff["rejected"])
    return head


def plain(eff: dict | None = None) -> str:
    """One plain line for Kenny, e.g. 'Settings: max exposure 40% (from now on); 3000 sims (today only)'."""
    eff = eff or current()
    ch = changed(eff)
    s = "Settings: all defaults" if not ch else "Settings: " + "; ".join(
        f"{c['label']} {_fmt(c['key'], c['value'])} ({c['source']})" for c in ch)
    if eff["rejected"]:
        s += " | rejected: " + "; ".join(f"{r['key']}={r['value']!r} ({r['reason']})" for r in eff["rejected"])
    return s


def _fmt(key, v):
    if key in ("portfolio.max_exposure",) or key.endswith(".gate_pct"):
        return f"{v:.0%}"
    return f"{v:,}" if isinstance(v, int) else f"{v:g}"


def import_dashboard(payload: dict, day: date | None = None, settings_dir: Path | None = None) -> dict:
    """Write the Slate Desk's settings into the settings files and return what will apply.

    payload = {"standing": {"values": {key: value}}, "today": {"date": "YYYY-MM-DD", "values": {...}}}
    The standing file is replaced by the page's "from now on" values; the today file is written only
    when the page's today date is this slate's date (otherwise those values are stale and ignored).
    Only valid values are written; invalid ones come back under "rejected"."""
    spec = schema()
    day = day or slate_date()
    sdir = settings_dir or _settings_dir()
    sdir.mkdir(parents=True, exist_ok=True)
    rejected, notes = [], []

    def clean(vals, layer):
        out = {}
        for k, v in (vals or {}).items():
            cv, why = check(k, v, spec.get(k))
            if why:
                rejected.append({"key": k, "value": v, "layer": layer, "reason": why})
            elif cv != spec[k]["default"] or layer == "today only":
                out[k] = cv
        return out

    _write(sdir / STANDING, clean((payload.get("standing") or {}).get("values"), "from now on"))
    today = payload.get("today") or {}
    today_file = sdir / f"{day.isoformat()}.yaml"
    if today.get("date") == day.isoformat():
        vals = clean(today.get("values"), "today only")
        if vals:
            _write(today_file, vals)
        elif today_file.exists():
            today_file.unlink()
    elif (today.get("values") or {}) and today.get("date"):
        notes.append(f"today-only changes dated {today.get('date')} ignored (slate is {day.isoformat()})")
    eff = load(day, sdir)
    eff["rejected"] = rejected + eff["rejected"]
    eff["notes"] = notes
    return eff


def payload_from(path: Path) -> dict:
    """The page's settings documents from a JSON file ({"standing": ..., "today": ...}) or from the
    folder an ArtifactData read with `out_dir` writes (`<dir>/settings/standing.json`, `today.json`)."""
    path = Path(path)
    if path.is_file():
        return json.loads(path.read_text())
    d = path / "settings" if (path / "settings").is_dir() else path
    out = {}
    for name in ("standing", "today"):
        f = d / f"{name}.json"
        if f.exists():
            doc = json.loads(f.read_text())
            out[name] = doc.get("data", doc) if isinstance(doc, dict) and "values" not in doc else doc
    return out


def _write(path: Path, vals: dict) -> None:
    head = ("# Written by `python -m nflmain settings --import` from the Slate Desk.\n"
            "# Keys and safe ranges: config/settings.yaml\n")
    path.write_text(head + (yaml.safe_dump(vals, sort_keys=False) if vals else "{}\n"))


def report(eff: dict | None = None) -> dict:
    """JSON-ready report of what applies (for the page's `settings/applied` document)."""
    eff = eff or current()
    return {"slate_date": eff["date"], "at": datetime.now(ET).strftime("%Y-%m-%d %I:%M %p ET"),
            "line": plain(eff), "changed": changed(eff), "rejected": eff["rejected"], "notes": eff.get("notes", [])}


def dumps(obj) -> str:
    return json.dumps(obj, indent=2, default=str)
