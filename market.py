"""Fade the Presser — the market test.

"Fade the presser" is a falsifiable claim: the press conference moves rates
and the minutes move them back. This module measures it with the 2-year
Treasury constant-maturity yield (FRED series DGS2, daily close, H.15):

    decision-day change  = close on the meeting's final day - prior close
    minutes-day change   = close on the minutes release day - prior close

The presser starts at 2:30 p.m. ET and the minutes are released at 2:00 p.m.
ET, so the daily close captures each event's session. A meeting is marked
"faded" when the two changes have opposite signs and neither is zero.
Coarse, free, and a real number with a real unit.
"""

from __future__ import annotations

import csv
import io
from datetime import date, timedelta

import requests

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
SERIES = "DGS2"
UA = {"User-Agent": "fade-the-presser/1.0 (FOMC communication tracker)"}


def load_series(series: str = SERIES) -> dict[date, float]:
    r = requests.get(FRED_CSV.format(series=series), headers=UA, timeout=60)
    r.raise_for_status()
    out: dict[date, float] = {}
    for row in csv.DictReader(io.StringIO(r.text)):
        val = row.get(series) or row.get("value") or ""
        if val.strip() in ("", "."):
            continue            # market holiday / not yet published
        out[date.fromisoformat(row["observation_date"] if "observation_date" in row else row["DATE"])] = float(val)
    return out


def change_bp(series: dict[date, float], day: date) -> int | None:
    """Close-to-close change on `day` in basis points, or None if not yet published."""
    if day not in series:
        return None
    prev = day - timedelta(days=1)
    for _ in range(7):
        if prev in series:
            return round((series[day] - series[prev]) * 100)
        prev -= timedelta(days=1)
    return None


def market_block(series: dict[date, float], decision_day: date, minutes_day: date | None) -> dict:
    d = change_bp(series, decision_day)
    m = change_bp(series, minutes_day) if minutes_day else None
    faded = None if d is None or m is None or d == 0 or m == 0 else (d > 0) != (m > 0)
    return {
        "series": SERIES,
        "decision_date": decision_day.isoformat(),
        "decision_chg_bp": d,
        "minutes_date": minutes_day.isoformat() if minutes_day else None,
        "minutes_chg_bp": m,
        "faded": faded,
    }
