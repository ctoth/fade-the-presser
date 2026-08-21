"""Fade the Presser — the market test.

"Fade the presser" is a falsifiable claim: the press conference moves rates
and the minutes move them back. This module measures it with the 2-year
Treasury constant-maturity yield as published daily by the U.S. Treasury (par yield curve, the series
FRED republishes as DGS2):

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
import time
from datetime import date, datetime, timedelta

import requests

TREASURY_CSV = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
                "daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
                "&field_tdr_date_value={year}&page&_format=csv")
START_YEAR = 2026          # the Warsh era
SERIES = "UST 2-year par yield"
COLUMN = "2 Yr"
UA = {"User-Agent": "fade-the-presser/1.0 (FOMC communication tracker)"}


def load_series(start_year: int = START_YEAR) -> dict[date, float]:
    """Daily 2-year par yields from the Treasury's own published curve (the
    primary source behind FRED's DGS2, which throttles CI runners)."""
    out: dict[date, float] = {}
    for year in range(start_year, date.today().year + 1):
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                r = requests.get(TREASURY_CSV.format(year=year), headers=UA, timeout=(15, 90))
                r.raise_for_status()
                break
            except requests.RequestException as e:
                last_err = e
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"Treasury fetch failed for {year} after 3 attempts: {last_err}")
        for row in csv.DictReader(io.StringIO(r.text)):
            val = (row.get(COLUMN) or "").strip()
            if not val:
                continue
            out[datetime.strptime(row["Date"], "%m/%d/%Y").date()] = float(val)
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
