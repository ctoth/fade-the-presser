"""Fade the Presser — unattended updater (runs in GitHub Actions).

Scrapes the Fed's FOMC calendar for meetings whose minutes are published,
skips meetings already in index.html, and for each new one: fetches the
minutes (HTML) and the press-conference transcript (PDF), runs the
claim-pair classification from analyze.py, and injects the meeting into
index.html. On every run it also refreshes the market test (market.py) for
all recorded meetings, since FRED publishes the minutes-day close a day
late. Exits 0 quietly when there is nothing new.

Usage:
    python auto_update.py            # full run (needs ANTHROPIC_API_KEY)
    python auto_update.py --dry-run  # fetch + parse + market only, no Claude call

Requires: pip install anthropic pydantic pypdf requests
"""

from __future__ import annotations

import argparse
import calendar as cal
import io
import os
import re
import sys
from datetime import date, datetime
from html.parser import HTMLParser

import requests
from pypdf import PdfReader

from analyze import classify_meeting, inject, load_data, save_data
from market import load_series, market_block

FED = "https://www.federalreserve.gov"
CALENDAR = f"{FED}/monetarypolicy/fomccalendars.htm"
MINUTES = f"{FED}/monetarypolicy/fomcminutes{{code}}.htm"
PRESSER = f"{FED}/mediacenter/files/FOMCpresconf{{code}}.pdf"
UA = {"User-Agent": "fade-the-presser/1.0 (FOMC communication tracker)"}
MAX_MEETINGS_PER_RUN = 2  # backstop against a surprise backlog
ERA_START = "20260601"   # the Warsh Fed only — his first meeting was June 2026

# Running header the Fed stamps on every transcript page, e.g.
# "July 29, 2026 Chairman Warsh's Press Conference FINAL Page 15 of 21".
# pypdf interleaves it mid-sentence at page breaks, which would break quote
# verification, so strip it wherever it appears.
RUNNING_HEADER = re.compile(
    r"[A-Z][a-z]+ \d{1,2}, \d{4}\s+Chair(?:man|woman)?\s+[^\n]{0,40}?Press Conference"
    r"\s+(?:FINAL|PRELIMINARY)?\s*Page \d+ of \d+"
)
# "fomcminutes20260729.htm ... (Released August 19, 2026)" on the calendar page
RELEASE_RE = re.compile(r"fomcminutes(\d{8})\.htm.{0,400}?\(Released ([A-Z][a-z]+ \d{1,2}, \d{4})\)", re.DOTALL)


class TextExtractor(HTMLParser):
    SKIP = {"script", "style", "nav", "header", "footer"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.depth:
            self.depth -= 1

    def handle_data(self, data):
        if not self.depth and data.strip():
            self.parts.append(data.strip())


def html_to_text(html: str) -> str:
    p = TextExtractor()
    p.feed(html)
    return "\n".join(p.parts)


def pdf_to_text(blob: bytes) -> str:
    text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(blob)).pages)
    return RUNNING_HEADER.sub(" ", text)


def published_minutes() -> dict[str, date | None]:
    """Meeting code -> minutes release date, for every meeting with minutes on the calendar."""
    page = requests.get(CALENDAR, headers=UA, timeout=60)
    page.raise_for_status()
    out: dict[str, date | None] = {c: None for c in set(re.findall(r"fomcminutes(\d{8})\.htm", page.text))}
    for code, released in RELEASE_RE.findall(page.text):
        out[code] = datetime.strptime(released, "%B %d, %Y").date()
    return out


def refresh_market(releases: dict[str, date | None]) -> None:
    """Recompute the market block for every recorded meeting (FRED lags a day)."""
    series = load_series()
    html, match, data = load_data()
    changed = False
    for m in data["meetings"]:
        if m.get("status") != "scored":
            continue
        code = m["code"]
        block = market_block(series, date.fromisoformat(m["decision_date"]), releases.get(code))
        if block != m.get("market"):
            m["market"] = block
            changed = True
            print(f"{m['id']}: market — decision {block['decision_chg_bp']} bp, "
                  f"minutes {block['minutes_chg_bp']} bp, faded={block['faded']}")
    if changed:
        save_data(html, match, data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="fetch and parse only; skip the Claude call")
    args = ap.parse_args()

    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Add it as a repo secret: "
                 "gh secret set ANTHROPIC_API_KEY --repo ctoth/fade-the-presser")

    releases = published_minutes()
    _, _, data = load_data()
    done = {m["id"] for m in data["meetings"] if m.get("status") == "scored"}
    new = 0
    for code in sorted(releases, reverse=True):
        if code < ERA_START:
            continue
        year, month, day = code[:4], int(code[4:6]), int(code[6:8])
        meeting_id = f"{year}-{month:02d}"
        if meeting_id in done:
            continue
        if new >= MAX_MEETINGS_PER_RUN:
            print(f"{meeting_id}: deferred to next run (per-run cap)")
            continue

        presser_url = PRESSER.format(code=code)
        head = requests.head(presser_url, headers=UA, timeout=60, allow_redirects=True)
        if head.status_code != 200:
            print(f"{meeting_id}: minutes out, presser transcript not posted yet "
                  f"({head.status_code}) — will retry next run")
            continue

        print(f"{meeting_id}: fetching minutes + presser transcript…")
        minutes_html = requests.get(MINUTES.format(code=code), headers=UA, timeout=120)
        minutes_html.raise_for_status()
        presser_pdf = requests.get(presser_url, headers=UA, timeout=120)
        presser_pdf.raise_for_status()

        minutes_text = html_to_text(minutes_html.text)
        presser_text = pdf_to_text(presser_pdf.content)
        if len(minutes_text) < 5000 or len(presser_text) < 5000:
            sys.exit(f"{meeting_id}: extracted text suspiciously short "
                     f"(minutes {len(minutes_text)}, presser {len(presser_text)}) — aborting")

        label = f"{cal.month_name[month]} {year} FOMC"
        date_str = f"Ended {cal.month_abbr[month]} {day}, {year}"
        if args.dry_run:
            print(f"{meeting_id}: DRY RUN ok — minutes {len(minutes_text):,} chars, "
                  f"presser {len(presser_text):,} chars ({label}, minutes released {releases[code]})")
            new += 1
            continue

        print(f"{meeting_id}: classifying ({label})…")
        result = classify_meeting(presser_text, minutes_text)
        inject({
            "id": meeting_id, "code": code, "label": label, "date": date_str,
            "decision_date": f"{year}-{month:02d}-{day:02d}",
            "minutes_released": releases[code].isoformat() if releases[code] else None,
            "status": "scored", **result,
        })
        k = result["counts"]
        print(f"{meeting_id}: {k['CONTRADICTION']} contradiction, {k['ATTRIBUTION']} attribution, "
              f"{k['EMPHASIS']} emphasis, {k['CONSISTENT']} consistent "
              f"(contradiction in {result['contradiction_runs']}/{result['runs']} runs)")
        new += 1

    refresh_market(releases)
    print("done — new meetings classified:" if new else "done — nothing new.", new or "")


if __name__ == "__main__":
    main()
