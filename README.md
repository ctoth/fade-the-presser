# Fade the Presser

**Read the minutes. Fade the presser.** A meeting-by-meeting ledger of what the
Fed Chair said at the press conference (T+0) against what the FOMC minutes
recorded three weeks later (T+21) — as verbatim, machine-verified quote pairs,
classified rather than scored — plus the market test the title implies.

Premise from Claudia Sahm's [Stay-At-Home Macro](https://stayathomemacro.substack.com/)
post "Read the Minutes, Fade the Presser" (Aug 20, 2026).

Live: https://ctoth.github.io/fade-the-presser/

## What it publishes

- **Claim pairs.** For each substantive topic, a condensed verbatim excerpt from
  the Chair beside one from the minutes. Every quoted fragment is checked
  verbatim against the Federal Reserve source text before publication; an
  analysis run with any unverifiable quote is discarded.
- **A classification per pair**, not a score: `CONTRADICTION` (both accounts
  can't be true), `ATTRIBUTION` (same facts, the Chair presents a minority or
  personal view as the Committee's, or vice versa), `EMPHASIS`, `CONSISTENT`.
  Meeting-level output is counts, with how many of the independent runs found a
  contradiction at all.
- **The market test.** 2-year Treasury yield change (FRED `DGS2`, daily close)
  on decision day vs. on minutes-release day. Opposite signs = the presser was
  faded. Coarse, free, and a real number with a real unit.

## Layout

- `index.html` — the whole site, self-contained; meeting data lives in an
  embedded JSON block.
- `analyze.py` — Claude extraction + classification, quote verification,
  multi-run selection, JSON injection.
- `market.py` — FRED fetch and the decision-day / minutes-day comparison.
- `auto_update.py` — scrapes the Fed calendar (meeting dates + minutes release
  dates), fetches documents, strips PDF running headers, runs the above,
  refreshes market data for all meetings (FRED lags a day).
- `.github/workflows/update.yml` — daily cron + manual dispatch (`dry_run`
  input); commits `index.html` when it changes.

## Running it yourself

```bash
pip install anthropic pydantic pypdf requests
export ANTHROPIC_API_KEY=...        # or `ant auth login`
python auto_update.py --dry-run     # fetch/parse/market only
python auto_update.py               # classify any new meeting
```

Not investment advice. Not affiliated with the Federal Reserve.
