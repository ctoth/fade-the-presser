# Fade the Presser

**Read the minutes. Fade the presser.** A meeting-by-meeting ledger of the gap
between what the Fed Chair says at the press conference (T+0) and what the FOMC
minutes record three weeks later (T+21) — scored in rhetorical basis points.

Premise from Claudia Sahm's [Stay-At-Home Macro](https://stayathomemacro.substack.com/)
post "Read the Minutes, Fade the Presser" (Aug 20, 2026), which seeded the
July 2026 meeting data.

## The site

`index.html` — fully self-contained static page. Open it in a browser, that's it.
Meeting data lives in the embedded `<script type="application/json" id="meeting-data">`
block, so the page needs no server and no fetch.

- **Presser–Minutes Basis**: mean divergence across scored claims for the
  latest meeting. 0 bp = the Chair speaks for the Committee.
- **Claim ledger**: per claim, the Chair's condensed quote vs. the minutes'
  condensed quote, a gist for each, a 0–100 divergence score, and a verdict note.

## The pipeline ("always and forever")

Each FOMC cycle, when the minutes drop:

```bash
pip install anthropic pydantic
export ANTHROPIC_API_KEY=...   # or `ant auth login`

python analyze.py \
  --presser transcripts/2026-09-presser.txt \
  --minutes transcripts/2026-09-minutes.txt \
  --id 2026-09 --label "September 2026 FOMC" --date "Sept 16-17, 2026"
```

`analyze.py` sends both documents to Claude (`claude-opus-5`) with a structured
output schema, gets back scored claim pairs, computes the basis, and injects the
meeting straight into `index.html` (replacing any pending placeholder with the
same id). Commit, push, done — repeat every six weeks until regime change ends
or the regime does.

## Scoring rubric

| Score | Reading |
|---|---|
| 0–30 | ALIGNED — the Chair's account matches the record |
| 31–60 | MODERATE — emphasis drifts from the record |
| 61–100 | WIDE — the account and the record materially disagree |

Not investment advice. Not affiliated with the Federal Reserve. Basis points
are rhetorical, not tradable — mostly.
