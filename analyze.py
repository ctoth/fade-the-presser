"""Fade the Presser — meeting analyzer.

Feed it a press-conference transcript and the FOMC minutes for the same
meeting. Claude extracts claim pairs (what the Chair said vs. what the
Committee recorded) and scores each divergence 0-100. The scoring is run
N_RUNS times independently; every quote fragment in every run is checked
against the source text, runs with any unverifiable quote are discarded,
and the published basis is the mean across surviving runs with its range.
The claim pairs shown are those of the run closest to that mean.

Usage:
    python analyze.py --presser transcripts/2026-09-presser.txt \
                      --minutes transcripts/2026-09-minutes.txt \
                      --id 2026-09 --label "September 2026 FOMC" \
                      --date "Sept 16-17, 2026"

Requires: pip install anthropic pydantic
Auth: ANTHROPIC_API_KEY, or `ant auth login` (the SDK picks up the profile).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import anthropic
from pydantic import BaseModel, Field

MODEL = "claude-opus-5"
N_RUNS = 5
MIN_VALID_RUNS = 3
MIN_FRAGMENT_CHARS = 8   # alphanumeric chars; shorter fragments aren't checkable
INDEX = Path(__file__).parent / "index.html"

DATA_RE = re.compile(
    r'(<script type="application/json" id="meeting-data">\s*)(.*?)(\s*</script>)',
    re.DOTALL,
)

QUOTE_RULES = ("Condensed verbatim excerpt from the source text. Copy words exactly; "
               "mark omitted spans with an ellipsis (...). No paraphrase, no bracketed "
               "insertions, no corrections of grammar.")


class Claim(BaseModel):
    topic: str = Field(description="Very short label, 1-3 words, e.g. 'Data dependence'")
    question: str = Field(description="The question this pair answers, under 10 words")
    presser_quote: str = Field(description="From the press conference. " + QUOTE_RULES)
    presser_gist: str = Field(description="One sentence: the Chair's position, plainly")
    minutes_quote: str = Field(description="From the minutes. " + QUOTE_RULES)
    minutes_gist: str = Field(description="One sentence: the Committee's recorded position, plainly")
    score: int = Field(description="Divergence 0-100. 0 = same substance; 50 = clear difference "
                                   "of emphasis or degree; 100 = direct contradiction of fact or rationale")
    note: str = Field(description="One sentence explaining the score. Sparing <b> allowed. No numbers.")


class MeetingAnalysis(BaseModel):
    decision: str = Field(description="e.g. 'HOLD', 'HIKE +25', 'CUT -25'")
    rate: str = Field(description="Target range after the meeting, e.g. '3.50-3.75%'")
    dissents: str = Field(description="e.g. '3 (hawkish - favored a hike)' or 'none'")
    inflation: str = Field(description="One short phrase on the inflation backdrop")
    summary: str = Field(description="2-3 sentences on the meeting and how the two accounts compare. "
                                     "Do not cite any numeric scores; they are computed separately.")
    claims: List[Claim] = Field(description="4-6 pairs covering the most substantive topics, "
                                            "in the order the press conference raised them")


SYSTEM = """You are a careful reader of Federal Reserve communications. You are given
the transcript of an FOMC press conference and the minutes of the same meeting,
published three weeks later. Your job is to compare the Chair's account of the
decision, its rationale and the outlook with the Committee's own record of the
same points.

Method:
- Identify the 4-6 most substantive topics the Chair addressed that the minutes
  also address. Choose by substance, not by how much the accounts differ. A
  meeting where the accounts agree should produce pairs that score low.
- For each topic, pair a condensed verbatim excerpt from the Chair with a
  condensed verbatim excerpt from the minutes, and score the divergence 0-100
  symmetrically: 0 means the same substance in different words; around 50 means
  a clear difference of emphasis, degree or attribution; 100 means a direct
  contradiction of fact or rationale. Differences of tone alone score low.
- Read minutes vocabulary the way Fed-watchers do: "most" and "almost all"
  indicate consensus; "many" a strong majority; "several" a meaningful bloc;
  "some", "various" and "a few" are well short of a majority. A view the minutes
  attribute to a minority is not the Committee's view, and a personal framework
  the Chair offers from the podium is not the Committee's unless the minutes
  record it as such.
- Quotes must be exact text from the documents with omissions marked by an
  ellipsis. Every quote will be machine-checked against the source; a quote that
  cannot be found verbatim invalidates the whole analysis."""


# ---------------------------------------------------------------- verification

def _norm(s: str) -> str:
    """Collapse to lowercase alphanumerics so line breaks, hyphenation, curly
    quotes and PDF spacing artefacts cannot cause spurious mismatches."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _fragments(quote: str) -> list[str]:
    quote = re.sub(r"\[[^\]]*\]", "", quote)          # drop bracketed insertions
    parts = re.split(r"(?:\.\s*){3,}|…", quote)  # split on ..., . . . or …
    return [p for p in (_norm(p) for p in parts) if len(p) >= MIN_FRAGMENT_CHARS]


def verify_quotes(a: MeetingAnalysis, presser: str, minutes: str) -> list[str]:
    """Return a list of human-readable misses; empty means every fragment was found."""
    np_, nm = _norm(presser), _norm(minutes)
    misses = []
    for c in a.claims:
        for frag in _fragments(c.presser_quote):
            if frag not in np_:
                misses.append(f"[{c.topic}] presser: …{frag[:60]}…")
        for frag in _fragments(c.minutes_quote):
            if frag not in nm:
                misses.append(f"[{c.topic}] minutes: …{frag[:60]}…")
        if not _fragments(c.presser_quote) or not _fragments(c.minutes_quote):
            misses.append(f"[{c.topic}] a quote had no checkable fragment")
    return misses


# ---------------------------------------------------------------- scoring

def verdict_for(mean: float) -> str:
    return "ALIGNED" if mean <= 30 else "MODERATE" if mean <= 60 else "WIDE"


def run_once(client: anthropic.Anthropic, presser: str, minutes: str) -> MeetingAnalysis:
    response = client.messages.parse(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        messages=[{
            "role": "user",
            "content": (
                "<press_conference_transcript>\n" + presser +
                "\n</press_conference_transcript>\n\n"
                "<fomc_minutes>\n" + minutes + "\n</fomc_minutes>\n\n"
                "Compare the Chair's account with the Committee's record."
            ),
        }],
        output_format=MeetingAnalysis,
    )
    if response.stop_reason == "refusal":
        sys.exit(f"Model declined the request (stop_details: {response.stop_details}).")
    return response.parsed_output


def score_meeting(presser: str, minutes: str, n_runs: int = N_RUNS) -> dict:
    """Run the analysis n_runs times, verify every run's quotes, aggregate."""
    client = anthropic.Anthropic()
    with ThreadPoolExecutor(max_workers=n_runs) as pool:
        runs = list(pool.map(lambda _: run_once(client, presser, minutes), range(n_runs)))

    valid: list[tuple[float, MeetingAnalysis]] = []
    for i, a in enumerate(runs, 1):
        misses = verify_quotes(a, presser, minutes)
        mean = sum(c.score for c in a.claims) / len(a.claims)
        if misses:
            print(f"  run {i}: REJECTED ({len(misses)} unverifiable quote fragment(s))")
            for m in misses[:4]:
                print("     ", m)
            continue
        print(f"  run {i}: ok — {len(a.claims)} claims, mean {mean:.1f}")
        valid.append((mean, a))

    if len(valid) < MIN_VALID_RUNS:
        sys.exit(f"Only {len(valid)}/{n_runs} runs passed quote verification "
                 f"(need {MIN_VALID_RUNS}); refusing to publish.")

    means = [m for m, _ in valid]
    basis = sum(means) / len(means)
    _, shown = min(valid, key=lambda mv: abs(mv[0] - basis))   # run closest to the mean
    return {
        "basis": round(basis),
        "basis_range": [round(min(means)), round(max(means))],
        "runs": len(valid),
        "run_means": [round(m, 1) for m in means],
        "verdict": verdict_for(basis),
        **shown.model_dump(),
    }


# ---------------------------------------------------------------- publish

def inject(meeting: dict) -> None:
    html = INDEX.read_text(encoding="utf-8")
    match = DATA_RE.search(html)
    if not match:
        sys.exit("index.html: meeting-data block not found.")
    data = json.loads(match.group(2))
    data["meetings"] = [m for m in data["meetings"] if m["id"] != meeting["id"]]
    data["meetings"].insert(0, meeting)
    data["meetings"].sort(key=lambda m: m["id"], reverse=True)
    blob = json.dumps(data, ensure_ascii=False, indent=1)
    INDEX.write_text(html[:match.start(2)] + blob + html[match.end(2):], encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Score presser-vs-minutes divergence.")
    ap.add_argument("--presser", required=True, type=Path)
    ap.add_argument("--minutes", required=True, type=Path)
    ap.add_argument("--id", required=True, help="meeting id, e.g. 2026-09")
    ap.add_argument("--label", required=True, help='e.g. "September 2026 FOMC"')
    ap.add_argument("--date", required=True, help='e.g. "Sept 16-17, 2026"')
    ap.add_argument("--runs", type=int, default=N_RUNS)
    args = ap.parse_args()

    result = score_meeting(
        args.presser.read_text(encoding="utf-8"),
        args.minutes.read_text(encoding="utf-8"),
        n_runs=args.runs,
    )
    inject({"id": args.id, "label": args.label, "date": args.date, "status": "scored", **result})
    lo, hi = result["basis_range"]
    print(f"{args.label}: basis {result['basis']} bp (range {lo}-{hi}, {result['runs']} runs) "
          f"- {result['verdict']} - index.html updated")


if __name__ == "__main__":
    main()
