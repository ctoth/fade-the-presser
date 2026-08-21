"""Fade the Presser — claim-pair extraction and classification.

Given a press-conference transcript and the FOMC minutes for the same
meeting, Claude pairs the Chair's account of each substantive point with the
Committee's record of it, as condensed verbatim quotes, and classifies each
pair into one of four defined types. There is no score: the output is the
pairs themselves, which a reader can check against the two quotes, plus the
counts by type.

Every quote fragment is verified verbatim against the source text; a run
with any unverifiable quote is discarded. The analysis runs N_RUNS times and
the published run is the one with the median count of contradictions; how
many runs found at least one contradiction is published alongside, as the
stability indicator.

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
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Literal

import anthropic
from pydantic import BaseModel, Field

MODEL = "claude-opus-5"
N_RUNS = 3
MIN_VALID_RUNS = 2
MIN_FRAGMENT_CHARS = 8   # alphanumeric chars; shorter fragments aren't checkable
INDEX = Path(__file__).parent / "index.html"
KINDS = ("CONTRADICTION", "ATTRIBUTION", "EMPHASIS", "CONSISTENT")

DATA_RE = re.compile(
    r'(<script type="application/json" id="meeting-data">\s*)(.*?)(\s*</script>)',
    re.DOTALL,
)

QUOTE_RULES = ("Condensed verbatim excerpt from the source text. Copy words exactly; "
               "mark omitted spans with an ellipsis (...). No paraphrase, no bracketed "
               "insertions, no corrections of grammar.")

Kind = Literal["CONTRADICTION", "ATTRIBUTION", "EMPHASIS", "CONSISTENT"]


class Claim(BaseModel):
    topic: str = Field(description="Very short label, 1-3 words, e.g. 'Data dependence'")
    question: str = Field(description="The question this pair answers, under 10 words")
    presser_quote: str = Field(description="From the press conference. " + QUOTE_RULES)
    presser_gist: str = Field(description="One sentence: the Chair's position, plainly")
    minutes_quote: str = Field(description="From the minutes. " + QUOTE_RULES)
    minutes_gist: str = Field(description="One sentence: the Committee's recorded position, plainly")
    kind: Kind = Field(description=(
        "CONTRADICTION: the two accounts cannot both be true. "
        "ATTRIBUTION: same facts, but the Chair presents as the Committee's view something the "
        "minutes attribute to a minority or to the Chair alone (or the reverse). "
        "EMPHASIS: same facts and attribution, materially different weight or framing. "
        "CONSISTENT: same substance in different words."))
    note: str = Field(description="One sentence justifying the classification from the two quotes. "
                                  "Sparing <b> allowed. No numbers.")


class MeetingAnalysis(BaseModel):
    decision: str = Field(description="e.g. 'HOLD', 'HIKE +25', 'CUT -25'")
    rate: str = Field(description="Target range after the meeting, e.g. '3.50-3.75%'")
    dissents: str = Field(description="e.g. '3 (hawkish - favored a hike)' or 'none'")
    inflation: str = Field(description="One short phrase on the inflation backdrop")
    summary: str = Field(description="2-3 sentences on the meeting and how the two accounts compare. "
                                     "No counts or numbers; those are computed separately.")
    claims: List[Claim] = Field(description="4-6 pairs covering the most substantive topics, "
                                            "in the order the press conference raised them")


SYSTEM = """You are a careful reader of Federal Reserve communications. You are given
the transcript of an FOMC press conference and the minutes of the same meeting,
published three weeks later. Your job is to pair the Chair's account of each
substantive point with the Committee's own record of the same point, and to
classify the relationship between the two.

Method:
- Identify the 4-6 most substantive topics the Chair addressed that the minutes
  also address. Choose by substance, not by how much the accounts differ. A
  meeting where the accounts agree should produce mostly CONSISTENT pairs.
- For each topic, pair a condensed verbatim excerpt from the Chair with a
  condensed verbatim excerpt from the minutes, then classify:
    CONTRADICTION - the two accounts cannot both be true (a fact, a count, a
      stated rationale, or whether something was said or proposed).
    ATTRIBUTION - the facts agree, but the Chair presents as the Committee's
      view something the minutes attribute to a minority, to "some" or "a few"
      participants, or to the Chair alone; or the Chair disowns as personal a
      view the minutes record as the Committee's.
    EMPHASIS - same facts and attribution; materially different weight, order
      or framing.
    CONSISTENT - the same substance in different words. Tone alone never
      changes a classification.
- Read minutes vocabulary the way Fed-watchers do: "most" and "almost all"
  indicate consensus; "many" a strong majority; "several" a meaningful bloc;
  "some", "various" and "a few" are well short of a majority.
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
    parts = re.split(r"(?:\.\s*){3,}|…", quote)       # split on ..., . . . or …
    return [p for p in (_norm(p) for p in parts) if len(p) >= MIN_FRAGMENT_CHARS]


def verify_quotes(a: MeetingAnalysis, presser: str, minutes: str) -> list[str]:
    """Return human-readable misses; empty means every fragment was found verbatim."""
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


# ---------------------------------------------------------------- analysis

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
                "Pair and classify the Chair's account against the Committee's record."
            ),
        }],
        output_format=MeetingAnalysis,
    )
    if response.stop_reason == "refusal":
        sys.exit(f"Model declined the request (stop_details: {response.stop_details}).")
    assert response.parsed_output is not None
    return response.parsed_output


def counts_of(a: MeetingAnalysis) -> dict[str, int]:
    c = Counter(cl.kind for cl in a.claims)
    return {k: c.get(k, 0) for k in KINDS}


def classify_meeting(presser: str, minutes: str, n_runs: int = N_RUNS) -> dict:
    """Run n_runs analyses, verify every quote, publish the median-contradiction run."""
    client = anthropic.Anthropic()
    with ThreadPoolExecutor(max_workers=n_runs) as pool:
        runs = list(pool.map(lambda _: run_once(client, presser, minutes), range(n_runs)))

    valid: list[MeetingAnalysis] = []
    for i, a in enumerate(runs, 1):
        misses = verify_quotes(a, presser, minutes)
        if misses:
            print(f"  run {i}: REJECTED ({len(misses)} unverifiable quote fragment(s))")
            for m in misses[:4]:
                print("     ", m)
            continue
        k = counts_of(a)
        print(f"  run {i}: ok — {len(a.claims)} pairs: "
              + ", ".join(f"{v} {n.lower()}" for n, v in k.items() if v))
        valid.append(a)

    if len(valid) < MIN_VALID_RUNS:
        sys.exit(f"Only {len(valid)}/{n_runs} runs passed quote verification "
                 f"(need {MIN_VALID_RUNS}); refusing to publish.")

    # Publish the run at the median on (contradictions, attributions); report stability.
    ranked = sorted(valid, key=lambda a: (counts_of(a)["CONTRADICTION"], counts_of(a)["ATTRIBUTION"]))
    shown = ranked[len(ranked) // 2]
    return {
        "counts": counts_of(shown),
        "runs": len(valid),
        "contradiction_runs": sum(1 for a in valid if counts_of(a)["CONTRADICTION"] > 0),
        **shown.model_dump(),
    }


# ---------------------------------------------------------------- publish

def load_data() -> tuple[str, re.Match, dict]:
    html = INDEX.read_text(encoding="utf-8")
    match = DATA_RE.search(html)
    if not match:
        sys.exit("index.html: meeting-data block not found.")
    return html, match, json.loads(match.group(2))


def save_data(html: str, match: re.Match, data: dict) -> None:
    data["meetings"].sort(key=lambda m: m["id"], reverse=True)
    blob = json.dumps(data, ensure_ascii=False, indent=1)
    INDEX.write_text(html[:match.start(2)] + blob + html[match.end(2):], encoding="utf-8")


def inject(meeting: dict) -> None:
    html, match, data = load_data()
    data["meetings"] = [m for m in data["meetings"] if m["id"] != meeting["id"]]
    data["meetings"].append(meeting)
    save_data(html, match, data)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pair and classify presser vs. minutes.")
    ap.add_argument("--presser", required=True, type=Path)
    ap.add_argument("--minutes", required=True, type=Path)
    ap.add_argument("--id", required=True, help="meeting id, e.g. 2026-09")
    ap.add_argument("--label", required=True, help='e.g. "September 2026 FOMC"')
    ap.add_argument("--date", required=True, help='e.g. "Sept 16-17, 2026"')
    ap.add_argument("--runs", type=int, default=N_RUNS)
    args = ap.parse_args()

    result = classify_meeting(
        args.presser.read_text(encoding="utf-8"),
        args.minutes.read_text(encoding="utf-8"),
        n_runs=args.runs,
    )
    inject({"id": args.id, "label": args.label, "date": args.date, "status": "scored", **result})
    k = result["counts"]
    print(f"{args.label}: {k['CONTRADICTION']} contradiction, {k['ATTRIBUTION']} attribution, "
          f"{k['EMPHASIS']} emphasis, {k['CONSISTENT']} consistent "
          f"(contradiction in {result['contradiction_runs']}/{result['runs']} runs) - index.html updated")


if __name__ == "__main__":
    main()
