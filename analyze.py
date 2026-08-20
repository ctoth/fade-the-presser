"""Fade the Presser — meeting analyzer.

Feed it a press-conference transcript and the FOMC minutes for the same
meeting; Claude extracts claim pairs (what the Chair said vs. what the
Committee recorded), scores each divergence in "basis points" (0-100),
and writes the scored meeting into index.html's embedded data block.

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
from pathlib import Path
from typing import List, Literal

import anthropic
from pydantic import BaseModel, Field

MODEL = "claude-opus-5"
INDEX = Path(__file__).parent / "index.html"

DATA_RE = re.compile(
    r'(<script type="application/json" id="meeting-data">\s*)(.*?)(\s*</script>)',
    re.DOTALL,
)


class Claim(BaseModel):
    topic: str = Field(description="Very short label, 1-3 words, e.g. 'Data dependence', 'The pause'")
    question: str = Field(description="The question the claim pair answers, under 10 words")
    presser_quote: str    # condensed verbatim quote from the Chair
    presser_gist: str     # one-line reading of the Chair's position
    minutes_quote: str    # condensed verbatim quote from the minutes
    minutes_gist: str     # one-line reading of the Committee's position
    score: int            # divergence, 0 (identical) .. 100 (contradiction)
    note: str             # one-sentence verdict; sparing <b> allowed


class MeetingAnalysis(BaseModel):
    decision: str         # e.g. "HOLD", "HIKE +25", "CUT -25"
    rate: str             # target range, e.g. "3.50-3.75%"
    dissents: str         # e.g. "3 (hawkish - favored a hike)" or "none"
    inflation: str        # one-phrase inflation backdrop
    verdict: Literal["WIDE", "MODERATE", "ALIGNED"]
    summary: str          # 2-3 sentences on the meeting and the gap
    claims: List[Claim]   # 3-6 claim pairs, most divergent first


SYSTEM = """You are the analyst behind "Fade the Presser," a tracker that
compares a Fed Chair's press-conference account of an FOMC decision with the
Committee's own record in the minutes, published three weeks later.

Working principles, per Claudia Sahm's "Read the Minutes, Fade the Presser":
- The minutes' "Participants' Views" section is the Committee's reaction
  function. Weigh minutes vocabulary precisely: "most" ~ consensus,
  "many" ~ strong majority, "several" ~ meaningful bloc, "various"/"a few"
  ~ far short of consensus.
- The Chair speaks with one vote out of twelve. A personal framework offered
  from the podium is not the Committee's framework unless the minutes say so.
- A claim pair diverges when the Chair's characterization of the decision,
  its rationale, or the reaction function differs from the minutes' record -
  not merely when tone differs.

Extract 3-6 claim pairs, most divergent first. Quotes must be condensed
verbatim (use ellipses), never paraphrase inside quote fields. Score each
divergence 0-100: 0-30 aligned, 31-60 moderate, 61-100 wide (the Chair's
account and the record materially disagree). Verdict thresholds on the mean
score: ALIGNED <= 30, MODERATE 31-60, WIDE > 60."""


def analyze(presser: str, minutes: str) -> MeetingAnalysis:
    client = anthropic.Anthropic()
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
                "Compare the Chair's account with the Committee's record "
                "and produce the scored meeting analysis."
            ),
        }],
        output_format=MeetingAnalysis,
    )
    if response.stop_reason == "refusal":
        sys.exit(f"Model declined the request (stop_details: {response.stop_details}).")
    return response.parsed_output


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
    args = ap.parse_args()

    result = analyze(
        args.presser.read_text(encoding="utf-8"),
        args.minutes.read_text(encoding="utf-8"),
    )
    basis = round(sum(c.score for c in result.claims) / len(result.claims))
    meeting = {
        "id": args.id, "label": args.label, "date": args.date,
        "status": "scored", "basis": basis,
        **result.model_dump(),
    }
    inject(meeting)
    print(f"{args.label}: {len(result.claims)} claims scored - "
          f"basis {basis} bp ({result.verdict}) - index.html updated")


if __name__ == "__main__":
    main()
