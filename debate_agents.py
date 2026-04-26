"""
Debate-based multi-agent pipeline.

For each dimension, two independent reviewer agents score the paper.
If their scores differ by more than DEBATE_THRESHOLD (default 1 point), a referee agent
reads both justifications and issues a final synthesized verdict.
If they agree, their scores are averaged and justifications concatenated.

This improves calibration on borderline papers at ~2-3x the API cost of single-agent scoring.

The output format is identical to dimension_agents.py, with extra debug fields stored
under "_details[dim].reviewer_a", ".reviewer_b", ".debate_triggered", ".resolution_method".
"""

import os
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from dimension_agents import (
    DimensionScore,
    DIMENSION_AGENTS,
    DIMENSION_PROMPT_TEMPLATE,
    VENUE_DIMENSIONS,
    VENUE_MAP,
    load_paper_json,
    call_dimension_agent,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEBATE_THRESHOLD = 1  # Score delta that triggers the referee


# ---------------------------------------------------------------------------
# Referee prompt
# ---------------------------------------------------------------------------

REFEREE_PROMPT = """\
### REFEREE TASK
Two independent reviewers disagreed on the dimension "{dimension}" for the paper below.
Your job is to read both perspectives and issue a final score.

Paper Title: {title}

### Reviewer A  (score: {score_a}/5)
{justification_a}

### Reviewer B  (score: {score_b}/5)
{justification_b}

### Instructions
Weigh both arguments. The final score must be an integer 1–5.
Write a synthesized justification (2–4 sentences) that acknowledges where reviewers agreed
and explains why you chose this score.

Output a JSON object:
- "score": integer 1-5
- "justification": synthesized 2-4 sentence justification

Output only the JSON object, nothing else."""


# ---------------------------------------------------------------------------
# Referee call
# ---------------------------------------------------------------------------

def _call_referee(
    client: OpenAI,
    title: str,
    dimension: str,
    score_a: int,
    justification_a: str,
    score_b: int,
    justification_b: str,
) -> DimensionScore:
    prompt = REFEREE_PROMPT.format(
        dimension=dimension,
        title=title,
        score_a=score_a,
        justification_a=justification_a,
        score_b=score_b,
        justification_b=justification_b,
    )
    response = client.responses.parse(
        model="gpt-5-mini",
        input=[
            {"role": "developer", "content": [{"type": "input_text", "text": prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": ""}]},
        ],
        text_format=DimensionScore,
        reasoning={"effort": "medium", "summary": "auto"},
        tools=[],
        store=True,
        include=["reasoning.encrypted_content"],
    )
    return response.output_parsed


# ---------------------------------------------------------------------------
# Single-dimension debate
# ---------------------------------------------------------------------------

def score_dimension_with_debate(
    client: OpenAI, title: str, parsed_text: str, dimension: str,
    novelty_context: str = None,
) -> dict:
    """
    Score one dimension using two independent agents. Invokes a referee if they
    disagree by more than DEBATE_THRESHOLD points.

    Returns a dict with score, justification, and full debate trace fields.
    """
    # Two independent reviews in parallel
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_a = ex.submit(call_dimension_agent, client, title, parsed_text, dimension, novelty_context)
        fut_b = ex.submit(call_dimension_agent, client, title, parsed_text, dimension, novelty_context)
        review_a: DimensionScore = fut_a.result()
        review_b: DimensionScore = fut_b.result()

    delta = abs(review_a.score - review_b.score)
    debate_triggered = delta > DEBATE_THRESHOLD

    if debate_triggered:
        final = _call_referee(
            client, title, dimension,
            review_a.score, review_a.justification,
            review_b.score, review_b.justification,
        )
        resolution = "referee"
    else:
        avg_score = round((review_a.score + review_b.score) / 2)
        final = DimensionScore(
            score=avg_score,
            justification=(
                f"[Consensus avg={avg_score}] "
                f"A({review_a.score}): {review_a.justification} "
                f"| B({review_b.score}): {review_b.justification}"
            ),
        )
        resolution = "consensus"

    return {
        "score": final.score,
        "justification": final.justification,
        "reviewer_a": review_a.model_dump(),
        "reviewer_b": review_b.model_dump(),
        "score_delta": delta,
        "debate_triggered": debate_triggered,
        "resolution_method": resolution,
    }


# ---------------------------------------------------------------------------
# Per-dimension worker
# ---------------------------------------------------------------------------

def _process_single_dimension(client, title, parsed_text, dimension):
    try:
        result = score_dimension_with_debate(client, title, parsed_text, dimension)
        return dimension, result
    except Exception as e:
        return dimension, {
            "score": None,
            "justification": f"[ERROR] {e}",
            "debate_triggered": False,
            "resolution_method": "error",
        }


# ---------------------------------------------------------------------------
# Paper-level processing
# ---------------------------------------------------------------------------

def process_paper(client, paper_json, dimensions, out_dir):
    title, parsed_text = load_paper_json(paper_json)
    results = {}

    with ThreadPoolExecutor(max_workers=len(dimensions)) as executor:
        futures = {
            executor.submit(_process_single_dimension, client, title, parsed_text, dim): dim
            for dim in dimensions
        }
        for future in as_completed(futures):
            dim, result = future.result()
            results[dim] = result

    n_debated = sum(1 for r in results.values() if r.get("debate_triggered"))

    out_file = out_dir / f"{paper_json.stem}_review.json"
    flat = {dim: results[dim]["score"] for dim in results}
    flat["_details"] = results
    flat["_debate_stats"] = {
        "dimensions_debated": n_debated,
        "total_dimensions": len(dimensions),
        "debate_rate": round(n_debated / len(dimensions), 3) if dimensions else 0,
    }
    with open(out_file, "w") as f:
        json.dump(flat, f, indent=2)
    return paper_json.name


# ---------------------------------------------------------------------------
# Venue-level processing
# ---------------------------------------------------------------------------

def process_venue(client, venue_folder, venue_name, output_dir):
    base = Path("PeerRead/data")
    split_dir = base / venue_folder / "test" / "parsed_pdfs"
    if not split_dir.exists():
        tqdm.write(f"Skipping {venue_folder}: no test/parsed_pdfs")
        return

    dimensions = VENUE_DIMENSIONS.get(venue_folder, [])
    if not dimensions:
        tqdm.write(f"Skipping {venue_folder}: no dimensions defined")
        return

    out_dir = output_dir / venue_folder / "test"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list(split_dir.glob("*.json"))

    pbar = tqdm(total=len(files), desc=f"{venue_name} (debate, {len(dimensions)} dims)", unit="paper")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(process_paper, client, paper_json, dimensions, out_dir): paper_json
            for paper_json in files
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                tqdm.write(f"[ERROR] {venue_name} {futures[future].name}: {e}")
            pbar.update(1)
    pbar.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    load_dotenv()
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://ai-gateway.andrew.cmu.edu",
    )
    output_dir = Path("debate_agent_prompts")

    print("Model       : gpt-5-mini")
    print(f"Debate threshold: delta > {DEBATE_THRESHOLD} triggers referee\n")

    venue_tasks = [(client, vf, vn, output_dir) for vf, vn in VENUE_MAP.items()]
    with ThreadPoolExecutor(max_workers=len(venue_tasks)) as executor:
        futures = [executor.submit(process_venue, *args) for args in venue_tasks]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Venue thread failed: {exc}")


if __name__ == "__main__":
    main()
