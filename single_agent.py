"""
Single-agent baseline with checklist rubrics.

A single LLM call evaluates ALL dimensions for a paper simultaneously, using the
same checklist rubric text as dimension_agents.py. This isolates the effect of
specialization: dimension_agents.py vs. this script differ only in whether each
dimension gets its own focused call or all dimensions share one context.

Outputs to single_agent_prompts/, using the same flat JSON format as all other
approaches so correlation_script.py can evaluate it directly.
"""

import os
import json
from pathlib import Path
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, create_model
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from dimension_agents import (
    DIMENSION_AGENTS,
    VENUE_DIMENSIONS,
    VENUE_MAP,
    load_paper_json,
)

# ---------------------------------------------------------------------------
# Build a single Pydantic model that holds all dimension scores at once
# ---------------------------------------------------------------------------

def _build_review_model(dimensions):
    """Dynamically create a Pydantic model with one DimensionScore field per dimension."""
    from dimension_agents import DimensionScore
    fields = {dim: (DimensionScore, ...) for dim in dimensions}
    return create_model("PaperReview", **fields)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SINGLE_AGENT_TEMPLATE = """\
### SYSTEM ROLE
You are an expert paper reviewer. Evaluate the paper on ALL of the following dimensions.
For each dimension, apply the specific checklist provided and assign a score.
Keep each dimension's evaluation independent — do not let your view of one dimension
bias your judgment on another.

### INPUT DATA
Paper Title: {title}
Paper Content: {parsed_text}

### DIMENSIONS TO EVALUATE
{dimension_blocks}

### OUTPUT FORMAT
Respond with a single JSON object. For each dimension listed above, include a nested object with:
  - "score": integer 1–5
  - "justification": 2–4 sentences citing specific parts of the paper

### CONSTRAINT
Base all judgments only on the provided paper text. Output only the JSON object, nothing else."""


def _build_dimension_blocks(dimensions):
    blocks = []
    for i, dim in enumerate(dimensions, 1):
        agent = DIMENSION_AGENTS[dim]
        blocks.append(
            f"--- Dimension {i}: {dim} ---\n"
            f"Focus: {agent['system']}\n\n"
            f"{agent['checklist']}"
        )
    return "\n\n".join(blocks)


def build_prompt(title, parsed_text, dimensions):
    return SINGLE_AGENT_TEMPLATE.format(
        title=title,
        parsed_text=parsed_text,
        dimension_blocks=_build_dimension_blocks(dimensions),
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_single_agent(client, title, parsed_text, dimensions):
    """One LLM call that scores all dimensions and returns a dict {dim: DimensionScore}."""
    review_model = _build_review_model(dimensions)
    prompt = build_prompt(title, parsed_text, dimensions)
    response = client.responses.parse(
        model="gpt-5-mini",
        input=[
            {"role": "developer", "content": [{"type": "input_text", "text": prompt}]},
            {"role": "user",      "content": [{"type": "input_text", "text": ""}]},
        ],
        text_format=review_model,
        reasoning={"effort": "medium", "summary": "auto"},
        tools=[],
        store=True,
        include=["reasoning.encrypted_content"],
    )
    return response.output_parsed


# ---------------------------------------------------------------------------
# Paper-level processing
# ---------------------------------------------------------------------------

def process_paper(client, paper_json, dimensions, out_dir):
    title, parsed_text = load_paper_json(paper_json)

    try:
        review = call_single_agent(client, title, parsed_text, dimensions)
        details = {dim: getattr(review, dim).model_dump() for dim in dimensions}
    except Exception as e:
        details = {dim: {"score": None, "justification": f"[ERROR] {e}"} for dim in dimensions}

    flat = {dim: details[dim]["score"] for dim in dimensions}
    flat["_details"] = details

    out_file = out_dir / f"{paper_json.stem}_review.json"
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

    pbar = tqdm(total=len(files), desc=f"{venue_name} (single-agent, {len(dimensions)} dims)", unit="paper")
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
    output_dir = Path("single_agent_prompts")

    print("Model: gpt-5-mini")
    print("All dimensions scored in a single call per paper.\n")

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
