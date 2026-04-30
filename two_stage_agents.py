"""
Two-stage multi-agent pipeline.

Stage 1 (small model): Extract a compact structured PaperSummary from the full paper text.
Stage 2 (routed):
  - LIGHT_DIMENSIONS (CLARITY, APPROPRIATENESS, REPLICABILITY) -> small model scores from summary
  - HEAVY_DIMENSIONS (everything else) -> large model scores from summary

Token savings: full paper (~8k tokens) is read only once by the small model; the large model
receives only the ~500-token structured summary times the number of heavy dimensions.

Environment variables:
  OPENAI_API_KEY          - API key for CMU gateway (large model)
  LARGE_MODEL             - model name for large model (default: gpt-5-mini)
  SMALL_MODEL_API_KEY     - API key for small model endpoint (default: "ollama")
  SMALL_MODEL_BASE_URL    - base URL for small model (default: http://localhost:11434/v1)
  SMALL_MODEL             - model name for small model (default: llama3:8b)
"""

import os
import json
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from dimension_agents import (
    DimensionScore,
    DIMENSION_AGENTS,
    VENUE_DIMENSIONS,
    VENUE_MAP,
    load_paper_json,
    has_ground_truth,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Max chars of paper text sent for extraction (Stage 1)
EXTRACTION_CHAR_LIMIT = 40_000


# ---------------------------------------------------------------------------
# Stage 1: Paper Summary schema
# ---------------------------------------------------------------------------

class PaperSummary(BaseModel):
    main_claims: List[str] = Field(description="3-5 main claims or contributions")
    methodology: str = Field(description="Core method description in 1-2 sentences")
    datasets: List[str] = Field(description="Dataset names used in experiments")
    baselines: List[str] = Field(description="Baseline method names compared against")
    metrics: List[str] = Field(description="Evaluation metric names used")
    limitations: List[str] = Field(description="Stated limitations or failure cases")
    code_available: str = Field(description="'yes', 'no', or 'not mentioned'")
    hyperparams_reported: bool = Field(description="True if key hyperparameters are reported")
    related_work_coverage: str = Field(description="'thorough', 'adequate', or 'sparse'")
    topic_area: str = Field(description="Main topic area, e.g. 'NLP - machine translation'")
    writing_quality: str = Field(description="'excellent', 'good', 'adequate', or 'poor'")


EXTRACTION_PROMPT = """\
You are a research paper analyzer. Extract structured factual information from the paper below.
Only extract what is explicitly stated; do not infer or fabricate.

Paper Title: {title}
Paper Content:
{parsed_text}

Return a JSON object with exactly these fields:
{{
  "main_claims": ["claim1", "claim2", ...],   // 3-5 main claims or contributions
  "methodology": "...",                         // core method in 1-2 sentences
  "datasets": ["name1", ...],                   // dataset names (empty list if none)
  "baselines": ["method1", ...],                // baseline names (empty list if none)
  "metrics": ["metric1", ...],                  // evaluation metrics (empty list if none)
  "limitations": ["...", ...],                  // stated limitations (empty list if none)
  "code_available": "yes" | "no" | "not mentioned",
  "hyperparams_reported": true | false,
  "related_work_coverage": "thorough" | "adequate" | "sparse",
  "topic_area": "...",
  "writing_quality": "excellent" | "good" | "adequate" | "poor"
}}

Output only the JSON object, nothing else."""


# ---------------------------------------------------------------------------
# Stage 2: Scoring from summary
# ---------------------------------------------------------------------------

SUMMARY_SCORE_TEMPLATE = """\
### SYSTEM ROLE
{system}

### PAPER SUMMARY (extracted facts)
Title: {title}
Main claims: {main_claims}
Methodology: {methodology}
Datasets used: {datasets}
Baselines compared: {baselines}
Evaluation metrics: {metrics}
Stated limitations: {limitations}
Code available: {code_available}
Hyperparameters reported: {hyperparams_reported}
Related work coverage: {related_work_coverage}
Topic area: {topic_area}
Writing quality: {writing_quality}

### TASK
{checklist}

Respond with a JSON object containing:
- "score": an integer from 1 to 5
- "justification": 2-4 sentences referencing specific facts above

### CONSTRAINT
Base your judgment only on the provided summary. Output only the JSON object, nothing else."""


def _fmt_list(items: List[str], fallback: str = "not specified") -> str:
    return "; ".join(items) if items else fallback


def _format_summary_kwargs(title: str, summary: PaperSummary) -> dict:
    return {
        "title": title,
        "main_claims": _fmt_list(summary.main_claims),
        "methodology": summary.methodology or "not described",
        "datasets": _fmt_list(summary.datasets),
        "baselines": _fmt_list(summary.baselines, "none mentioned"),
        "metrics": _fmt_list(summary.metrics),
        "limitations": _fmt_list(summary.limitations, "none stated"),
        "code_available": summary.code_available,
        "hyperparams_reported": summary.hyperparams_reported,
        "related_work_coverage": summary.related_work_coverage,
        "topic_area": summary.topic_area,
        "writing_quality": summary.writing_quality,
    }


# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------

LLM_TIMEOUT = 180  # seconds — raises httpx.TimeoutException if exceeded

def _call_structured(client: OpenAI, prompt: str, schema_class):
    """responses.parse with gpt-5-mini.
    Parameters are always identical to avoid breaking the CMU gateway's cached
    prepared statement (401 'cached plan must not change result type')."""
    response = client.responses.parse(
        model="gpt-5-mini",
        input=[
            {"role": "developer", "content": [{"type": "input_text", "text": prompt}]},
            {"role": "user",      "content": [{"type": "input_text", "text": ""}]},
        ],
        text_format=schema_class,
        reasoning={"effort": "low", "summary": "auto"},
        tools=[],
        store=True,
        include=["reasoning.encrypted_content"],
        timeout=LLM_TIMEOUT,
    )
    return response.output_parsed


# ---------------------------------------------------------------------------
# Stage 1: extraction
# ---------------------------------------------------------------------------

def extract_paper_facts(client: OpenAI, title: str, parsed_text: str) -> PaperSummary:
    prompt = EXTRACTION_PROMPT.format(
        title=title,
        parsed_text=parsed_text[:EXTRACTION_CHAR_LIMIT],
    )
    return _call_structured(client, prompt, PaperSummary)


# ---------------------------------------------------------------------------
# Stage 2: dimension scoring from summary
# ---------------------------------------------------------------------------

def score_dimension_from_summary(
    client: OpenAI,
    title: str,
    summary: PaperSummary,
    dimension: str,
) -> DimensionScore:
    agent_def = DIMENSION_AGENTS[dimension]
    prompt = SUMMARY_SCORE_TEMPLATE.format(
        system=agent_def["system"],
        checklist=agent_def["checklist"],
        **_format_summary_kwargs(title, summary),
    )
    return _call_structured(client, prompt, DimensionScore)


# ---------------------------------------------------------------------------
# Per-dimension worker
# ---------------------------------------------------------------------------

def _process_single_dimension(client, title, summary, dimension):
    try:
        score_obj = score_dimension_from_summary(client, title, summary, dimension)
        return dimension, score_obj.model_dump()
    except Exception as e:
        return dimension, {"score": None, "justification": f"[ERROR] {e}"}


# ---------------------------------------------------------------------------
# Paper-level processing
# ---------------------------------------------------------------------------

def process_paper(client, paper_json, dimensions, out_dir):
    out_file = out_dir / f"{paper_json.stem}_review.json"
    if out_file.exists():
        return paper_json.name

    title, parsed_text = load_paper_json(paper_json)

    # Stage 1: extract structured summary
    extraction_error = None
    try:
        summary = extract_paper_facts(client, title, parsed_text)
    except Exception as e:
        tqdm.write(f"[WARN] Extraction failed for {paper_json.name}: {e}")
        extraction_error = str(e)
        summary = None

    results = {}
    if summary is not None:
        with ThreadPoolExecutor(max_workers=len(dimensions)) as executor:
            futures = {
                executor.submit(_process_single_dimension, client, title, summary, dim): dim
                for dim in dimensions
            }
            for future in as_completed(futures):
                dim, result = future.result()
                results[dim] = result

    out_file = out_dir / f"{paper_json.stem}_review.json"
    flat = {dim: results[dim]["score"] for dim in results}
    flat["_details"] = results
    flat["_summary"] = summary.model_dump() if summary else {"error": extraction_error}
    with open(out_file, "w") as f:
        json.dump(flat, f, indent=2)
    return paper_json.name


# ---------------------------------------------------------------------------
# Venue-level processing
# ---------------------------------------------------------------------------

def process_venue(client, venue_folder, venue_name, output_dir):
    base = Path("PeerRead/data")
    venue_dir = base / venue_folder

    dimensions = VENUE_DIMENSIONS.get(venue_folder, [])
    if not dimensions:
        tqdm.write(f"Skipping {venue_folder}: no dimensions defined")
        return

    for split in ["train", "dev", "test"]:
        split_dir = venue_dir / split / "parsed_pdfs"
        reviews_dir = venue_dir / split / "reviews"
        if not split_dir.exists():
            continue

        all_files = list(split_dir.glob("*.json"))
        if reviews_dir.exists():
            files = [f for f in all_files if has_ground_truth(f, reviews_dir, dimensions)]
            skipped = len(all_files) - len(files)
            if skipped:
                tqdm.write(f"[INFO] {venue_name}/{split}: skipping {skipped}/{len(all_files)} papers with no GT scores")
        else:
            files = all_files

        if not files:
            tqdm.write(f"[INFO] {venue_name}/{split}: no papers with GT scores, skipping split")
            continue

        out_dir = output_dir / venue_folder / split
        out_dir.mkdir(parents=True, exist_ok=True)

        pbar = tqdm(total=len(files), desc=f"{venue_name}/{split} (2-stage, {len(dimensions)} dims)", unit="paper")
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(process_paper, client, paper_json, dimensions, out_dir): paper_json
                for paper_json in files
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    tqdm.write(f"[ERROR] {venue_name}/{split} {futures[future].name}: {e}")
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
    output_dir = Path("two_stage_agent_prompts")

    print("Model : gpt-5-mini @ ai-gateway.andrew.cmu.edu")
    print("Stage 1: extract PaperSummary from full text")
    print("Stage 2: score all dimensions from compressed summary\n")

    venue_tasks = [
        (client, vf, vn, output_dir)
        for vf, vn in VENUE_MAP.items()
    ]
    with ThreadPoolExecutor(max_workers=len(venue_tasks)) as executor:
        futures = [executor.submit(process_venue, *args_) for args_ in venue_tasks]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Venue thread failed: {exc}")


if __name__ == "__main__":
    main()
