"""
Literature-Survey-Augmented Dimension Agents.

Uses the same specialist dimension agents as dimension_agents.py, but injects
pre-computed literature survey context into the three dimensions where knowledge
of prior work matters most:

  MEANINGFUL_COMPARISON  — did the paper compare against the relevant baselines?
  ORIGINALITY            — is the contribution novel given known prior work?
  SUBSTANCE              — are there missing experiments against relevant methods?

For all other dimensions the prompt is identical to dimension_agents.py.

Prerequisites:
  Run `python run_lit_survey_batch.py` first to populate:
    lit_survey_results/<venue>/test/<paper_id>_survey.json

If a survey file is missing for a paper, the dimension falls back silently to
the standard unaugmented prompt (no hard failure).

Outputs to: lit_augmented_agent_prompts/
Output format is identical to all other approaches (flat JSON + _details),
with an extra _augmentation_stats field per paper.
"""

import os
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Dimensions where lit survey context materially helps
SURVEY_AUGMENTED_DIMS = {"MEANINGFUL_COMPARISON", "ORIGINALITY", "SUBSTANCE"}

# How many top papers from the survey pool to inject per prompt
TOP_K_PAPERS = 6

# Default survey results location (produced by run_lit_survey_batch.py)
DEFAULT_SURVEY_DIR = "lit_survey_results"


# ---------------------------------------------------------------------------
# Survey loading
# ---------------------------------------------------------------------------

def load_survey(paper_json_path: Path, survey_dir: str) -> dict | None:
    """
    Load the pre-computed survey JSON for a paper.
    Paper path: .../parsed_pdfs/358.pdf.json  → stem: 358.pdf
    Survey path: {survey_dir}/{venue}/test/358.pdf_survey.json
    Returns the parsed dict, or None if not found / errored.
    """
    # Reconstruct the venue/split path from the paper_json location
    # Expected structure: PeerRead/data/{venue}/test/parsed_pdfs/{paper}.json
    parts = paper_json_path.parts
    try:
        pdfs_idx = parts.index("parsed_pdfs")
        venue    = parts[pdfs_idx - 2]   # e.g. "acl_2017"
        split    = parts[pdfs_idx - 1]   # e.g. "test"
    except ValueError:
        return None

    paper_id    = paper_json_path.stem           # e.g. "358.pdf"
    survey_file = Path(survey_dir) / venue / split / f"{paper_id}_survey.json"

    if not survey_file.exists():
        return None
    try:
        with open(survey_file) as f:
            data = json.load(f)
        if "error" in data:
            return None
        return data
    except Exception:
        return None


def format_survey_context(survey: dict, top_k: int = TOP_K_PAPERS) -> str:
    """
    Format the top-K papers from the survey pool into a compact context block.
    Papers are already sorted by final_score descending by the orchestrator.
    """
    pool = survey.get("paper_pool", [])[:top_k]
    if not pool:
        return "(no relevant prior works found by the survey agent)"

    lines = []
    for i, p in enumerate(pool, 1):
        year     = f"({p.get('year', '?')})" if p.get("year") else ""
        score    = p.get("final_score", 0.0)
        rationale = p.get("score_rationale", "").strip()
        title    = p.get("title", "Unknown")[:90]
        lines.append(
            f"[{i}] \"{title}\" {year}  relevance={score:.2f}\n"
            f"     {rationale}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Augmented prompt template
# ---------------------------------------------------------------------------

AUGMENTED_PROMPT_TEMPLATE = """\
### SYSTEM ROLE
{system}

### INPUT DATA
Paper Title: {title}
Paper Content: {parsed_text}

### LITERATURE SURVEY CONTEXT
A retrieval agent searched Semantic Scholar, arXiv, and OpenAlex and found the \
following {n_papers} relevant prior works (ranked by topical + methodological relevance):

{survey_context}

Use this context to assess whether the paper properly engages with the relevant literature.

### TASK
{checklist}

Respond with a JSON object containing:
- "score": an integer from 1 to 5
- "justification": a concise explanation (2-4 sentences) citing specific parts of the \
paper and referencing relevant prior works from the context above where applicable

### CONSTRAINT
Base your judgment only on the provided paper text and the survey context above. \
Output only the JSON object, nothing else."""


# ---------------------------------------------------------------------------
# Agent call helpers
# ---------------------------------------------------------------------------

def _call_standard_agent(client, title, parsed_text, dimension) -> DimensionScore:
    """Standard unaugmented call — identical to dimension_agents.call_dimension_agent."""
    agent_def = DIMENSION_AGENTS[dimension]
    prompt = DIMENSION_PROMPT_TEMPLATE.format(
        system=agent_def["system"],
        title=title,
        parsed_text=parsed_text,
        checklist=agent_def["checklist"],
    )
    model = os.getenv("LARGE_MODEL", "gpt-5-mini")
    response = client.responses.parse(
        model=model,
        input=[
            {"role": "developer", "content": [{"type": "input_text", "text": prompt}]},
            {"role": "user",      "content": [{"type": "input_text", "text": ""}]},
        ],
        text_format=DimensionScore,
        reasoning={"effort": "medium", "summary": "auto"},
        tools=[],
        store=True,
        include=["reasoning.encrypted_content"],
    )
    return response.output_parsed


def _call_augmented_agent(
    client, title, parsed_text, dimension, survey: dict
) -> tuple[DimensionScore, int]:
    """Survey-augmented call. Returns (DimensionScore, n_papers_injected)."""
    agent_def     = DIMENSION_AGENTS[dimension]
    survey_ctx    = format_survey_context(survey)
    n_papers      = min(len(survey.get("paper_pool", [])), TOP_K_PAPERS)

    prompt = AUGMENTED_PROMPT_TEMPLATE.format(
        system=agent_def["system"],
        title=title,
        parsed_text=parsed_text,
        survey_context=survey_ctx,
        n_papers=n_papers,
        checklist=agent_def["checklist"],
    )
    model = os.getenv("LARGE_MODEL", "gpt-5-mini")
    response = client.responses.parse(
        model=model,
        input=[
            {"role": "developer", "content": [{"type": "input_text", "text": prompt}]},
            {"role": "user",      "content": [{"type": "input_text", "text": ""}]},
        ],
        text_format=DimensionScore,
        reasoning={"effort": "medium", "summary": "auto"},
        tools=[],
        store=True,
        include=["reasoning.encrypted_content"],
    )
    return response.output_parsed, n_papers


# ---------------------------------------------------------------------------
# Per-dimension worker
# ---------------------------------------------------------------------------

def _process_single_dimension(client, title, parsed_text, dimension, survey):
    try:
        augmented = dimension in SURVEY_AUGMENTED_DIMS and survey is not None
        if augmented:
            score_obj, n_injected = _call_augmented_agent(
                client, title, parsed_text, dimension, survey
            )
        else:
            score_obj  = _call_standard_agent(client, title, parsed_text, dimension)
            n_injected = 0

        result = score_obj.model_dump()
        result["augmented"]   = augmented
        result["n_injected"]  = n_injected
        return dimension, result
    except Exception as e:
        return dimension, {
            "score": None,
            "justification": f"[ERROR] {e}",
            "augmented": False,
            "n_injected": 0,
        }


# ---------------------------------------------------------------------------
# Paper-level processing
# ---------------------------------------------------------------------------

def process_paper(client, paper_json, dimensions, out_dir, survey_dir):
    title, parsed_text = load_paper_json(paper_json)
    survey = load_survey(paper_json, survey_dir)

    results = {}
    with ThreadPoolExecutor(max_workers=len(dimensions)) as executor:
        futures = {
            executor.submit(
                _process_single_dimension, client, title, parsed_text, dim, survey
            ): dim
            for dim in dimensions
        }
        for future in as_completed(futures):
            dim, result = future.result()
            results[dim] = result

    augmented_dims = [d for d, r in results.items() if r.get("augmented")]

    out_file = out_dir / f"{paper_json.stem}_review.json"
    flat = {dim: results[dim]["score"] for dim in results}
    flat["_details"] = results
    flat["_augmentation_stats"] = {
        "survey_found":      survey is not None,
        "survey_pool_size":  len(survey.get("paper_pool", [])) if survey else 0,
        "augmented_dims":    augmented_dims,
        "total_dims":        len(dimensions),
    }
    with open(out_file, "w") as f:
        json.dump(flat, f, indent=2)
    return paper_json.name


# ---------------------------------------------------------------------------
# Venue-level processing
# ---------------------------------------------------------------------------

def process_venue(client, venue_folder, venue_name, output_dir, survey_dir):
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

    # Count how many papers have surveys available
    surveys_found = sum(
        1 for f in files if load_survey(f, survey_dir) is not None
    )
    tqdm.write(
        f"{venue_name}: {surveys_found}/{len(files)} papers have survey results "
        f"(augmenting {sorted(SURVEY_AUGMENTED_DIMS)})"
    )

    pbar = tqdm(
        total=len(files),
        desc=f"{venue_name} (lit-augmented, {len(dimensions)} dims)",
        unit="paper",
    )
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                process_paper, client, paper_json, dimensions, out_dir, survey_dir
            ): paper_json
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
# Augmentation coverage report (printed after each venue)
# ---------------------------------------------------------------------------

def print_augmentation_report(output_dir: Path, venues: list[str]):
    print(f"\n{'='*60}")
    print("  Lit-Survey Augmentation Coverage")
    print(f"{'='*60}")
    for venue in venues:
        reviews_dir = output_dir / venue / "test"
        if not reviews_dir.exists():
            continue
        total = surveys_ok = 0
        for f in reviews_dir.glob("*_review.json"):
            data = json.load(open(f))
            stats = data.get("_augmentation_stats", {})
            total += 1
            if stats.get("survey_found"):
                surveys_ok += 1
        if total:
            print(f"  {venue}: {surveys_ok}/{total} papers augmented ({surveys_ok/total:.0%})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Lit-survey-augmented dimension agents."
    )
    parser.add_argument(
        "--survey-dir", default=DEFAULT_SURVEY_DIR,
        help=f"Directory with pre-computed survey results (default: {DEFAULT_SURVEY_DIR}). "
             f"Run run_lit_survey_batch.py first."
    )
    args = parser.parse_args()

    load_dotenv()
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://ai-gateway.andrew.cmu.edu",
    )
    output_dir = Path("lit_augmented_agent_prompts")
    survey_dir = args.survey_dir

    print(f"Model      : {os.getenv('LARGE_MODEL', 'gpt-5-mini')}")
    print(f"Survey dir : {survey_dir}")
    print(f"Augmenting : {sorted(SURVEY_AUGMENTED_DIMS)} (top {TOP_K_PAPERS} papers)\n")

    venues = list(VENUE_MAP.keys())
    venue_tasks = [
        (client, vf, vn, output_dir, survey_dir)
        for vf, vn in VENUE_MAP.items()
    ]
    with ThreadPoolExecutor(max_workers=len(venue_tasks)) as executor:
        futures = [executor.submit(process_venue, *args_) for args_ in venue_tasks]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Venue thread failed: {exc}")

    print_augmentation_report(output_dir, venues)


if __name__ == "__main__":
    main()
