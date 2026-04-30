"""
Novelty-augmented two-stage pipeline.

Same as two_stage_agents.py but injects pre-computed literature survey and
novelty classifier context into the dimensions where prior-work knowledge helps:

  MEANINGFUL_COMPARISON  — sees ranked literature survey (related prior works)
  ORIGINALITY            — sees literature survey + claim-level novelty scores
  SUBSTANCE              — sees literature survey (check for missing comparisons)

All other dimensions use the standard unaugmented prompt.

Prerequisites:
  Run `python run_novelty_batch.py` first to populate:
    novelty_outputs/{venue}/{split}/{paper_id}/survey.json
    novelty_outputs/{venue}/{split}/{paper_id}/novelty.json

Only papers with a valid (non-error) novelty.json are processed.

Outputs to: novelty_augmented_two_stage_prompts/
"""

import os
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from dimension_agents import (
    DIMENSION_AGENTS,
    VENUE_DIMENSIONS,
    VENUE_MAP,
    load_paper_json,
    has_ground_truth,
)
from two_stage_agents import (
    PaperSummary,
    DimensionScore,
    extract_paper_facts,
    _call_structured,
    _format_summary_kwargs,
    score_dimension_from_summary,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SURVEY_AUGMENTED_DIMS = {"MEANINGFUL_COMPARISON", "ORIGINALITY", "SUBSTANCE"}
NOVELTY_AUGMENTED_DIMS = {"ORIGINALITY"}
TOP_K_PAPERS = 6
NOVELTY_DIR = Path("novelty_outputs")


# ---------------------------------------------------------------------------
# Novelty output loading
# ---------------------------------------------------------------------------

def load_novelty_data(paper_json_path: Path) -> tuple[dict | None, dict | None]:
    """Return (survey_dict, novelty_dict) for a paper, or (None, None) on failure.

    survey_dict is None if the survey has no paper_pool entries.
    novelty_dict is None if novelty.json contains an 'error' key.
    """
    parts = paper_json_path.parts
    try:
        pdfs_idx = parts.index("parsed_pdfs")
        venue = parts[pdfs_idx - 2]
        split = parts[pdfs_idx - 1]
    except ValueError:
        return None, None

    paper_id = paper_json_path.stem  # e.g. "107.pdf"
    base = NOVELTY_DIR / venue / split / paper_id

    survey = None
    survey_file = base / "survey.json"
    if survey_file.exists():
        try:
            data = json.loads(survey_file.read_text())
            if data.get("paper_pool"):
                survey = data
        except Exception:
            pass

    novelty = None
    novelty_file = base / "novelty.json"
    if novelty_file.exists():
        try:
            data = json.loads(novelty_file.read_text())
            if "error" not in data:
                novelty = data
        except Exception:
            pass

    return survey, novelty


def has_valid_novelty(paper_json_path: Path) -> bool:
    _, novelty = load_novelty_data(paper_json_path)
    return novelty is not None


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def format_survey_context(survey: dict, top_k: int = TOP_K_PAPERS) -> str:
    pool = survey.get("paper_pool", [])[:top_k]
    if not pool:
        return "(no relevant prior works found)"
    lines = []
    for i, p in enumerate(pool, 1):
        year = f"({p.get('year', '?')})" if p.get("year") else ""
        score = p.get("final_score", 0.0)
        rationale = p.get("score_rationale", "").strip()
        title = p.get("title", "Unknown")[:90]
        lines.append(
            f"[{i}] \"{title}\" {year}  relevance={score:.2f}\n"
            f"     {rationale}"
        )
    return "\n".join(lines)


def format_novelty_context(novelty: dict) -> str:
    num_claims = novelty.get("num_claims", 0)
    num_novel = novelty.get("num_novel_claims", 0)
    mean = novelty.get("paper_novelty_mean") or 0.0
    threshold = novelty.get("threshold", 0.35)
    novel_claims = [c["claim"] for c in novelty.get("claims", []) if c.get("is_novel")][:5]
    claim_lines = "\n".join(f"  - {c}" for c in novel_claims) if novel_claims else "  (none)"
    return (
        f"{num_novel}/{num_claims} claims flagged as novel "
        f"(mean score: {mean:.2f}; threshold: {threshold}).\n"
        f"Novel claims:\n{claim_lines}"
    )


def build_context_suffix(dimension: str, survey: dict | None, novelty: dict | None) -> str:
    """Return extra context blocks to append to a prompt for the given dimension."""
    parts = []

    if dimension in SURVEY_AUGMENTED_DIMS and survey is not None:
        n = min(len(survey.get("paper_pool", [])), TOP_K_PAPERS)
        parts.append(
            f"### LITERATURE SURVEY CONTEXT\n"
            f"A retrieval agent found {n} relevant prior works (ranked by relevance):\n\n"
            f"{format_survey_context(survey)}\n\n"
            f"Use this to assess how the paper relates to and builds upon existing work."
        )

    if dimension in NOVELTY_AUGMENTED_DIMS and novelty is not None:
        parts.append(
            f"### NOVELTY ANALYSIS\n"
            f"Automated claim-level analysis against the retrieved prior works:\n"
            f"{format_novelty_context(novelty)}"
        )

    return ("\n\n" + "\n\n".join(parts)) if parts else ""


# ---------------------------------------------------------------------------
# Augmented prompt template
# ---------------------------------------------------------------------------

AUGMENTED_SUMMARY_SCORE_TEMPLATE = """\
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
{context_suffix}

### TASK
{checklist}

Respond with a JSON object containing:
- "score": an integer from 1 to 5
- "justification": 2-4 sentences referencing specific facts above and relevant prior works where applicable

### CONSTRAINT
Base your judgment only on the provided summary and context above. Output only the JSON object, nothing else."""


# ---------------------------------------------------------------------------
# Dimension scoring
# ---------------------------------------------------------------------------

def score_dimension_augmented(
    client: OpenAI,
    title: str,
    summary: PaperSummary,
    dimension: str,
    survey: dict | None,
    novelty: dict | None,
) -> tuple[dict, bool, bool]:
    """Score one dimension, injecting survey/novelty context when relevant.
    Returns (score_dict, survey_injected, novelty_injected).
    """
    use_survey = dimension in SURVEY_AUGMENTED_DIMS and survey is not None
    use_novelty = dimension in NOVELTY_AUGMENTED_DIMS and novelty is not None

    if use_survey or use_novelty:
        context_suffix = build_context_suffix(dimension, survey, novelty)
        agent_def = DIMENSION_AGENTS[dimension]
        prompt = AUGMENTED_SUMMARY_SCORE_TEMPLATE.format(
            system=agent_def["system"],
            checklist=agent_def["checklist"],
            context_suffix=context_suffix,
            **_format_summary_kwargs(title, summary),
        )
        score_obj = _call_structured(client, prompt, DimensionScore)
    else:
        score_obj = score_dimension_from_summary(client, title, summary, dimension)

    return score_obj.model_dump(), use_survey, use_novelty


# ---------------------------------------------------------------------------
# Per-dimension worker
# ---------------------------------------------------------------------------

def _process_single_dimension(client, title, summary, dimension, survey, novelty):
    try:
        result, survey_injected, novelty_injected = score_dimension_augmented(
            client, title, summary, dimension, survey, novelty
        )
        result["survey_injected"] = survey_injected
        result["novelty_injected"] = novelty_injected
        return dimension, result
    except Exception as e:
        return dimension, {
            "score": None,
            "justification": f"[ERROR] {e}",
            "survey_injected": False,
            "novelty_injected": False,
        }


# ---------------------------------------------------------------------------
# Paper-level processing
# ---------------------------------------------------------------------------

def process_paper(client, paper_json, dimensions, out_dir):
    out_file = out_dir / f"{paper_json.stem}_review.json"
    if out_file.exists():
        return paper_json.name

    title, parsed_text = load_paper_json(paper_json)
    survey, novelty = load_novelty_data(paper_json)

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
                executor.submit(
                    _process_single_dimension, client, title, summary, dim, survey, novelty
                ): dim
                for dim in dimensions
            }
            for future in as_completed(futures):
                dim, result = future.result()
                results[dim] = result

    augmented_dims = [d for d, r in results.items() if r.get("survey_injected")]

    flat = {dim: results[dim]["score"] for dim in results}
    flat["_details"] = results
    flat["_summary"] = summary.model_dump() if summary else {"error": extraction_error}
    flat["_augmentation_stats"] = {
        "survey_found": survey is not None,
        "novelty_found": novelty is not None,
        "survey_pool_size": len(survey.get("paper_pool", [])) if survey else 0,
        "paper_novelty_mean": novelty.get("paper_novelty_mean") if novelty else None,
        "augmented_dims": augmented_dims,
        "total_dims": len(dimensions),
    }
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
            gt_files = [f for f in all_files if has_ground_truth(f, reviews_dir, dimensions)]
        else:
            gt_files = all_files

        # Only process papers with a valid (non-error) novelty output
        files = [f for f in gt_files if has_valid_novelty(f)]
        if not files:
            tqdm.write(f"[INFO] {venue_name}/{split}: no papers with valid novelty output, skipping")
            continue

        tqdm.write(f"[INFO] {venue_name}/{split}: {len(files)}/{len(gt_files)} papers have valid novelty")

        out_dir = output_dir / venue_folder / split
        out_dir.mkdir(parents=True, exist_ok=True)

        pbar = tqdm(
            total=len(files),
            desc=f"{venue_name}/{split} (novelty-augmented 2-stage, {len(dimensions)} dims)",
            unit="paper",
        )
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
    output_dir = Path("novelty_augmented_two_stage_prompts")

    print("Model                 : gpt-5-mini")
    print(f"Survey-augmented dims : {sorted(SURVEY_AUGMENTED_DIMS)}")
    print(f"Novelty-augmented dims: {sorted(NOVELTY_AUGMENTED_DIMS)}")
    print(f"Novelty outputs dir   : {NOVELTY_DIR}")
    print("Only papers with valid (non-error) novelty.json are processed.\n")

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
