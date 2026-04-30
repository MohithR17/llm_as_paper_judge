"""
Novelty-augmented debate-based multi-agent pipeline.

Same as debate_agents.py but injects pre-computed literature survey and
novelty classifier context into the dimensions where prior-work knowledge helps:

  MEANINGFUL_COMPARISON  — sees ranked literature survey (related prior works)
  ORIGINALITY            — sees literature survey + claim-level novelty scores
  SUBSTANCE              — sees literature survey (check for missing comparisons)

Context is injected into every prompt that scores an augmented dimension:
  - Reviewer A and B persona scoring prompts
  - Referee prompt (when debate is triggered)

All other dimensions use the standard unaugmented debate pipeline.

Prerequisites:
  Run `python run_novelty_batch.py` first to populate:
    novelty_outputs/{venue}/{split}/{paper_id}/survey.json
    novelty_outputs/{venue}/{split}/{paper_id}/novelty.json

Only papers with a valid (non-error) novelty.json are processed.

Outputs to: novelty_augmented_debate_agent_prompts/
"""

import os
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
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
    LLM_TIMEOUT,
)
from debate_agents import (
    ReviewerPersona,
    ReviewerPersonaPair,
    generate_reviewer_personas,
    DEBATE_THRESHOLD,
)
from novelty_augmented_two_stage import (
    load_novelty_data,
    has_valid_novelty,
    build_context_suffix,
    SURVEY_AUGMENTED_DIMS,
    NOVELTY_AUGMENTED_DIMS,
)

# ---------------------------------------------------------------------------
# Augmented persona scoring prompt
# ---------------------------------------------------------------------------

AUGMENTED_PERSONA_SCORE_TEMPLATE = """\
### YOUR REVIEWER PERSONA
Name            : {persona_name}
Background      : {persona_background}
Reviewing style : {persona_reviewing_style}
Key traits      :
{persona_traits}

Adopt this persona fully. Let your background and traits shape which checklist items
you weight most heavily and how strictly you apply each criterion.

### SYSTEM ROLE
{system}

### PAPER SUMMARY (extracted facts)
Title                : {title}
Main claims          : {main_claims}
Methodology          : {methodology}
Datasets used        : {datasets}
Baselines compared   : {baselines}
Evaluation metrics   : {metrics}
Stated limitations   : {limitations}
Code available       : {code_available}
Hyperparameters reported: {hyperparams_reported}
Related work coverage: {related_work_coverage}
Topic area           : {topic_area}
Writing quality      : {writing_quality}
{context_suffix}

### TASK
{checklist}

Respond with a JSON object containing:
- "score": an integer from 1 to 5
- "justification": 2-4 sentences referencing specific facts above and prior works where applicable, written from your persona's perspective

### CONSTRAINT
Base your judgment only on the provided summary and context. Output only the JSON object, nothing else."""


# ---------------------------------------------------------------------------
# Augmented referee prompt
# ---------------------------------------------------------------------------

AUGMENTED_REFEREE_PROMPT = """\
### REFEREE TASK
Two independent reviewers disagreed on the dimension "{dimension}" for the paper below.
Your job is to read both perspectives AND the paper's factual summary, then issue a final score.

Paper Title: {title}

### PAPER SUMMARY (ground truth facts)
Main claims          : {main_claims}
Methodology          : {methodology}
Datasets used        : {datasets}
Baselines compared   : {baselines}
Evaluation metrics   : {metrics}
Stated limitations   : {limitations}
Code available       : {code_available}
Hyperparameters reported: {hyperparams_reported}
Related work coverage: {related_work_coverage}
Topic area           : {topic_area}
Writing quality      : {writing_quality}
{context_suffix}

### Reviewer A  (score: {score_a}/5)
{justification_a}

### Reviewer B  (score: {score_b}/5)
{justification_b}

### Instructions
Use the paper summary and context as evidence to evaluate which reviewer's argument is better grounded.
Do NOT simply split the difference — choose the score most defensible given the facts.
Write a synthesized justification (2–4 sentences) referencing specific paper facts
and prior works where applicable.

Output a JSON object:
- "score": integer 1-5
- "justification": synthesized 2-4 sentence justification

Output only the JSON object, nothing else."""


# ---------------------------------------------------------------------------
# Augmented scorer and referee
# ---------------------------------------------------------------------------

def _score_with_persona_augmented(
    client: OpenAI,
    title: str,
    summary: PaperSummary,
    dimension: str,
    persona: ReviewerPersona,
    survey: dict | None,
    novelty: dict | None,
) -> DimensionScore:
    agent_def = DIMENSION_AGENTS[dimension]
    traits_block = "\n".join(f"  - {t}" for t in persona.traits)
    context_suffix = build_context_suffix(dimension, survey, novelty)
    prompt = AUGMENTED_PERSONA_SCORE_TEMPLATE.format(
        persona_name=persona.name,
        persona_background=persona.background,
        persona_reviewing_style=persona.reviewing_style,
        persona_traits=traits_block,
        system=agent_def["system"],
        checklist=agent_def["checklist"],
        context_suffix=context_suffix,
        **_format_summary_kwargs(title, summary),
    )
    return _call_structured(client, prompt, DimensionScore)


def _call_referee_augmented(
    client: OpenAI,
    title: str,
    summary: PaperSummary,
    dimension: str,
    score_a: int,
    justification_a: str,
    score_b: int,
    justification_b: str,
    survey: dict | None,
    novelty: dict | None,
) -> DimensionScore:
    context_suffix = build_context_suffix(dimension, survey, novelty)
    prompt = AUGMENTED_REFEREE_PROMPT.format(
        dimension=dimension,
        score_a=score_a,
        justification_a=justification_a,
        score_b=score_b,
        justification_b=justification_b,
        context_suffix=context_suffix,
        **_format_summary_kwargs(title, summary),
    )
    response = client.responses.parse(
        model="gpt-5-mini",
        input=[
            {"role": "developer", "content": [{"type": "input_text", "text": prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": ""}]},
        ],
        text_format=DimensionScore,
        reasoning={"effort": "low", "summary": "auto"},
        tools=[],
        store=True,
        include=["reasoning.encrypted_content"],
        timeout=LLM_TIMEOUT,
    )
    return response.output_parsed


# ---------------------------------------------------------------------------
# Single-dimension debate (augmented)
# ---------------------------------------------------------------------------

def score_dimension_with_debate_augmented(
    client: OpenAI,
    title: str,
    summary: PaperSummary,
    dimension: str,
    personas: ReviewerPersonaPair | None,
    survey: dict | None,
    novelty: dict | None,
) -> dict:
    """Debate scoring with survey/novelty context injected for augmented dimensions."""
    use_augmentation = (
        (dimension in SURVEY_AUGMENTED_DIMS and survey is not None) or
        (dimension in NOVELTY_AUGMENTED_DIMS and novelty is not None)
    )

    with ThreadPoolExecutor(max_workers=2) as ex:
        if personas is not None and use_augmentation:
            fut_a = ex.submit(
                _score_with_persona_augmented,
                client, title, summary, dimension, personas.reviewer_a, survey, novelty,
            )
            fut_b = ex.submit(
                _score_with_persona_augmented,
                client, title, summary, dimension, personas.reviewer_b, survey, novelty,
            )
        elif personas is not None:
            from debate_agents import _score_with_persona
            fut_a = ex.submit(_score_with_persona, client, title, summary, dimension, personas.reviewer_a)
            fut_b = ex.submit(_score_with_persona, client, title, summary, dimension, personas.reviewer_b)
        else:
            from two_stage_agents import score_dimension_from_summary
            fut_a = ex.submit(score_dimension_from_summary, client, title, summary, dimension)
            fut_b = ex.submit(score_dimension_from_summary, client, title, summary, dimension)
        review_a: DimensionScore = fut_a.result()
        review_b: DimensionScore = fut_b.result()

    delta = abs(review_a.score - review_b.score)
    debate_triggered = delta > DEBATE_THRESHOLD

    if debate_triggered:
        if use_augmentation:
            final = _call_referee_augmented(
                client, title, summary, dimension,
                review_a.score, review_a.justification,
                review_b.score, review_b.justification,
                survey, novelty,
            )
        else:
            from debate_agents import _call_referee
            final = _call_referee(
                client, title, summary, dimension,
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
        "survey_injected": use_augmentation and survey is not None and dimension in SURVEY_AUGMENTED_DIMS,
        "novelty_injected": use_augmentation and novelty is not None and dimension in NOVELTY_AUGMENTED_DIMS,
    }


# ---------------------------------------------------------------------------
# Per-dimension worker
# ---------------------------------------------------------------------------

def _process_single_dimension(client, title, summary, dimension, personas, survey, novelty):
    try:
        result = score_dimension_with_debate_augmented(
            client, title, summary, dimension, personas, survey, novelty
        )
        return dimension, result
    except Exception as e:
        return dimension, {
            "score": None,
            "justification": f"[ERROR] {e}",
            "debate_triggered": False,
            "resolution_method": "error",
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

    personas = None
    if summary is not None:
        try:
            personas = generate_reviewer_personas(client, title, summary)
        except Exception as e:
            tqdm.write(f"[WARN] Persona generation failed for {paper_json.name}: {e}")

    results = {}
    if summary is not None:
        with ThreadPoolExecutor(max_workers=len(dimensions)) as executor:
            futures = {
                executor.submit(
                    _process_single_dimension,
                    client, title, summary, dim, personas, survey, novelty,
                ): dim
                for dim in dimensions
            }
            for future in as_completed(futures):
                dim, result = future.result()
                results[dim] = result

    n_debated = sum(1 for r in results.values() if r.get("debate_triggered"))
    augmented_dims = [d for d, r in results.items() if r.get("survey_injected")]

    flat = {dim: results[dim]["score"] for dim in results}
    flat["_details"] = results
    flat["_summary"] = summary.model_dump() if summary else {"error": extraction_error}
    flat["_personas"] = personas.model_dump() if personas else None
    flat["_debate_stats"] = {
        "dimensions_debated": n_debated,
        "total_dimensions": len(dimensions),
        "debate_rate": round(n_debated / len(dimensions), 3) if dimensions else 0,
    }
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
            desc=f"{venue_name}/{split} (novelty-augmented debate, {len(dimensions)} dims)",
            unit="paper",
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
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
    output_dir = Path("novelty_augmented_debate_agent_prompts")

    print("Model                 : gpt-5-mini")
    print(f"Survey-augmented dims : {sorted(SURVEY_AUGMENTED_DIMS)}")
    print(f"Novelty-augmented dims: {sorted(NOVELTY_AUGMENTED_DIMS)}")
    print(f"Debate threshold      : delta > {DEBATE_THRESHOLD} triggers referee")
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
