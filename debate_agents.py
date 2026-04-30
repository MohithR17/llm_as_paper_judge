"""
Debate-based multi-agent pipeline with two-stage summarisation and reviewer personas.

Stage 1 (once per paper): extract a compact PaperSummary from the full text.
Stage 1b (once per paper): generate two distinct reviewer personas matched to the paper topic.
Stage 2 (per dimension):  two independent reviewer agents, each embodying their persona,
  score from the summary.
  - If scores differ by more than DEBATE_THRESHOLD, a referee reads both and issues a verdict.
  - If they agree, scores are averaged and justifications concatenated.

Personas force genuinely different evaluation lenses (e.g., empiricist vs. theorist),
increasing calibrated disagreement and making the referee more meaningful.
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
    DimensionScore,
    DIMENSION_AGENTS,
    VENUE_DIMENSIONS,
    VENUE_MAP,
    load_paper_json,
    has_ground_truth,
)
from two_stage_agents import (
    PaperSummary,
    extract_paper_facts,
    _call_structured,
    _format_summary_kwargs,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEBATE_THRESHOLD = 1  # Trigger referee only when delta >= 2 (skip referee for 1-point gaps)


# ---------------------------------------------------------------------------
# Reviewer persona models
# ---------------------------------------------------------------------------

class ReviewerPersona(BaseModel):
    name: str = Field(description="Short label, e.g. 'Skeptical Empiricist'")
    background: str = Field(description="1-2 sentence academic/professional background")
    traits: List[str] = Field(description="3-5 concrete reviewing tendencies or biases")
    reviewing_style: str = Field(description="One sentence describing overall reviewing approach")


class ReviewerPersonaPair(BaseModel):
    reviewer_a: ReviewerPersona
    reviewer_b: ReviewerPersona


# ---------------------------------------------------------------------------
# Persona generation
# ---------------------------------------------------------------------------

PERSONA_GENERATION_PROMPT = """\
You are assigning two peer reviewers with COMPLEMENTARY evaluation lenses to this paper.
They should bring genuinely different expertise to the review — not opposite biases —
so that their combined perspective is more complete than either alone.

Paper title  : {title}
Topic area   : {topic_area}
Main claims  : {main_claims}
Methodology  : {methodology}

Design the personas as complementary domain experts:
- Reviewer A should be a METHODOLOGY specialist: someone whose expertise centers on
  experimental design, statistical rigor, and reproducibility for this type of paper
  (e.g. for an NLP paper: an ML engineer who cares about baselines, ablations, and
  implementation details).
- Reviewer B should be a DOMAIN / IMPACT specialist: someone whose expertise centers on
  the research landscape, novelty relative to prior work, and broader significance
  (e.g. for an NLP paper: a senior researcher who tracks what the field needs and where
  this work fits in the literature).

Rules:
- Traits must describe what each reviewer NOTICES and VALUES, not systematic up/down bias.
  Good: "requires ablation studies before accepting empirical claims"
  Bad: "gives low scores to papers without baselines" (that is adversarial bias, not lens)
- Both reviewers should be fair; they may disagree because they emphasize different
  dimensions of quality, not because one is an advocate and one is a skeptic.
- Personas must be realistic for ACL / ICLR / CoNLL / NeurIPS senior reviewers.

Output only the JSON object, nothing else."""


def generate_reviewer_personas(
    client: OpenAI, title: str, summary: PaperSummary
) -> ReviewerPersonaPair:
    """Generate two topic-matched reviewer personas from the paper summary."""
    kwargs = _format_summary_kwargs(title, summary)
    prompt = PERSONA_GENERATION_PROMPT.format(
        title=kwargs["title"],
        topic_area=kwargs["topic_area"],
        main_claims=kwargs["main_claims"],
        methodology=kwargs["methodology"],
    )
    return _call_structured(client, prompt, ReviewerPersonaPair)


# ---------------------------------------------------------------------------
# Persona-aware dimension scoring
# ---------------------------------------------------------------------------

PERSONA_SCORE_TEMPLATE = """\
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

### TASK
{checklist}

Respond with a JSON object containing:
- "score": an integer from 1 to 5
- "justification": 2-4 sentences referencing specific facts above, written from your persona's perspective

### CONSTRAINT
Base your judgment only on the provided summary. Output only the JSON object, nothing else."""


def _score_with_persona(
    client: OpenAI,
    title: str,
    summary: PaperSummary,
    dimension: str,
    persona: ReviewerPersona,
) -> DimensionScore:
    """Score one dimension from the summary, embodying the given reviewer persona."""
    agent_def = DIMENSION_AGENTS[dimension]
    traits_block = "\n".join(f"  - {t}" for t in persona.traits)
    prompt = PERSONA_SCORE_TEMPLATE.format(
        persona_name=persona.name,
        persona_background=persona.background,
        persona_reviewing_style=persona.reviewing_style,
        persona_traits=traits_block,
        system=agent_def["system"],
        checklist=agent_def["checklist"],
        **_format_summary_kwargs(title, summary),
    )
    return _call_structured(client, prompt, DimensionScore)


# ---------------------------------------------------------------------------
# Referee prompt
# ---------------------------------------------------------------------------

REFEREE_PROMPT = """\
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

### Reviewer A  (score: {score_a}/5)
{justification_a}

### Reviewer B  (score: {score_b}/5)
{justification_b}

### Instructions
Use the paper summary as evidence to evaluate which reviewer's argument is better grounded
in the facts. Do NOT simply split the difference — choose the score that is most defensible
given the actual paper facts above. The final score must be an integer 1–5.
Write a synthesized justification (2–4 sentences) that references specific paper facts
and explains why you chose this score over the alternative.

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
    summary: PaperSummary,
    dimension: str,
    score_a: int,
    justification_a: str,
    score_b: int,
    justification_b: str,
) -> DimensionScore:
    prompt = REFEREE_PROMPT.format(
        dimension=dimension,
        score_a=score_a,
        justification_a=justification_a,
        score_b=score_b,
        justification_b=justification_b,
        **_format_summary_kwargs(title, summary),
    )
    from two_stage_agents import LLM_TIMEOUT
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
# Single-dimension debate
# ---------------------------------------------------------------------------

def score_dimension_with_debate(
    client: OpenAI,
    title: str,
    summary: PaperSummary,
    dimension: str,
    personas: ReviewerPersonaPair | None = None,
) -> dict:
    """
    Score one dimension with two sequential reviewer calls then an optional referee.
    Reviewers run sequentially (not in parallel) to avoid nested thread pool deadlocks.
    """
    with ThreadPoolExecutor(max_workers=2) as ex:
        if personas is not None:
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
    }


# ---------------------------------------------------------------------------
# Per-dimension worker
# ---------------------------------------------------------------------------

def _process_single_dimension(client, title, summary, dimension, personas):
    try:
        result = score_dimension_with_debate(client, title, summary, dimension, personas)
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
    out_file = out_dir / f"{paper_json.stem}_review.json"
    if out_file.exists():
        return paper_json.name

    title, parsed_text = load_paper_json(paper_json)

    # Stage 1: extract structured summary once
    extraction_error = None
    try:
        summary = extract_paper_facts(client, title, parsed_text)
    except Exception as e:
        tqdm.write(f"[WARN] Extraction failed for {paper_json.name}: {e}")
        extraction_error = str(e)
        summary = None

    # Stage 1b: generate reviewer personas from the summary (once per paper)
    personas = None
    if summary is not None:
        try:
            personas = generate_reviewer_personas(client, title, summary)
        except Exception as e:
            tqdm.write(f"[WARN] Persona generation failed for {paper_json.name}: {e}")

    results = {}
    if summary is not None:
        # Stage 2: dimensions run in parallel (safe — only 2 executor levels: paper + dimension).
        # Reviewers A/B within each dimension are sequential to avoid a 3rd nesting level.
        with ThreadPoolExecutor(max_workers=len(dimensions)) as executor:
            futures = {
                executor.submit(_process_single_dimension, client, title, summary, dim, personas): dim
                for dim in dimensions
            }
            for future in as_completed(futures):
                dim, result = future.result()
                results[dim] = result

    n_debated = sum(1 for r in results.values() if r.get("debate_triggered"))

    flat = {dim: results[dim]["score"] for dim in results}
    flat["_details"] = results
    flat["_summary"] = summary.model_dump() if summary else {"error": extraction_error}
    flat["_personas"] = personas.model_dump() if personas else None
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

        pbar = tqdm(total=len(files), desc=f"{venue_name}/{split} (2-stage debate, {len(dimensions)} dims)", unit="paper")
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
    output_dir = Path("debate_agent_prompts")

    print("Model           : gpt-5-mini")
    print("Stage 1         : extract PaperSummary from full text (once per paper)")
    print("Stage 1b        : generate two reviewer personas matched to paper topic")
    print("Stage 2         : each persona reviews each dimension from the summary")
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
