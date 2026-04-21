import os
import json
from pathlib import Path
from typing import Union
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- Structured output models (one score + justification per dimension) ---

class DimensionScore(BaseModel):
    score: int
    justification: str


# --- Dimension agent definitions with checklist-based rubrics ---

DIMENSION_AGENTS = {
    "SOUNDNESS_CORRECTNESS": {
        "system": (
            "You are a specialist reviewer evaluating ONLY the technical soundness and "
            "correctness of a research paper. Do not evaluate writing quality, novelty, "
            "or impact — focus exclusively on whether the claims are technically valid."
        ),
        "checklist": (
            "Evaluate the paper on these checklist items, then give an overall score (1-5):\n"
            "1. Are the main claims clearly stated and supported by evidence (proofs, experiments, analysis)?\n"
            "2. Are the mathematical derivations and proofs (if any) correct and complete?\n"
            "3. Is the experimental methodology sound (proper baselines, controls, statistical tests)?\n"
            "4. Are there logical gaps or unsupported assumptions in the arguments?\n"
            "5. Are the conclusions justified by the results presented?\n\n"
            "Score scale:\n"
            "  5 = Technically flawless, all claims well-supported\n"
            "  4 = Minor issues that don't affect main conclusions\n"
            "  3 = Some concerns about correctness or missing evidence\n"
            "  2 = Significant technical flaws or unsupported claims\n"
            "  1 = Fundamental errors that invalidate the contribution"
        ),
    },
    "ORIGINALITY": {
        "system": (
            "You are a specialist reviewer evaluating ONLY the originality and novelty "
            "of a research paper. Do not evaluate writing quality, soundness, or impact — "
            "focus exclusively on how novel the contribution is."
        ),
        "checklist": (
            "Evaluate the paper on these checklist items, then give an overall score (1-5):\n"
            "1. Does the paper introduce a genuinely new idea, method, or perspective?\n"
            "2. Is the problem formulation itself novel, or is it a well-studied problem?\n"
            "3. Does the paper clearly differentiate its contributions from prior work?\n"
            "4. Are the techniques a non-trivial extension beyond existing methods?\n"
            "5. Would the contribution surprise or inform an expert in the field?\n\n"
            "Score scale:\n"
            "  5 = Highly original, significant new ideas\n"
            "  4 = Notable novelty with clear differentiation from prior work\n"
            "  3 = Incremental contribution, moderate novelty\n"
            "  2 = Mostly combines known techniques with limited novelty\n"
            "  1 = No meaningful novelty over existing work"
        ),
    },
    "CLARITY": {
        "system": (
            "You are a specialist reviewer evaluating ONLY the clarity and presentation "
            "quality of a research paper. Do not evaluate technical soundness, novelty, "
            "or impact — focus exclusively on how clearly the paper communicates its ideas."
        ),
        "checklist": (
            "Evaluate the paper on these checklist items, then give an overall score (1-5):\n"
            "1. Is the paper well-organized with a logical flow (intro -> method -> experiments -> conclusion)?\n"
            "2. Are the main contributions clearly stated early in the paper?\n"
            "3. Is the notation consistent and well-defined?\n"
            "4. Are figures, tables, and examples effective and informative?\n"
            "5. Is the writing free of ambiguity, jargon overload, and grammatical issues?\n\n"
            "Score scale:\n"
            "  5 = Exceptionally clear, a pleasure to read\n"
            "  4 = Well-written with minor presentation issues\n"
            "  3 = Adequate but some sections hard to follow\n"
            "  2 = Poorly organized or unclear in important sections\n"
            "  1 = Very difficult to understand, major rewriting needed"
        ),
    },
    "SUBSTANCE": {
        "system": (
            "You are a specialist reviewer evaluating ONLY the substance and thoroughness "
            "of a research paper. Do not evaluate writing quality, novelty, or impact — "
            "focus exclusively on the depth and completeness of the work."
        ),
        "checklist": (
            "Evaluate the paper on these checklist items, then give an overall score (1-5):\n"
            "1. Are the experiments comprehensive enough to support the claims?\n"
            "2. Are ablation studies or sensitivity analyses provided where appropriate?\n"
            "3. Is the related work coverage thorough and fair?\n"
            "4. Are failure cases, limitations, or negative results discussed?\n"
            "5. Is there sufficient detail to reproduce the work?\n\n"
            "Score scale:\n"
            "  5 = Extremely thorough, comprehensive experiments and analysis\n"
            "  4 = Solid depth with minor gaps\n"
            "  3 = Adequate but missing some important experiments or analysis\n"
            "  2 = Shallow, important aspects left unexamined\n"
            "  1 = Severely lacking in depth or evidence"
        ),
    },
    "IMPACT": {
        "system": (
            "You are a specialist reviewer evaluating ONLY the potential impact and "
            "significance of a research paper. Do not evaluate writing quality, soundness, "
            "or novelty — focus exclusively on how important the contribution is."
        ),
        "checklist": (
            "Evaluate the paper on these checklist items, then give an overall score (1-5):\n"
            "1. Does the paper address an important problem in the field?\n"
            "2. Would the results change how researchers or practitioners think about this area?\n"
            "3. Does the work open up significant new research directions?\n"
            "4. Are the results likely to be widely used or built upon?\n"
            "5. Does the contribution have broader implications beyond the specific subfield?\n\n"
            "Score scale:\n"
            "  5 = Transformative, will significantly influence the field\n"
            "  4 = Strong impact, addresses an important problem well\n"
            "  3 = Moderate impact, useful but not field-changing\n"
            "  2 = Limited impact, niche contribution\n"
            "  1 = Negligible impact"
        ),
    },
    "APPROPRIATENESS": {
        "system": (
            "You are a specialist reviewer evaluating ONLY whether a research paper is "
            "appropriate for a top-tier NLP/ML venue. Do not evaluate technical soundness, "
            "novelty, or clarity — focus exclusively on fit and relevance."
        ),
        "checklist": (
            "Evaluate the paper on these checklist items, then give an overall score (1-5):\n"
            "1. Is the topic within the scope of a top NLP/ML conference?\n"
            "2. Does the paper address the right audience for this venue?\n"
            "3. Is the problem formulation appropriate for the venue's standards?\n"
            "4. Does the paper use methods and evaluation standards expected at this venue?\n"
            "5. Is the paper positioned correctly within the venue's research community?\n\n"
            "Score scale:\n"
            "  5 = Perfect fit for the venue\n"
            "  4 = Good fit with minor scope concerns\n"
            "  3 = Borderline appropriate\n"
            "  2 = Questionable fit, may belong at a different venue\n"
            "  1 = Clearly outside the venue's scope"
        ),
    },
    "MEANINGFUL_COMPARISON": {
        "system": (
            "You are a specialist reviewer evaluating ONLY whether a research paper makes "
            "meaningful comparisons with prior work. Do not evaluate other aspects."
        ),
        "checklist": (
            "Evaluate the paper on these checklist items, then give an overall score (1-5):\n"
            "1. Does the paper compare against appropriate and recent baselines?\n"
            "2. Are the comparisons fair (same data, same evaluation metrics, same conditions)?\n"
            "3. Does the paper acknowledge when prior methods perform comparably or better?\n"
            "4. Are the differences between the proposed method and baselines clearly analyzed?\n"
            "5. Does the paper cite and discuss the most relevant competing approaches?\n\n"
            "Score scale:\n"
            "  5 = Comprehensive and fair comparisons with all relevant baselines\n"
            "  4 = Good comparisons with minor omissions\n"
            "  3 = Some comparisons but missing important baselines or unfair setup\n"
            "  2 = Inadequate comparisons, key baselines missing\n"
            "  1 = No meaningful comparison with prior work"
        ),
    },
    "REPLICABILITY": {
        "system": (
            "You are a specialist reviewer evaluating ONLY the replicability of a research "
            "paper. Do not evaluate other aspects — focus on whether someone could reproduce this work."
        ),
        "checklist": (
            "Evaluate the paper on these checklist items, then give an overall score (1-5):\n"
            "1. Are all hyperparameters, model configurations, and training details specified?\n"
            "2. Is the data preprocessing pipeline clearly described?\n"
            "3. Are the datasets publicly available or clearly described for recreation?\n"
            "4. Is the code available or could the method be reimplemented from the description?\n"
            "5. Are the evaluation metrics and procedures precisely defined?\n\n"
            "Score scale:\n"
            "  5 = Fully reproducible, all details provided\n"
            "  4 = Mostly reproducible with minor details missing\n"
            "  3 = Partially reproducible, some important details absent\n"
            "  2 = Difficult to reproduce, many details missing\n"
            "  1 = Not reproducible from the paper alone"
        ),
    },
}

# Which dimensions to run for each venue (matching ground truth availability)
VENUE_DIMENSIONS = {
    "acl_2017": [
        "SOUNDNESS_CORRECTNESS", "ORIGINALITY", "CLARITY", "SUBSTANCE",
        "IMPACT", "APPROPRIATENESS", "MEANINGFUL_COMPARISON",
    ],
    "conll_2016": [
        "SOUNDNESS_CORRECTNESS", "ORIGINALITY", "CLARITY", "SUBSTANCE",
        "IMPACT", "APPROPRIATENESS", "MEANINGFUL_COMPARISON", "REPLICABILITY",
    ],
    "iclr_2017": [
        "SOUNDNESS_CORRECTNESS", "ORIGINALITY", "CLARITY",
    ],
}

VENUE_MAP = {
    "acl_2017": "ACL",
    "iclr_2017": "ICLR",
    "conll_2016": "CONLL",
}

DIMENSION_PROMPT_TEMPLATE = """### SYSTEM ROLE
{system}

### INPUT DATA
Paper Title: {title}
Paper Content: {parsed_text}

### TASK
{checklist}

Respond with a JSON object containing:
- "score": an integer from 1 to 5
- "justification": a concise explanation (2-4 sentences) citing specific parts of the paper

### CONSTRAINT
Base your judgment only on the provided text. Output only the JSON object, nothing else.
"""


def load_paper_json(json_path: Path):
    """Load and parse a paper JSON file, same as monolithic baseline."""
    import re

    with open(json_path) as f:
        data = json.load(f)
    title = data.get("title") or data.get("metadata", {}).get("title", "Unknown Title")

    def is_line_number_block(text):
        if not isinstance(text, str):
            return False
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines or len(lines) < 3:
            return False
        num_number_lines = sum(1 for line in lines if re.fullmatch(r'\d{1,5}', line))
        return num_number_lines / len(lines) > 0.7

    def concat_values(obj):
        if isinstance(obj, dict):
            if 'sections' in obj and isinstance(obj['sections'], list):
                filtered_sections = [s for s in obj['sections'] if not is_line_number_block(s.get('text', ''))]
                obj = dict(obj)
                obj['sections'] = filtered_sections
            return " ".join(concat_values(v) for k, v in obj.items() if k != "references")
        elif isinstance(obj, list):
            return " ".join(concat_values(v) for v in obj)
        elif isinstance(obj, str):
            blocks = obj.split('\n\n')
            filtered_blocks = []
            for block in blocks:
                lines = [line.strip() for line in block.splitlines() if line.strip()]
                if not lines:
                    continue
                num_number_lines = sum(1 for line in lines if re.fullmatch(r'\d{1,5}', line))
                if len(lines) >= 3 and num_number_lines / len(lines) > 0.7:
                    continue
                filtered_blocks.append(block)
            return " ".join(filtered_blocks)
        else:
            return ""

    parsed_text = concat_values({k: v for k, v in data.items() if k != "references"})
    # Strip lone surrogates produced by broken PDF extraction (e.g. \ud835 math chars)
    parsed_text = parsed_text.encode("utf-8", errors="replace").decode("utf-8")
    return title, parsed_text.strip()


def call_dimension_agent(client, title, parsed_text, dimension):
    """Call the LLM for a single dimension, returning a DimensionScore."""
    agent_def = DIMENSION_AGENTS[dimension]
    prompt = DIMENSION_PROMPT_TEMPLATE.format(
        system=agent_def["system"],
        title=title,
        parsed_text=parsed_text,
        checklist=agent_def["checklist"],
    )
    response = client.responses.parse(
        model="gpt-5-mini",
        input=[
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": ""}],
            },
        ],
        text_format=DimensionScore,
        reasoning={"effort": "medium", "summary": "auto"},
        tools=[],
        store=True,
        include=["reasoning.encrypted_content"],
    )
    return response.output_parsed


def _process_single_dimension(client, title, parsed_text, dimension):
    """Call a single dimension agent. Returns (dimension, result_dict)."""
    try:
        score_obj = call_dimension_agent(client, title, parsed_text, dimension)
        return dimension, score_obj.model_dump()
    except Exception as e:
        return dimension, {"score": None, "justification": f"[ERROR] {e}"}


def process_paper(client, paper_json, dimensions, out_dir):
    """Run all dimension agents on a single paper in parallel and save combined output."""
    title, parsed_text = load_paper_json(paper_json)
    results = {}

    # Run all dimensions for this paper in parallel
    with ThreadPoolExecutor(max_workers=len(dimensions)) as dim_executor:
        dim_futures = {
            dim_executor.submit(_process_single_dimension, client, title, parsed_text, dim): dim
            for dim in dimensions
        }
        for future in as_completed(dim_futures):
            dim, result = future.result()
            results[dim] = result

    # Save combined review (flat scores + justifications)
    out_file = out_dir / f"{paper_json.stem}_review.json"
    flat = {dim: results[dim]["score"] for dim in results}
    flat["_details"] = results
    with open(out_file, "w") as f:
        json.dump(flat, f, indent=2)
    return paper_json.name


def process_venue(client, venue_folder, venue_name, output_dir):
    """Process all test papers for a venue with parallel paper processing."""
    base = Path("PeerRead/data")
    venue_dir = base / venue_folder
    split_dir = venue_dir / "test" / "parsed_pdfs"
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

    pbar = tqdm(total=len(files), desc=f"{venue_name} ({len(dimensions)} dims)", unit="paper")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(process_paper, client, paper_json, dimensions, out_dir): paper_json
            for paper_json in files
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                paper = futures[future]
                tqdm.write(f"[ERROR] {venue_name} {paper.name}: {e}")
            pbar.update(1)
    pbar.close()


def main():
    load_dotenv()
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://ai-gateway.andrew.cmu.edu",
    )
    output_dir = Path("dimension_agent_prompts")

    # Process venues in parallel
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
