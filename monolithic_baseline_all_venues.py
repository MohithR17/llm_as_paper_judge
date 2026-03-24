import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Review base models for all venues, matching PeerRead golden review schemas
from typing import Optional, Union


# ACL review schema (matches PeerRead ACL ground truth fields)
class ACLReviewBaseModel(BaseModel):
    IMPACT: Union[str, int]
    SUBSTANCE: Union[str, int]
    APPROPRIATENESS: Union[str, int]
    MEANINGFUL_COMPARISON: Union[str, int]
    PRESENTATION_FORMAT: Optional[str] = None
    CLARITY: Union[str, int]
    REVIEWER_CONFIDENCE: Union[str, int]
    SOUNDNESS_CORRECTNESS: Union[str, int]
    ORIGINALITY: Union[str, int]
    comments: str
    RECOMMENDATION: Union[str, int]


# CoNLL has the same fields as ACL plus REPLICABILITY
class CONLLReviewBaseModel(BaseModel):
    IMPACT: Union[str, int]
    SUBSTANCE: Union[str, int]
    APPROPRIATENESS: Union[str, int]
    MEANINGFUL_COMPARISON: Union[str, int]
    PRESENTATION_FORMAT: Optional[str] = None
    CLARITY: Union[str, int]
    REVIEWER_CONFIDENCE: Union[str, int]
    SOUNDNESS_CORRECTNESS: Union[str, int]
    ORIGINALITY: Union[str, int]
    REPLICABILITY: Union[str, int]
    comments: str
    RECOMMENDATION: Union[str, int]


# ICLR review schema (matches the numeric dimensions actually scored in PeerRead ICLR)
class ICLRReviewBaseModel(BaseModel):
    RECOMMENDATION: Union[str, int]
    REVIEWER_CONFIDENCE: Union[str, int]
    CLARITY: Union[str, int]
    SOUNDNESS_CORRECTNESS: Union[str, int]
    ORIGINALITY: Union[str, int]
    comments: str


# arXiv: use ACL schema since there are no ground truth reviews (only accept/reject)
ArXivReviewBaseModel = ACLReviewBaseModel

# Venue-specific rubrics
VENUE_RUBRICS = {
    "ACL": (
        "Please provide a review with the following fields: "
        "IMPACT (1-5), SUBSTANCE (1-5), APPROPRIATENESS (1-5), MEANINGFUL_COMPARISON (1-5), "
        "PRESENTATION_FORMAT (e.g. 'Oral Presentation' or 'Poster'), CLARITY (1-5), "
        "REVIEWER_CONFIDENCE (1-5), SOUNDNESS_CORRECTNESS (1-5), ORIGINALITY (1-5), "
        "comments (detailed textual review), RECOMMENDATION (1-10). "
        "For RECOMMENDATION, provide just the integer score (1-10). "
        "For all other numeric fields, provide just the integer score."
    ),
    "ICLR": (
        "Please provide a review with the following fields: "
        "RECOMMENDATION (1-10), REVIEWER_CONFIDENCE (1-5), CLARITY (1-5), "
        "SOUNDNESS_CORRECTNESS (1-5), ORIGINALITY (1-5), "
        "comments (detailed textual review). "
        "For RECOMMENDATION, provide just the integer score (1-10). "
        "For all other numeric fields, provide just the integer score."
    ),
    "CONLL": (
        "Please provide a review with the following fields: "
        "IMPACT (1-5), SUBSTANCE (1-5), APPROPRIATENESS (1-5), MEANINGFUL_COMPARISON (1-5), "
        "PRESENTATION_FORMAT (e.g. 'Oral Presentation' or 'Poster'), CLARITY (1-5), "
        "REVIEWER_CONFIDENCE (1-5), SOUNDNESS_CORRECTNESS (1-5), ORIGINALITY (1-5), "
        "REPLICABILITY (1-5), comments (detailed textual review), RECOMMENDATION (1-10). "
        "For RECOMMENDATION, provide just the integer score (1-10). "
        "For all other numeric fields, provide just the integer score."
    ),
    "arXiv": (
        "Please provide a review with the following fields: "
        "IMPACT (1-5), SUBSTANCE (1-5), APPROPRIATENESS (1-5), MEANINGFUL_COMPARISON (1-5), "
        "PRESENTATION_FORMAT (e.g. 'Oral Presentation' or 'Poster'), CLARITY (1-5), "
        "REVIEWER_CONFIDENCE (1-5), SOUNDNESS_CORRECTNESS (1-5), ORIGINALITY (1-5), "
        "comments (detailed textual review), RECOMMENDATION (1-10). "
        "For RECOMMENDATION, provide just the integer score (1-10). "
        "For all other numeric fields, provide just the integer score."
    ),
}

VENUE_MAP = {
    "acl_2017": "ACL",
    "iclr_2017": "ICLR",
    "conll_2016": "CONLL",
    "arxiv.cs.ai_2007-2017": "arXiv",
    "arxiv.cs.cl_2007-2017": "arXiv",
    "arxiv.cs.lg_2007-2017": "arXiv"
}

PROMPT_TEMPLATE = """
### SYSTEM ROLE
You are an expert senior reviewer for {venue}. Your goal is to provide a critical, objective, and constructive review.

### INPUT DATA
Paper Title: {title}
Paper Content: {parsed_text}

### TASK
Provide your review as a JSON object with the following fields, matching the official review format for {venue}:
{standard_rubric}

### CONSTRAINT
Base your judgment only on the provided text. Do not use external tools. Output only the JSON object, nothing else.
"""

def load_paper_json(json_path: Path):
    with open(json_path) as f:
        data = json.load(f)
    # Title logic as before
    title = data.get("title") or data.get("metadata", {}).get("title", "Unknown Title")
    # Concatenate all values except references
    import re

    def is_line_number_block(text):
        # Returns True if most lines in the text are just numbers (optionally with leading zeros)
        if not isinstance(text, str):
            return False
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines or len(lines) < 3:
            return False
        num_lines = len(lines)
        num_number_lines = sum(1 for line in lines if re.fullmatch(r'\d{1,5}', line))
        # If more than 70% of lines are just numbers, treat as a number block
        return num_number_lines / num_lines > 0.7

    def concat_values(obj):
        if isinstance(obj, dict):
            # Special handling for 'sections' key: filter out numeric/noisy sections
            if 'sections' in obj and isinstance(obj['sections'], list):
                filtered_sections = [s for s in obj['sections'] if not is_line_number_block(s.get('text', ''))]
                obj = dict(obj)
                obj['sections'] = filtered_sections
            return " ".join(concat_values(v) for k, v in obj.items() if k != "references")
        elif isinstance(obj, list):
            return " ".join(concat_values(v) for v in obj)
        elif isinstance(obj, str):
            # Remove any block of line-separated numbers
            blocks = obj.split('\n\n')
            filtered_blocks = []
            for block in blocks:
                lines = [line.strip() for line in block.splitlines() if line.strip()]
                if not lines:
                    continue
                num_lines = len(lines)
                num_number_lines = sum(1 for line in lines if re.fullmatch(r'\d{1,5}', line))
                # If more than 70% of lines in the block are just numbers, skip this block
                if num_lines >= 3 and num_number_lines / num_lines > 0.7:
                    continue
                filtered_blocks.append(block)
            return " ".join(filtered_blocks)
        else:
            return ""
    parsed_text = concat_values({k: v for k, v in data.items() if k != "references"})
    return title, parsed_text.strip()

def build_prompt(title: str, parsed_text: str, standard_rubric: str, venue: str) -> str:
    return PROMPT_TEMPLATE.format(title=title, parsed_text=parsed_text, standard_rubric=standard_rubric, venue=venue)

def _process_single_paper(paper_json, venue_name, rubric, out_split_dir, text_format):
    """Process a single paper: build prompt, call LLM, save output. Returns paper name."""
    title, parsed_text = load_paper_json(paper_json)
    prompt = build_prompt(title, parsed_text, rubric, venue_name)

    out_file = out_split_dir / f"{paper_json.stem}_prompt.txt"
    with open(out_file, "w") as f:
        f.write(prompt)

    try:
        review = prompt_gpt5_structured(prompt, text_format)
    except Exception as e:
        review = f"[ERROR] LLM call failed: {e}"

    review_to_save = review
    if hasattr(review_to_save, 'model_dump'):
        review_to_save = review_to_save.model_dump()
    elif hasattr(review_to_save, 'dict'):
        review_to_save = review_to_save.dict()

    out_review_file = out_split_dir / f"{paper_json.stem}_review.json"
    with open(out_review_file, "w") as f:
        json.dump(review_to_save, f, indent=2)
    return paper_json.name


def process_venue(venue_dir: Path, venue_name: str, rubric: str, output_dir: Path):
    venue_class_map = {
        "ACL": ACLReviewBaseModel,
        "ICLR": ICLRReviewBaseModel,
        "CONLL": CONLLReviewBaseModel,
        "arXiv": ArXivReviewBaseModel
    }
    for split in ["test"]:
        split_dir = venue_dir / split / "parsed_pdfs"
        if not split_dir.exists():
            continue
        out_split_dir = output_dir / venue_dir.name / split
        out_split_dir.mkdir(parents=True, exist_ok=True)
        files = list(split_dir.glob("*.json"))
        text_format = venue_class_map.get(venue_name, ACLReviewBaseModel)

        pbar = tqdm(total=len(files), desc=f"{venue_name}", unit="paper", position=0)
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(
                    _process_single_paper, paper_json, venue_name, rubric, out_split_dir, text_format
                ): paper_json
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
    global _CLIENT
    _CLIENT = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url="https://ai-gateway.andrew.cmu.edu")
    base = Path("PeerRead/data")
    output_dir = Path("monolithic_prompts")

    # Collect all venue tasks, then run venues in parallel
    venue_tasks = []
    for venue_folder, venue_name in VENUE_MAP.items():
        rubric = VENUE_RUBRICS[venue_name]
        venue_dir = base / venue_folder
        if not venue_dir.exists():
            continue
        venue_tasks.append((venue_dir, venue_name, rubric, output_dir))

    with ThreadPoolExecutor(max_workers=len(venue_tasks)) as executor:
        futures = [executor.submit(process_venue, *args) for args in venue_tasks]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"A venue thread failed: {exc}")

# Example structured prompt function for GPT-5
def prompt_gpt5_structured(text, text_format):
    # text_format should be the class name string, e.g., "ACLReviewBaseModel"
    response = _CLIENT.responses.parse(
        model="gpt-5-mini",  # or "gpt-5" if available
        input=[
            {
                "role": "developer",
                "content": [
                    {"type": "input_text", "text": text}
                ]
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": ""}]
            },
        ],
        text_format=text_format,
        reasoning={
            "effort": "medium",
            "summary": "auto"
        },
        tools=[],
        store=True,
        include=[
            "reasoning.encrypted_content",
        ]
    )
    return response.output_parsed

if __name__ == "__main__":
    main()
