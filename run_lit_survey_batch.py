"""
Batch Literature Survey Runner
================================
Runs the literature survey agent on all test papers across venues.

Supports two input modes:
  --source pdf          Use original PDFs (requires pdfplumber/pypdf)
  --source parsed_json  Use PeerRead parsed JSON files (no PDF dependency)

Outputs one SurveyResult JSON per paper to:
    lit_survey_results/<venue>/test/<paper_id>_survey.json
"""

import argparse
import os
import sys
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from tqdm import tqdm

# Add the literature_survey_agent directory to path so we can import its modules
sys.path.insert(0, str(Path(__file__).parent / "literature_survey_agent"))

from orchestrator import LiteratureSurveyOrchestrator
from pdf_loader import load_pdf, load_pdf_metadata


VENUE_MAP = {
    "acl_2017": "ACL",
    "iclr_2017": "ICLR",
    "conll_2016": "CONLL",
}


# --- Paper text loaders ---

def load_paper_text_from_json(json_path: Path) -> tuple[str, str, int | None]:
    """
    Load paper text from a PeerRead parsed JSON file.
    Returns (title, full_text, year).
    Includes references — useful for lit survey topic extraction.
    """
    with open(json_path) as f:
        data = json.load(f)

    title = data.get("title") or data.get("metadata", {}).get("title", "Unknown Title")

    def concat_values(obj):
        if isinstance(obj, dict):
            return " ".join(concat_values(v) for v in obj.values())
        elif isinstance(obj, list):
            return " ".join(concat_values(v) for v in obj)
        elif isinstance(obj, str):
            return obj
        else:
            return ""

    full_text = concat_values(data)
    return title, full_text.strip(), None


def load_paper_text_from_pdf(pdf_path: Path) -> tuple[str, str, int | None]:
    """
    Load paper text from a PDF file using the lit survey agent's pdf_loader.
    Returns (title, full_text, year).
    """
    text = load_pdf(pdf_path)
    meta = load_pdf_metadata(pdf_path)
    title = meta.get("title") or pdf_path.stem
    year = meta.get("year")
    return title, text, year


# --- Core logic ---

def run_survey_for_paper(orch, paper_path: Path, out_dir: Path, source: str) -> str:
    """Run the lit survey pipeline on a single paper. Returns paper ID."""
    paper_id = paper_path.stem  # e.g. "358.pdf" or "358"
    out_file = out_dir / f"{paper_id}_survey.json"

    # Skip if already processed
    if out_file.exists():
        return f"{paper_id} (cached)"

    if source == "parsed_json":
        title, full_text, year = load_paper_text_from_json(paper_path)
    else:
        title, full_text, year = load_paper_text_from_pdf(paper_path)

    try:
        result = orch.run(paper_text=full_text, paper_year=year)
        with open(out_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
    except Exception as e:
        # Save error so we don't retry on next run
        with open(out_file, "w") as f:
            json.dump({"error": str(e), "paper_id": paper_id}, f, indent=2)

    return paper_id


def process_venue(api_key: str, venue_folder: str, venue_name: str,
                  output_base: Path, source: str):
    """Process all test papers for a venue."""
    base = Path("PeerRead/data")

    if source == "parsed_json":
        input_dir = base / venue_folder / "test" / "parsed_pdfs"
        glob_pattern = "*.json"
    else:
        input_dir = base / venue_folder / "test" / "pdfs"
        glob_pattern = "*.pdf"

    if not input_dir.exists():
        tqdm.write(f"Skipping {venue_folder}: {input_dir} does not exist")
        return

    out_dir = output_base / venue_folder / "test"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_dir.glob(glob_pattern))

    # Each paper's survey involves multiple API calls (Semantic Scholar, arXiv, OpenAlex + LLM),
    # so we run papers sequentially per venue to avoid rate limiting
    orch = LiteratureSurveyOrchestrator(
        api_key=api_key,
        base_url="https://ai-gateway.andrew.cmu.edu",
        model="gpt-5-mini",
        max_iterations=2,
        new_paper_threshold=3,
        max_llm_calls=20,
        results_per_query=5,
        max_concurrent=3,
    )

    pbar = tqdm(total=len(files), desc=f"LitSurvey {venue_name}", unit="paper")
    for paper_path in files:
        try:
            run_survey_for_paper(orch, paper_path, out_dir, source)
        except Exception as e:
            tqdm.write(f"[ERROR] {venue_name} {paper_path.name}: {e}")
        pbar.update(1)
    pbar.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run literature survey agent on all PeerRead test papers."
    )
    parser.add_argument(
        "--source", choices=["pdf", "parsed_json"], default="parsed_json",
        help="Input source: 'pdf' uses original PDFs (requires pdfplumber), "
             "'parsed_json' uses PeerRead parsed JSON files (default: parsed_json)"
    )
    parser.add_argument(
        "--output-dir", default="lit_survey_results",
        help="Output directory for survey results (default: lit_survey_results)"
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in .env")
        sys.exit(1)

    output_base = Path(args.output_dir)

    # Run venues in parallel (papers within a venue run sequentially to respect rate limits)
    with ThreadPoolExecutor(max_workers=len(VENUE_MAP)) as executor:
        futures = []
        for venue_folder, venue_name in VENUE_MAP.items():
            futures.append(
                executor.submit(
                    process_venue, api_key, venue_folder, venue_name, output_base, args.source
                )
            )
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Venue thread failed: {exc}")

    print(f"\nDone! Results saved to {output_base}/")


if __name__ == "__main__":
    main()
