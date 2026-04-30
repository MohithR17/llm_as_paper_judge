#!/usr/bin/env python3
"""
run_novelty_batch.py
====================
Runs literature survey + novelty classification on all PeerRead papers
that have ground-truth dimension scores, across train/dev/test splits.

Pipeline per paper
------------------
  Stage 1 — Literature survey (LiteratureSurveyOrchestrator):
      Retrieves and scores related prior work for the paper.
      Saved to: novelty_outputs/{venue}/{split}/{paper_id}/survey.json

  Stage 2 — Novelty scoring (run_novelty_pipeline):
      Extracts research claims from the paper and checks each claim
      against the survey's paper pool using LLM groundedness scoring.
      Saved to: novelty_outputs/{venue}/{split}/{paper_id}/novelty.json

Both files are saved as intermediate + final results so the run can be
resumed if interrupted. Papers where both files already exist are skipped.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# ── path setup ────────────────────────────────────────────────────────────────
# Both sub-packages use bare relative imports, so we inject their directories.
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "literature_survey_agent"))
sys.path.insert(0, str(ROOT / "novelty_classifier"))

from dimension_agents import (       # noqa: E402
    VENUE_DIMENSIONS,
    VENUE_MAP,
    load_paper_json,
    has_ground_truth,
)
from orchestrator import LiteratureSurveyOrchestrator   # noqa: E402
from run_novelty_pipeline import (                       # noqa: E402
    extract_claims_with_llm,
    score_claims_with_llm,
    load_literature_entries,
)

# ── configuration ─────────────────────────────────────────────────────────────

BASE_URL          = "https://ai-gateway.andrew.cmu.edu"
SURVEY_MODEL      = "gpt-5.4-nano"   # smaller model for survey (cheaper)
NOVELTY_MODEL     = "gpt-5-mini"     # claim extraction + groundedness scoring
MAX_PDF_CHARS     = 24_000           # chars sent to claim extractor
NOVELTY_THRESHOLD = 0.35             # claims above this are flagged novel
TOP_K_MATCHES     = 3                # top prior-work matches kept per claim
RETRIEVAL_K       = 5                # lexical retrieval candidates per claim

PAPER_WORKERS = 3

OUTPUT_DIR        = ROOT / "novelty_outputs"


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=BASE_URL)


def _extract_paper_year(paper_json: Path) -> int | None:
    """Best-effort year extraction from a parsed PDF JSON."""
    try:
        with open(paper_json) as f:
            data = json.load(f)
        year = (
            data.get("year")
            or data.get("metadata", {}).get("year")
            or data.get("metadata", {}).get("dateOfPublication", "")
        )
        if isinstance(year, int) and 1900 < year < 2100:
            return year
        if isinstance(year, str):
            import re
            m = re.search(r"((?:19|20)\d{2})", year)
            if m:
                return int(m.group(1))
    except Exception:
        pass
    return None


# ── per-paper pipeline ────────────────────────────────────────────────────────

def _run_survey(api_key: str, title: str, parsed_text: str,
                paper_year: int | None, survey_file: Path) -> bool:
    """
    Run the literature survey and save survey.json.
    Returns True on success, False on error.
    """
    try:
        orch = LiteratureSurveyOrchestrator(
            api_key=api_key,
            base_url=BASE_URL,
            model=SURVEY_MODEL,
        )
        result = orch.run(paper_text=parsed_text, paper_year=paper_year)
        survey_file.write_text(
            json.dumps(result.to_dict(), indent=2), encoding="utf-8"
        )
        return True
    except Exception as e:
        survey_file.write_text(
            json.dumps({"error": str(e), "stage": "survey"}), encoding="utf-8"
        )
        tqdm.write(f"  [WARN] survey failed for {survey_file.parent.name}: {e}")
        return False


def _run_novelty(api_key: str, title: str, parsed_text: str,
                 survey_file: Path, novelty_file: Path, paper_id: str) -> None:
    """
    Extract claims and score novelty against the survey pool. Save novelty.json.
    """
    try:
        survey_data = json.loads(survey_file.read_text(encoding="utf-8"))
        if "error" in survey_data and "paper_pool" not in survey_data:
            novelty_file.write_text(
                json.dumps({
                    "error": "survey failed, novelty skipped",
                    "paper_id": paper_id,
                }),
                encoding="utf-8",
            )
            return

        paper_pool = survey_data.get("paper_pool", [])
        if not paper_pool:
            novelty_file.write_text(
                json.dumps({
                    "error": "empty survey pool, novelty skipped",
                    "paper_id": paper_id,
                    "pool_size": 0,
                }),
                encoding="utf-8",
            )
            return

        client = _make_client(api_key)

        # load_literature_entries reads the survey JSON — it handles the
        # "paper_pool" key that SurveyResult.to_dict() produces.
        literature = load_literature_entries(str(survey_file))

        claims = extract_claims_with_llm(
            client,
            model=NOVELTY_MODEL,
            pdf_text=parsed_text,
            max_chars=MAX_PDF_CHARS,
        )

        report = score_claims_with_llm(
            client=client,
            model=NOVELTY_MODEL,
            claims=claims,
            literature=literature,
            novelty_threshold=NOVELTY_THRESHOLD,
            top_k_matches=TOP_K_MATCHES,
            retrieval_k=RETRIEVAL_K,
        )

        # Paper-level novelty summary
        claim_scores = [c["novelty_score"] for c in report.get("claims", [])]
        report["paper_id"]              = paper_id
        report["title"]                 = title
        report["survey_pool_size"]      = len(paper_pool)
        report["paper_novelty_mean"]    = round(sum(claim_scores) / len(claim_scores), 4) if claim_scores else None
        report["paper_novelty_max"]     = round(max(claim_scores), 4) if claim_scores else None
        report["paper_novelty_min"]     = round(min(claim_scores), 4) if claim_scores else None
        report["num_novel_claims"]      = sum(1 for c in report.get("claims", []) if c.get("is_novel"))

        novelty_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    except Exception as e:
        novelty_file.write_text(
            json.dumps({"error": str(e), "stage": "novelty", "paper_id": paper_id}),
            encoding="utf-8",
        )
        tqdm.write(f"  [WARN] novelty failed for {paper_id}: {e}")


def _survey_is_complete(survey_file: Path) -> bool:
    """True only if survey.json exists and contains a non-empty paper_pool."""
    if not survey_file.exists():
        return False
    try:
        data = json.loads(survey_file.read_text(encoding="utf-8"))
        return bool(data.get("paper_pool"))
    except Exception:
        return False


def _novelty_is_complete(novelty_file: Path) -> bool:
    """True only if novelty.json exists and contains no 'error' key."""
    if not novelty_file.exists():
        return False
    try:
        data = json.loads(novelty_file.read_text(encoding="utf-8"))
        return "error" not in data
    except Exception:
        return False


def process_paper(api_key: str, paper_json: Path, out_dir: Path) -> str:
    """
    Run the full two-stage pipeline for one paper.
    Both intermediate outputs are saved separately so partial runs are resumable.
    Stages with error outputs are retried on subsequent runs.
    """
    paper_id     = paper_json.stem          # e.g. "148.pdf"
    survey_file  = out_dir / "survey.json"
    novelty_file = out_dir / "novelty.json"

    # Skip only if both stages completed successfully
    if _survey_is_complete(survey_file) and _novelty_is_complete(novelty_file):
        return paper_id

    out_dir.mkdir(parents=True, exist_ok=True)
    title, parsed_text = load_paper_json(paper_json)
    paper_year = _extract_paper_year(paper_json)

    # Stage 1: run survey if missing or previously errored/empty
    if not _survey_is_complete(survey_file):
        _run_survey(api_key, title, parsed_text, paper_year, survey_file)

    # Stage 2: run novelty if survey succeeded and novelty is missing or errored
    if _survey_is_complete(survey_file) and not _novelty_is_complete(novelty_file):
        _run_novelty(api_key, title, parsed_text, survey_file, novelty_file, paper_id)

    return paper_id


# ── venue / split processing ──────────────────────────────────────────────────

def process_split(api_key: str, venue_folder: str, split: str,
                  dimensions: list[str], output_dir: Path) -> None:
    base = Path("PeerRead/data")
    split_dir   = base / venue_folder / split / "parsed_pdfs"
    reviews_dir = base / venue_folder / split / "reviews"

    if not split_dir.exists():
        return

    all_files = list(split_dir.glob("*.json"))
    if reviews_dir.exists():
        files = [f for f in all_files if has_ground_truth(f, reviews_dir, dimensions)]
        skipped = len(all_files) - len(files)
        if skipped:
            tqdm.write(
                f"[INFO] {venue_folder}/{split}: skipping {skipped}/{len(all_files)} "
                f"papers with no GT scores"
            )
    else:
        files = all_files

    if not files:
        return

    venue_name = VENUE_MAP.get(venue_folder, venue_folder)
    pbar = tqdm(
        total=len(files),
        desc=f"{venue_name}/{split} (survey+novelty)",
        unit="paper",
    )

    with ThreadPoolExecutor(max_workers=PAPER_WORKERS) as executor:
        futures = {
            executor.submit(
                process_paper,
                api_key,
                paper_json,
                output_dir / venue_folder / split / paper_json.stem,
            ): paper_json
            for paper_json in files
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                tqdm.write(
                    f"[ERROR] {venue_folder}/{split} "
                    f"{futures[future].name}: {e}"
                )
            pbar.update(1)
    pbar.close()


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set.")

    print(f"Survey model      : {SURVEY_MODEL}")
    print(f"Novelty model     : {NOVELTY_MODEL}")
    print(f"Output dir        : {OUTPUT_DIR}")
    print(f"Paper workers     : {PAPER_WORKERS}")
    s2_key = os.getenv("S2_API_KEY")
    s2_status = "enabled (API key set)" if s2_key else "disabled (no API key — set S2_API_KEY to enable)"
    print(f"Semantic Scholar  : {s2_status}")
    print(f"Retrieval sources : OpenAlex only (arXiv disabled)")
    print(f"Retrieval         : 1 iteration, 2 variants/slot (broad+narrow)\n")

    for venue_folder, dimensions in VENUE_DIMENSIONS.items():
        for split in ["train", "dev", "test"]:
            process_split(api_key, venue_folder, split, dimensions, OUTPUT_DIR)

    print("\nDone. Results in novelty_outputs/")


if __name__ == "__main__":
    main()
