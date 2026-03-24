"""
Literature Survey Orchestrator
================================
Ties together all four pipeline components into a single iterative loop:

    TopicExtractor → [QueryGenerator → RetrievalLayer → RelevanceFilter] × N

Each iteration:
  1. QueryGenerator produces a fresh batch (skipping already-issued queries,
     boosting low-yield slots from the previous iteration).
  2. RetrievalLayer fetches candidates from Semantic Scholar / arXiv / OpenAlex.
  3. RelevanceFilter heuristically prunes then LLM-scores survivors.
  4. New included papers are merged into the running paper pool (dedup by key).

Loop stops when ANY of these conditions fires:
  - Δ new papers this iteration < NEW_PAPER_THRESHOLD  (convergence)
  - iteration count reaches MAX_ITERATIONS             (depth limit)
  - total LLM batch calls reaches MAX_LLM_CALLS        (budget cap)

Final output — SurveyResult:
  - paper pool   : deduplicated ScoredPaper records, sorted by final_score
  - taxonomy     : the extracted TopicTaxonomy dict
  - run stats    : per-iteration breakdown for debugging
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import os
from dataclasses import dataclass, field
from typing import Optional

# ── pipeline imports ──────────────────────────────────────────────────────────
from topic_extractor  import TopicExtractor,  TopicTaxonomy
from query_generator  import QueryGenerator,  QueryBatch, SearchQuery
from retrieval_layer  import RetrievalLayer
from relevance_filter import RelevanceFilter, ScoredPaper, FilterResult
from pdf_loader       import load_pdf, load_pdf_metadata


# ── Stop-condition defaults (all overridable via CLI / constructor) ────────────
MAX_ITERATIONS       = 2
NEW_PAPER_THRESHOLD  = 5     # stop if fewer than this many new papers in an iteration
MAX_LLM_CALLS        = 40    # budget cap: total LLM scoring batches across all iterations


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class IterationStats:
    iteration: int
    queries_issued:   int = 0
    candidates_found: int = 0
    heuristic_drops:  int = 0
    llm_scored:       int = 0
    new_included:     int = 0    # papers not seen in any prior iteration
    total_pool_size:  int = 0
    elapsed_sec:      float = 0.0
    stop_reason:      str = ""   # set on the final iteration


@dataclass
class SurveyResult:
    """Complete output of one literature survey run."""
    paper_title: str
    paper_year:  Optional[int]
    taxonomy:    dict
    paper_pool:  list[ScoredPaper]        # all included papers, sorted by score
    borderline:  list[ScoredPaper]        # borderline papers from all iterations
    iterations:  list[IterationStats]
    stop_reason: str

    def to_dict(self) -> dict:
        return {
            "paper_title": self.paper_title,
            "paper_year":  self.paper_year,
            "taxonomy":    self.taxonomy,
            "stop_reason": self.stop_reason,
            "iterations":  [vars(s) for s in self.iterations],
            "pool_size":   len(self.paper_pool),
            "paper_pool":  [p.to_dict() for p in self.paper_pool],
            "borderline":  [p.to_dict() for p in self.borderline],
        }

    def pretty(self) -> str:
        lines = [
            "=" * 64,
            f"  Literature Survey — {self.paper_title}",
            "=" * 64,
            f"  Paper year : {self.paper_year or 'unknown'}",
            f"  Stop reason: {self.stop_reason}",
            f"  Pool size  : {len(self.paper_pool)} included  "
            f"+ {len(self.borderline)} borderline",
            "",
            "  Per-iteration breakdown:",
            f"  {'iter':>4}  {'queries':>7}  {'cands':>6}  "
            f"{'h-drop':>6}  {'scored':>6}  {'new':>5}  {'pool':>5}  {'sec':>5}",
        ]
        for s in self.iterations:
            lines.append(
                f"  {s.iteration:>4}  {s.queries_issued:>7}  {s.candidates_found:>6}  "
                f"{s.heuristic_drops:>6}  {s.llm_scored:>6}  {s.new_included:>5}  "
                f"{s.total_pool_size:>5}  {s.elapsed_sec:>5.1f}"
            )
        lines += [
            "",
            "  Top 10 included papers:",
        ]
        for p in self.paper_pool[:10]:
            year = f"({p.year})" if p.year else ""
            lines.append(
                f"    [{p.final_score:.2f}] {p.title[:62]:<62} "
                f"{year:>6}  {p.query_slot}"
            )

        slot_counts: dict[str, int] = {}
        for p in self.paper_pool:
            slot_counts[p.query_slot] = slot_counts.get(p.query_slot, 0) + 1
        lines += ["", "  Pool by slot:"]
        for slot, count in sorted(slot_counts.items(), key=lambda x: -x[1]):
            lines.append(f"    {slot:<25} {count}")

        return "\n".join(lines)


# ── Orchestrator ──────────────────────────────────────────────────────────────

class LiteratureSurveyOrchestrator:
    """
    Usage:
        orch = LiteratureSurveyOrchestrator(
            api_key="...", base_url="...", model="gpt-4o",
            s2_api_key="...",          # optional, raises S2 rate limit
        )
        result = orch.run(paper_text="...", paper_year=2022)
        print(result.pretty())
    """

    def __init__(
        self,
        api_key:    str,
        base_url:   str  = "https://ai-gateway.andrew.cmu.edu",
        model:      str  = "gpt-5-mini",
        s2_api_key: Optional[str] = None,
        # stop conditions
        max_iterations:      int = MAX_ITERATIONS,
        new_paper_threshold: int = NEW_PAPER_THRESHOLD,
        max_llm_calls:       int = MAX_LLM_CALLS,
        # retrieval tuning
        results_per_query:   int = 10,
        max_concurrent:      int = 5,
    ) -> None:
        self.max_iterations      = max_iterations
        self.new_paper_threshold = new_paper_threshold
        self.max_llm_calls       = max_llm_calls

        # Instantiate each component once — they are stateless across iterations
        self.extractor  = TopicExtractor( api_key=api_key, base_url=base_url, model=model)
        self.gen        = QueryGenerator( api_key=api_key, base_url=base_url, model=model)
        self.retriever  = RetrievalLayer( s2_api_key=s2_api_key,
                                          results_per_query=results_per_query,
                                          max_concurrent=max_concurrent)
        self.filter     = RelevanceFilter(api_key=api_key, base_url=base_url, model=model)

    # ── Public ────────────────────────────────────────────────────────────────

    def run(
        self,
        paper_text: str,
        paper_year: Optional[int] = None,
    ) -> SurveyResult:

        # ── Step 1: extract topic taxonomy (once) ─────────────────────────────
        print("\n[1/4] Extracting topic taxonomy…")
        
        # Truncate to first ~2,500 chars (approx 500 tokens) to prevent timeouts
        # This is usually enough to capture the Abstract + Introduction + Related Work
        truncated_text = paper_text[:2500]
        print(f"truncated_text: {truncated_text}")
        
        taxonomy: TopicTaxonomy = self.extractor.extract(truncated_text)
        taxonomy_dict = taxonomy.to_dict()
        print(taxonomy.pretty())

        paper_title = taxonomy.paper_title
        low_conf    = taxonomy.low_confidence_slots(threshold=0.6)

        # ── Shared state across iterations ────────────────────────────────────
        pool:       dict[str, ScoredPaper] = {}   # dedup_key → ScoredPaper
        borderline: dict[str, ScoredPaper] = {}
        all_issued: list[SearchQuery]      = []   # every query ever sent
        stats_log:  list[IterationStats]   = []
        llm_calls_used = 0
        stop_reason    = "max_iterations"

        # ── Step 2: iterative retrieve → filter loop ──────────────────────────
        for iteration in range(1, self.max_iterations + 1):
            t0 = time.time()
            print(f"\n{'─'*64}")
            print(f"[iteration {iteration}/{self.max_iterations}]")
            stats = IterationStats(iteration=iteration)

            # ── 2a: determine which slots need more coverage ──────────────────
            # Any slot that had < 3 included papers last iteration gets boosted
            slot_yield   = {p.query_slot for p in pool.values()}
            expand_slots = [
                slot for slot in low_conf
                if sum(1 for p in pool.values() if p.query_slot == slot) < 3
            ]

            # ── 2b: query generation ──────────────────────────────────────────
            print(f"\n[{iteration}/A] Generating queries…")
            batch: QueryBatch = self.gen.generate(
                taxonomy          = taxonomy_dict,
                iteration         = iteration,
                previously_issued = all_issued,
                low_confidence_slots = expand_slots or low_conf,
            )
            all_issued.extend(batch.queries)
            stats.queries_issued = len(batch.queries)
            print(batch.pretty())

            if not batch.queries:
                stop_reason = "no_new_queries"
                stats.stop_reason = stop_reason
                stats_log.append(stats)
                break

            # ── 2c: retrieval ─────────────────────────────────────────────────
            print(f"\n[{iteration}/B] Retrieving papers…")
            retrieval = self.retriever.retrieve_sync(batch)
            stats.candidates_found = len(retrieval.papers)
            print(retrieval.pretty())

            # ── 2d: relevance filter ──────────────────────────────────────────
            print(f"\n[{iteration}/C] Filtering…")

            # Budget check before firing LLM calls
            from relevance_filter import LLM_BATCH_SIZE
            batches_needed = (
                max(0, len(retrieval.papers) - len(retrieval.papers) // 2)  # rough estimate after heuristic
                // LLM_BATCH_SIZE + 1
            )
            if llm_calls_used + batches_needed > self.max_llm_calls:
                stop_reason = "budget_cap"
                stats.stop_reason = stop_reason
                stats_log.append(stats)
                print(f"  Budget cap reached ({llm_calls_used} calls used). Stopping.")
                break

            filter_result: FilterResult = self.filter.filter(
                papers     = retrieval.papers,
                taxonomy   = taxonomy_dict,
                paper_year = paper_year,
                iteration  = iteration,
            )
            stats.heuristic_drops = len(filter_result.dropped_heuristic)
            stats.llm_scored      = (len(filter_result.included)
                                     + len(filter_result.borderline)
                                     + len(filter_result.dropped_by_llm))

            # Track LLM batch calls used
            from math import ceil
            llm_calls_used += ceil(stats.llm_scored / LLM_BATCH_SIZE)

            # ── 2e: merge into pool (dedup) ────────────────────────────────────
            new_count = 0
            for paper in filter_result.included:
                if paper.dedup_key not in pool:
                    pool[paper.dedup_key] = paper
                    new_count += 1
                else:
                    # Keep the higher-scored version
                    if paper.final_score > pool[paper.dedup_key].final_score:
                        pool[paper.dedup_key] = paper

            for paper in filter_result.borderline:
                if paper.dedup_key not in pool and paper.dedup_key not in borderline:
                    borderline[paper.dedup_key] = paper

            stats.new_included   = new_count
            stats.total_pool_size = len(pool)
            stats.elapsed_sec    = round(time.time() - t0, 1)
            stats_log.append(stats)

            print(filter_result.pretty())
            print(f"\n  → {new_count} new papers added  |  pool: {len(pool)}  |  "
                  f"borderline: {len(borderline)}  |  LLM calls used: {llm_calls_used}")

            # ── 2f: convergence check ──────────────────────────────────────────
            if new_count < self.new_paper_threshold:
                stop_reason = f"converged (Δ={new_count} < threshold={self.new_paper_threshold})"
                stats.stop_reason = stop_reason
                print(f"\n  Convergence: {stop_reason}. Stopping.")
                break

        # ── Step 3: assemble final result ─────────────────────────────────────
        sorted_pool = sorted(pool.values(), key=lambda p: p.final_score, reverse=True)
        sorted_bl   = sorted(borderline.values(), key=lambda p: p.final_score, reverse=True)

        return SurveyResult(
            paper_title = paper_title,
            paper_year  = paper_year,
            taxonomy    = taxonomy_dict,
            paper_pool  = sorted_pool,
            borderline  = sorted_bl,
            iterations  = stats_log,
            stop_reason = stop_reason,
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Literature survey agent — runs the full pipeline end-to-end."
    )
    # LLM access
    parser.add_argument("--api-key",    required=True)
    parser.add_argument("--base-url",   default="https://ai-gateway.andrew.cmu.edu")
    parser.add_argument("--model",      default="gpt-5-mini")
    # Retrieval
    parser.add_argument("--s2-api-key", default=None,
                        help="Semantic Scholar API key (optional, raises rate limit)")
    parser.add_argument("--results-per-query", type=int, default=10)
    parser.add_argument("--concurrency",       type=int, default=5)
    # Paper input
    parser.add_argument("--pdf",        required=True,
                        help="Path to the paper PDF file under review")
    parser.add_argument("--paper-year", type=int, default=None,
                        help="Publication year of the paper (auto-detected from PDF "
                             "metadata if omitted)")
    # Stop conditions
    parser.add_argument("--max-iterations",      type=int, default=MAX_ITERATIONS)
    parser.add_argument("--new-paper-threshold", type=int, default=NEW_PAPER_THRESHOLD)
    parser.add_argument("--max-llm-calls",       type=int, default=MAX_LLM_CALLS)
    # Output
    parser.add_argument("--out-json",   default=None,
                        help="Write full SurveyResult JSON to this file")
    args = parser.parse_args()

    # ── Load PDF ──────────────────────────────────────────────────────────────
    print(f"Loading PDF: {args.pdf}")
    try:
        paper_text = load_pdf(args.pdf)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # ── Resolve paper year ────────────────────────────────────────────────────
    paper_year = args.paper_year
    if paper_year is None:
        meta = load_pdf_metadata(args.pdf)
        paper_year = meta.get("year")
        if paper_year:
            print(f"  Auto-detected paper year from PDF metadata: {paper_year}")
        else:
            print("  Warning: could not detect paper year from PDF metadata. "
                  "Pass --paper-year to enable the temporal heuristic filter.",
                  file=sys.stderr)

    orch = LiteratureSurveyOrchestrator(
        api_key    = args.api_key,
        base_url   = args.base_url,
        model      = args.model,
        s2_api_key = args.s2_api_key,
        max_iterations      = args.max_iterations,
        new_paper_threshold = args.new_paper_threshold,
        max_llm_calls       = args.max_llm_calls,
        results_per_query   = args.results_per_query,
        max_concurrent      = args.concurrency,
    )

    result = orch.run(paper_text=paper_text, paper_year=paper_year)

    print("\n" + result.pretty())

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nSurvey result written to {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())