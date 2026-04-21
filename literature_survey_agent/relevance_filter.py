"""
Relevance Filter — Literature Survey Agent
==========================================
Two-stage filter that runs after the retrieval layer.

Stage 1 — Heuristic pre-filter (free, no LLM):
  • Drops papers published AFTER the paper under review (can't be prior work)
  • Drops papers below a citation floor, with year-aware thresholds:
      - recent papers (≤2 years old) are exempt from citation floor
      - older papers need a minimum citation count proportional to age
  • Drops papers with no abstract (nothing to score)

Stage 2 — LLM relevance scoring:
  • Batches surviving papers (up to BATCH_SIZE per call) to keep cost low
  • Scores each paper 0.0–1.0 across three dimensions:
      topical_relevance, methodological_fit, problem_proximity
  • Final score = weighted average of the three dimensions
  • Routes papers into: included (≥0.7) | borderline (0.4–0.7) | dropped (<0.4)

Output — FilterResult:
  • included    papers ready for the paper registry
  • borderline  papers to re-evaluate if a slot is underpopulated later
  • dropped     papers with drop reason (for debugging / audit)
  • yield stats per slot so the query generator knows where to expand
"""

from __future__ import annotations

import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime

from openai import OpenAI


# ── Constants ─────────────────────────────────────────────────────────────────

INCLUDE_THRESHOLD    = 0.7    # score ≥ this → included
BORDERLINE_THRESHOLD = 0.4    # score ≥ this → borderline; below → dropped

# Citation floor by paper age in years.
# Papers published within RECENT_YEARS are exempt (they haven't had time to accumulate).
RECENT_YEARS = 2
CITATION_FLOOR_PER_YEAR = 1   # e.g. a 5-year-old paper needs ≥ 5 citations
CITATION_FLOOR_MAX = 20       # cap so we don't exclude niche-but-relevant older work

LLM_BATCH_SIZE = 16           # abstracts per LLM call — balances cost vs latency


# ── Data Models ───────────────────────────────────────────────────────────────

@dataclass
class ScoredPaper:
    """A PaperRecord that has passed heuristic filtering and been LLM-scored."""
    # Core identity — mirrors PaperRecord fields we carry forward
    title: str
    dedup_key: str
    doi: Optional[str]
    arxiv_id: Optional[str]
    abstract: str
    year: Optional[int]
    venue: str
    authors: list[str]
    citation_count: int
    source_apis: list[str]
    query_text: str
    query_slot: str
    query_variant: str
    iteration: int

    # Relevance scores (set after LLM scoring)
    topical_relevance: float = 0.0      # does it address the same topic/task?
    methodological_fit: float = 0.0     # does it use a related approach?
    problem_proximity: float = 0.0      # is it solving the same or adjacent problem?
    final_score: float = 0.0            # weighted composite
    score_rationale: str = ""           # one-sentence LLM rationale

    # Routing
    bucket: str = ""                    # included | borderline | dropped

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HeuristicDrop:
    """A paper dropped by the heuristic pre-filter before LLM scoring."""
    title: str
    dedup_key: str
    year: Optional[int]
    citation_count: int
    query_slot: str
    reason: str         # too_recent | low_citations | no_abstract

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FilterResult:
    """Complete output of the relevance filter for one retrieval iteration."""
    iteration: int
    paper_year: Optional[int]           # year of the paper under review

    included:   list[ScoredPaper] = field(default_factory=list)
    borderline: list[ScoredPaper] = field(default_factory=list)
    dropped_by_llm:   list[ScoredPaper]   = field(default_factory=list)
    dropped_heuristic: list[HeuristicDrop] = field(default_factory=list)

    def slot_yield(self) -> dict[str, int]:
        """Number of included papers per query slot — used by query generator."""
        counts: dict[str, int] = {}
        for p in self.included:
            counts[p.query_slot] = counts.get(p.query_slot, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "iteration":          self.iteration,
            "paper_year":         self.paper_year,
            "included_count":     len(self.included),
            "borderline_count":   len(self.borderline),
            "dropped_llm_count":  len(self.dropped_by_llm),
            "dropped_heuristic_count": len(self.dropped_heuristic),
            "slot_yield":         self.slot_yield(),
            "included":           [p.to_dict() for p in self.included],
            "borderline":         [p.to_dict() for p in self.borderline],
            "dropped_by_llm":     [p.to_dict() for p in self.dropped_by_llm],
            "dropped_heuristic":  [p.to_dict() for p in self.dropped_heuristic],
        }

    def pretty(self) -> str:
        lines = [
            f"Filter result — iteration {self.iteration}  "
            f"(reviewing paper year: {self.paper_year or 'unknown'})",
            "",
            f"  Heuristic drops : {len(self.dropped_heuristic)}",
        ]
        reason_counts: dict[str, int] = {}
        for d in self.dropped_heuristic:
            reason_counts[d.reason] = reason_counts.get(d.reason, 0) + 1
        for reason, count in sorted(reason_counts.items()):
            lines.append(f"    {reason:<20} {count}")

        lines += [
            "",
            f"  LLM scored      : {len(self.included) + len(self.borderline) + len(self.dropped_by_llm)}",
            f"    included   (≥{INCLUDE_THRESHOLD})    : {len(self.included)}",
            f"    borderline ({BORDERLINE_THRESHOLD}–{INCLUDE_THRESHOLD}) : {len(self.borderline)}",
            f"    dropped    (<{BORDERLINE_THRESHOLD})    : {len(self.dropped_by_llm)}",
            "",
            "  Slot yield (included papers):",
        ]
        for slot, count in sorted(self.slot_yield().items()):
            lines.append(f"    {slot:<25} {count}")

        if self.included:
            lines += ["", "  Top included papers:"]
            for p in sorted(self.included, key=lambda x: x.final_score, reverse=True)[:8]:
                lines.append(
                    f"    [{p.final_score:.2f}] {p.title[:65]}"
                    f"  ({p.year})  {p.query_slot}/{p.query_variant}"
                )
        return "\n".join(lines)


# ── Prompts ───────────────────────────────────────────────────────────────────

SCORING_PROMPT = """You are a research assistant evaluating whether retrieved papers are relevant
prior work for a paper under review.

Paper under review:
  Title    : {paper_title}
  Abstract : {paper_abstract}

Score each of the following candidate papers on three dimensions (each 0.0–1.0):
  topical_relevance   — does the candidate address the same task or topic?
  methodological_fit  — does it use a related technique or approach?
  problem_proximity   — is it solving the same or a closely adjacent problem?

Return a JSON array with one object per candidate, in the SAME ORDER as the input list:
[
  {{
    "title": "<candidate title>",
    "topical_relevance": <float>,
    "methodological_fit": <float>,
    "problem_proximity": <float>,
    "rationale": "<one sentence explaining the score>"
  }},
  ...
]

Scoring guidance:
  0.9–1.0  Directly relevant — addresses same task, method, or problem.
  0.7–0.9  Closely related — significant overlap in topic or approach.
  0.5–0.7  Partially relevant — related domain but different focus.
  0.3–0.5  Tangentially related — same broad field, little direct overlap.
  0.0–0.3  Not relevant — different task, domain, or approach entirely.

Candidates:
{candidates}

Respond with the JSON array only. No markdown, no preamble."""


# ── Filter ────────────────────────────────────────────────────────────────────

class RelevanceFilter:
    """
    Two-stage relevance filter.

    Usage:
        f = RelevanceFilter(api_key=..., base_url=..., model=...)
        result = f.filter(
            papers       = retrieval_result.papers,   # list[PaperRecord]
            taxonomy     = taxonomy_dict,
            paper_year   = 2022,                      # year of the paper under review
            iteration    = 1,
        )
        print(result.pretty())
    """

    MAX_RETRIES = 3

    # Dimension weights for the final composite score
    WEIGHTS = {
        "topical_relevance":   0.45,
        "methodological_fit":  0.30,
        "problem_proximity":   0.25,
    }

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ── Public ────────────────────────────────────────────────────────────────

    def filter(
        self,
        papers: list,                   # list[PaperRecord] from retrieval_layer
        taxonomy: dict,                 # TopicTaxonomy.to_dict()
        paper_year: Optional[int],      # publication year of the paper under review
        iteration: int = 1,
    ) -> FilterResult:
        result = FilterResult(iteration=iteration, paper_year=paper_year)

        # ── Stage 1: heuristic pre-filter ─────────────────────────────────────
        survivors = []
        current_year = datetime.now().year

        for p in papers:
            drop_reason = self._heuristic_check(p, paper_year, current_year)
            if drop_reason:
                result.dropped_heuristic.append(HeuristicDrop(
                    title=p.title,
                    dedup_key=p.dedup_key,
                    year=p.year,
                    citation_count=p.citation_count,
                    query_slot=p.query_slot,
                    reason=drop_reason,
                ))
            else:
                survivors.append(p)

        print(
            f"  Heuristic filter: {len(papers)} in → "
            f"{len(survivors)} survive, {len(result.dropped_heuristic)} dropped"
        )

        # ── Stage 2: LLM relevance scoring in batches ─────────────────────────
        paper_title    = taxonomy.get("paper_title", "")
        paper_abstract = self._build_paper_abstract(taxonomy)

        batches = [
            survivors[i : i + LLM_BATCH_SIZE]
            for i in range(0, len(survivors), LLM_BATCH_SIZE)
        ]
        print(f"  LLM scoring: {len(survivors)} papers in {len(batches)} batch(es) (parallel)…")

        def score_one(args):
            batch_idx, batch = args
            print(f"    batch {batch_idx + 1}/{len(batches)} ({len(batch)} papers)")
            return batch_idx, batch, self._score_batch(paper_title, paper_abstract, batch)

        batch_outputs: list[tuple[int, list, list[dict]]] = []
        with ThreadPoolExecutor(max_workers=min(len(batches), 8)) as executor:
            futures = [executor.submit(score_one, (i, b)) for i, b in enumerate(batches)]
            for future in as_completed(futures):
                batch_outputs.append(future.result())

        # Restore original order before routing papers
        batch_outputs.sort(key=lambda x: x[0])
        for _, batch, scores in batch_outputs:
            for paper, score in zip(batch, scores):
                sp = self._make_scored_paper(paper, score)

                if sp.final_score >= INCLUDE_THRESHOLD:
                    sp.bucket = "included"
                    result.included.append(sp)
                elif sp.final_score >= BORDERLINE_THRESHOLD:
                    sp.bucket = "borderline"
                    result.borderline.append(sp)
                else:
                    sp.bucket = "dropped"
                    result.dropped_by_llm.append(sp)

        # Sort included by score descending
        result.included.sort(key=lambda x: x.final_score, reverse=True)
        result.borderline.sort(key=lambda x: x.final_score, reverse=True)

        return result

    # ── Private: heuristic ────────────────────────────────────────────────────

    def _heuristic_check(
        self,
        paper,                      # PaperRecord
        review_year: Optional[int],
        current_year: int,
    ) -> Optional[str]:
        """
        Returns a drop reason string if the paper should be discarded,
        or None if it passes.
        """
        # Must have an abstract — nothing to score otherwise
        if not paper.abstract or len(paper.abstract.strip()) < 30:
            return "no_abstract"

        # Must not be published after the paper under review
        if paper.year is not None and review_year is not None:
            if paper.year > review_year:
                return "published_after_review_paper"

        # Citation floor — year-aware
        if paper.year is not None:
            age = current_year - paper.year
            if age > RECENT_YEARS:
                floor = min(age * CITATION_FLOOR_PER_YEAR, CITATION_FLOOR_MAX)
                if paper.citation_count < floor:
                    return "low_citations"

        return None

    # ── Private: LLM scoring ──────────────────────────────────────────────────

    def _build_paper_abstract(self, taxonomy: dict) -> str:
        """
        Reconstruct a short context string from the taxonomy for the scoring prompt.
        This avoids sending the full paper text in every batch call.
        """
        parts = []
        for slot in ("core_problem", "proposed_method", "application_domain"):
            terms = taxonomy.get(slot, {}).get("terms", [])
            if terms:
                parts.append(f"{slot.replace('_', ' ')}: {', '.join(terms)}")
        return "; ".join(parts) if parts else "See paper title."

    def _score_batch(
        self,
        paper_title: str,
        paper_abstract: str,
        batch: list,                # list[PaperRecord]
    ) -> list[dict]:
        """
        Score a batch of PaperRecords. Returns a list of score dicts,
        one per paper, in the same order. Falls back to zero scores on failure.
        """
        candidates_text = "\n\n".join(
            f"[{i+1}] Title: {p.title}\n"
            f"     Year: {p.year or 'unknown'}\n"
            f"     Abstract: {p.abstract[:400].strip()}{'…' if len(p.abstract) > 400 else ''}"
            for i, p in enumerate(batch)
        )

        prompt = SCORING_PROMPT.format(
            paper_title=paper_title,
            paper_abstract=paper_abstract,
            candidates=candidates_text,
        )

        raw = self._call_with_retry(prompt)
        if not raw:
            return [self._zero_score(p) for p in batch]

        try:
            scores = json.loads(raw)
            if not isinstance(scores, list):
                raise ValueError("Expected a JSON array")
            # Pad with zeros if LLM returned fewer items than expected
            while len(scores) < len(batch):
                scores.append({})
            return scores[:len(batch)]
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"    [score parse error] {exc} — falling back to zero scores")
            return [self._zero_score(p) for p in batch]

    def _call_with_retry(self, prompt: str) -> str:
        last_error: Exception = RuntimeError("No attempts made")
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = (response.choices[0].message.content or "").strip()

                preview = raw[:120].replace("\n", "↵") if raw else "(empty)"
                print(f"      [attempt {attempt}] raw: {preview!r}")

                if not raw:
                    raise json.JSONDecodeError("Empty response", "", 0)

                # Strip markdown fences
                if "```" in raw:
                    parts = raw.split("```")
                    raw = parts[1] if len(parts) >= 3 else parts[-1]
                    if raw.lstrip().startswith("json"):
                        raw = raw.lstrip()[4:]
                    raw = raw.strip()

                json.loads(raw)
                return raw

            except json.JSONDecodeError as exc:
                last_error = exc
                print(f"      [attempt {attempt}/{self.MAX_RETRIES}] parse error: {exc} — retrying…")
                time.sleep(1.5 * attempt)
            except Exception as exc:
                last_error = exc
                print(f"      [attempt {attempt}/{self.MAX_RETRIES}] API error: {exc} — retrying…")
                time.sleep(2.0 * attempt)

        print(f"    [scoring failed after {self.MAX_RETRIES} attempts: {last_error}]")
        return ""

    # ── Private: helpers ──────────────────────────────────────────────────────

    def _make_scored_paper(self, paper, score: dict) -> ScoredPaper:
        """Combine a PaperRecord with its LLM score dict into a ScoredPaper."""
        tr  = max(0.0, min(1.0, float(score.get("topical_relevance",   0.0))))
        mf  = max(0.0, min(1.0, float(score.get("methodological_fit",  0.0))))
        pp  = max(0.0, min(1.0, float(score.get("problem_proximity",   0.0))))
        final = (
            tr * self.WEIGHTS["topical_relevance"] +
            mf * self.WEIGHTS["methodological_fit"] +
            pp * self.WEIGHTS["problem_proximity"]
        )
        return ScoredPaper(
            title=paper.title,
            dedup_key=paper.dedup_key,
            doi=paper.doi,
            arxiv_id=paper.arxiv_id,
            abstract=paper.abstract,
            year=paper.year,
            venue=paper.venue,
            authors=paper.authors,
            citation_count=paper.citation_count,
            source_apis=paper.source_apis,
            query_text=paper.query_text,
            query_slot=paper.query_slot,
            query_variant=paper.query_variant,
            iteration=paper.iteration,
            topical_relevance=round(tr, 3),
            methodological_fit=round(mf, 3),
            problem_proximity=round(pp, 3),
            final_score=round(final, 3),
            score_rationale=str(score.get("rationale", "")),
        )

    @staticmethod
    def _zero_score(paper) -> dict:
        return {
            "title": paper.title,
            "topical_relevance": 0.0,
            "methodological_fit": 0.0,
            "problem_proximity": 0.0,
            "rationale": "scoring failed",
        }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Relevance filter for the literature survey agent.")
    parser.add_argument("--api-key",       required=True)
    parser.add_argument("--base-url",      default="https://ai-gateway.andrew.cmu.edu")
    parser.add_argument("--model",         default="gpt-5-mini")
    parser.add_argument("--retrieval",     required=True,
                        help="Path to retrieval result JSON (from retrieval_layer.py)")
    parser.add_argument("--taxonomy",      required=True,
                        help="Path to taxonomy JSON (from topic_extractor.py)")
    parser.add_argument("--paper-year",    type=int, default=None,
                        help="Publication year of the paper under review (e.g. 2022). "
                             "Papers published after this year are dropped.")
    parser.add_argument("--out-json",      default=None,
                        help="Write filter result JSON to this file.")
    args = parser.parse_args()

    # Load inputs
    with open(args.retrieval, "r", encoding="utf-8") as f:
        retrieval_dict = json.load(f)
    with open(args.taxonomy, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    # Reconstruct PaperRecord-like objects from the retrieval JSON.
    # We use SimpleNamespace so we don't need to import retrieval_layer.
    from types import SimpleNamespace
    papers = [
        SimpleNamespace(**p)
        for p in retrieval_dict.get("papers", [])
    ]

    iteration = retrieval_dict.get("iteration", 1)

    # Infer paper_year from taxonomy title if not supplied, or leave None
    paper_year = args.paper_year

    f = RelevanceFilter(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )

    print(
        f"Running relevance filter — {len(papers)} candidates, "
        f"review paper year: {paper_year or 'not set'}\n"
    )

    result = f.filter(
        papers=papers,
        taxonomy=taxonomy,
        paper_year=paper_year,
        iteration=iteration,
    )

    print()
    print(result.pretty())

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=2)
        print(f"\nFilter result written to {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())