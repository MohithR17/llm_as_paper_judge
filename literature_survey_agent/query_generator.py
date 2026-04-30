"""
Search Query Generator — Literature Survey Agent
=================================================
Takes a TopicTaxonomy (as dict) and produces a prioritised batch of search
queries per slot, with lexical diversity across variants.

Query variants per slot:
  - broad keyword      broad topic terms, good for recall
  - narrow / specific  method name or exact technique, good for precision
  - survey framing     "survey of X" — finds overview papers
  - benchmark framing  "X benchmark / evaluation" — finds dataset papers

Iteration-awareness:
  Pass previously_issued_queries on subsequent calls so the generator
  skips near-duplicate queries (simple substring dedup — embeddings not
  required, keeps the component dependency-free).
"""

from __future__ import annotations

import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel


# ── Data Models ───────────────────────────────────────────────────────────────

@dataclass
class SearchQuery:
    """A single search query with metadata for the retrieval layer."""
    text: str                   # the actual query string to send to an API
    slot: str                   # which taxonomy slot this targets
    variant: str                # broad | narrow | survey | benchmark
    priority: float             # 0.0–1.0  (higher = run first)
    iteration: int = 1          # which retrieval iteration this belongs to


@dataclass
class QueryBatch:
    """All queries for one retrieval iteration, ordered by priority."""
    iteration: int
    queries: list[SearchQuery] = field(default_factory=list)

    def sorted_queries(self) -> list[SearchQuery]:
        return sorted(self.queries, key=lambda q: q.priority, reverse=True)

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "queries": [asdict(q) for q in self.sorted_queries()],
        }

    def pretty(self) -> str:
        lines = [f"Query batch — iteration {self.iteration} ({len(self.queries)} queries)\n"]
        current_slot = None
        for q in self.sorted_queries():
            if q.slot != current_slot:
                current_slot = q.slot
                lines.append(f"  [{q.slot}]")
            lines.append(f"    ({q.variant:<10}) p={q.priority:.2f}  {q.text!r}")
        return "\n".join(lines)


# ── Pydantic models for structured output ────────────────────────────────────

class QueryVariant(BaseModel):
    variant: str
    text: str

class QueryBatchModel(BaseModel):
    queries: list[QueryVariant]


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a research librarian generating academic search queries.
Given a topic slot and its terms from a research paper, produce diverse search queries
to find related prior work on academic search engines (Semantic Scholar, arXiv).
Respond with ONLY valid JSON — no markdown fences, no preamble."""

QUERY_GEN_PROMPT = """Generate search queries for the following topic slot extracted from a research paper.

Paper title: {paper_title}
Slot: {slot_name}
Terms: {terms}
Slot description: {slot_description}

Generate exactly 2 query variants as a JSON object with a "queries" array:
{{
  "queries": [
    {{
      "variant": "broad",
      "text": "<2-5 word query capturing the general topic>"
    }},
    {{
      "variant": "narrow",
      "text": "<specific query using exact method/dataset/metric names from the terms>"
    }}
  ]
}}

Rules:
- Each query must be meaningfully different from the other — no paraphrasing the same idea.
- Prefer short, precise queries (2–6 words). Academic search engines work better with fewer, more targeted terms.
- Use exact names from the terms list where possible (model names, dataset names, metric names).
- Do NOT include author names or paper titles.
- Do NOT add boolean operators (AND, OR) or field specifiers (title:, abstract:).
- The broad variant should be general enough to find tangentially related work.
- The narrow variant should be specific enough that only closely related papers appear.

Previously issued queries to AVOID duplicating:
{previous_queries}
"""

SLOT_DESCRIPTIONS = {
    "core_problem":       "The main task or problem the paper addresses",
    "proposed_method":    "The paper's own technique, model, or architectural contribution",
    "baselines":          "Competing methods the paper compares against",
    "datasets":           "Benchmark datasets or corpora used in experiments",
    "evaluation_metrics": "Metrics used to measure performance",
    "application_domain": "The broad research field or application area",
}


# ── Generator ─────────────────────────────────────────────────────────────────

class QueryGenerator:
    """
    Generates a QueryBatch from a TopicTaxonomy dict.

    Usage (first iteration):
        gen   = QueryGenerator(api_key=..., base_url=..., model=...)
        batch = gen.generate(taxonomy_dict, iteration=1)
        print(batch.pretty())

    Usage (subsequent iterations — skip already-issued queries):
        batch2 = gen.generate(taxonomy_dict, iteration=2,
                              previously_issued=batch.queries,
                              low_confidence_slots=["baselines", "datasets"])
    """

    MAX_RETRIES = 5
    # Priority weights: slots that drive the most search diversity get higher base priority.
    SLOT_BASE_PRIORITY = {
        "core_problem":       1.0,
        "proposed_method":    0.9,
        "application_domain": 0.8,
        "baselines":          0.7,
        "datasets":           0.6,
        "evaluation_metrics": 0.5,
    }
    # Low-confidence slots get a priority boost so the generator expands them first.
    LOW_CONF_BOOST = 0.2

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-5.4-nano",
        temperature: float = 0.3,   # slight creativity for query diversity
        max_tokens: int = 512,
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ── Public ────────────────────────────────────────────────────────────────

    def generate(
        self,
        taxonomy: dict,
        iteration: int = 1,
        previously_issued: Optional[list[SearchQuery]] = None,
        low_confidence_slots: Optional[list[str]] = None,
    ) -> QueryBatch:
        """
        Generate a QueryBatch for one retrieval iteration.

        Args:
            taxonomy:             Output of TopicTaxonomy.to_dict()
            iteration:            Current iteration number (1-indexed)
            previously_issued:    All queries issued in prior iterations (for dedup)
            low_confidence_slots: Slot names flagged as low-confidence by the extractor —
                                  these get priority-boosted and broader query variants
        """
        previously_issued = previously_issued or []
        low_confidence_slots = low_confidence_slots or []
        prior_texts = {q.text.lower() for q in previously_issued}

        batch = QueryBatch(iteration=iteration)

        # Collect active slots first so we can launch them in parallel
        active_slots = []
        for slot_name, slot_desc in SLOT_DESCRIPTIONS.items():
            slot_data = taxonomy.get(slot_name, {})
            terms = slot_data.get("terms", [])
            confidence = slot_data.get("confidence", 0.0)
            if not terms or (confidence < 0.3 and slot_name not in low_confidence_slots):
                continue
            active_slots.append((slot_name, slot_desc, terms))

        paper_title = taxonomy.get("paper_title", "")

        def fetch_slot(slot_name, slot_desc, terms):
            return slot_name, self._generate_slot_queries(
                paper_title=paper_title,
                slot_name=slot_name,
                slot_description=slot_desc,
                terms=terms,
                prior_texts=set(prior_texts),  # snapshot — read-only in threads
            )

        # Run all slot LLM calls in parallel
        slot_results: dict[str, list[dict]] = {}
        with ThreadPoolExecutor(max_workers=len(active_slots) or 1) as executor:
            futures = {executor.submit(fetch_slot, *s): s[0] for s in active_slots}
            for future in as_completed(futures):
                slot_name, raw_queries = future.result()
                slot_results[slot_name] = raw_queries

        # Merge in canonical slot order so priority/dedup is deterministic
        for slot_name, slot_desc, terms in active_slots:
            raw_queries = slot_results.get(slot_name, [])
            base_priority = self.SLOT_BASE_PRIORITY.get(slot_name, 0.5)
            if slot_name in low_confidence_slots:
                base_priority = min(1.0, base_priority + self.LOW_CONF_BOOST)

            for q in raw_queries:
                text = q.get("text", "").strip()
                variant = q.get("variant", "broad")

                if not text:
                    continue

                if self._is_duplicate(text, prior_texts):
                    continue

                variant_offset = {"broad": 0.05, "narrow": 0.0, "survey": -0.05, "benchmark": -0.10}
                priority = round(base_priority + variant_offset.get(variant, 0.0), 3)

                batch.queries.append(SearchQuery(
                    text=text,
                    slot=slot_name,
                    variant=variant,
                    priority=priority,
                    iteration=iteration,
                ))
                prior_texts.add(text.lower())

        return batch

    # ── Private ───────────────────────────────────────────────────────────────

    def _generate_slot_queries(
        self,
        paper_title: str,
        slot_name: str,
        slot_description: str,
        terms: list[str],
        prior_texts: set[str],
    ) -> list[dict]:
        prior_str = "\n".join(f"  - {t}" for t in sorted(prior_texts)) or "  (none yet)"
        prompt = QUERY_GEN_PROMPT.format(
            paper_title=paper_title,
            slot_name=slot_name,
            slot_description=slot_description,
            terms=", ".join(terms),
            previous_queries=prior_str,
        )
        parsed = self._call_with_retry(prompt)
        return [q.model_dump() for q in parsed.queries]

    def _call_with_retry(self, prompt: str) -> QueryBatchModel:
        last_error: Exception = RuntimeError("No attempts made")
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = self.client.responses.parse(
                    model=self.model,
                    input=[
                        {"role": "developer", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                        {"role": "user",       "content": [{"type": "input_text", "text": prompt}]},
                    ],
                    text_format=QueryBatchModel,
                )
                print(f"response received (attempt {attempt}): {len(response.output_parsed.queries)} queries")
                return response.output_parsed
            except Exception as exc:
                last_error = exc
                print(f"  [attempt {attempt}/{self.MAX_RETRIES}] error: {exc} — retrying…")
                time.sleep(2.0 * attempt)
        raise RuntimeError(
            f"Query generation failed after {self.MAX_RETRIES} attempts. Last error: {last_error}"
        )

    @staticmethod
    def _is_duplicate(text: str, prior_texts: set[str]) -> bool:
        """
        True if `text` is a near-duplicate of any previously issued query.
        Uses bidirectional substring containment as a cheap proxy for semantic similarity.
        """
        t = text.lower().strip()
        for prior in prior_texts:
            if t == prior:
                return True
            # One is a substring of the other and they're close in length
            if (t in prior or prior in t) and abs(len(t) - len(prior)) < 10:
                return True
        return False


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Search query generator for the literature survey agent.")
    parser.add_argument("--api-key",   required=True)
    parser.add_argument("--base-url",  default="https://ai-gateway.andrew.cmu.edu")
    parser.add_argument("--model",     default="gpt-5.4-nano")
    parser.add_argument("--taxonomy",  required=True, help="Path to taxonomy JSON (from topic_extractor.py --out-json)")
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--out-json",  default=None, help="Write query batch JSON to this file")
    args = parser.parse_args()

    with open(args.taxonomy, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    # Reconstruct low-confidence slots from the saved taxonomy
    low_conf = [
        slot for slot in SLOT_DESCRIPTIONS
        if taxonomy.get(slot, {}).get("confidence", 1.0) < 0.6
    ]

    gen = QueryGenerator(api_key=args.api_key, base_url=args.base_url, model=args.model)

    print(f"Generating queries (iteration {args.iteration})…\n")
    batch = gen.generate(taxonomy, iteration=args.iteration, low_confidence_slots=low_conf)

    print(batch.pretty())

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(batch.to_dict(), f, indent=2)
        print(f"\nQuery batch written to {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())