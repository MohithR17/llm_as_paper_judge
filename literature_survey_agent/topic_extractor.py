"""
Topic Extractor — Literature Survey Agent
==========================================
Extracts a structured topic taxonomy from a research paper.
Uses OpenAI-compatible client (same pattern as the project's other agents).
"""

from __future__ import annotations

import json
import time
import argparse
from dataclasses import dataclass, asdict
from typing import Optional

from openai import OpenAI


# ── Data Models ────────────────────────────────────────────────────────────────

@dataclass
class TopicEntry:
    """A single topic slot with its extracted terms and confidence."""
    terms: list[str]
    confidence: float          # 0.0 – 1.0
    notes: str = ""            # one-sentence rationale from LLM


@dataclass
class TopicTaxonomy:
    """
    Structured topic taxonomy extracted from a paper.

    Slots:
      core_problem       — the task / problem being solved
      proposed_method    — the paper's own technique / architecture
      baselines          — methods the paper explicitly compares against
      datasets           — benchmarks / corpora used in experiments
      evaluation_metrics — how results are measured (BLEU, F1, accuracy…)
      application_domain — broad research field (NLP, CV, robotics…)
    """
    paper_title: str
    core_problem: TopicEntry
    proposed_method: TopicEntry
    baselines: TopicEntry
    datasets: TopicEntry
    evaluation_metrics: TopicEntry
    application_domain: TopicEntry

    def low_confidence_slots(self, threshold: float = 0.6) -> list[str]:
        """Return slot names whose confidence is below threshold.
        Used by the query generator to flag slots needing broader search."""
        slots = [
            "core_problem", "proposed_method", "baselines",
            "datasets", "evaluation_metrics", "application_domain",
        ]
        return [s for s in slots if getattr(self, s).confidence < threshold]

    def to_dict(self) -> dict:
        return asdict(self)

    def pretty(self) -> str:
        """Human-readable summary."""
        lines = [f"Paper: {self.paper_title}", ""]
        slots = [
            ("Core problem",       self.core_problem),
            ("Proposed method",    self.proposed_method),
            ("Baselines",          self.baselines),
            ("Datasets",           self.datasets),
            ("Evaluation metrics", self.evaluation_metrics),
            ("Application domain", self.application_domain),
        ]
        for label, entry in slots:
            filled = int(entry.confidence * 10)
            bar = "█" * filled + "░" * (10 - filled)
            lines.append(f"  {label}")
            lines.append(f"    Terms:      {', '.join(entry.terms) if entry.terms else '—'}")
            lines.append(f"    Confidence: [{bar}] {entry.confidence:.2f}")
            if entry.notes:
                lines.append(f"    Notes:      {entry.notes}")
            lines.append("")

        low = self.low_confidence_slots()
        if low:
            lines.append(f"  ⚠  Low-confidence slots (need broader search): {', '.join(low)}")
        return "\n".join(lines)


# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a research assistant specialised in academic paper analysis.
Your task is to extract a structured topic taxonomy from a research paper.
You must respond with ONLY valid JSON — no markdown fences, no preamble, no trailing text."""

EXTRACTION_PROMPT = """Analyse the following research paper and extract a structured topic taxonomy.

Return a JSON object with exactly this structure:
{{
  "paper_title": "<string>",
  "core_problem": {{
    "terms": ["<term1>", "<term2>", ...],
    "confidence": <float 0.0-1.0>,
    "notes": "<one sentence rationale>"
  }},
  "proposed_method": {{
    "terms": ["<term1>", "<term2>", ...],
    "confidence": <float 0.0-1.0>,
    "notes": "<one sentence rationale>"
  }},
  "baselines": {{
    "terms": ["<term1>", "<term2>", ...],
    "confidence": <float 0.0-1.0>,
    "notes": "<one sentence rationale>"
  }},
  "datasets": {{
    "terms": ["<term1>", "<term2>", ...],
    "confidence": <float 0.0-1.0>,
    "notes": "<one sentence rationale>"
  }},
  "evaluation_metrics": {{
    "terms": ["<term1>", "<term2>", ...],
    "confidence": <float 0.0-1.0>,
    "notes": "<one sentence rationale>"
  }},
  "application_domain": {{
    "terms": ["<term1>", "<term2>", ...],
    "confidence": <float 0.0-1.0>,
    "notes": "<one sentence rationale>"
  }}
}}

Guidance per slot:
- core_problem:       The task the paper solves. E.g. "machine translation", "text summarisation".
- proposed_method:    The paper's own technique, model name, or architecture. Be specific.
- baselines:          Every method or model this paper explicitly compares against. List all — do not summarise.
- datasets:           Every benchmark dataset or corpus used in experiments. List all.
- evaluation_metrics: Metrics used to measure performance. E.g. BLEU, F1, accuracy, perplexity.
- application_domain: Broad research field. E.g. "natural language processing", "computer vision".

Confidence rubric:
- 0.9–1.0: Explicitly stated, unambiguous.
- 0.7–0.9: Clearly implied or directly inferable.
- 0.5–0.7: Partially mentioned or requires inference.
- 0.0–0.5: Absent or very unclear. Return empty terms list if below 0.3.

Be precise: prefer specific technical terms over vague descriptions.
List ALL baselines and datasets individually — never collapse them into a summary phrase.

PAPER TEXT:
-----------
{paper_text}
-----------"""


# ── Extractor ──────────────────────────────────────────────────────────────────

class TopicExtractor:
    """
    Extracts a TopicTaxonomy from paper text using an OpenAI-compatible API.

    Usage:
        extractor = TopicExtractor(api_key="...", base_url="...", model="...")
        taxonomy  = extractor.extract(paper_text)
        print(taxonomy.pretty())

        # Feed weak slots to the query generator
        weak_slots = taxonomy.low_confidence_slots(threshold=0.6)

        # Serialise to dict for downstream components
        data = taxonomy.to_dict()
    """

    MAX_RETRIES = 3

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://ai-gateway.andrew.cmu.edu",
        model: str = "gpt-5-mini",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ── Public ────────────────────────────────────────────────────────────────

    def extract(self, paper_text: str) -> TopicTaxonomy:
        """Run full-paper extraction. Retries up to MAX_RETRIES on parse failures."""
        prompt = EXTRACTION_PROMPT.format(paper_text=paper_text.strip())
        raw_json = self._call_with_retry(prompt)
        return self._parse(raw_json)

    # ── Private ───────────────────────────────────────────────────────────────

    def _call_with_retry(self, prompt: str) -> str:
        last_error: Exception = RuntimeError("No attempts made")

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                )
                text = (response.choices[0].message.content or "").strip()

                # Strip accidental markdown fences if the model ignores the instruction
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.strip()

                json.loads(text)   # validate before returning — raises JSONDecodeError if bad
                return text

            except json.JSONDecodeError as exc:
                last_error = exc
                print(f"  [attempt {attempt}/{self.MAX_RETRIES}] JSON parse error: {exc} — retrying…")
                time.sleep(1.5 * attempt)

            except Exception as exc:
                last_error = exc
                print(f"  [attempt {attempt}/{self.MAX_RETRIES}] API error: {exc} — retrying…")
                time.sleep(2.0 * attempt)

        raise RuntimeError(
            f"Topic extraction failed after {self.MAX_RETRIES} attempts. "
            f"Last error: {last_error}"
        )

    def _parse(self, raw_json: str) -> TopicTaxonomy:
        """Deserialise JSON → TopicTaxonomy with per-field defensive validation."""
        data = json.loads(raw_json)

        def parse_entry(slot_name: str) -> TopicEntry:
            slot = data.get(slot_name, {})
            if not isinstance(slot, dict):
                raise ValueError(f"Slot '{slot_name}' must be a dict, got: {type(slot)}")
            terms = slot.get("terms", [])
            if not isinstance(terms, list):
                terms = [str(terms)]           # coerce single string → list
            terms = [str(t) for t in terms]    # ensure all items are strings
            confidence = max(0.0, min(1.0, float(slot.get("confidence", 0.0))))
            notes = str(slot.get("notes", ""))
            return TopicEntry(terms=terms, confidence=confidence, notes=notes)

        return TopicTaxonomy(
            paper_title        = str(data.get("paper_title", "Unknown")),
            core_problem       = parse_entry("core_problem"),
            proposed_method    = parse_entry("proposed_method"),
            baselines          = parse_entry("baselines"),
            datasets           = parse_entry("datasets"),
            evaluation_metrics = parse_entry("evaluation_metrics"),
            application_domain = parse_entry("application_domain"),
        )


# ── CLI ────────────────────────────────────────────────────────────────────────

SAMPLE_ABSTRACT = """
Title: Attention Is All You Need

We propose a new simple network architecture, the Transformer, based solely on
attention mechanisms, dispensing with recurrence and convolutions entirely.
Experiments on two machine translation tasks show these models to be superior
in quality while being more parallelizable and requiring significantly less time
to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German
translation task, improving over the existing best results, including ensembles,
by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model
establishes a new single-model state-of-the-art BLEU score of 41.0 after
training for 3.5 days on 8 GPUs. We also show the Transformer generalizes well
to other tasks by applying it successfully to English constituency parsing with
both large and limited training data. Baselines compared include ByteNet,
ConvS2S, and GNMT+RL. Evaluation uses BLEU score on WMT 2014 benchmarks.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Topic extractor for the literature survey agent.")
    parser.add_argument("--api-key",  required=True,  help="OpenAI (or compatible) API key")
    parser.add_argument("--base-url", default="https://ai-gateway.andrew.cmu.edu",
                        help="API base URL (e.g. https://ai-gateway.andrew.cmu.edu/v1)")
    parser.add_argument("--model",    default="gpt-5-mini")
    parser.add_argument("--paper",    default=None,
                        help="Path to a plain-text paper file. Omit to run on the built-in sample.")
    parser.add_argument("--out-json", default=None,
                        help="Optional path to write the taxonomy JSON output.")
    args = parser.parse_args()

    if args.paper:
        with open(args.paper, "r", encoding="utf-8") as f:
            paper_text = f.read()
    else:
        print("No --paper provided — running on built-in sample abstract.\n")
        paper_text = SAMPLE_ABSTRACT

    extractor = TopicExtractor(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )

    print("Extracting topic taxonomy…\n")
    taxonomy = extractor.extract(paper_text)

    print(taxonomy.pretty())

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(taxonomy.to_dict(), f, indent=2)
        print(f"\nTaxonomy written to {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())