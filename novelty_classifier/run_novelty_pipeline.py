#!/usr/bin/env python3
"""
Prototype novelty classification pipeline.

Pipeline:
1. Extract text from a PDF.
2. Ask an LLM to extract claim sentences.
3. Embed each claim and each literature abstract.
4. Compute novelty(claim) = 1 - max cosine_similarity(claim, abstract).
5. Flag claims above a threshold as novel.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
load_dotenv()
DEFAULT_BASE_URL = "https://ai-gateway.andrew.cmu.edu"

api_key=os.getenv("OPENAI_API_KEY", "")
@dataclass
class LiteratureEntry:
    title: str
    abstract: str
    raw: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract claims from a PDF and score novelty against literature abstracts.")
    parser.add_argument("--pdf", required=True, help="Path to the input PDF.")
    parser.add_argument("--survey-json", required=True, help="Path to the literature survey JSON.")
    # parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"), help="OpenAI-compatible API key.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL.")
    parser.add_argument("--llm-model", required=True, help="Model used for claim extraction.")
    parser.add_argument(
        "--scoring-mode",
        choices=["llm_judge", "embeddings"],
        default="llm_judge",
        help="Novelty scoring backend. Use llm_judge if embeddings are unavailable on your gateway.",
    )
    parser.add_argument("--embedding-model", help="Embedding model used as the dense encoder when scoring-mode=embeddings.")
    parser.add_argument("--novelty-threshold", type=float, default=0.35, help="Claims above this novelty score are flagged as novel.")
    parser.add_argument("--max-pdf-chars", type=int, default=24000, help="Max PDF text sent to the claim extraction LLM.")
    parser.add_argument("--top-k-matches", type=int, default=3, help="Number of nearest abstracts to report per claim.")
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=5,
        help="Number of candidate abstracts shortlisted by lexical retrieval before LLM judging.",
    )
    parser.add_argument("--output-json", help="Optional output path for the full report.")
    return parser.parse_args()


def require_dependency(module_name: str, install_hint: str) -> Any:
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise SystemExit(f"Missing dependency '{module_name}'. Install it with: {install_hint}") from exc


def build_client(api_key: str, base_url: str):
    if not api_key:
        raise SystemExit("Missing API key. Pass --api-key or set OPENAI_API_KEY.")
    openai_module = require_dependency("openai", "pip install openai")
    return openai_module.OpenAI(api_key=api_key, base_url=base_url)


def extract_pdf_text(pdf_path: str) -> str:
    pypdf_module = require_dependency("pypdf", "pip install pypdf")
    reader = pypdf_module.PdfReader(pdf_path)
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    text = "\n".join(pages).strip()
    if not text:
        raise SystemExit(f"No text could be extracted from PDF: {pdf_path}")
    return text


def load_literature_entries(path: str) -> list[LiteratureEntry]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "paper_pool" in payload and isinstance(payload["paper_pool"], list):
            items = payload["paper_pool"]
        elif "papers" in payload and isinstance(payload["papers"], list):
            items = payload["papers"]
        elif "data" in payload and isinstance(payload["data"], list):
            items = payload["data"]
        else:
            items = [payload]
    elif isinstance(payload, list):
        items = payload
    else:
        raise SystemExit("Survey JSON must be a list or an object containing a paper list.")

    entries: list[LiteratureEntry] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = first_present(item, ["paper_title", "title", "name"]) or "Untitled"
        abstract = first_present(item, ["abstract", "paper_abstract", "summary", "description"])
        if not abstract:
            continue
        entries.append(LiteratureEntry(title=title.strip(), abstract=abstract.strip(), raw=item))

    if not entries:
        raise SystemExit("No usable abstracts found in the survey JSON.")
    return entries


def first_present(item: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def extract_claims_with_llm(client: Any, model: str, pdf_text: str, max_chars: int) -> list[str]:
    prompt_text = pdf_text[:max_chars]
    system_prompt = (
        "You extract research claims from academic papers. "
        "Return only a JSON object with key 'claims' whose value is a list of claim sentences. "
        "IMPORTANT: copy each claim VERBATIM (word-for-word) from the source text — do NOT paraphrase, summarize, or rewrite."
    )
    user_prompt = (
        "Extract the paper's main claim sentences from the text below.\n"
        "Rules:\n"
        "- Return 5 to 15 claims when possible.\n"
        "- Copy each sentence EXACTLY as it appears in the text, with no changes.\n"
        "- Prefer contribution statements, findings, method claims, or performance claims.\n"
        "- Do not include citations or section headers.\n"
        "- Do not merge or split sentences.\n\n"
        f"PDF text:\n{prompt_text}"
    )

    request_kwargs = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    try:
        completion = client.chat.completions.create(
            response_format={"type": "json_object"},
            **request_kwargs,
        )
    except Exception:
        completion = client.chat.completions.create(**request_kwargs)

    content = completion.choices[0].message.content or "{}"
    try:
        data = parse_json_object(content)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Claim extractor returned invalid JSON: {content}") from exc

    claims = data.get("claims", [])
    if not isinstance(claims, list):
        raise SystemExit("Claim extractor response is missing a valid 'claims' list.")
    cleaned = [normalize_sentence(claim) for claim in claims if isinstance(claim, str) and claim.strip()]
    cleaned = dedupe_preserve_order(cleaned)
    if not cleaned:
        raise SystemExit("No claims were extracted from the PDF.")
    return cleaned


def normalize_sentence(text: str) -> str:
    cleaned = " ".join(text.split()).strip()
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return json.loads(stripped)


def embed_texts(client: Any, model: str, texts: list[str]) -> list[list[float]]:
    embeddings: list[list[float]] = []
    batch_size = 64
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        ordered = sorted(response.data, key=lambda row: row.index)
        embeddings.extend([list(row.embedding) for row in ordered])
    return embeddings


def tokenize_for_retrieval(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def lexical_retrieval(claim: str, literature: list[LiteratureEntry], top_k: int) -> list[tuple[LiteratureEntry, float]]:
    claim_tokens = tokenize_for_retrieval(claim)
    scored: list[tuple[LiteratureEntry, float]] = []
    for entry in literature:
        abstract_tokens = tokenize_for_retrieval(entry.abstract)
        if not claim_tokens or not abstract_tokens:
            score = 0.0
        else:
            overlap = len(claim_tokens & abstract_tokens)
            score = overlap / math.sqrt(len(claim_tokens) * len(abstract_tokens))
        scored.append((entry, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]


def llm_groundedness_score(client: Any, model: str, claim: str, abstract: str) -> dict[str, Any]:
    system_prompt = (
        "You compare a research claim against a prior-work abstract. "
        "Return only JSON with keys: groundedness_score, label, rationale. "
        "groundedness_score must be a number from 0 to 1, where 1 means the abstract clearly contains the same core proposition "
        "and 0 means the abstract does not support that proposition."
    )
    user_prompt = (
        f"Claim:\n{claim}\n\n"
        f"Prior-work abstract:\n{abstract}\n\n"
        "Judge whether the claim's core proposition is already present in the abstract.\n"
        "Use labels: grounded, partially_grounded, or not_grounded.\n"
        "Keep the rationale brief."
    )
    request_kwargs = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    try:
        completion = client.chat.completions.create(
            response_format={"type": "json_object"},
            **request_kwargs,
        )
    except Exception:
        completion = client.chat.completions.create(**request_kwargs)
    content = completion.choices[0].message.content or "{}"
    data = parse_json_object(content)
    score = data.get("groundedness_score", 0.0)
    try:
        groundedness_score = min(1.0, max(0.0, float(score)))
    except (TypeError, ValueError):
        groundedness_score = 0.0
    return {
        "groundedness_score": groundedness_score,
        "label": str(data.get("label", "not_grounded")),
        "rationale": str(data.get("rationale", "")),
    }


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def score_claims(
    claims: list[str],
    claim_embeddings: list[list[float]],
    literature: list[LiteratureEntry],
    abstract_embeddings: list[list[float]],
    novelty_threshold: float,
    top_k_matches: int,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []

    for claim, claim_embedding in zip(claims, claim_embeddings):
        matches: list[dict[str, Any]] = []
        best_similarity = -1.0
        best_entry: LiteratureEntry | None = None

        for entry, abstract_embedding in zip(literature, abstract_embeddings):
            similarity = cosine_similarity(claim_embedding, abstract_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_entry = entry
            matches.append(
                {
                    "title": entry.title,
                    "similarity": round(similarity, 4),
                    "abstract": entry.abstract,
                }
            )

        matches.sort(key=lambda item: item["similarity"], reverse=True)
        novelty = 1 - max(best_similarity, 0.0)
        results.append(
            {
                "claim": claim,
                "novelty_score": round(novelty, 4),
                "is_novel": novelty > novelty_threshold,
                "best_match_title": best_entry.title if best_entry else None,
                "best_match_similarity": round(max(best_similarity, 0.0), 4),
                "top_matches": matches[:top_k_matches],
            }
        )

    return {
        "threshold": novelty_threshold,
        "num_claims": len(results),
        "num_literature_entries": len(literature),
        "claims": results,
    }


def _batch_groundedness_score(
    client: Any,
    model: str,
    claim: str,
    candidates: list[tuple["LiteratureEntry", float]],
) -> list[dict[str, Any]]:
    """
    Score one claim against all candidates in a single LLM call.
    Returns a list of judgment dicts in the same order as candidates.
    """
    abstracts_block = "\n\n".join(
        f"[{i + 1}] {entry.abstract[:600]}"
        for i, (entry, _) in enumerate(candidates)
    )
    system_prompt = (
        "You compare a research claim against a numbered list of prior-work abstracts. "
        'Return ONLY a JSON object: {"scores": [{"groundedness_score": <0-1>, '
        '"label": "grounded|partially_grounded|not_grounded", "rationale": "<brief>"}]}. '
        "One entry per abstract, in the same order. "
        "groundedness_score=1 means the abstract clearly supports the claim; 0 means it does not."
    )
    user_prompt = (
        f"Claim:\n{claim}\n\n"
        f"Abstracts:\n{abstracts_block}\n\n"
        "Rate the groundedness of the claim in each abstract."
    )
    request_kwargs = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    try:
        completion = client.chat.completions.create(
            response_format={"type": "json_object"}, **request_kwargs
        )
    except Exception:
        completion = client.chat.completions.create(**request_kwargs)

    content = completion.choices[0].message.content or "{}"
    try:
        data = parse_json_object(content)
        raw_scores = data.get("scores", [])
    except json.JSONDecodeError:
        raw_scores = []

    judgments: list[dict[str, Any]] = []
    for i in range(len(candidates)):
        item = raw_scores[i] if i < len(raw_scores) else {}
        try:
            gs = min(1.0, max(0.0, float(item.get("groundedness_score", 0.0))))
        except (TypeError, ValueError):
            gs = 0.0
        judgments.append({
            "groundedness_score": gs,
            "label": str(item.get("label", "not_grounded")),
            "rationale": str(item.get("rationale", "")),
        })
    return judgments


def _score_single_claim(
    client: Any,
    model: str,
    claim: str,
    literature: list[LiteratureEntry],
    novelty_threshold: float,
    top_k_matches: int,
    retrieval_k: int,
) -> dict[str, Any]:
    """Score one claim: one batched LLM call covers all retrieval_k candidates."""
    candidates = lexical_retrieval(claim, literature, top_k=retrieval_k)
    if not candidates:
        return {
            "claim": claim, "novelty_score": 1.0, "is_novel": True,
            "best_match_title": None, "best_match_groundedness": 0.0, "top_matches": [],
        }

    judgments = _batch_groundedness_score(client, model, claim, candidates)

    matches: list[dict[str, Any]] = []
    best_score = 0.0
    best_entry: LiteratureEntry | None = None

    for (entry, retrieval_score), judgment in zip(candidates, judgments):
        gs = judgment["groundedness_score"]
        if gs > best_score:
            best_score = gs
            best_entry = entry
        matches.append({
            "title": entry.title,
            "retrieval_score": round(retrieval_score, 4),
            "groundedness_score": round(gs, 4),
            "label": judgment["label"],
            "rationale": judgment["rationale"],
            "abstract": entry.abstract,
        })

    matches.sort(key=lambda m: m["groundedness_score"], reverse=True)
    novelty = 1 - best_score
    return {
        "claim": claim,
        "novelty_score": round(novelty, 4),
        "is_novel": novelty > novelty_threshold,
        "best_match_title": best_entry.title if best_entry else None,
        "best_match_groundedness": round(best_score, 4),
        "top_matches": matches[:top_k_matches],
    }


def score_claims_with_llm(
    client: Any,
    model: str,
    claims: list[str],
    literature: list[LiteratureEntry],
    novelty_threshold: float,
    top_k_matches: int,
    retrieval_k: int,
) -> dict[str, Any]:
    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

    # Score all claims in parallel; each claim also parallelises its candidate judgments
    indexed: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=len(claims)) as ex:
        futs = {
            ex.submit(
                _score_single_claim, client, model, claim, literature,
                novelty_threshold, top_k_matches, retrieval_k
            ): i
            for i, claim in enumerate(claims)
        }
        for fut in _as_completed(futs):
            indexed[futs[fut]] = fut.result()

    results = [indexed[i] for i in range(len(claims))]
    return {
        "threshold": novelty_threshold,
        "scoring_mode": "llm_judge",
        "num_claims": len(results),
        "num_literature_entries": len(literature),
        "claims": results,
    }


def main() -> None:
    args = parse_args()
    client = build_client(api_key=api_key, base_url=DEFAULT_BASE_URL)

    literature = load_literature_entries(args.survey_json)
    pdf_text = extract_pdf_text(args.pdf)
    claims = extract_claims_with_llm(client, model=args.llm_model, pdf_text=pdf_text, max_chars=args.max_pdf_chars)
    if args.scoring_mode == "embeddings":
        if not args.embedding_model:
            raise SystemExit("--embedding-model is required when --scoring-mode=embeddings")
        claim_embeddings = embed_texts(client, model=args.embedding_model, texts=claims)
        abstract_embeddings = embed_texts(client, model=args.embedding_model, texts=[entry.abstract for entry in literature])
        report = score_claims(
            claims=claims,
            claim_embeddings=claim_embeddings,
            literature=literature,
            abstract_embeddings=abstract_embeddings,
            novelty_threshold=args.novelty_threshold,
            top_k_matches=args.top_k_matches,
        )
        report["scoring_mode"] = "embeddings"
    else:
        report = score_claims_with_llm(
            client=client,
            model=args.llm_model,
            claims=claims,
            literature=literature,
            novelty_threshold=args.novelty_threshold,
            top_k_matches=args.top_k_matches,
            retrieval_k=args.retrieval_k,
        )

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")

    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
