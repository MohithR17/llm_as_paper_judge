"""
Retrieval Layer — Literature Survey Agent
==========================================
Takes a QueryBatch and retrieves candidate papers from multiple APIs in parallel.

Sources:
  - Semantic Scholar   richest metadata: citations, venue, fields of study
  - arXiv              preprints; best coverage for recent ML/AI work
  - OpenAlex           open-access, broad coverage across all fields

All three sources are queried concurrently per query.
Results are merged and deduplicated by DOI / arXiv ID / title hash.

Each returned PaperRecord carries full provenance so the relevance filter
and paper registry know exactly where and how each paper was found.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import argparse
import time
import urllib.parse
from dataclasses import dataclass, field, asdict
from typing import Optional

import httpx


# ── Data Models ───────────────────────────────────────────────────────────────

@dataclass
class PaperRecord:
    """
    Canonical record for a retrieved paper.
    Dedup key priority: doi > arxiv_id > title_hash
    """
    # Identity
    title: str
    dedup_key: str              # doi:<DOI> | arxiv:<ID> | title:<hash>
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None

    # Content for relevance scoring
    abstract: str = ""
    year: Optional[int] = None
    venue: str = ""
    authors: list[str] = field(default_factory=list)

    # Signals for heuristic pre-filter
    citation_count: int = 0

    # Provenance
    source_apis: list[str] = field(default_factory=list)   # may be found by multiple APIs
    query_text: str = ""
    query_slot: str = ""
    query_variant: str = ""
    iteration: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievalResult:
    """All papers retrieved for one QueryBatch, deduplicated."""
    iteration: int
    papers: list[PaperRecord] = field(default_factory=list)
    query_count: int = 0
    api_call_count: int = 0

    def to_dict(self) -> dict:
        return {
            "iteration":      self.iteration,
            "query_count":    self.query_count,
            "api_call_count": self.api_call_count,
            "paper_count":    len(self.papers),
            "papers":         [p.to_dict() for p in self.papers],
        }

    def pretty(self) -> str:
        lines = [
            f"Retrieval result — iteration {self.iteration}",
            f"  Queries issued : {self.query_count}",
            f"  API calls made : {self.api_call_count}",
            f"  Papers found   : {len(self.papers)} (after dedup)",
            "",
        ]
        by_slot: dict[str, list[PaperRecord]] = {}
        for p in self.papers:
            by_slot.setdefault(p.query_slot, []).append(p)
        for slot, slot_papers in sorted(by_slot.items()):
            lines.append(f"  [{slot}]  {len(slot_papers)} papers")
            for p in slot_papers[:5]:
                year = f" ({p.year})" if p.year else ""
                lines.append(f"    • {p.title[:72]}{year}  [{', '.join(p.source_apis)}]")
            if len(slot_papers) > 5:
                lines.append(f"    … and {len(slot_papers) - 5} more")
            lines.append("")
        return "\n".join(lines)


# ── API Clients ───────────────────────────────────────────────────────────────

class SemanticScholarClient:
    BASE = "https://api.semanticscholar.org/graph/v1"
    FIELDS = "title,abstract,year,venue,authors,citationCount,externalIds"

    def __init__(self, client: httpx.AsyncClient, api_key: Optional[str] = None) -> None:
        self.client = client
        self.headers = {"x-api-key": api_key} if api_key else {}

    async def search(self, query: str, limit: int = 10) -> list[dict]:
        url = f"{self.BASE}/paper/search"
        params = {"query": query, "limit": limit, "fields": self.FIELDS}
        try:
            r = await self.client.get(url, params=params, headers=self.headers, timeout=15.0)
            r.raise_for_status()
            return r.json().get("data", [])
        except Exception as exc:
            print(f"    [S2 error] {query!r}: {exc}")
            return []

    def to_record(self, raw: dict, query_text: str, query_slot: str,
                  query_variant: str, iteration: int) -> Optional[PaperRecord]:
        title = (raw.get("title") or "").strip()
        if not title:
            return None

        ext = raw.get("externalIds") or {}
        doi      = ext.get("DOI")
        arxiv_id = ext.get("ArXiv")
        dedup    = _make_dedup_key(doi, arxiv_id, title)

        authors = [a.get("name", "") for a in (raw.get("authors") or [])]
        return PaperRecord(
            title=title, dedup_key=dedup, doi=doi, arxiv_id=arxiv_id,
            abstract=raw.get("abstract") or "",
            year=raw.get("year"),
            venue=raw.get("venue") or "",
            authors=authors[:6],
            citation_count=raw.get("citationCount") or 0,
            source_apis=["semantic_scholar"],
            query_text=query_text, query_slot=query_slot,
            query_variant=query_variant, iteration=iteration,
        )


class ArxivClient:
    BASE = "https://export.arxiv.org/api/query"

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def search(self, query: str, limit: int = 10) -> list[dict]:
        params = {
            "search_query": f"all:{urllib.parse.quote(query)}",
            "start": 0,
            "max_results": limit,
        }
        try:
            r = await self.client.get(self.BASE, params=params, timeout=15.0)
            r.raise_for_status()
            return self._parse_atom(r.text)
        except Exception as exc:
            print(f"    [arXiv error] {query!r}: {exc}")
            return []

    @staticmethod
    def _parse_atom(xml: str) -> list[dict]:
        """Minimal Atom XML parser — avoids the xml stdlib for simplicity."""
        import re
        entries = []
        for entry in re.findall(r"<entry>(.*?)</entry>", xml, re.DOTALL):
            def tag(t: str) -> str:
                m = re.search(rf"<{t}[^>]*>(.*?)</{t}>", entry, re.DOTALL)
                return m.group(1).strip() if m else ""

            arxiv_url = tag("id")
            arxiv_id  = arxiv_url.split("/abs/")[-1].split("v")[0] if "/abs/" in arxiv_url else ""
            authors   = re.findall(r"<name>(.*?)</name>", entry)
            published = tag("published")
            year      = int(published[:4]) if published else None

            entries.append({
                "title":    tag("title").replace("\n", " "),
                "abstract": tag("summary").replace("\n", " "),
                "arxiv_id": arxiv_id,
                "year":     year,
                "authors":  authors[:6],
            })
        return entries

    def to_record(self, raw: dict, query_text: str, query_slot: str,
                  query_variant: str, iteration: int) -> Optional[PaperRecord]:
        title = (raw.get("title") or "").strip()
        if not title:
            return None
        arxiv_id = raw.get("arxiv_id") or ""
        dedup    = _make_dedup_key(None, arxiv_id or None, title)
        return PaperRecord(
            title=title, dedup_key=dedup, doi=None,
            arxiv_id=arxiv_id or None,
            abstract=raw.get("abstract") or "",
            year=raw.get("year"),
            venue="arXiv",
            authors=raw.get("authors") or [],
            citation_count=0,
            source_apis=["arxiv"],
            query_text=query_text, query_slot=query_slot,
            query_variant=query_variant, iteration=iteration,
        )


class OpenAlexClient:
    BASE = "https://api.openalex.org/works"

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def search(self, query: str, limit: int = 10) -> list[dict]:
        params = {
            "search":       query,
            "per-page":     limit,
            "select":       "title,abstract_inverted_index,publication_year,primary_location,authorships,cited_by_count,ids",
            "mailto":       "literature-survey-agent@example.com",  # polite pool
        }
        try:
            r = await self.client.get(self.BASE, params=params, timeout=15.0)
            r.raise_for_status()
            return r.json().get("results", [])
        except Exception as exc:
            print(f"    [OpenAlex error] {query!r}: {exc}")
            return []

    def to_record(self, raw: dict, query_text: str, query_slot: str,
                  query_variant: str, iteration: int) -> Optional[PaperRecord]:
        title = (raw.get("title") or "").strip()
        if not title:
            return None

        ids      = raw.get("ids") or {}
        doi      = (ids.get("doi") or "").replace("https://doi.org/", "") or None
        arxiv_id = None
        if ids.get("arxiv"):
            arxiv_id = ids["arxiv"].split("/abs/")[-1].split("v")[0]

        abstract = _reconstruct_abstract(raw.get("abstract_inverted_index"))

        loc      = raw.get("primary_location") or {}
        source   = loc.get("source") or {}
        venue    = source.get("display_name") or ""

        authors = [
            a.get("author", {}).get("display_name", "")
            for a in (raw.get("authorships") or [])
        ]

        dedup = _make_dedup_key(doi, arxiv_id, title)
        return PaperRecord(
            title=title, dedup_key=dedup, doi=doi, arxiv_id=arxiv_id,
            abstract=abstract,
            year=raw.get("publication_year"),
            venue=venue,
            authors=authors[:6],
            citation_count=raw.get("cited_by_count") or 0,
            source_apis=["openalex"],
            query_text=query_text, query_slot=query_slot,
            query_variant=query_variant, iteration=iteration,
        )


# ── Retrieval Layer ───────────────────────────────────────────────────────────

class RetrievalLayer:
    """
    Executes a QueryBatch against all configured APIs in parallel and returns
    a deduplicated RetrievalResult.

    Usage:
        layer  = RetrievalLayer(results_per_query=10)
        result = asyncio.run(layer.retrieve(batch))
        print(result.pretty())

        # Or from synchronous code:
        result = layer.retrieve_sync(batch)
    """

    def __init__(
        self,
        results_per_query: int = 10,
        s2_api_key: Optional[str] = None,
        max_concurrent: int = 5,       # cap parallel API calls to be polite
    ) -> None:
        self.results_per_query = results_per_query
        self.s2_api_key        = s2_api_key
        self.semaphore         = asyncio.Semaphore(max_concurrent)

    # ── Public ────────────────────────────────────────────────────────────────

    async def retrieve(self, batch: "QueryBatch") -> RetrievalResult:  # noqa: F821
        """Async entry point — call from async code or via retrieve_sync."""
        result = RetrievalResult(iteration=batch.iteration)
        registry: dict[str, PaperRecord] = {}   # dedup_key → record

        async with httpx.AsyncClient() as http:
            s2     = SemanticScholarClient(http, self.s2_api_key)
            arxiv  = ArxivClient(http)
            openalex = OpenAlexClient(http)

            tasks = [
                self._fetch_query(q, s2, arxiv, openalex)
                for q in batch.sorted_queries()
            ]
            all_batches = await asyncio.gather(*tasks)

        api_calls = 0
        for records, n_calls in all_batches:
            api_calls += n_calls
            result.query_count += 1
            for rec in records:
                if rec.dedup_key not in registry:
                    registry[rec.dedup_key] = rec
                else:
                    # Merge: add source_api if not already listed, keep richer abstract
                    existing = registry[rec.dedup_key]
                    for src in rec.source_apis:
                        if src not in existing.source_apis:
                            existing.source_apis.append(src)
                    if len(rec.abstract) > len(existing.abstract):
                        existing.abstract = rec.abstract
                    if rec.citation_count > existing.citation_count:
                        existing.citation_count = rec.citation_count

        result.papers = list(registry.values())
        result.api_call_count = api_calls
        return result

    def retrieve_sync(self, batch: "QueryBatch") -> RetrievalResult:  # noqa: F821
        """Synchronous wrapper — use when not already inside an event loop."""
        return asyncio.run(self.retrieve(batch))

    # ── Private ───────────────────────────────────────────────────────────────

    async def _fetch_query(
        self,
        query: "SearchQuery",  # noqa: F821
        s2: SemanticScholarClient,
        arxiv: ArxivClient,
        openalex: OpenAlexClient,
    ) -> tuple[list[PaperRecord], int]:
        """Fire all three APIs concurrently for one query under the semaphore."""
        async with self.semaphore:
            print(f"  → [{query.slot}/{query.variant}] {query.text!r}")
            raw_s2, raw_ax, raw_oa = await asyncio.gather(
                s2.search(query.text,       self.results_per_query),
                arxiv.search(query.text,    self.results_per_query),
                openalex.search(query.text, self.results_per_query),
            )

        records: list[PaperRecord] = []
        kwargs = dict(
            query_text=query.text, query_slot=query.slot,
            query_variant=query.variant, iteration=query.iteration,
        )
        for raw in raw_s2:
            r = s2.to_record(raw, **kwargs)
            if r:
                records.append(r)
        for raw in raw_ax:
            r = arxiv.to_record(raw, **kwargs)
            if r:
                records.append(r)
        for raw in raw_oa:
            r = openalex.to_record(raw, **kwargs)
            if r:
                records.append(r)

        return records, 3   # 3 API calls per query


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_dedup_key(
    doi: Optional[str],
    arxiv_id: Optional[str],
    title: str,
) -> str:
    if doi:
        return f"doi:{doi.lower().strip()}"
    if arxiv_id:
        return f"arxiv:{arxiv_id.strip()}"
    h = hashlib.md5(title.lower().strip().encode()).hexdigest()[:12]
    return f"title:{h}"


def _reconstruct_abstract(inverted_index: Optional[dict]) -> str:
    """Reconstruct OpenAlex abstract from its inverted index format."""
    if not inverted_index:
        return ""
    try:
        max_pos = max(pos for positions in inverted_index.values() for pos in positions)
        words = [""] * (max_pos + 1)
        for word, positions in inverted_index.items():
            for pos in positions:
                words[pos] = word
        return " ".join(w for w in words if w)
    except Exception:
        return ""


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieval layer for the literature survey agent.")
    parser.add_argument("--query-batch",  required=True, help="Path to query batch JSON (from query_generator.py)")
    parser.add_argument("--s2-api-key",   default=None,  help="Optional Semantic Scholar API key (higher rate limit)")
    parser.add_argument("--results",      type=int, default=10, help="Results per query per API (default: 10)")
    parser.add_argument("--concurrency",  type=int, default=5,  help="Max parallel API calls (default: 5)")
    parser.add_argument("--out-json",     default=None,  help="Write retrieval result JSON to this file")
    args = parser.parse_args()

    with open(args.query_batch, "r", encoding="utf-8") as f:
        batch_dict = json.load(f)

    # Reconstruct QueryBatch from dict
    from query_generator import QueryBatch, SearchQuery
    batch = QueryBatch(iteration=batch_dict["iteration"])
    for q in batch_dict["queries"]:
        batch.queries.append(SearchQuery(**q))

    layer = RetrievalLayer(
        results_per_query=args.results,
        s2_api_key=args.s2_api_key,
        max_concurrent=args.concurrency,
    )

    print(f"Retrieving papers for iteration {batch.iteration} "
          f"({len(batch.queries)} queries × 3 APIs)…\n")

    start = time.time()
    result = layer.retrieve_sync(batch)
    elapsed = time.time() - start

    print(f"\nDone in {elapsed:.1f}s\n")
    print(result.pretty())

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"Retrieval result written to {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())