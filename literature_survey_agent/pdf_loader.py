"""
PDF Loader — Literature Survey Agent
======================================
Extracts clean plain text from a research paper PDF for the topic extractor.

Strategy (per the pdf-reading skill):
  1. pdfplumber  — primary extractor; handles multi-column layout well,
                   which is common in two-column conference papers.
  2. pypdf       — fallback if pdfplumber yields too little text
                   (some PDFs have encoding quirks pdfplumber chokes on).

Post-extraction cleaning:
  - Collapses hyphenated line-breaks ("meth-\nod" → "method")
  - Strips headers/footers: lines that repeat on every page (page numbers,
    journal names, conference names) are removed.
  - Normalises whitespace without destroying paragraph structure.

The loader also surfaces warnings when:
  - The PDF appears to be a scan (very little extractable text)
  - Extraction quality looks poor (average chars/page too low)
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional


# ── Constants ─────────────────────────────────────────────────────────────────

MIN_CHARS_PER_PAGE   = 200    # below this average → likely a scan
GOOD_CHARS_PER_PAGE  = 800    # above this → extraction looks healthy
MAX_PAGES            = 50     # cap for very long documents (proceedings, theses)
FOOTER_REPEAT_THRESH = 0.6    # a line appearing on >60% of pages → header/footer


# ── Public API ────────────────────────────────────────────────────────────────

def load_pdf(path: str | Path) -> str:
    """
    Extract and clean text from a research paper PDF.

    Returns a single string ready to pass to TopicExtractor.extract().
    Raises ValueError if the file cannot be read or appears to be a scan.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    # Try pdfplumber first, fall back to pypdf
    pages_text = _extract_pdfplumber(path)
    if _quality(pages_text) < MIN_CHARS_PER_PAGE:
        print(f"  [pdf] pdfplumber yield low ({_quality(pages_text):.0f} chars/page) "
              f"— trying pypdf fallback…", file=sys.stderr)
        pages_text_fb = _extract_pypdf(path)
        if _quality(pages_text_fb) > _quality(pages_text):
            pages_text = pages_text_fb

    avg = _quality(pages_text)
    if avg < MIN_CHARS_PER_PAGE:
        raise ValueError(
            f"PDF appears to be a scan or has unextractable text "
            f"({avg:.0f} chars/page average). "
            f"OCR pre-processing is required before running the survey agent."
        )
    if avg < GOOD_CHARS_PER_PAGE:
        print(f"  [pdf] warning: low extraction quality ({avg:.0f} chars/page). "
              f"Results may be incomplete.", file=sys.stderr)

    # Cap to MAX_PAGES to keep LLM context manageable
    if len(pages_text) > MAX_PAGES:
        print(f"  [pdf] truncating to first {MAX_PAGES} pages "
              f"(total: {len(pages_text)})", file=sys.stderr)
        pages_text = pages_text[:MAX_PAGES]

    clean = _clean(pages_text)
    print(f"  [pdf] extracted {len(pages_text)} pages · "
          f"{len(clean):,} chars · {avg:.0f} avg chars/page", file=sys.stderr)
    return clean


def load_pdf_metadata(path: str | Path) -> dict:
    """
    Extract document metadata (title, authors, year) from PDF properties.
    Useful for pre-filling paper_year without manual --paper-year flag.
    Returns a dict with keys: title, author, year (all Optional[str]).
    """
    path = Path(path)
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            meta = pdf.metadata or {}
    except Exception:
        try:
            from pypdf import PdfReader
            meta = PdfReader(str(path)).metadata or {}
        except Exception:
            return {"title": None, "author": None, "year": None}

    # PDF metadata keys vary: /Title, /Author, /CreationDate
    title  = meta.get("/Title")  or meta.get("Title")
    author = meta.get("/Author") or meta.get("Author")
    raw_date = meta.get("/CreationDate") or meta.get("CreationDate") or ""

    year = None
    m = re.search(r"(\d{4})", str(raw_date))
    if m:
        y = int(m.group(1))
        if 1900 < y < 2100:
            year = y

    return {"title": title, "author": author, "year": year}


# ── Extractors ────────────────────────────────────────────────────────────────

def _extract_pdfplumber(path: Path) -> list[str]:
    """Extract per-page text using pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        return []

    pages: list[str] = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages.append(text)
    except Exception as exc:
        print(f"  [pdf] pdfplumber error: {exc}", file=sys.stderr)
    return pages


def _extract_pypdf(path: Path) -> list[str]:
    """Extract per-page text using pypdf as fallback."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return []

    pages: list[str] = []
    try:
        reader = PdfReader(str(path))
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
    except Exception as exc:
        print(f"  [pdf] pypdf error: {exc}", file=sys.stderr)
    return pages


# ── Cleaning ──────────────────────────────────────────────────────────────────

def _clean(pages: list[str]) -> str:
    """
    Clean and join per-page text into a single string suitable for the LLM.
    """
    # 1. Identify repeated header/footer lines to strip
    line_page_count: Counter = Counter()
    all_page_lines: list[list[str]] = []
    for page_text in pages:
        lines = [l.strip() for l in page_text.splitlines()]
        all_page_lines.append(lines)
        # Only check short lines (headers/footers are rarely long)
        for line in lines:
            if line and len(line) < 120:
                line_page_count[line] += 1

    n_pages = max(len(pages), 1)
    repeated = set()
    if n_pages > 1:
        repeated = {
            line for line, count in line_page_count.items()
            if count / n_pages >= FOOTER_REPEAT_THRESH and len(line) < 120
        }

    # 2. Reconstruct text page by page, stripping repeated lines
    cleaned_pages: list[str] = []
    for lines in all_page_lines:
        kept = [l for l in lines if l not in repeated]
        cleaned_pages.append("\n".join(kept))

    full_text = "\n\n".join(cleaned_pages)

    # 3. Fix hyphenated line-breaks (common in two-column papers)
    #    "meth-\nod" → "method"
    full_text = re.sub(r"(\w)-\n(\w)", r"\1\2", full_text)

    # 4. Collapse runs of blank lines to a single blank line
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)

    # 5. Strip leading/trailing whitespace per line while preserving structure
    lines = [l.rstrip() for l in full_text.splitlines()]
    full_text = "\n".join(lines).strip()

    return full_text


# ── Quality metric ────────────────────────────────────────────────────────────

def _quality(pages: list[str]) -> float:
    """Average characters per page — primary quality signal."""
    if not pages:
        return 0.0
    return sum(len(p) for p in pages) / len(pages)


# ── CLI (standalone diagnostic) ───────────────────────────────────────────────

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Extract text from a research paper PDF.")
    parser.add_argument("pdf",           help="Path to PDF file")
    parser.add_argument("--out",         default=None, help="Write extracted text to file")
    parser.add_argument("--show-meta",   action="store_true", help="Print PDF metadata")
    parser.add_argument("--first-chars", type=int, default=2000,
                        help="Preview first N chars of extracted text (default: 2000)")
    args = parser.parse_args()

    if args.show_meta:
        meta = load_pdf_metadata(args.pdf)
        print("PDF metadata:")
        for k, v in meta.items():
            print(f"  {k}: {v}")
        print()

    text = load_pdf(args.pdf)

    print(f"Extracted text preview (first {args.first_chars} chars):\n")
    print(text[:args.first_chars])
    if len(text) > args.first_chars:
        print(f"\n… [{len(text) - args.first_chars:,} more chars]")

    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\nFull text written to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())