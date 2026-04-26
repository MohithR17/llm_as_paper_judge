"""Shared JSON extraction utility for the literature survey agent."""
from __future__ import annotations

import json
import re


def extract_json(text: str) -> str:
    """
    Extract the first valid JSON object or array from text that may contain
    markdown fences, prose, or other surrounding content.

    Raises ValueError if no valid JSON is found.
    """
    if not text or not text.strip():
        raise ValueError("Empty response")

    # 1. Strip markdown code fences (```json ... ``` or ``` ... ```)
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        candidate = fenced.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # 2. Try the whole string first (model followed instructions perfectly)
    stripped = text.strip()
    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass

    # 3. Find the first { ... } or [ ... ] block using brace matching
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        break  # malformed — don't try further

    raise ValueError(f"No valid JSON found in response: {text[:200]!r}")
