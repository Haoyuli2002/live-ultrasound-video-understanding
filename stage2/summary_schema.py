"""Versioned prose and legacy facet targets for Stage 2 annotation/data."""
from __future__ import annotations

import math
import re

SCHEMA_VERSION = "stage2_facets_v1"  # Legacy datasets remain readable.
PROSE_SCHEMA_VERSION = "stage2_prose_v2"
SUPPORTED_SCHEMAS = (SCHEMA_VERSION, PROSE_SCHEMA_VERSION)

PERSPECTIVES = {
    "View": "the scan plane or standard view that is visible",
    "Anatomy": "the anatomical structures and organs that are visible",
    "Probe": "probe position or movement, only when it explains a finding",
    "Finding": "static sonographic findings or signs that are visible",
    "Dynamic": "dynamic signs such as motion, flow, pulsation, or filling",
    "Measure": "on-screen measurements, calipers, labels, or machine settings",
    "Change": "visual change or comparison relative to earlier in the video",
}

SUMMARY_PLANS = {
    "local": {
        "prompt_kind": "current 10-second local",
        "facets": ["View", "Anatomy", "Probe", "Finding", "Dynamic"],
        "words": 40,
        "focus": ("Independently summarize relevant evidence in these ten seconds; "
                  "do not judge whether it is new relative to a previous window."),
    },
    "block": {
        "prompt_kind": "independent current 60-second block",
        "facets": ["View", "Anatomy", "Finding", "Dynamic", "Measure"],
        "words": 80,
        "focus": ("Integrate this minute, merge repeated states, and do not list "
                  "individual local summaries."),
    },
    "global": {
        "prompt_kind": "cumulative video prefix",
        "facets": ["Anatomy", "Finding", "Dynamic", "Measure", "Change"],
        "words": 90,
        "focus": ("Summarize the entire video prefix so far, retaining key evidence "
                  "and persistent states without expanding unchanged information."),
    },
}

_CLAUSE = re.compile(r"^([A-Za-z]+)\s*:\s*(.+)$")


def parse_summary(text: str, kind: str, *, word_tolerance: float = 0.10) -> dict[str, str]:
    """Parse and strictly validate one labelled-clause teacher summary."""
    return _parse_summary(text, kind, word_tolerance=word_tolerance, require_order=True)


def _parse_summary(text: str, kind: str, *, word_tolerance: float,
                   require_order: bool) -> dict[str, str]:
    if kind not in SUMMARY_PLANS:
        raise ValueError(f"Unknown summary kind: {kind}")
    value = str(text or "").strip()
    if not value:
        raise ValueError("summary is empty")
    clauses = [part.strip() for part in re.split(r"[;\n]+", value) if part.strip()]
    parsed: dict[str, str] = {}
    allowed = SUMMARY_PLANS[kind]["facets"]
    last_index = -1
    for clause in clauses:
        match = _CLAUSE.fullmatch(clause)
        if match is None:
            raise ValueError(f"unlabelled clause: {clause!r}")
        label, content = match.groups()
        if label not in PERSPECTIVES:
            raise ValueError(f"unknown facet: {label}")
        if label not in allowed:
            raise ValueError(f"facet {label} is not allowed for {kind}")
        if label in parsed:
            raise ValueError(f"duplicate facet: {label}")
        index = allowed.index(label)
        if require_order and index <= last_index:
            raise ValueError(f"facet {label} is out of order for {kind}")
        if not content.strip():
            raise ValueError(f"facet {label} is empty")
        if re.search(r"\b(?:" + "|".join(PERSPECTIVES) + r")\s*:", content):
            raise ValueError("embedded facet label; clauses must be separated by semicolons")
        parsed[label] = content.strip()
        last_index = index
    if not parsed:
        raise ValueError("summary has no valid facets")
    limit = math.ceil(SUMMARY_PLANS[kind]["words"] * (1 + word_tolerance))
    if len(value.split()) > limit:
        raise ValueError(f"summary exceeds the {limit}-word limit including tolerance")
    return parsed


def normalize_summary(text: str, kind: str) -> tuple[str, dict[str, str]]:
    """Canonicalize teacher clause order without dropping or rewriting evidence.

    Stored labels still pass the strict parser. Only the annotation boundary
    accepts a permutation; duplicates, illegal labels, empty clauses and word
    limits remain errors. Already valid output is preserved verbatim.
    """
    value = str(text or "").strip()
    facets = _parse_summary(value, kind, word_tolerance=0.10, require_order=False)
    ordered = {label: facets[label] for label in SUMMARY_PLANS[kind]["facets"]
               if label in facets}
    if list(facets) != list(ordered):
        value = "; ".join(f"{label}: {content}" for label, content in ordered.items())
    return value, parse_summary(value, kind)


def validate_prose(text: str, kind: str) -> str:
    """Validate output shape, not the clinical truth of the teacher's prose."""
    if kind not in SUMMARY_PLANS:
        raise ValueError(f"Unknown summary kind: {kind}")
    value = str(text or "").strip()
    if not value:
        raise ValueError("summary is empty")
    if re.search(r"\b(?:" + "|".join(PERSPECTIVES) + r")\s*:", value, re.I):
        raise ValueError("write connected prose without perspective labels")
    if value.startswith(("{", "[", "```", "#")) or re.search(r"(?m)^\s*(?:[-*•]|\d+[.)])\s", value):
        raise ValueError("write one prose paragraph, not JSON, headings or a list")
    if re.search(r"<\/?think>", value, re.I):
        raise ValueError("reasoning tags are not allowed")
    if re.search(r"\n\s*\n", value):
        raise ValueError("write a single paragraph")
    words = value.split()
    if len(words) < 4:
        raise ValueError("summary is too short to convey a visual observation")
    return value


def validate_target(text: str, kind: str, schema: str | None = None) -> dict[str, str]:
    """Dispatch by explicit schema; never guess a format from the text."""
    schema = schema or SCHEMA_VERSION
    if schema == PROSE_SCHEMA_VERSION:
        validate_prose(text, kind)
        return {}
    if schema == SCHEMA_VERSION:
        return parse_summary(text, kind)
    raise ValueError(f"unsupported schema_version={schema!r}")
