"""Structured teacher-summary schema shared by annotation and data building."""
from __future__ import annotations

import math
import re

SCHEMA_VERSION = "stage2_facets_v1"

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
        "focus": ("Capture only what appears or changes within these ten seconds; "
                  "keep it to a single short line."),
    },
    "block": {
        "prompt_kind": "independent current 60-second block",
        "facets": ["View", "Anatomy", "Finding", "Dynamic", "Measure"],
        "words": 80,
        "focus": ("Integrate this minute as a whole and give its overall visual "
                  "assessment; do not list the individual ten-second observations."),
    },
    "global": {
        "prompt_kind": "cumulative video prefix",
        "facets": ["Anatomy", "Finding", "Dynamic", "Measure", "Change"],
        "words": 90,
        "focus": ("Maintain a fixed-size visual summary of the whole video so far: "
                  "keep useful persistent evidence, merge repetition, rewrite rather "
                  "than append, and drop single-minute transients."),
    },
}

_CLAUSE = re.compile(r"^([A-Za-z]+)\s*:\s*(.+)$")


def parse_summary(text: str, kind: str, *, word_tolerance: float = 0.10) -> dict[str, str]:
    """Parse and strictly validate one labelled-clause teacher summary."""
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
        if index <= last_index:
            raise ValueError(f"facet {label} is out of order for {kind}")
        if not content.strip():
            raise ValueError(f"facet {label} is empty")
        parsed[label] = content.strip()
        last_index = index
    if not parsed:
        raise ValueError("summary has no valid facets")
    limit = math.ceil(SUMMARY_PLANS[kind]["words"] * (1 + word_tolerance))
    if len(value.split()) > limit:
        raise ValueError(f"summary exceeds the {limit}-word limit including tolerance")
    return parsed
