from __future__ import annotations

import re
from typing import Iterable

from .config import (
    CATEGORY_KEYWORDS_EN,
    CATEGORY_KEYWORDS_ZH,
    NEGATIVE_NON_VISUAL_HINTS,
    ULTRASOUND_KEYWORDS_EN,
    ULTRASOUND_KEYWORDS_ZH,
    VISUAL_HINTS_EN,
    VISUAL_HINTS_ZH,
)


CJK_RE = re.compile(r"[\u4e00-\u9fff]")
EN_WORD_RE = re.compile(r"[a-zA-Z]{2,}")


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip().lower()


def merge_text_parts(*parts: str | None) -> str:
    return " ".join(normalize_text(p) for p in parts if p)


def _keyword_hits(text: str, keywords: Iterable[str]) -> list[str]:
    return [kw for kw in keywords if kw.lower() in text]


def looks_like_language(text: str, lang: str) -> bool:
    cjk_count = len(CJK_RE.findall(text))
    en_words = len(EN_WORD_RE.findall(text))
    if lang == "en":
        return en_words >= 4 and en_words >= cjk_count
    if lang == "zh":
        return cjk_count >= 2 or (cjk_count >= 1 and en_words <= 10)
    return True


def is_ultrasound_related(text: str, chinese: bool) -> tuple[bool, list[str]]:
    keys = ULTRASOUND_KEYWORDS_ZH if chinese else ULTRASOUND_KEYWORDS_EN
    hits = _keyword_hits(text, keys)
    return bool(hits), hits[:8]


def is_visual_ultrasound_likely(text: str, chinese: bool) -> tuple[bool, str]:
    visual_keys = VISUAL_HINTS_ZH if chinese else VISUAL_HINTS_EN
    visual_hits = _keyword_hits(text, visual_keys)
    negative_hits = _keyword_hits(text, NEGATIVE_NON_VISUAL_HINTS)

    if negative_hits and not visual_hits:
        return False, f"命中非画面线索: {', '.join(negative_hits[:3])}"
    if visual_hits:
        return True, f"命中图像线索: {', '.join(visual_hits[:4])}"
    return True, "未命中强负面词，暂保留"


def classify_category(text: str, chinese: bool) -> tuple[str, str]:
    mapping = CATEGORY_KEYWORDS_ZH if chinese else CATEGORY_KEYWORDS_EN
    best_category = "待人工分类"
    best_hits: list[str] = []
    best_score = 0

    for category, keywords in mapping.items():
        hits = _keyword_hits(text, keywords)
        score = len(hits)
        if score > best_score:
            best_score = score
            best_category = category
            best_hits = hits

    if best_score == 0:
        return "待人工分类", "分类关键词不足"
    return best_category, ", ".join(best_hits[:5])

