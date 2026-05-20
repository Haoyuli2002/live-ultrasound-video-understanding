from __future__ import annotations

import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import requests
import yt_dlp

from .classifier import (
    classify_category,
    is_ultrasound_related,
    is_visual_ultrasound_likely,
    looks_like_language,
    merge_text_parts,
)
from .config import BILIBILI_SEARCH_TERMS
from .models import CrawlConfig, CrawlResult, VideoRecord


BILIBILI_SEARCH_API = "https://api.bilibili.com/x/web-interface/search/type"
HTML_TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
CATEGORY_FOLDER_MAP = {
    "扫查教学型": "scan_tutorial",
    "病例讲解型": "case_reasoning",
    "器官系统教学型": "organ_system_lecture",
    "待人工分类": "uncategorized",
}

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)


class QuietLogger:
    def debug(self, _msg: str) -> None:
        return

    def warning(self, _msg: str) -> None:
        return

    def error(self, _msg: str) -> None:
        return


def _safe_int(value: Any) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(value))
    except Exception:
        return 0


def _log(config: CrawlConfig, message: str) -> None:
    if config.log_callback:
        config.log_callback(message)


def _effective_terms(config: CrawlConfig) -> list[str]:
    raw = [str(x).strip() for x in (config.custom_keywords or []) if str(x).strip()]
    if raw:
        return list(dict.fromkeys(raw))
    return list(BILIBILI_SEARCH_TERMS)


def _category_folder_name(category: str) -> str:
    return CATEGORY_FOLDER_MAP.get(category, "uncategorized")


def _parse_count(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value or "").strip().lower().replace(",", "")
    if not text:
        return 0
    mul = 1
    if text.endswith("万"):
        mul = 10000
        text = text[:-1]
    elif text.endswith("亿"):
        mul = 100000000
        text = text[:-1]
    try:
        return int(float(text) * mul)
    except Exception:
        return 0


def _strip_html(text: str | None) -> str:
    if not text:
        return ""
    return SPACE_RE.sub(" ", HTML_TAG_RE.sub(" ", text)).strip()


def _parse_duration(duration: Any) -> int:
    text = str(duration or "").strip()
    if not text:
        return 0
    parts = text.split(":")
    if not all(p.isdigit() for p in parts):
        return 0
    if len(parts) == 3:
        h, m, s = [int(v) for v in parts]
        return h * 3600 + m * 60 + s
    if len(parts) == 2:
        m, s = [int(v) for v in parts]
        return m * 60 + s
    if len(parts) == 1:
        return int(parts[0])
    return 0


def _sleep(config: CrawlConfig) -> None:
    time.sleep(random.uniform(config.min_delay_sec, config.max_delay_sec))


def _with_retry(
    config: CrawlConfig,
    stage: str,
    target: str,
    func: Callable[[], Any],
) -> Any:
    last_exc: Exception | None = None
    for attempt in range(1, config.max_retries + 1):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            if attempt < config.max_retries:
                cooldown = min(2 * attempt, 8) + random.uniform(0.2, 0.8)
                time.sleep(cooldown)
    raise RuntimeError(f"{stage}失败({target})") from last_exc


def _extract_detail(url: str) -> dict[str, Any]:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        "skip_download": True,
        "extract_flat": False,
        "socket_timeout": 25,
        "retries": 1,
        "extractor_retries": 1,
        "logger": QuietLogger(),
        "http_headers": {"User-Agent": USER_AGENT, "Referer": "https://www.bilibili.com/"},
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=False) or {}


def _extract_bilisearch(term: str, max_items: int) -> dict[str, Any]:
    query = f"bilisearch{max_items}:{term}"
    opts = {
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        "skip_download": True,
        "extract_flat": True,
        "socket_timeout": 25,
        "retries": 1,
        "extractor_retries": 1,
        "logger": QuietLogger(),
        "http_headers": {"User-Agent": USER_AGENT, "Referer": "https://www.bilibili.com/"},
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(query, download=False) or {}


def _download_media(url: str, video_id: str, media_dir: Path, timeout_sec: int) -> str:
    media_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str((media_dir / f"{video_id}.%(ext)s").resolve())
    opts = {
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "outtmpl": outtmpl,
        "retries": 2,
        "extractor_retries": 2,
        "socket_timeout": max(10, _safe_int(timeout_sec)),
        "logger": QuietLogger(),
        "http_headers": {"User-Agent": USER_AGENT, "Referer": "https://www.bilibili.com/"},
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.extract_info(url, download=True)

    candidates = [
        p
        for p in media_dir.glob(f"{video_id}.*")
        if p.is_file() and p.suffix.lower() not in {".part", ".ytdl"}
    ]
    if not candidates:
        return ""
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return str(candidates[0].resolve())


def _download_thumbnail(url: str, video_id: str, thumbnail_dir: Path) -> str:
    if not url:
        return ""
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    fixed = url if url.startswith("http") else f"https:{url}"

    path = urlparse(fixed).path
    suffix = Path(path).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"

    file_path = thumbnail_dir / f"{video_id}{suffix}"
    resp = requests.get(fixed, headers={"User-Agent": USER_AGENT}, timeout=20)
    resp.raise_for_status()
    file_path.write_bytes(resp.content)
    return str(file_path.resolve())


def _has_paired_av(info: dict[str, Any]) -> bool:
    formats = info.get("formats") or []
    for fmt in formats:
        if fmt.get("acodec") not in (None, "none") and fmt.get("vcodec") not in (None, "none"):
            return True
    return False


def _build_chapters(info: dict[str, Any]) -> list[dict[str, Any]]:
    chapters = info.get("chapters") or []
    normalized: list[dict[str, Any]] = []
    for chapter in chapters:
        normalized.append(
            {
                "title": str(chapter.get("title") or ""),
                "start_time": _safe_int(chapter.get("start_time")),
                "end_time": _safe_int(chapter.get("end_time")),
            }
        )
    return normalized


def _build_subtitles_langs(info: dict[str, Any]) -> list[str]:
    langs: list[str] = []
    subtitles = info.get("subtitles") or {}
    auto_caps = info.get("automatic_captions") or {}
    for key in list(subtitles.keys()) + list(auto_caps.keys()):
        if key not in langs:
            langs.append(key)
    return langs


def _collect_search_seeds(config: CrawlConfig, failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Referer": "https://www.bilibili.com/",
            "Accept": "application/json, text/plain, */*",
        }
    )

    seeds: list[dict[str, Any]] = []
    seen: set[str] = set()
    fallback_done_terms: set[str] = set()

    for term in _effective_terms(config):
        _log(config, f"[B站] 搜索关键词: {term}")
        for page in range(1, config.pages_per_term + 1):
            _sleep(config)
            params = {
                "search_type": "video",
                "keyword": term,
                "order": "pubdate",
                "page": page,
            }
            target = f"{term}-P{page}"
            try:
                payload = _with_retry(
                    config,
                    "bilibili_list_search",
                    target,
                    lambda: session.get(BILIBILI_SEARCH_API, params=params, timeout=20).json(),
                )
            except Exception as exc:
                failures.append({"stage": "bilibili_list_search", "target": target, "error": str(exc)})
                _log(config, f"[B站] 搜索失败: {target} -> {exc}")
                continue

            if payload.get("code") != 0:
                code = payload.get("code")
                msg = str(payload.get("message") or "")
                failures.append(
                    {
                        "stage": "bilibili_list_search",
                        "target": target,
                        "error": f"code={code} message={msg}",
                    }
                )
                if code == -412 and term not in fallback_done_terms:
                    _log(config, f"[B站] 触发反爬(-412)，降级 bilisearch: {term}")
                    fallback_done_terms.add(term)
                    try:
                        fallback_payload = _with_retry(
                            config,
                            "bilibili_list_fallback",
                            term,
                            lambda: _extract_bilisearch(term, config.search_per_term),
                        )
                    except Exception as exc:
                        failures.append(
                            {
                                "stage": "bilibili_list_fallback",
                                "target": term,
                                "error": str(exc),
                            }
                        )
                        _log(config, f"[B站] bilisearch 降级失败: {term} -> {exc}")
                        continue

                    for entry in fallback_payload.get("entries") or []:
                        if not entry:
                            continue
                        bvid = str(entry.get("id") or "").strip()
                        if not bvid or bvid in seen:
                            continue
                        seen.add(bvid)

                        title = _strip_html(str(entry.get("title") or ""))
                        desc = _strip_html(str(entry.get("description") or ""))
                        url = str(entry.get("webpage_url") or entry.get("url") or "").strip()
                        if not url.startswith("http"):
                            url = f"https://www.bilibili.com/video/{bvid}"

                        seeds.append(
                            {
                                "video_id": bvid,
                                "url": url,
                                "title": title,
                                "description": desc,
                                "duration_sec": _safe_int(entry.get("duration")),
                                "view_count": _parse_count(entry.get("view_count")),
                                "comment_count": _parse_count(entry.get("comment_count")),
                                "like_count": _parse_count(entry.get("like_count")),
                                "publish_time": "",
                                "thumbnail_url": str(entry.get("thumbnail") or ""),
                                "seed_source": f"bilisearch:{term}",
                                "raw_seed": entry,
                            }
                        )
                    _log(config, f"[B站] bilisearch 降级完成: {term}")
                continue

            entries = (payload.get("data") or {}).get("result") or []
            if not entries:
                break
            _log(config, f"[B站] {target} 返回 {len(entries)} 条")

            for entry in entries:
                bvid = str(entry.get("bvid") or "").strip()
                if not bvid or bvid in seen:
                    continue
                seen.add(bvid)
                title = _strip_html(str(entry.get("title") or ""))
                desc = _strip_html(str(entry.get("description") or ""))
                seeds.append(
                    {
                        "video_id": bvid,
                        "url": f"https://www.bilibili.com/video/{bvid}",
                        "title": title,
                        "description": desc,
                        "duration_sec": _parse_duration(entry.get("duration")),
                        "view_count": _parse_count(entry.get("play")),
                        "comment_count": _parse_count(entry.get("video_review")),
                        "like_count": _parse_count(entry.get("like")),
                        "publish_time": datetime.fromtimestamp(_safe_int(entry.get("pubdate"))).isoformat()
                        if _safe_int(entry.get("pubdate")) > 0
                        else "",
                        "thumbnail_url": str(entry.get("pic") or ""),
                        "seed_source": f"search:{term}:P{page}",
                        "raw_seed": entry,
                    }
                )
    return seeds


def crawl_bilibili(config: CrawlConfig, run_dir: Path) -> CrawlResult:
    result = CrawlResult(source="bilibili")
    media_dir = run_dir / "media"
    thumbnail_dir = run_dir / "thumbnails"

    seeds = _collect_search_seeds(config, result.failures)
    result.raw_seeds = seeds
    result.metrics["seed_total"] = len(seeds)
    _log(config, f"[B站] 种子去重后共 {len(seeds)} 条，开始详情抓取。")

    for idx, seed in enumerate(seeds, 1):
        if len(result.records) >= config.max_results:
            break

        video_id = seed["video_id"]
        url = seed["url"]
        _log(config, f"[B站] 详情处理中 {idx}/{len(seeds)}: {video_id}")
        _sleep(config)

        info: dict[str, Any] | None = None
        detail_failed = False
        try:
            info = _with_retry(config, "bilibili_detail", video_id, lambda: _extract_detail(url))
        except Exception as exc:
            detail_failed = True
            result.failures.append({"stage": "bilibili_detail", "target": video_id, "error": str(exc)})
            _log(config, f"[B站] 详情失败: {video_id} -> {exc}")
        detail_fallback_used = detail_failed or not bool(info)

        if info:
            result.raw_details.append({"video_id": video_id, "payload": info})

        title = str((info or {}).get("title") or seed.get("title") or "")
        description = str((info or {}).get("description") or seed.get("description") or "")
        tags = [str(tag) for tag in ((info or {}).get("tags") or []) if tag]
        chapters = _build_chapters(info or {})
        text_blob = merge_text_parts(title, description, " ".join(tags), " ".join(c.get("title", "") for c in chapters))

        if not looks_like_language(text_blob, "zh"):
            result.filtered.append({"video_id": video_id, "reason": "非中文内容", "title": title})
            _log(config, f"[B站] 过滤(非中文): {video_id}")
            continue

        related, related_hits = is_ultrasound_related(text_blob, chinese=True)
        if not related:
            result.filtered.append({"video_id": video_id, "reason": "未命中超声关键词", "title": title})
            _log(config, f"[B站] 过滤(非超声): {video_id}")
            continue

        visual_ok, visual_reason = is_visual_ultrasound_likely(text_blob, chinese=True)
        if not visual_ok:
            result.filtered.append({"video_id": video_id, "reason": visual_reason, "title": title})
            _log(config, f"[B站] 过滤(疑似无图像): {video_id}")
            continue

        category, category_reason = classify_category(text_blob, chinese=True)
        subtitles_langs = _build_subtitles_langs(info or {})
        explanation_available = bool(description.strip() or chapters)
        thumbnail_url = str((info or {}).get("thumbnail") or seed.get("thumbnail_url") or "")
        category_media_dir = media_dir / _category_folder_name(category)
        media_file = ""
        thumbnail_file = ""

        if config.download_media:
            try:
                media_file = _with_retry(
                    config,
                    "bilibili_media",
                    video_id,
                    lambda: _download_media(url, video_id, category_media_dir, config.download_timeout_sec),
                )
                if media_file:
                    _log(config, f"[B站] 媒体下载完成: {video_id}")
            except Exception as exc:
                result.failures.append({"stage": "bilibili_media", "target": video_id, "error": str(exc)})
                _log(config, f"[B站] 媒体下载失败: {video_id} -> {exc}")

        if config.download_thumbnails and thumbnail_url:
            try:
                thumbnail_file = _with_retry(
                    config,
                    "bilibili_thumbnail",
                    video_id,
                    lambda: _download_thumbnail(thumbnail_url, video_id, thumbnail_dir),
                )
                if thumbnail_file:
                    _log(config, f"[B站] 缩略图下载完成: {video_id}")
            except Exception as exc:
                result.failures.append({"stage": "bilibili_thumbnail", "target": video_id, "error": str(exc)})
                _log(config, f"[B站] 缩略图下载失败: {video_id} -> {exc}")

        timestamp = _safe_int((info or {}).get("timestamp"))
        publish_time = ""
        if timestamp > 0:
            publish_time = datetime.fromtimestamp(timestamp).isoformat()
        elif seed.get("publish_time"):
            publish_time = str(seed["publish_time"])

        record = VideoRecord(
            source_platform="Bilibili",
            language="zh",
            video_id=video_id,
            url=str((info or {}).get("webpage_url") or url),
            title=title,
            description=description,
            channel=str((info or {}).get("channel") or ""),
            channel_id=str((info or {}).get("channel_id") or ""),
            channel_url=str((info or {}).get("channel_url") or ""),
            uploader=str((info or {}).get("uploader") or ""),
            uploader_id=str((info or {}).get("uploader_id") or ""),
            publish_time=publish_time,
            duration_sec=_safe_int((info or {}).get("duration") or seed.get("duration_sec")),
            view_count=_safe_int((info or {}).get("view_count") or seed.get("view_count")),
            like_count=_safe_int((info or {}).get("like_count") or seed.get("like_count")),
            comment_count=_safe_int((info or {}).get("comment_count") or seed.get("comment_count")),
            tags=tags,
            chapters=chapters,
            subtitles_langs=subtitles_langs,
            category_pred=category,
            category_reason=f"{category_reason}; 超声词: {', '.join(related_hits[:5])}",
            ultrasound_visual_likely=visual_ok,
            explanation_available=explanation_available,
            seed_source=str(seed.get("seed_source") or ""),
            thumbnail_url=thumbnail_url,
            thumbnail_file=thumbnail_file,
            media_file=media_file,
            audio_video_paired=bool(media_file) or _has_paired_av(info or {}),
            extra={
                "chapters_count": len(chapters),
                "subtitle_lang_count": len(subtitles_langs),
                "detail_fallback_used": detail_fallback_used,
            },
        )
        result.records.append(record)
        _log(
            config,
            f"[B站] 保留 {len(result.records)}/{config.max_results}: {video_id} | "
            f"分类={category} | 当前失败={len(result.failures)} 过滤={len(result.filtered)}",
        )

    result.metrics["kept"] = len(result.records)
    result.metrics["filtered"] = len(result.filtered)
    result.metrics["failures"] = len(result.failures)
    return result
