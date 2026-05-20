from __future__ import annotations

import json
import random
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
from .config import YOUTUBE_CHANNEL_URLS, YOUTUBE_SEARCH_TERMS
from .models import CrawlConfig, CrawlResult, VideoRecord


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)

EXTRA_ORGANS = [
    "lung",
    "cardiac",
    "heart",
    "abdomen",
    "abdominal",
    "liver",
    "kidney",
    "ivc",
    "eFAST",
    "vascular",
    "thyroid",
    "pelvic",
]

EXTRA_FOCUS = [
    "scan tutorial",
    "how to scan",
    "probe position",
    "case reasoning",
    "ultrasound lecture",
    "POCUS teaching",
]

STATE_FILE_NAME = "_resume_state_youtube.json"
CATEGORY_FOLDER_MAP = {
    "扫查教学型": "scan_tutorial",
    "病例讲解型": "case_reasoning",
    "器官系统教学型": "organ_system_lecture",
    "待人工分类": "uncategorized",
}


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


def _state_path(config: CrawlConfig) -> Path:
    return config.output_root / STATE_FILE_NAME


def _config_signature(config: CrawlConfig) -> dict[str, Any]:
    return {
        "source": "youtube",
        "max_results": config.max_results,
        "search_per_term": config.search_per_term,
        "download_media": bool(config.download_media),
        "download_thumbnails": bool(config.download_thumbnails),
        "download_timeout_sec": _safe_int(config.download_timeout_sec),
        "custom_keywords": list(config.custom_keywords or []),
    }


def _record_from_dict(payload: dict[str, Any]) -> VideoRecord:
    known = {
        "source_platform",
        "language",
        "video_id",
        "url",
        "title",
        "description",
        "channel",
        "channel_id",
        "channel_url",
        "uploader",
        "uploader_id",
        "publish_time",
        "duration_sec",
        "view_count",
        "like_count",
        "comment_count",
        "tags",
        "chapters",
        "subtitles_langs",
        "category_pred",
        "category_reason",
        "ultrasound_visual_likely",
        "explanation_available",
        "seed_source",
        "thumbnail_url",
        "thumbnail_file",
        "media_file",
        "audio_video_paired",
    }
    extra = {k: v for k, v in payload.items() if k not in known}
    return VideoRecord(
        source_platform=str(payload.get("source_platform") or "YouTube"),
        language=str(payload.get("language") or "en"),
        video_id=str(payload.get("video_id") or ""),
        url=str(payload.get("url") or ""),
        title=str(payload.get("title") or ""),
        description=str(payload.get("description") or ""),
        channel=str(payload.get("channel") or ""),
        channel_id=str(payload.get("channel_id") or ""),
        channel_url=str(payload.get("channel_url") or ""),
        uploader=str(payload.get("uploader") or ""),
        uploader_id=str(payload.get("uploader_id") or ""),
        publish_time=str(payload.get("publish_time") or ""),
        duration_sec=_safe_int(payload.get("duration_sec")),
        view_count=_safe_int(payload.get("view_count")),
        like_count=_safe_int(payload.get("like_count")),
        comment_count=_safe_int(payload.get("comment_count")),
        tags=list(payload.get("tags") or []),
        chapters=list(payload.get("chapters") or []),
        subtitles_langs=list(payload.get("subtitles_langs") or []),
        category_pred=str(payload.get("category_pred") or "待人工分类"),
        category_reason=str(payload.get("category_reason") or ""),
        ultrasound_visual_likely=bool(payload.get("ultrasound_visual_likely", True)),
        explanation_available=bool(payload.get("explanation_available", True)),
        seed_source=str(payload.get("seed_source") or ""),
        thumbnail_url=str(payload.get("thumbnail_url") or ""),
        thumbnail_file=str(payload.get("thumbnail_file") or ""),
        media_file=str(payload.get("media_file") or ""),
        audio_video_paired=bool(payload.get("audio_video_paired", False)),
        extra=extra,
    )


def _save_state(
    config: CrawlConfig,
    *,
    processed_cursor: int,
    seeds: list[dict[str, Any]],
    records: list[VideoRecord],
    filtered: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    metrics: dict[str, Any],
    query_cursor: int,
) -> None:
    path = _state_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "signature": _config_signature(config),
        "processed_cursor": processed_cursor,
        "query_cursor": query_cursor,
        "seeds": seeds,
        "records": [r.to_dict() for r in records],
        "filtered": filtered,
        "failures": failures,
        "metrics": metrics,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _load_state(config: CrawlConfig) -> dict[str, Any] | None:
    path = _state_path(config)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("signature") != _config_signature(config):
        return None
    return payload


def _clear_state(config: CrawlConfig) -> None:
    path = _state_path(config)
    if path.exists():
        path.unlink()


def _effective_terms(config: CrawlConfig) -> list[str]:
    raw = [str(x).strip() for x in (config.custom_keywords or []) if str(x).strip()]
    if raw:
        return list(dict.fromkeys(raw))
    return list(YOUTUBE_SEARCH_TERMS)


def _category_folder_name(category: str) -> str:
    return CATEGORY_FOLDER_MAP.get(category, "uncategorized")


def _build_query_plan(config: CrawlConfig) -> list[dict[str, str]]:
    year_now = datetime.now().year
    years = [str(y) for y in range(year_now, max(2016, year_now - 9) - 1, -1)]
    terms: list[str] = []
    base_terms = _effective_terms(config)
    terms.extend(base_terms)

    for base in base_terms:
        for organ in EXTRA_ORGANS:
            terms.append(f"{organ} {base}")
        for focus in EXTRA_FOCUS:
            terms.append(f"{base} {focus}")
        for year in years:
            terms.append(f"{base} {year}")

    for organ in EXTRA_ORGANS:
        for focus in EXTRA_FOCUS:
            terms.append(f"POCUS {organ} {focus}")
        for year in years:
            terms.append(f"{organ} ultrasound tutorial {year}")
            terms.append(f"{organ} pocus scan {year}")

    unique_terms = list(dict.fromkeys(t.strip() for t in terms if t and t.strip()))
    plan: list[dict[str, str]] = []
    for term in unique_terms:
        plan.append({"type": "search", "mode": "ytsearch", "term": term})
        plan.append({"type": "search", "mode": "ytsearchdate", "term": term})
    for channel_url in YOUTUBE_CHANNEL_URLS:
        plan.append({"type": "channel", "mode": "channel", "term": channel_url})
    return plan


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


def _parse_upload_date(upload_date: Any) -> str:
    text = str(upload_date or "")
    if len(text) >= 8 and text[:8].isdigit():
        try:
            return datetime.strptime(text[:8], "%Y%m%d").date().isoformat()
        except Exception:
            return ""
    return ""


def _extract_flat(target: str) -> dict[str, Any]:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        "skip_download": True,
        "extract_flat": True,
        "socket_timeout": 20,
        "retries": 1,
        "extractor_retries": 1,
        "logger": QuietLogger(),
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(target, download=False) or {}


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
        "http_headers": {"User-Agent": USER_AGENT},
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=False) or {}


def _download_media(url: str, video_id: str, media_dir: Path, timeout_sec: int) -> str:
    media_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str((media_dir / f"{video_id}.%(ext)s").resolve())
    opts = {
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        "format": "bv*+ba/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "outtmpl": outtmpl,
        "retries": 2,
        "extractor_retries": 2,
        "socket_timeout": max(10, _safe_int(timeout_sec)),
        "logger": QuietLogger(),
        "http_headers": {"User-Agent": USER_AGENT},
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


def _pick_thumbnail_url(info: dict[str, Any]) -> str:
    thumbnail = str(info.get("thumbnail") or "").strip()
    if thumbnail:
        return thumbnail
    thumbs = info.get("thumbnails") or []
    if thumbs:
        url = thumbs[-1].get("url")
        if url:
            return str(url)
    return ""


def _download_thumbnail(url: str, video_id: str, thumbnail_dir: Path) -> str:
    if not url:
        return ""
    thumbnail_dir.mkdir(parents=True, exist_ok=True)

    path = urlparse(url).path
    suffix = Path(path).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"

    file_path = thumbnail_dir / f"{video_id}{suffix}"
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
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


def _collect_more_seeds(
    config: CrawlConfig,
    *,
    failures: list[dict[str, Any]],
    seeds: list[dict[str, Any]],
    seen_ids: set[str],
    plan: list[dict[str, str]],
    query_cursor: int,
    target_total: int,
) -> int:
    cursor = query_cursor
    while cursor < len(plan) and len(seeds) < target_total:
        item = plan[cursor]
        cursor += 1
        tp = item["type"]
        term = item["term"]

        if tp == "search":
            mode = item["mode"]
            query = f"{mode}{config.search_per_term}:{term}"
            _log(config, f"[YouTube] 扩种检索: {mode} {term}")
            try:
                info = _with_retry(config, "youtube_list_search", term, lambda: _extract_flat(query))
            except Exception as exc:
                failures.append({"stage": "youtube_list_search", "target": term, "error": str(exc)})
                _log(config, f"[YouTube] 扩种失败: {term} -> {exc}")
                continue

            entries = info.get("entries") or []
            _log(config, f"[YouTube] 扩种返回 {len(entries)} 条: {term}")
            for entry in entries:
                if not entry:
                    continue
                video_id = str(entry.get("id") or "").strip()
                if not video_id or video_id in seen_ids:
                    continue
                seen_ids.add(video_id)
                seeds.append(
                    {
                        "video_id": video_id,
                        "url": f"https://www.youtube.com/watch?v={video_id}",
                        "title": str(entry.get("title") or ""),
                        "seed_source": f"{mode}:{term}",
                        "raw_seed": entry,
                    }
                )
                if len(seeds) >= target_total:
                    break

        elif tp == "channel":
            channel_url = term
            _log(config, f"[YouTube] 扫描频道扩种: {channel_url}")
            target = f"{channel_url.rstrip('/')}/videos"
            try:
                info = _with_retry(config, "youtube_list_channel", channel_url, lambda: _extract_flat(target))
            except Exception as exc:
                failures.append({"stage": "youtube_list_channel", "target": channel_url, "error": str(exc)})
                _log(config, f"[YouTube] 频道扩种失败: {channel_url} -> {exc}")
                continue

            limit = min(max(config.search_per_term * 6, 80), 1200)
            entries = (info.get("entries") or [])[:limit]
            _log(config, f"[YouTube] 频道扩种返回 {len(entries)} 条: {channel_url}")
            for entry in entries:
                if not entry:
                    continue
                video_id = str(entry.get("id") or "").strip()
                if not video_id or video_id in seen_ids:
                    continue
                seen_ids.add(video_id)
                seeds.append(
                    {
                        "video_id": video_id,
                        "url": f"https://www.youtube.com/watch?v={video_id}",
                        "title": str(entry.get("title") or ""),
                        "seed_source": f"channel:{channel_url}",
                        "raw_seed": entry,
                    }
                )
                if len(seeds) >= target_total:
                    break
    return cursor


def crawl_youtube(config: CrawlConfig, run_dir: Path) -> CrawlResult:
    result = CrawlResult(source="youtube")
    media_dir = run_dir / "media"
    thumbnail_dir = run_dir / "thumbnails"
    initial_target_seed_total = min(max(config.max_results + 30, config.search_per_term * 4, 50), 300)
    plan = _build_query_plan(config)
    query_cursor = 0
    seeds: list[dict[str, Any]] = []
    processed_cursor = 0
    seen_ids: set[str] = set()

    result.metrics = {
        "seed_total": 0,
        "detail_attempted": 0,
        "detail_success": 0,
        "detail_empty": 0,
        "download_media_success": 0,
        "download_media_fail": 0,
        "download_thumb_success": 0,
        "download_thumb_fail": 0,
        "resume_used": False,
    }

    state = _load_state(config)
    if state:
        _log(config, "[YouTube] 发现断点状态，开始续跑。")
        seeds = list(state.get("seeds") or [])
        processed_cursor = _safe_int(state.get("processed_cursor"))
        query_cursor = _safe_int(state.get("query_cursor"))
        result.records = [_record_from_dict(x) for x in (state.get("records") or [])]
        result.filtered = list(state.get("filtered") or [])
        result.failures = list(state.get("failures") or [])
        restored_metrics = state.get("metrics") or {}
        for key, value in restored_metrics.items():
            result.metrics[key] = value
        result.metrics["resume_used"] = True
        _log(
            config,
            f"[YouTube] 断点恢复: 已处理={processed_cursor}, 已保留={len(result.records)}, "
            f"已有种子={len(seeds)}, 查询进度={query_cursor}/{len(plan)}",
        )

    for seed in seeds:
        sid = str(seed.get("video_id") or "").strip()
        if sid:
            seen_ids.add(sid)

    if len(seeds) < initial_target_seed_total and query_cursor < len(plan):
        prev_count = len(seeds)
        query_cursor = _collect_more_seeds(
            config,
            failures=result.failures,
            seeds=seeds,
            seen_ids=seen_ids,
            plan=plan,
            query_cursor=query_cursor,
            target_total=initial_target_seed_total,
        )
        _log(config, f"[YouTube] 初始扩种完成: {prev_count} -> {len(seeds)}")

    result.raw_seeds = seeds
    result.metrics["seed_total"] = len(seeds)
    _save_state(
        config,
        processed_cursor=processed_cursor,
        seeds=seeds,
        records=result.records,
        filtered=result.filtered,
        failures=result.failures,
        metrics=result.metrics,
        query_cursor=query_cursor,
    )

    _log(
        config,
        f"[YouTube] 种子总数={len(seeds)}，目标保留={config.max_results}，开始详情抓取。",
    )

    while len(result.records) < config.max_results:
        if processed_cursor >= len(seeds):
            if query_cursor >= len(plan):
                _log(config, "[YouTube] 查询计划已耗尽，无法继续扩种。")
                break
            prev_count = len(seeds)
            query_cursor = _collect_more_seeds(
                config,
                failures=result.failures,
                seeds=seeds,
                seen_ids=seen_ids,
                plan=plan,
                query_cursor=query_cursor,
                target_total=min(len(seeds) + max(config.search_per_term * 10, 120), 7000),
            )
            result.metrics["seed_total"] = len(seeds)
            _log(config, f"[YouTube] 处理队列耗尽，补充种子: {prev_count} -> {len(seeds)}")
            if len(seeds) <= prev_count:
                break
            result.raw_seeds = seeds

        if (len(seeds) - processed_cursor) <= max(config.search_per_term, 20) and query_cursor < len(plan):
            prev_count = len(seeds)
            query_cursor = _collect_more_seeds(
                config,
                failures=result.failures,
                seeds=seeds,
                seen_ids=seen_ids,
                plan=plan,
                query_cursor=query_cursor,
                target_total=min(len(seeds) + max(config.search_per_term * 8, 80), 7000),
            )
            result.metrics["seed_total"] = len(seeds)
            if len(seeds) > prev_count:
                _log(config, f"[YouTube] 前瞻补种: {prev_count} -> {len(seeds)}")
                result.raw_seeds = seeds

        seed = seeds[processed_cursor]
        video_id = str(seed.get("video_id") or "").strip()
        url = str(seed.get("url") or f"https://www.youtube.com/watch?v={video_id}")
        processed_cursor += 1
        result.metrics["detail_attempted"] = _safe_int(result.metrics.get("detail_attempted")) + 1

        _log(config, f"[YouTube] 详情处理中 {processed_cursor}/{len(seeds)}: {video_id}")
        _sleep(config)

        try:
            info = _with_retry(config, "youtube_detail", video_id, lambda: _extract_detail(url))
        except Exception as exc:
            result.failures.append({"stage": "youtube_detail", "target": video_id, "error": str(exc)})
            _log(config, f"[YouTube] 详情失败: {video_id} -> {exc}")
            if processed_cursor % 3 == 0:
                _save_state(
                    config,
                    processed_cursor=processed_cursor,
                    seeds=seeds,
                    records=result.records,
                    filtered=result.filtered,
                    failures=result.failures,
                    metrics=result.metrics,
                    query_cursor=query_cursor,
                )
            continue

        if not info:
            recovered = False
            for alt_url in (
                f"https://www.youtube.com/shorts/{video_id}",
                f"https://www.youtube.com/watch?v={video_id}&bpctr=9999999999&has_verified=1",
            ):
                try:
                    info = _with_retry(config, "youtube_detail_recover", video_id, lambda u=alt_url: _extract_detail(u))
                except Exception:
                    info = {}
                if info:
                    recovered = True
                    _log(config, f"[YouTube] 详情回补成功: {video_id}")
                    break

            if not recovered:
                result.failures.append({"stage": "youtube_detail", "target": video_id, "error": "empty payload"})
                result.metrics["detail_empty"] = _safe_int(result.metrics.get("detail_empty")) + 1
                _log(config, f"[YouTube] 详情为空: {video_id}")
                if processed_cursor % 3 == 0:
                    _save_state(
                        config,
                        processed_cursor=processed_cursor,
                        seeds=seeds,
                        records=result.records,
                        filtered=result.filtered,
                        failures=result.failures,
                        metrics=result.metrics,
                        query_cursor=query_cursor,
                    )
                continue

        result.metrics["detail_success"] = _safe_int(result.metrics.get("detail_success")) + 1
        result.raw_details.append({"video_id": video_id, "payload": info})

        title = str(info.get("title") or seed.get("title") or "")
        description = str(info.get("description") or "")
        tags = [str(tag) for tag in (info.get("tags") or []) if tag]
        chapters = _build_chapters(info)
        text_blob = merge_text_parts(title, description, " ".join(tags), " ".join(c.get("title", "") for c in chapters))

        if not looks_like_language(text_blob, "en"):
            result.filtered.append({"video_id": video_id, "reason": "非英文内容", "title": title})
            _log(config, f"[YouTube] 过滤(非英文): {video_id}")
            if processed_cursor % 3 == 0:
                _save_state(
                    config,
                    processed_cursor=processed_cursor,
                    seeds=seeds,
                    records=result.records,
                    filtered=result.filtered,
                    failures=result.failures,
                    metrics=result.metrics,
                    query_cursor=query_cursor,
                )
            continue

        related, related_hits = is_ultrasound_related(text_blob, chinese=False)
        if not related:
            result.filtered.append({"video_id": video_id, "reason": "未命中超声关键词", "title": title})
            _log(config, f"[YouTube] 过滤(非超声): {video_id}")
            if processed_cursor % 3 == 0:
                _save_state(
                    config,
                    processed_cursor=processed_cursor,
                    seeds=seeds,
                    records=result.records,
                    filtered=result.filtered,
                    failures=result.failures,
                    metrics=result.metrics,
                    query_cursor=query_cursor,
                )
            continue

        visual_ok, visual_reason = is_visual_ultrasound_likely(text_blob, chinese=False)
        if not visual_ok:
            result.filtered.append({"video_id": video_id, "reason": visual_reason, "title": title})
            _log(config, f"[YouTube] 过滤(疑似无图像): {video_id}")
            if processed_cursor % 3 == 0:
                _save_state(
                    config,
                    processed_cursor=processed_cursor,
                    seeds=seeds,
                    records=result.records,
                    filtered=result.filtered,
                    failures=result.failures,
                    metrics=result.metrics,
                    query_cursor=query_cursor,
                )
            continue

        category, category_reason = classify_category(text_blob, chinese=False)
        subtitles_langs = _build_subtitles_langs(info)
        explanation_available = bool(description.strip() or chapters)
        thumbnail_url = _pick_thumbnail_url(info)
        category_media_dir = media_dir / _category_folder_name(category)
        media_file = ""
        thumbnail_file = ""

        if config.download_media:
            try:
                media_file = _with_retry(
                    config,
                    "youtube_media",
                    video_id,
                    lambda: _download_media(url, video_id, category_media_dir, config.download_timeout_sec),
                )
                if media_file:
                    result.metrics["download_media_success"] = _safe_int(result.metrics.get("download_media_success")) + 1
                    _log(config, f"[YouTube] 媒体下载完成: {video_id}")
            except Exception as exc:
                result.failures.append({"stage": "youtube_media", "target": video_id, "error": str(exc)})
                result.metrics["download_media_fail"] = _safe_int(result.metrics.get("download_media_fail")) + 1
                _log(config, f"[YouTube] 媒体下载失败: {video_id} -> {exc}")

        if config.download_thumbnails and thumbnail_url:
            try:
                thumbnail_file = _with_retry(
                    config,
                    "youtube_thumbnail",
                    video_id,
                    lambda: _download_thumbnail(thumbnail_url, video_id, thumbnail_dir),
                )
                if thumbnail_file:
                    result.metrics["download_thumb_success"] = _safe_int(result.metrics.get("download_thumb_success")) + 1
                    _log(config, f"[YouTube] 缩略图下载完成: {video_id}")
            except Exception as exc:
                result.failures.append({"stage": "youtube_thumbnail", "target": video_id, "error": str(exc)})
                result.metrics["download_thumb_fail"] = _safe_int(result.metrics.get("download_thumb_fail")) + 1
                _log(config, f"[YouTube] 缩略图下载失败: {video_id} -> {exc}")

        record = VideoRecord(
            source_platform="YouTube",
            language="en",
            video_id=video_id,
            url=str(info.get("webpage_url") or url),
            title=title,
            description=description,
            channel=str(info.get("channel") or ""),
            channel_id=str(info.get("channel_id") or ""),
            channel_url=str(info.get("channel_url") or ""),
            uploader=str(info.get("uploader") or ""),
            uploader_id=str(info.get("uploader_id") or ""),
            publish_time=_parse_upload_date(info.get("upload_date")),
            duration_sec=_safe_int(info.get("duration")),
            view_count=_safe_int(info.get("view_count")),
            like_count=_safe_int(info.get("like_count")),
            comment_count=_safe_int(info.get("comment_count")),
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
            audio_video_paired=bool(media_file) or _has_paired_av(info),
            extra={
                "chapters_count": len(chapters),
                "subtitle_lang_count": len(subtitles_langs),
            },
        )
        result.records.append(record)
        _log(
            config,
            f"[YouTube] 保留 {len(result.records)}/{config.max_results}: {video_id} | 分类={category} | "
            f"详情成功={result.metrics['detail_success']}/{result.metrics['detail_attempted']} | "
            f"当前失败={len(result.failures)} 过滤={len(result.filtered)}",
        )

        if processed_cursor % 2 == 0 or len(result.records) >= config.max_results:
            _save_state(
                config,
                processed_cursor=processed_cursor,
                seeds=seeds,
                records=result.records,
                filtered=result.filtered,
                failures=result.failures,
                metrics=result.metrics,
                query_cursor=query_cursor,
            )

    result.metrics["seed_total"] = len(seeds)
    result.metrics["kept"] = len(result.records)
    result.metrics["filtered"] = len(result.filtered)
    result.metrics["failures"] = len(result.failures)
    _clear_state(config)
    return result
