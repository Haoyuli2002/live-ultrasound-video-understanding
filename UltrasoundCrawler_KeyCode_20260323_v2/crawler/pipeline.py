from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from .bilibili import crawl_bilibili
from .models import CrawlConfig
from .storage import build_run_dir, read_json, write_csv, write_json, write_jsonl
from .youtube import crawl_youtube


def _log(config: CrawlConfig, message: str) -> None:
    if config.log_callback:
        config.log_callback(message)


def _build_summary(config: CrawlConfig, run_dir: Path, result: Any) -> dict[str, Any]:
    rows = [record.to_dict() for record in result.records]
    category_counter = Counter(row.get("category_pred", "待人工分类") for row in rows)
    with_media = sum(1 for row in rows if row.get("media_file"))
    with_thumbnail = sum(1 for row in rows if row.get("thumbnail_file"))
    paired_av = sum(1 for row in rows if row.get("audio_video_paired"))

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": config.source,
        "run_dir": str(run_dir.resolve()),
        "config": {
            "max_results": config.max_results,
            "search_per_term": config.search_per_term,
            "pages_per_term": config.pages_per_term,
            "download_media": config.download_media,
            "download_thumbnails": config.download_thumbnails,
            "download_timeout_sec": config.download_timeout_sec,
            "custom_keywords_count": len(config.custom_keywords or []),
            "max_retries": config.max_retries,
        },
        "metrics": {
            "seed_total": result.metrics.get("seed_total", 0),
            "kept_records": len(rows),
            "filtered_records": len(result.filtered),
            "failures": len(result.failures),
            "with_media_files": with_media,
            "with_thumbnails": with_thumbnail,
            "audio_video_paired": paired_av,
            "detail_attempted": result.metrics.get("detail_attempted", 0),
            "detail_success": result.metrics.get("detail_success", 0),
            "detail_empty": result.metrics.get("detail_empty", 0),
            "download_media_success": result.metrics.get("download_media_success", 0),
            "download_media_fail": result.metrics.get("download_media_fail", 0),
            "download_thumb_success": result.metrics.get("download_thumb_success", 0),
            "download_thumb_fail": result.metrics.get("download_thumb_fail", 0),
            "resume_used": bool(result.metrics.get("resume_used", False)),
        },
        "category_distribution": dict(sorted(category_counter.items(), key=lambda x: x[0])),
        "notes": [
            "无超声图像视频采用启发式过滤，建议交付前进行一次人工抽检。",
            "分类为关键词启发式分类，客户可基于 title/description/chapters 再二次校正。",
        ],
    }
    return summary


def run_crawl(config: CrawlConfig) -> dict[str, Any]:
    _log(config, f"初始化任务，来源={config.source}")
    run_dir = build_run_dir(config.output_root, config.source)
    _log(config, f"输出目录: {run_dir.resolve()}")
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if config.source == "youtube":
        _log(config, "开始抓取 YouTube 种子与详情...")
        result = crawl_youtube(config, run_dir)
    elif config.source == "bilibili":
        _log(config, "开始抓取 B站 种子与详情...")
        result = crawl_bilibili(config, run_dir)
    else:
        raise ValueError(f"未知数据源: {config.source}")

    rows = [record.to_dict() for record in result.records]
    _log(
        config,
        "抓取结束，准备写出文件。"
        f"保留={len(rows)} 过滤={len(result.filtered)} 失败={len(result.failures)} "
        f"详情成功/尝试={result.metrics.get('detail_success', 0)}/{result.metrics.get('detail_attempted', 0)}",
    )
    write_jsonl(run_dir / "videos.jsonl", rows)
    write_csv(run_dir / "videos.csv", rows)
    write_json(run_dir / "failures.json", result.failures)
    write_jsonl(run_dir / "filtered_out.jsonl", result.filtered)
    write_jsonl(raw_dir / "seed_payloads.jsonl", result.raw_seeds)
    write_jsonl(raw_dir / "detail_payloads.jsonl", result.raw_details)

    summary = _build_summary(config, run_dir, result)
    write_json(run_dir / "summary.json", summary)
    _log(config, "summary.json 写出完成，任务结束。")
    return summary


def list_recent_runs(output_root: Path, limit: int = 8) -> list[dict[str, Any]]:
    summaries: list[tuple[float, dict[str, Any]]] = []
    if not output_root.exists():
        return []

    for summary_file in output_root.glob("*/summary.json"):
        summary = read_json(summary_file)
        if not summary:
            continue
        try:
            mtime = summary_file.stat().st_mtime
        except Exception:
            mtime = 0.0
        summary["_summary_path"] = str(summary_file.resolve())
        summaries.append((mtime, summary))

    summaries.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in summaries[:limit]]
