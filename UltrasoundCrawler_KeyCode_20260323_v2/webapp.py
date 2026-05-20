from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import re
import sys
from typing import Any

from flask import Flask, jsonify, redirect, render_template, request, url_for

from crawler.config import BILIBILI_SEARCH_TERMS, YOUTUBE_SEARCH_TERMS
from crawler.models import CrawlConfig
from crawler.pipeline import list_recent_runs, run_crawl


APP_TITLE = "超声扫描讲解视频抓取器"


def _runtime_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _resource_path(name: str) -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS")) / name
    return _runtime_base_dir() / name


BASE_DIR = _runtime_base_dir()
OUTPUT_ROOT = BASE_DIR / "output"
TEMPLATE_DIR = _resource_path("templates")
STATIC_DIR = _resource_path("static")
DEFAULTS = {
    "max_results": 60,
    "search_per_term": 20,
    "pages_per_term": 3,
    "download_timeout_sec": 60,
    "download_media": True,
    "download_thumbnails": True,
    "youtube_keywords_text": "\n".join(YOUTUBE_SEARCH_TERMS),
    "bilibili_keywords_text": "\n".join(BILIBILI_SEARCH_TERMS),
}

app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
MAX_LOG_LINES = 600
MAX_JOBS = 50


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _parse_keywords(raw: str) -> list[str] | None:
    text = (raw or "").strip()
    if not text:
        return None
    parts = [x.strip() for x in re.split(r"[\n,，;；|]+", text) if x.strip()]
    if not parts:
        return None
    return list(dict.fromkeys(parts))


def _append_job_log(job_id: str, message: str) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job["logs"].append(f"[{_now_text()}] {message}")
        if len(job["logs"]) > MAX_LOG_LINES:
            job["logs"] = job["logs"][-MAX_LOG_LINES:]
        job["updated_at"] = _now_text()


def _job_snapshot(job_id: str) -> dict[str, Any] | None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return None
        return {
            "id": job["id"],
            "source": job["source"],
            "status": job["status"],
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
            "logs": list(job["logs"]),
            "summary": job["summary"],
            "error": job["error"],
        }


def _latest_job_snapshot(prefer_running: bool = True) -> dict[str, Any] | None:
    with JOBS_LOCK:
        if not JOBS:
            return None
        jobs = list(JOBS.values())

    jobs.sort(key=lambda item: (item.get("updated_at", ""), item.get("created_at", "")), reverse=True)
    if prefer_running:
        for item in jobs:
            if item.get("status") in {"queued", "running"}:
                return _job_snapshot(item["id"])
    return _job_snapshot(jobs[0]["id"])


def _prune_jobs() -> None:
    with JOBS_LOCK:
        if len(JOBS) <= MAX_JOBS:
            return
        ordered = sorted(JOBS.values(), key=lambda item: item["created_at"])
        remove_count = len(JOBS) - MAX_JOBS
        for item in ordered[:remove_count]:
            JOBS.pop(item["id"], None)


def _set_job_status(job_id: str, status: str, summary: dict[str, Any] | None = None, error: str | None = None) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job["status"] = status
        job["updated_at"] = _now_text()
        if summary is not None:
            job["summary"] = summary
        if error is not None:
            job["error"] = error


def _start_background_job(job_id: str, config: CrawlConfig) -> None:
    _set_job_status(job_id, "running")
    _append_job_log(job_id, "任务已启动，开始抓取。")

    cfg = replace(config, log_callback=lambda msg: _append_job_log(job_id, msg))
    try:
        summary = run_crawl(cfg)
        _set_job_status(job_id, "completed", summary=summary)
        _append_job_log(job_id, "任务完成。")
    except Exception as exc:
        _set_job_status(job_id, "failed", error=str(exc))
        _append_job_log(job_id, f"任务失败: {exc}")
        _append_job_log(job_id, traceback.format_exc(limit=5))


@app.get("/")
def index():
    active_job_id = str(request.args.get("job_id", "") or "").strip()
    active_job = _job_snapshot(active_job_id) if active_job_id else None
    if not active_job:
        active_job = _latest_job_snapshot(prefer_running=True)
    recent = list_recent_runs(OUTPUT_ROOT, limit=10)
    summary = None
    error = None
    if active_job:
        if active_job["status"] == "completed":
            summary = active_job.get("summary")
        elif active_job["status"] == "failed":
            error = active_job.get("error") or "任务失败"

    return render_template(
        "index.html",
        app_title=APP_TITLE,
        defaults=DEFAULTS,
        recent=recent,
        summary=summary,
        error=error,
        active_job=active_job,
    )


@app.after_request
def disable_cache(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.post("/run/<source>")
def run_source(source: str):
    if source not in {"youtube", "bilibili"}:
        return redirect(url_for("index"))

    try:
        max_results = max(1, int(request.form.get("max_results", DEFAULTS["max_results"])))
        search_per_term = max(1, int(request.form.get("search_per_term", DEFAULTS["search_per_term"])))
        pages_per_term = max(1, int(request.form.get("pages_per_term", DEFAULTS["pages_per_term"])))
        download_timeout_sec = max(10, int(request.form.get("download_timeout_sec", DEFAULTS["download_timeout_sec"])))
    except ValueError:
        recent = list_recent_runs(OUTPUT_ROOT, limit=10)
        return render_template(
            "index.html",
            app_title=APP_TITLE,
            defaults=DEFAULTS,
            recent=recent,
            summary=None,
            error="输入参数格式错误，请填写整数。",
            active_job=None,
        )

    download_media = request.form.get("download_media") == "on"
    download_thumbnails = request.form.get("download_thumbnails") == "on"
    custom_keywords = _parse_keywords(str(request.form.get("keywords", "") or ""))

    config = CrawlConfig(
        source=source,
        max_results=max_results,
        search_per_term=search_per_term,
        pages_per_term=pages_per_term,
        download_media=download_media,
        download_thumbnails=download_thumbnails,
        download_timeout_sec=download_timeout_sec,
        custom_keywords=custom_keywords,
        output_root=OUTPUT_ROOT,
    )

    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "source": source,
            "status": "queued",
            "created_at": _now_text(),
            "updated_at": _now_text(),
            "logs": [],
            "summary": None,
            "error": None,
        }
    _append_job_log(job_id, f"任务创建成功，job_id={job_id}")
    _append_job_log(job_id, f"参数: max_results={max_results}, search_per_term={search_per_term}, pages_per_term={pages_per_term}")
    _append_job_log(job_id, f"下载音视频={download_media}, 下载缩略图={download_thumbnails}, 下载超时={download_timeout_sec}s")
    _append_job_log(job_id, f"自定义关键词数={len(custom_keywords or [])}")
    _prune_jobs()

    thread = threading.Thread(target=_start_background_job, args=(job_id, config), daemon=True)
    thread.start()
    return redirect(url_for("index", job_id=job_id))


@app.get("/job/<job_id>/status")
def job_status(job_id: str):
    snapshot = _job_snapshot(job_id)
    if not snapshot:
        return jsonify({"error": "job_not_found"}), 404
    return jsonify(snapshot)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5088, debug=False)
