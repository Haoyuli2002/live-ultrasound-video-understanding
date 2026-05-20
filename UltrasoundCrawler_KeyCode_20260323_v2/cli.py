from __future__ import annotations

import argparse
from pathlib import Path
import re

from crawler.models import CrawlConfig
from crawler.pipeline import run_crawl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="超声讲解视频抓取工具")
    parser.add_argument("--source", choices=["youtube", "bilibili"], required=True, help="数据源")
    parser.add_argument("--max-results", type=int, default=60, help="最多保留视频数")
    parser.add_argument("--search-per-term", type=int, default=20, help="YouTube 每关键词检索数量")
    parser.add_argument("--pages-per-term", type=int, default=3, help="B站每关键词检索页数")
    parser.add_argument("--download-media", action="store_true", help="下载音视频文件")
    parser.add_argument("--no-download-media", action="store_true", help="不下载音视频文件")
    parser.add_argument("--download-thumbnails", action="store_true", help="下载缩略图")
    parser.add_argument("--no-download-thumbnails", action="store_true", help="不下载缩略图")
    parser.add_argument("--download-timeout-sec", type=int, default=60, help="视频下载超时时间(秒)")
    parser.add_argument("--keywords", default="", help="自定义关键词，支持逗号/分号/换行分隔")
    parser.add_argument("--output-root", default="output", help="输出目录")
    return parser


def parse_keywords(raw: str) -> list[str] | None:
    if not raw.strip():
        return None
    parts = [x.strip() for x in re.split(r"[\n,，;；|]+", raw) if x.strip()]
    if not parts:
        return None
    return list(dict.fromkeys(parts))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    download_media = True
    if args.no_download_media:
        download_media = False
    elif args.download_media:
        download_media = True

    download_thumbnails = True
    if args.no_download_thumbnails:
        download_thumbnails = False
    elif args.download_thumbnails:
        download_thumbnails = True

    config = CrawlConfig(
        source=args.source,
        max_results=max(1, args.max_results),
        search_per_term=max(1, args.search_per_term),
        pages_per_term=max(1, args.pages_per_term),
        download_media=download_media,
        download_thumbnails=download_thumbnails,
        download_timeout_sec=max(10, args.download_timeout_sec),
        custom_keywords=parse_keywords(args.keywords or ""),
        output_root=Path(args.output_root),
    )
    summary = run_crawl(config)

    metrics = summary["metrics"]
    print(f"source: {summary['source']}")
    print(f"run_dir: {summary['run_dir']}")
    print(f"kept_records: {metrics['kept_records']}")
    print(f"filtered_records: {metrics['filtered_records']}")
    print(f"failures: {metrics['failures']}")
    print(f"with_media_files: {metrics['with_media_files']}")
    print(f"with_thumbnails: {metrics['with_thumbnails']}")


if __name__ == "__main__":
    main()
