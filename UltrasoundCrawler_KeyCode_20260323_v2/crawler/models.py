from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class CrawlConfig:
    source: str
    max_results: int = 60
    search_per_term: int = 20
    pages_per_term: int = 3
    download_media: bool = True
    download_thumbnails: bool = True
    output_root: Path = Path("output")
    min_delay_sec: float = 0.8
    max_delay_sec: float = 1.6
    max_retries: int = 3
    download_timeout_sec: int = 60
    custom_keywords: list[str] | None = None
    log_callback: Callable[[str], None] | None = None


@dataclass
class VideoRecord:
    source_platform: str
    language: str
    video_id: str
    url: str
    title: str
    description: str = ""
    channel: str = ""
    channel_id: str = ""
    channel_url: str = ""
    uploader: str = ""
    uploader_id: str = ""
    publish_time: str = ""
    duration_sec: int = 0
    view_count: int = 0
    like_count: int = 0
    comment_count: int = 0
    tags: list[str] = field(default_factory=list)
    chapters: list[dict[str, Any]] = field(default_factory=list)
    subtitles_langs: list[str] = field(default_factory=list)
    category_pred: str = "待人工分类"
    category_reason: str = ""
    ultrasound_visual_likely: bool = True
    explanation_available: bool = True
    seed_source: str = ""
    thumbnail_url: str = ""
    thumbnail_file: str = ""
    media_file: str = ""
    audio_video_paired: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "source_platform": self.source_platform,
            "language": self.language,
            "video_id": self.video_id,
            "url": self.url,
            "title": self.title,
            "description": self.description,
            "channel": self.channel,
            "channel_id": self.channel_id,
            "channel_url": self.channel_url,
            "uploader": self.uploader,
            "uploader_id": self.uploader_id,
            "publish_time": self.publish_time,
            "duration_sec": self.duration_sec,
            "view_count": self.view_count,
            "like_count": self.like_count,
            "comment_count": self.comment_count,
            "tags": self.tags,
            "chapters": self.chapters,
            "subtitles_langs": self.subtitles_langs,
            "category_pred": self.category_pred,
            "category_reason": self.category_reason,
            "ultrasound_visual_likely": self.ultrasound_visual_likely,
            "explanation_available": self.explanation_available,
            "seed_source": self.seed_source,
            "thumbnail_url": self.thumbnail_url,
            "thumbnail_file": self.thumbnail_file,
            "media_file": self.media_file,
            "audio_video_paired": self.audio_video_paired,
        }
        payload.update(self.extra)
        return payload


@dataclass
class CrawlResult:
    source: str
    records: list[VideoRecord] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    filtered: list[dict[str, Any]] = field(default_factory=list)
    raw_seeds: list[dict[str, Any]] = field(default_factory=list)
    raw_details: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
