"""Stable names and safety gates for the two Stage 2 supervision variants."""
from __future__ import annotations

from pathlib import Path
import os


DATASET_VERSIONS = ("visual_only", "visual_asr", "visual")

def normalize_version(version: str) -> str:
    if version not in DATASET_VERSIONS:
        raise ValueError(f"Unknown Stage 2 dataset version: {version!r}")
    return "visual_only" if version == "visual" else version

VISUAL_RELEASE_NAME = "accepted_v4_20261105_990b9b48"
VISUAL_ASR_RELEASE_NAME = "accepted_av_asr_v1"
VISUAL_ASR_MODALITIES = {"video", "audio_via_window_asr", "asr"}


def default_release(version: str, repo: str | Path, data_root: str | Path | None = None) -> Path:
    """Return the canonical immutable release location for a dataset version.

    AV+ASR snapshots are frozen separately with stage2.freeze_snapshot.
    Dataset selection never silently falls back to another supervision version.
    """
    version = normalize_version(version)
    repo = Path(repo)
    if version == "visual_only":
        return repo / "stage2" / "releases" / VISUAL_RELEASE_NAME
    if version == "visual_asr":
        override = os.environ.get("VISUAL_ASR_DATASET_RELEASE")
        if override:
            return Path(override).expanduser()
        if data_root:
            return Path(data_root) / "cluster_data" / "stage2" / "releases" / VISUAL_ASR_RELEASE_NAME
        return repo / "cluster_data" / "stage2" / "releases" / VISUAL_ASR_RELEASE_NAME
    raise ValueError(f"Unknown Stage 2 dataset version: {version!r}; choose {DATASET_VERSIONS}")


def validate_dataset_version(manifest: dict, version: str) -> None:
    """Prevent a training run from silently mixing the visual and AV+ASR labels."""
    version = normalize_version(version)
    declared = manifest.get("dataset_version")
    if declared is not None and normalize_version(declared) != version:
        raise ValueError("Manifest dataset_version does not match selected dataset")
    if manifest.get("training_ready") is not True or manifest.get("full_dataset_complete") is not True:
        raise ValueError("Stage 2 training requires a complete frozen release")
    if manifest.get("schema") != "stage2_accepted_release_v1":
        raise ValueError("Unsupported Stage 2 release schema")
    modalities = set(manifest.get("label_modalities", []))
    if version == "visual_only":
        if modalities & {"audio", "asr", "audio_via_window_asr"}:
            raise ValueError("Visual-only training cannot use an AV+ASR release")
    else:
        if modalities != VISUAL_ASR_MODALITIES:
            raise ValueError(
                "Visual+ASR training requires a frozen release declaring exactly "
                "video, audio_via_window_asr, and asr modalities"
            )
