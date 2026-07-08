"""
QA/_shared
==========
Thin re-export layer over the vendor code in ``scripts/`` so the new QA
pipeline can reuse the video I/O + OpenRouter helper without copying it.

This keeps a *single* implementation of ffmpeg cutting, base64 encoding,
retry logic, etc. — the one in ``scripts/_video_llm.py``. If that file
changes, we get the changes for free here.

If you ever remove ``scripts/_video_llm.py``, this module will start
raising ``ImportError`` — that is intentional; you should either put the
helper back or copy it into QA/_shared/ and update the re-exports.

Usage in QA/*.py:
    from _shared import (
        DEFAULT_MODEL,
        build_openrouter_client,
        build_video_block,
        call_with_content,
        cut_clip,
        temp_clip_path,
        text_block,
    )
"""

from __future__ import annotations

import sys
from pathlib import Path

# Locate the sibling `scripts/` directory (two levels up from this file:
# QA/_shared/__init__.py -> QA/ -> repo root -> repo root/scripts/).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"

if not _SCRIPTS_DIR.exists():
    raise RuntimeError(
        f"QA/_shared expected to find sibling scripts/ at {_SCRIPTS_DIR}, "
        f"but the directory does not exist. If you moved things, either "
        f"restore scripts/ or copy _video_llm.py and _env_loader.py into "
        f"QA/_shared/ and update this re-export module."
    )

if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# _env_loader has side effects at import time (loads .env into os.environ),
# so importing it here means every QA/*.py script that does
#   from _shared import ...
# will get the .env variables populated automatically.
import _env_loader  # noqa: F401,E402

from _video_llm import (  # noqa: E402
    DEFAULT_MODEL,
    OPENROUTER_BASE_URL,
    TEMP_ROOT,
    build_openrouter_client,
    build_video_block,
    call_with_content,
    call_with_videos,
    cut_clip,
    temp_clip_path,
    text_block,
    video_to_data_url,
)

__all__ = [
    "DEFAULT_MODEL",
    "OPENROUTER_BASE_URL",
    "TEMP_ROOT",
    "build_openrouter_client",
    "build_video_block",
    "call_with_content",
    "call_with_videos",
    "cut_clip",
    "temp_clip_path",
    "text_block",
    "video_to_data_url",
]