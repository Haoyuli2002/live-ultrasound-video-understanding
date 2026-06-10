"""
Project-wide .env loader.

Import this module from any script that needs API keys; it will load
the .env file at the project root (one level above the scripts/ folder).
The file is searched once per Python process; subsequent imports are no-ops.

Usage:
    import _env_loader   # noqa: F401  (imported for side effect)
"""

from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def _load():
    if load_dotenv is None:
        return
    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


_load()