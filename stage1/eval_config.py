"""Resolve evaluation inputs from the training run without loading model libraries."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DEFAULTS = {
    "model_name": "Qwen/Qwen3.5-2B",
    "frame_budget": 30,
    "frame_size": 224,
    "frame_sampling": "recent_sparse",
    "recent_seconds": 15.0,
    "max_asr_chars": 4000,
    "dtype": "bf16",
}


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--training-config", help="Training run's training_config.json; "
                        "auto-detected beside an adapter's final/checkpoint directory")
    parser.add_argument("--model-name")
    parser.add_argument("--frame-budget", type=int)
    parser.add_argument("--frame-size", type=int)
    parser.add_argument("--frame-sampling", choices=("uniform", "recent_sparse"))
    parser.add_argument("--recent-seconds", type=float)
    parser.add_argument("--max-asr-chars", type=int)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"))


def resolve_input_settings(args: argparse.Namespace) -> dict:
    """Mutate args to effective settings; reject overrides of a saved experiment."""
    path = Path(args.training_config) if args.training_config else None
    adapter = getattr(args, "adapter_path", None)
    if path is None and adapter:
        path = Path(adapter).parent / "training_config.json"
    saved = None
    if path is not None:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing training configuration: {path}; pass --training-config "
                "when evaluating a relocated adapter")
        saved = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(saved, dict):
            raise ValueError("training_config.json must contain a JSON object")
        # A single-GPU OOM retry may have reduced the requested frame budget.
        if saved.get("active_frame_budget") is not None:
            saved = {**saved, "frame_budget": saved["active_frame_budget"]}
        missing = sorted(set(DEFAULTS) - saved.keys())
        if missing:
            raise ValueError(f"Training configuration is missing input settings: {missing}")
    for key, default in DEFAULTS.items():
        requested = getattr(args, key, None)
        if saved is not None:
            value = saved[key]
            if requested is not None and requested != value:
                raise ValueError(
                    f"Evaluation {key}={requested!r} differs from training {value!r}; "
                    "remove the override to use the saved training setting")
        else:
            value = default if requested is None else requested
        setattr(args, key, value)
    if any(type(getattr(args, key)) is not int or getattr(args, key) <= 0
           for key in ("frame_budget", "frame_size")):
        raise ValueError("Frame budget and frame size must be positive integers")
    if (args.frame_sampling not in {"uniform", "recent_sparse"}
            or not isinstance(args.recent_seconds, (int, float))
            or not math.isfinite(args.recent_seconds) or args.recent_seconds <= 0
            or type(args.max_asr_chars) is not int
            or args.dtype not in {"bf16", "fp16", "fp32"}
            or not isinstance(args.model_name, str) or not args.model_name.strip()):
        raise ValueError("Invalid saved evaluation input settings")
    args.training_config = str(path.resolve()) if path is not None else None
    return {key: getattr(args, key) for key in DEFAULTS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(parser)
    parser.add_argument("--adapter-path")
    args = parser.parse_args()
    settings = resolve_input_settings(args)
    print(json.dumps({"training_config": args.training_config, **settings}, indent=2))


if __name__ == "__main__":
    main()
