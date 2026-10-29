"""Small, CPU-only helpers for an isolated local teacher service."""
from __future__ import annotations

import argparse
import json
import os
import socket
import urllib.request


def available_port(requested: str = "auto") -> int:
    port = 0 if requested in {"", "auto", "0"} else int(requested)
    if not 0 <= port <= 65535:
        raise ValueError("Invalid vLLM port")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", port))
        return listener.getsockname()[1]


def endpoint_ready(base_url: str, model: str, key: str) -> bool:
    request = urllib.request.Request(base_url.rstrip("/") + "/models",
                                     headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            value = json.load(response)
        return model in {item.get("id") for item in value.get("data", [])}
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    port = sub.add_parser("port")
    port.add_argument("requested", nargs="?", default="auto")
    ready = sub.add_parser("ready")
    ready.add_argument("--base-url", required=True)
    ready.add_argument("--model", required=True)
    ready.add_argument("--api-key-env", default="VLLM_API_KEY")
    args = parser.parse_args()
    if args.action == "port":
        print(available_port(args.requested))
    else:
        raise SystemExit(0 if endpoint_ready(args.base_url, args.model,
                                            os.environ.get(args.api_key_env, "")) else 1)


if __name__ == "__main__":
    main()
