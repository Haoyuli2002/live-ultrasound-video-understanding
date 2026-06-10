"""Smoke test: call google/gemini-2.5-pro on OpenRouter with image + JSON prompt."""

import os
import json
import base64
import sys
import re
from pathlib import Path

# Auto-load .env from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _env_loader  # noqa: F401

import cv2
import numpy as np
from openai import OpenAI


def main():
    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY not set")

    # Synthetic image
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    b64 = base64.b64encode(buf).decode()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        default_headers={
            "HTTP-Referer": "https://github.com/Haoyuli2002/live-ultrasound-video-understanding",
            "X-Title": "Ultrasound QA Validator Test",
        },
    )

    print("Calling google/gemini-2.5-pro with 1 image + JSON output prompt...")
    resp = client.chat.completions.create(
        model="google/gemini-2.5-pro",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            'Look at the image. Output STRICTLY this JSON object '
                            'and nothing else: '
                            '{"verdict": "pass", "reason": "smoke test"}'
                        ),
                    },
                ],
            }
        ],
        temperature=0.1,
        max_tokens=800,  # Gemini 2.5 Pro is a reasoning model; needs CoT headroom
    )

    raw = resp.choices[0].message.content
    tokens = resp.usage.total_tokens if resp.usage else 0
    print(f"tokens: {tokens}")
    print(f"raw: {raw!r}")

    try:
        parsed = json.loads(raw)
    except Exception:
        m = re.search(r"\{.+\}", raw or "", re.DOTALL)
        parsed = json.loads(m.group(0)) if m else None

    if parsed is None:
        print("FAIL: could not parse JSON")
        sys.exit(1)
    print("parsed JSON:", parsed)
    print("OK")


if __name__ == "__main__":
    main()