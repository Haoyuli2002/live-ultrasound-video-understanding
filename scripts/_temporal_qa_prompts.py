"""
Prompt templates for scripts/temporal_qa_generation.py.

Single-stage design: the MLLM receives the FULL clip (and optional ASR) and
directly produces temporal QA pairs grounded in the clip — no separate
event-extraction stage.

Supported qa_type values:
    - given_when_ask_what
    - given_what_ask_when
    - visible_anatomy
    - next_action_guidance
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Temporal QA generation prompt
# ---------------------------------------------------------------------------

QA_SYSTEM_PROMPT = (
    "You are generating temporal online QA for ultrasound video understanding.\n"
    "You will receive a single ultrasound video clip and optionally its narration.\n"
    "\n"
    "Produce a small, high-quality set of QA pairs grounded in the clip.\n"
    "Each QA pair must include a `query_time` (in absolute seconds, measured\n"
    "from the start of the original full video). At training/evaluation time,\n"
    "the model will only see the video from clip_start up to query_time, so\n"
    "questions and answers must respect that streaming constraint.\n"
    "\n"
    "Supported qa_type values (use only these):\n"
    "  - given_when_ask_what\n"
    "  - given_what_ask_when\n"
    "  - visible_anatomy\n"
    "  - next_action_guidance\n"
    "\n"
    "Per-type semantics:\n"
    "  given_when_ask_what:\n"
    "    - The question describes a time window or moment and asks what is\n"
    "      happening at that time.\n"
    "    - Set query_time to the end of that window.\n"
    "  given_what_ask_when:\n"
    "    - The question describes an event in natural language and asks when\n"
    "      it occurs in the clip.\n"
    "    - The answer MUST include a concrete time range like 'X.Xs - Y.Ys',\n"
    "      with both bounds inside [clip_start, clip_end].\n"
    "    - Set query_time = clip_end (this is a localization task, so the\n"
    "      whole clip is allowed as the seen window).\n"
    "  visible_anatomy:\n"
    "    - The question asks which anatomical structures are visible at the\n"
    "      time of querying.\n"
    "    - Set query_time inside [clip_start, clip_end].\n"
    "  next_action_guidance:\n"
    "    - The question asks what the sonographer should do next, given what\n"
    "      has been seen so far in the clip.\n"
    "    - The answer must be medically reasonable and CAUTIOUS. Do not give\n"
    "      definitive diagnoses. If the current view is insufficient, say so.\n"
    "    - Set query_time inside [clip_start, clip_end].\n"
    "\n"
    "Strict rules:\n"
    "- query_time MUST be a float (absolute seconds) inside\n"
    "      [clip_start, clip_end].\n"
    "- Never use 0.0 as a placeholder unless clip_start is 0.0.\n"
    "- For all qa_types except given_what_ask_when, the answer must be\n"
    "  derivable from the video up to query_time. Do not cite future-only\n"
    "  findings.\n"
    "- Use cautious medical language. Do not make definitive diagnoses.\n"
    "- Do not invent anatomy / findings that are not visible.\n"
    "- If evidence is insufficient, say so in the answer.\n"
)


_QA_SCHEMA = """\
{
  "qa_pairs": [
    {
      "qa_id":      "q1",
      "qa_type":    "given_when_ask_what",
      "query_time": 0.0,
      "question":   "...",
      "answer":     "...",
      "evidence":   "brief reference to what is visible/audible at this time",
      "asr_used":   false
    }
  ]
}
"""


def build_qa_instruction(clip: dict, asr: str, max_qa: int) -> str:
    """Build the user-side instruction for the single-stage QA generator."""
    asr_block = (asr or "").strip() or "(no narration provided)"
    head = (
        "Clip metadata\n"
        "-------------\n"
        f"video_id   : {clip.get('video_id')}\n"
        f"clip_idx   : {clip.get('clip_idx')}\n"
        f"clip_start : {float(clip['start']):.2f}s\n"
        f"clip_end   : {float(clip['end']):.2f}s\n"
        f"clip_topic : {clip.get('topic', '')}\n"
        "\n"
        "Narration (optional, may be empty)\n"
        "----------------------------------\n"
        f"{asr_block}\n"
        "\n"
        f"Generate AT MOST {max_qa} QA pairs across the supported qa_type\n"
        "values. Try to cover multiple moments inside the clip and use all 4\n"
        "qa_type values when possible. For given_what_ask_when, ensure the\n"
        "answer contains a concrete 'X.Xs - Y.Ys' time range.\n"
        "\n"
        "Return ONLY valid JSON with this exact schema:\n"
        "\n"
    )
    tail = "\nDo NOT include text outside the JSON object."
    return head + _QA_SCHEMA + tail