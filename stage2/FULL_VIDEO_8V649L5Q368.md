# Generate Stage 2 labels for 8V649L5Q368

Run on the LRZ login node after pulling `main`:

```bash
cd ~/haoyu/live-ultrasound-video-understanding
git pull --ff-only origin main
bash scripts/slurm/submit_stage2_8V649L5Q368.sh
```

The submitter finds the original video in the cluster path maps, creates a fresh
run directory, and submits one H100 job. If lookup fails, set
`VIDEO_PATH=/absolute/path/8V649L5Q368.mp4` before rerunning. No API key or external
teacher service is needed; vLLM serves `Qwen/Qwen3.5-27B` on the allocated node.
The `ultrasound` Conda environment must contain a Qwen3.5-compatible vLLM,
Transformers, OpenAI client, OpenCV, and ffmpeg.

The source video is unchanged. Temporary clips have no audio, preserve their
original time interval and aspect ratio, use 1 fps and a maximum side of 512
pixels. Qwen receives at most 120 frames per request, sampled across the entire
window. This bounds long-prefix decoding and visual context. Sparse sampling
can miss fast motion; this run is for inspecting the generated labels, not a
claim that every visual event has been captured. Thinking is disabled via the
Qwen chat template and responses have a 1024-token limit.

For the 1136.8-second source, the current Stage 2 time grid produces:

| Output | Count | Coverage |
| --- | ---: | --- |
| Local 10-second targets | 113 | [0,10) through [1120,1130) |
| Independent 60-second targets | 18 | [0,60) through [1020,1080) |
| Independent cumulative targets | 18 | [0,60) through [0,1080) |
| Training block rows | 19 | 18 complete minutes + one tail |

The final partial minute contains five Local targets; it has no Block/Global
target. The final approximately 6.8 seconds are outside the complete 10-second
grid. A Global target always comes from its own video prefix, not from earlier
text summaries. ASR and target sentences are never sent to the teacher.

Outputs live under the run directory printed by the submitter:

- `teacher/8V649L5Q368.jsonl`: 149 teacher events with facets and sampling config.
- `blocks.jsonl`: 19 chronological training rows, validated by `load_blocks`.
- `audit/8V649L5Q368.jsonl`: completion or failure audit.
- `video_paths.json`, `submit_env.sh`, `job_id.txt`: source mapping and run metadata.

Logs persist in the checkout at `logs/stage2_annotate_JOBID.{out,err}` and
`logs/stage2_vllm_JOBID.log`. Every completed minute is flushed to the teacher
file. To resume a stopped job, first ensure the previous job is no longer
running, then:

```bash
source /absolute/path/to/run/submit_env.sh
bash scripts/slurm/submit_stage2_8V649L5Q368.sh
```

The annotator automatically reorders valid teacher clauses into the schema's
canonical facet order without changing their content. Duplicate/illegal labels,
empty clauses and overlong output still trigger validation retries. Normalized
and invalid raw answers are printed in the annotation log for diagnosis. This
format repair keeps the existing schema and resume settings compatible.

Resumption rejects changed annotation settings and invalid saved chronology.
Already-complete labels, including the tail, are reused without teacher calls.
A failure within a minute reruns that minute. `--dry-run` writes and checks the
submission configuration without calling Slurm; it does not test GPU inference.

CPU checks: `python -m unittest stage2.test_annotation_job stage2.test_stage2`.
Actual Qwen/vLLM inference must be verified in the cluster job.

API references: [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-27B)
and [vLLM multimodal input documentation](https://docs.vllm.ai/en/latest/features/multimodal_inputs/).
