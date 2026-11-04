# Short-only and independent-target overfit experiments

These two independent jobs test whether separate reconstruction objectives and
fresh visual memories improve free-running Short decoding. Both initialize
from official `Qwen/Qwen3.5-2B`, with the same random seed, LoRA configuration,
learning rate and frozen P3FoSFIrfQA labels covering [0,120). Neither starts
from the previous overfit adapter or from the other experiment.

| Experiment | One epoch | 20 epochs | Final generated outputs |
|---|---|---|---|
| `short_only` | 12 Short updates | 240 updates | 12 targets × 3 controls = 36 |
| `independent` | 12 Short + 2 Block + 2 Global updates | 320 updates | 16 targets × 3 controls = 48 |

Each target has its own mean token CE loss, backward pass and AdamW step.
There is no gradient accumulation or combined Short/Block/Global loss. Within
each minute, six Short updates precede one Block update and one Global update.
The existing combined-loss trainer and old experiment remain available.

Every update re-encodes its current 10/60 seconds using current parameters.
All current-window visual readouts receive gradients. Before a Global update,
earlier complete minutes are replayed under current weights without gradients;
the previous long memory is detached. The current-minute long update remains
differentiable. This preserves causal, truncated historical gradients without
reusing memories computed before an optimizer step. No target text is used in
visual encoding or long-memory updates.

For this two-minute clip, Short-only encodes 120 frames per epoch. Independent
training encodes 420: 120 for Short, 120 for Block, 120 for current-minute
Global inputs, and 60 for replaying Global history. Frame input stays **224 ×
224 with aspect-ratio padding, one frame per second**, as in the previous
training experiment. This is distinct from native-resolution label review.
Independent training costs more than the old combined-loss run and is not a
pure loss-weighting ablation: update counts, fresh encoding and encoder gradient
coverage also differ. Identical seeds do not guarantee bitwise GPU determinism.

On the cluster:

```bash
(
set -euo pipefail
cd ~/haoyu/live-ultrasound-video-understanding
git pull --ff-only origin main
GPU_TYPE=auto EPOCHS=20 WALLTIME=04:00:00 \
  bash scripts/slurm/submit_stage2_overfit_ab.sh
)
```

The script submits **two jobs, each one full H100 or A100 80GB**, eight CPUs,
120GB host RAM, four-hour limit including evaluation. `GPU_TYPE=h100` or
`a100` restricts the partition choice. No MIG GPU is requested. Four hours is
an allocation limit, not a measured runtime or a guarantee. `DRY_RUN=1` validates
and freezes both plans without submitting. Set `VIDEO_FILE` if the default
cluster path cannot resolve the P3 video.

One common committed code snapshot and two copies of identical label inputs
are saved under the printed `Pair:` directory. The script checks both plans
before submitting either job, and saves each job ID immediately. If the second
submission fails, inspect `experiments.json` rather than rerunning blindly and
duplicating the first job. This entry has no optimizer-state resume.

Each job saves epoch-1 and final adapters, evaluates teacher-forced losses at
both checkpoints, and then greedily reconstructs from final real, zero and
other-window memories. Short-only does not train or evaluate Block/Global.
References never enter free-running decoding. A noncausal other-window donor
is a diagnostic control, not valid online evidence. Outputs include:

- `training/training_metrics.jsonl`: one row per optimizer update.
- `reconstruction_epoch-1.jsonl`, `reconstruction_final.jsonl`: teacher-forced losses.
- `reconstruct_final/predictions.jsonl`, `review.txt`, `memory_*.pt`: generated
  texts and actual memory tensors used for diagnosis.
- `comparison.tsv`: full GT / real / zero / other-window horizontal comparison.
- `overfit_report.json`: completeness checks, exact matches, unique outputs,
  wrong-window GT repetition, empty and truncated outputs.

Inspect both experiments after they finish, using the printed Pair path:

```bash
export PAIR=/dss/.../overfit_AB_P3_...
python -m stage2.overfit_ab_report --pair-dir "$PAIR"
cat "$PAIR/short_only/overfit_report.json"
cat "$PAIR/independent/overfit_report.json"
```

The bundled labels retain `draft_debug_only` status. These are training-video
diagnostics; decreasing CE or exact text reconstruction does not establish
medical correctness, visual dependence, or held-out generalization. Inspect
Short collapse and its control sensitivity before claiming improvement.
