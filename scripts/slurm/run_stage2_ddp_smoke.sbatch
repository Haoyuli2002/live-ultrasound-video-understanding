#!/bin/bash
#SBATCH --job-name=stage2-ddp-smoke
#SBATCH --partition=mcml-hgx-h100-94x4
#SBATCH --qos=mcml
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/stage2_ddp_smoke_%j.out
#SBATCH --error=logs/stage2_ddp_smoke_%j.err

# Real Qwen/LoRA/NCCL two-GPU smoke. Uses two real videos and synthetic labels:
# one full minute (6 updates and long-memory branch), one ten-second interval
# (1 update, then DDP Join). Nothing is written to the source video or labels.
# Run `mkdir -p logs` in REPO before sbatch, since Slurm opens logs first.

set -euo pipefail

REPO=${REPO:-/dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding}
DATA=${DATA:-/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding}
VIDEO_MAP=${VIDEO_MAP:-$DATA/cluster_data/stage2/train_full295_video_paths.json}
STAGE1_ADAPTER=${STAGE1_ADAPTER:-}
MODEL=${MODEL:-Qwen/Qwen3.5-2B}
SMOKE_PARENT=${SMOKE_PARENT:-$DATA/results/stage2_ddp_smoke}
SMOKE_ROOT=$SMOKE_PARENT/${SLURM_JOB_ID:?This script must run inside Slurm}
SMOKE_VIDEO_IDS=${SMOKE_VIDEO_IDS:-}

cd "$REPO"
if [[ -f /dss/dsshome1/04/ge75vid2/miniconda3/etc/profile.d/conda.sh ]]; then
  source /dss/dsshome1/04/ge75vid2/miniconda3/etc/profile.d/conda.sh
else
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate ultrasound

export HF_HOME=${HF_HOME:-$DATA/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$SMOKE_PARENT"

[[ -f "$VIDEO_MAP" ]] || { echo "[error] VIDEO_MAP not found: $VIDEO_MAP" >&2; exit 1; }
if [[ -n "$STAGE1_ADAPTER" && ! -d "$STAGE1_ADAPTER" ]]; then
  echo "[error] STAGE1_ADAPTER not found: $STAGE1_ADAPTER" >&2
  exit 1
fi
[[ ! -e "$SMOKE_ROOT" ]] || { echo "[error] smoke output already exists: $SMOKE_ROOT" >&2; exit 1; }

echo "[smoke] host=$(hostname) job=$SLURM_JOB_ID model=$MODEL"
echo "[smoke] Stage 1 adapter=${STAGE1_ADAPTER:-<from-base test>}"
nvidia-smi -L

prepare_args=(--video-path-map "$VIDEO_MAP" --output-dir "$SMOKE_ROOT")
if [[ -n "${SMOKE_VIDEO_IDS// /}" ]]; then
  read -r -a ids <<< "$SMOKE_VIDEO_IDS"
  if [[ ${#ids[@]} != 2 ]]; then
    echo "[error] SMOKE_VIDEO_IDS must contain exactly two space-separated IDs" >&2
    exit 1
  fi
  prepare_args+=(--video-id "${ids[0]}" --video-id "${ids[1]}")
fi
python -m stage2.prepare_ddp_smoke "${prepare_args[@]}"

train_args=(
  --model-name "$MODEL"
  --train-jsonl "$SMOKE_ROOT/smoke_blocks.jsonl"
  --video-path-map "$SMOKE_ROOT/smoke_video_paths.json"
  --output-dir "$SMOKE_ROOT/training"
  --epochs 1
  --frame-size 224
  --learning-rate 1e-4
  --validation-fraction 0
  --warmup-ratio 0
  --verify-ddp-gradients
)
if [[ -n "$STAGE1_ADAPTER" ]]; then
  train_args+=(--init-adapter "$STAGE1_ADAPTER")
fi
torchrun --standalone --nnodes=1 --nproc_per_node=2 -m stage2.train "${train_args[@]}"

python - "$SMOKE_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
train = root / "training"
manifest = json.loads((root / "manifest.json").read_text())
config = json.loads((train / "training_config.json").read_text())
assert config["world_size"] == 2, config["world_size"]
metrics = [
    [json.loads(line) for line in
     (train / f"training_metrics.rank{rank}.jsonl").read_text().splitlines()]
    for rank in range(2)
]
assert [len(items) for items in metrics] == [6, 1]
assert metrics[0][-1]["block_loss"] is not None
assert metrics[0][-1]["global"] is not None
assert (train / "epoch-1" / "trainer_state.pt").is_file()
assert (train / "final" / "adapter_config.json").is_file()
print(f"[smoke] PASS: 2 ranks, 6+1 updates, gradient check, minute update, "
      f"uneven Join, checkpoint; output={root}")
print(f"[smoke] videos: {manifest['full_minute_video_id']} "
      f"and {manifest['ten_second_video_id']}")
PY
