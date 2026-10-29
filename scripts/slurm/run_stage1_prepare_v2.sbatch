#!/bin/bash
#SBATCH --job-name=stage1-data-v2
#SBATCH --partition=lrz-dgx-a100-40x8-mig
#SBATCH --qos=mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/stage1_data_%j.out
#SBATCH --error=logs/stage1_data_%j.err

# Run separately for train and eval maps. This job prepares data; it does not train.
set -euo pipefail
export REPO=${REPO:-/dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding}
export DATA=${DATA:-/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding}
cd "$REPO"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ultrasound
source scripts/slurm/stage1_storage.sh
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false

# CTranslate2 uses CUDA 12/cuDNN 9 independently of the PyTorch CUDA build.
# Prefer the environment's installed CUDA 12 runtime libraries in this job only.
ASR_CUDA_LIB_DIRS=$(python - <<'PY'
from pathlib import Path
import nvidia.cublas
import nvidia.cudnn
paths = [str(Path(p) / "lib") for package in (nvidia.cublas, nvidia.cudnn)
         for p in package.__path__ if (Path(p) / "lib").is_dir()]
print(":".join(paths))
PY
)
export LD_LIBRARY_PATH="$ASR_CUDA_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python - <<'PY'
import ctranslate2
count = ctranslate2.get_cuda_device_count()
if count < 1:
    raise RuntimeError("CTranslate2 cannot see the allocated GPU")
supported = ctranslate2.get_supported_compute_types("cuda", 0)
if "float16" not in supported:
    raise RuntimeError(f"Allocated GPU does not support the requested FP16 ASR: {supported}")
print(f"ASR GPU preflight: visible_devices={count}; compute_types={sorted(supported)}", flush=True)
PY

SPLIT=${SPLIT:-train_full295}
VIDEO_MAP=${VIDEO_MAP:-$DATA/cluster_data/splits/${SPLIT}_qwen35_pretrain_keep_videos.json}
RUN_ROOT=${RUN_ROOT:-$DATA/cluster_data/stage1_datasets/${SPLIT}_${SLURM_JOB_ID}}
ASR_DIR=${ASR_DIR:-$RUN_ROOT/asr}
DATASET_DIR=${DATASET_DIR:-$RUN_ROOT/dataset}
[[ -f "$VIDEO_MAP" ]] || { echo "Missing VIDEO_MAP: $VIDEO_MAP" >&2; exit 1; }

asr_args=(--video-map "$VIDEO_MAP" --video-root "${VIDEO_ROOT:-$REPO}" --output-dir "$ASR_DIR"
          --model "${WHISPER_MODEL:-large-v3}" --device cuda --language "${LANGUAGE:-en}" --resume)
if [[ -n "${LIMIT_VIDEOS:-}" ]]; then asr_args+=(--limit-videos "$LIMIT_VIDEOS"); fi
python -u -m stage1.transcribe "${asr_args[@]}"

# Native Whisper punctuation is the default. LLM boundaries can instead be built
# later with stage1.prepare_dataset --boundary-mode llm against a served endpoint.
python -u -m stage1.prepare_dataset \
  --transcripts "$ASR_DIR" \
  --video-map "$ASR_DIR/selected_video_map.json" \
  --output-dir "$DATASET_DIR"
echo "Dataset: $DATASET_DIR/samples.jsonl"
echo "Video map: $DATASET_DIR/video_map.json"
echo "Review: $DATASET_DIR/report.json and quarantine.jsonl"
echo "Use identical training/eval sampling: --frame-sampling recent_sparse --recent-seconds 15"
