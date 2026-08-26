#!/bin/bash
#SBATCH --job-name=prep_extra_pretrain
#SBATCH --partition=mcml-hgx-h100-94x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_DIR="/dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding"
cd "${REPO_DIR}"

# DATA_ROOT should usually be the cluster_data symlink that points to mcmlscratch.
DATA_ROOT="${DATA_ROOT:-cluster_data}"
DATASET_NAME="${DATASET_NAME:-extra100}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train_${DATASET_NAME}}"
EVAL_SPLIT="${EVAL_SPLIT:-eval_${DATASET_NAME}}"

TRAIN_MAP="${DATA_ROOT}/splits/${TRAIN_SPLIT}_videos.json"
EVAL_MAP="${DATA_ROOT}/splits/${EVAL_SPLIT}_videos.json"
TRAIN_QA_DIR="${DATA_ROOT}/QA/${TRAIN_SPLIT}"
EVAL_QA_DIR="${DATA_ROOT}/QA/${EVAL_SPLIT}"
PRETRAIN_DIR="${DATA_ROOT}/pretrain"

TRAIN_SAMPLES="${PRETRAIN_DIR}/${TRAIN_SPLIT}_punct_sentence_mixedmask_samples.jsonl"
EVAL_SAMPLES="${PRETRAIN_DIR}/${EVAL_SPLIT}_punct_sentence_mixedmask_samples.jsonl"

WHISPER_MODEL="${WHISPER_MODEL:-large-v3}"
LANGUAGE="${LANGUAGE:-en}"
export TRAIN_MAP EVAL_MAP

source /dss/dsshome1/04/ge75vid2/miniconda3/etc/profile.d/conda.sh
conda activate ultrasound

mkdir -p logs "${TRAIN_QA_DIR}" "${EVAL_QA_DIR}" "${PRETRAIN_DIR}"

echo "======================================================================"
echo "[job] Prepare extra pretrain data"
echo "[job] job_id=${SLURM_JOB_ID:-unknown}"
echo "[job] node=${SLURMD_NODENAME:-unknown}"
echo "[job] start_time=$(date)"
echo "[job] repo=${REPO_DIR}"
echo "[job] DATA_ROOT=${DATA_ROOT}"
echo "[job] DATASET_NAME=${DATASET_NAME}"
echo "[job] TRAIN_MAP=${TRAIN_MAP}"
echo "[job] EVAL_MAP=${EVAL_MAP}"
echo "[job] WHISPER_MODEL=${WHISPER_MODEL}"
echo "======================================================================"

python - <<'PY'
import torch
print("[env] torch:", torch.__version__)
print("[env] cuda:", torch.cuda.is_available())
print("[env] gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

echo "== check split maps =="
test -f "${TRAIN_MAP}"
test -f "${EVAL_MAP}"
python - <<'PY'
import json, os
for name, path in [("train", os.environ["TRAIN_MAP"]), ("eval", os.environ["EVAL_MAP"])]:
    data = json.load(open(path, encoding="utf-8"))
    print(f"[split] {name}: {len(data)} videos from {path}")
PY

echo "== verify/transcode videos for Qwen3/OpenCV compatibility =="
python scripts/transcode_videos_for_opencv.py \
  --video-path-map "${TRAIN_MAP}" \
  --replace \
  --delete-backup \
  --quiet-opencv
python scripts/transcode_videos_for_opencv.py \
  --video-path-map "${EVAL_MAP}" \
  --replace \
  --delete-backup \
  --quiet-opencv

prepare_split() {
  local split_name="$1"
  local map_path="$2"
  local output_dir="$3"

  echo "======================================================================"
  echo "[prepare] split=${split_name} map=${map_path} output=${output_dir}"
  echo "======================================================================"

  python - <<PY
import json
with open("${map_path}", encoding="utf-8") as f:
    items = list(json.load(f).items())
for video_id, path in items:
    print(f"{video_id}\t{path}")
PY

  python - <<PY | while IFS=$'\t' read -r video_id video_path; do
import json
with open("${map_path}", encoding="utf-8") as f:
    items = list(json.load(f).items())
for video_id, path in items:
    print(f"{video_id}\t{path}")
PY
    transcript="${output_dir}/transcripts/${video_id}.json"
    clips="${output_dir}/clips/${video_id}_clips.json"
    if [[ -f "${transcript}" && -f "${clips}" ]]; then
      echo "[prepare] skip existing ${video_id}"
      continue
    fi

    echo "[prepare] ${video_id}: ${video_path}"
    python QA/prepare/run_prepare.py \
      --video "${video_path}" \
      --output-dir "${output_dir}" \
      --whisper-model "${WHISPER_MODEL}" \
      --language "${LANGUAGE}" \
      --no-llm-clipping
  done
}

prepare_split "${TRAIN_SPLIT}" "${TRAIN_MAP}" "${TRAIN_QA_DIR}"
prepare_split "${EVAL_SPLIT}" "${EVAL_MAP}" "${EVAL_QA_DIR}"

echo "== build train V4 mixedmask samples =="
python pretrain/build_samples.py \
  --transcripts "${TRAIN_QA_DIR}/transcripts" \
  --output "${TRAIN_SAMPLES}" \
  --unit sentence \
  --sentence-mode punctuation \
  --format mixedmask \
  --history-units 3 \
  --frames-per-sentence 3 \
  --min-words 3 \
  --sentence-max-words 80 \
  --split-punctuation '.?!;:' \
  --mask-current-visual-prob 0.33 \
  --no-mask-prob 0.33 \
  --random-unit-modality-mask-prob 0.34

echo "== build eval V4 mixedmask samples =="
python pretrain/build_samples.py \
  --transcripts "${EVAL_QA_DIR}/transcripts" \
  --output "${EVAL_SAMPLES}" \
  --unit sentence \
  --sentence-mode punctuation \
  --format mixedmask \
  --history-units 3 \
  --frames-per-sentence 3 \
  --min-words 3 \
  --sentence-max-words 80 \
  --split-punctuation '.?!;:' \
  --mask-current-visual-prob 0.33 \
  --no-mask-prob 0.33 \
  --random-unit-modality-mask-prob 0.34

echo "== summary =="
echo "train transcripts: $(find "${TRAIN_QA_DIR}/transcripts" -name '*.json' 2>/dev/null | wc -l)"
echo "eval transcripts : $(find "${EVAL_QA_DIR}/transcripts" -name '*.json' 2>/dev/null | wc -l)"
wc -l "${TRAIN_SAMPLES}" "${EVAL_SAMPLES}"
echo "[job] finished_time=$(date)"