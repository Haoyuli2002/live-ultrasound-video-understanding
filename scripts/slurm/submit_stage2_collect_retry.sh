#!/usr/bin/env bash
# Retry only selected inactive shards; leave old jobs and snapshots untouched.
set -euo pipefail
export STAGE2_RUN_DIR=${STAGE2_RUN_DIR:?Set STAGE2_RUN_DIR to the existing full run}
SOURCE_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$SOURCE_REPO"
[[ $# -gt 0 ]] || { echo 'Usage: submit_stage2_collect_retry.sh <shard> [shard ...] [--dry-run]' >&2; exit 2; }
SHARDS=()
PREVIEW=()
for arg in "$@"; do
  if [[ "$arg" == --dry-run ]]; then PREVIEW=(--dry-run)
  elif [[ "$arg" =~ ^[0-9]+$ ]]; then SHARDS+=("$arg")
  else echo "Invalid shard: $arg" >&2; exit 2
  fi
done
[[ ${#SHARDS[@]} -gt 0 && -f "$STAGE2_RUN_DIR/manifest.json" ]] || exit 2
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-ultrasound}"
REVISION=$(git rev-parse HEAD)
git cat-file -e "$REVISION:stage2/raw_collection.py"
SNAPSHOT="$STAGE2_RUN_DIR/code_keepall_${REVISION:0:12}"
if [[ ! -d "$SNAPSHOT" ]]; then
  TEMP_SNAPSHOT=$(mktemp -d "$STAGE2_RUN_DIR/.code_keepall.XXXXXX")
  trap 'if [[ -n "${TEMP_SNAPSHOT:-}" ]]; then rm -rf -- "$TEMP_SNAPSHOT"; fi' EXIT
  git archive "$REVISION" stage2 scripts | tar -x -C "$TEMP_SNAPSHOT"
  printf '%s\n' "$REVISION" > "$TEMP_SNAPSHOT/snapshot_commit.txt"
  mv "$TEMP_SNAPSHOT" "$SNAPSHOT"
  TEMP_SNAPSHOT=''
fi
[[ $(cat "$SNAPSHOT/snapshot_commit.txt") == "$REVISION" ]] || exit 2
export REPO="$SNAPSHOT"
export STAGE2_KEEP_ALL=1
export PYTHONUNBUFFERED=1
cd "$SNAPSHOT"
python -m stage2.full_run submit --repo "$REPO" --run-dir "$STAGE2_RUN_DIR" \
  --retry-shards "${SHARDS[@]}" --keep-all --concurrency "${STAGE2_CONCURRENCY:-1}" \
  --walltime "${STAGE2_WALLTIME:-04:00:00}" --gpu-type "${GPU_TYPE:-h100}" "${PREVIEW[@]}"
echo "Candidate data: $STAGE2_RUN_DIR/blocks.unreviewed.jsonl"
echo "Quality report: $STAGE2_RUN_DIR/quality_report.json"
