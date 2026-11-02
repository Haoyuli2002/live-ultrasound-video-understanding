#!/usr/bin/env bash
set -euo pipefail
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$CODE_DIR"
unset STAGE1_ADAPTER STAGE2_ADAPTER
export MODEL_NAME=Qwen/Qwen3.5-2B
export HF_HOME=${HF_HOME:-/dss/mcmlscratch/04/ge75vid2/haoyu/hf_cache}
export HF_HUB_CACHE="$HF_HOME/hub"
unset TRANSFORMERS_CACHE
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
SHORT_TMP=$(mktemp -d /tmp/s2of.XXXXXXXX)
trap 'rm -rf -- "$SHORT_TMP"' EXIT
export TMPDIR="$SHORT_TMP" TMP="$SHORT_TMP" TEMP="$SHORT_TMP"
python -m scripts.gpu_runtime
python -m stage2.train --train-jsonl "$TRAIN_JSONL" --video-path-map "$VIDEO_MAP" \
  --output-dir "$OUTPUT_DIR" --model-name "$MODEL_NAME" \
  --epochs "$EPOCHS" --learning-rate "$LEARNING_RATE" --frame-size "$FRAME_SIZE" --dtype bf16
for checkpoint in epoch-1 final; do
  python -m stage2.evaluate --eval-jsonl "$TRAIN_JSONL" --video-path-map "$VIDEO_MAP" \
    --adapter-path "$OUTPUT_DIR/$checkpoint" \
    --model-name "$MODEL_NAME" --frame-size "$FRAME_SIZE" --output "$RUN_DIR/reconstruction_$checkpoint.jsonl"
done
python - <<'PY'
import json,os,statistics
from pathlib import Path
r=Path(os.environ['RUN_DIR']); out=Path(os.environ['OUTPUT_DIR'])
rows=[json.loads(s) for s in (out/'training_metrics.jsonl').read_text().splitlines() if s.strip()]
expected=12*int(os.environ['EPOCHS'])
assert len(rows)==expected,(len(rows),expected)
assert (out/'final/adapter_config.json').is_file()
summary={'evaluation':'teacher-forced reconstruction on training video; not held-out accuracy or generated-answer quality','initialization':'official_checkpoint','model_name':os.environ['MODEL_NAME'],'init_adapter':None,'optimizer_updates':len(rows),'checkpoints':{}}
for name in ['epoch-1','final']:
 values=[json.loads(s) for s in (r/f'reconstruction_{name}.jsonl').read_text().splitlines() if s.strip()]
 assert len(values)==16,len(values)
 summary['checkpoints'][name]={kind:statistics.mean(v['loss'] for v in values if v['kind']==kind) for kind in ['short','block','global']}
(r/'overfit_report.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2),flush=True)
print('Completed: adapter saved; compare epoch-1 vs final losses. No automatic quality approval.',flush=True)
PY
