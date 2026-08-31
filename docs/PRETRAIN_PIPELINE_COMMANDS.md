# Pretrain Pipeline Commands

当前目标：从 YouTube 超声教学视频构建 Qwen3-VL pretrain 数据，并训练 V4 mixedmask LoRA。

集群 repo：

```bash
cd /dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding
source /dss/dsshome1/04/ge75vid2/miniconda3/etc/profile.d/conda.sh
conda activate ultrasound
```

当前 `cluster_data` 已软链接到 scratch：

```bash
readlink -f cluster_data
# /dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding/cluster_data
```

---

## 1. 视频爬取

入口：

```text
UltrasoundCrawler_KeyCode_20260323_v2/cli.py
```

示例：

```bash
cd /dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding/UltrasoundCrawler_KeyCode_20260323_v2

python cli.py \
  --source youtube \
  --max-results 180 \
  --search-per-term 60 \
  --download-media \
  --download-timeout-sec 300 \
  --keywords "ultrasound guided central line,ultrasound guided nerve block,ultrasound guided procedure,vascular access ultrasound,DVT ultrasound tutorial,renal ultrasound scan,renal POCUS,IVC ultrasound scan,FAST exam ultrasound,eFAST ultrasound,cardiac ultrasound apical view,parasternal long axis ultrasound,lung ultrasound pneumothorax,lung ultrasound pleural effusion,gallbladder ultrasound tutorial,OB ultrasound basics,soft tissue ultrasound,MSK ultrasound tutorial" \
  --output-root /dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding/crawl_runs \
  2>&1 | tee /dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding/crawl_runs/crawl_$(date +%Y%m%d_%H%M%S).log
```

输出：

```text
/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding/crawl_runs/<timestamp>_youtube/
  videos.jsonl
  videos.csv
  summary.json
  failures.json
  media/
  thumbnails/
```

---

## 2. YouTube 补下载（如需要）

如果 crawler 只拿到 metadata，或者下载数太少，可用 `yt-dlp` android client 补下载。

设置当前 crawl run：

```bash
CRAWL_RUN=/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding/crawl_runs/<timestamp>_youtube
```

生成去重后的缺失下载列表：

```bash
cd /dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding

python - <<'PY' > /tmp/missing_new_youtube_urls.tsv
import json
from pathlib import Path
import os, sys

run = Path(os.environ["CRAWL_RUN"])
data_root = Path("cluster_data")
crawl_root = Path("/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding/crawl_runs")

existing = set()
for p in (data_root / "videos").rglob("*"):
    if p.is_file() and p.suffix.lower() in {".mp4", ".webm", ".mkv", ".avi", ".mov"}:
        existing.add(p.stem)
for p in (data_root / "splits").glob("*_videos.json"):
    try:
        m = json.load(open(p, encoding="utf-8"))
        existing.update(m.keys())
        existing.update(Path(v).stem for v in m.values())
    except Exception:
        pass
for media in crawl_root.glob("*_youtube/media"):
    for p in media.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".mp4", ".webm", ".mkv", ".avi", ".mov"}:
            existing.add(p.stem)

for line in open(run / "videos.jsonl", encoding="utf-8"):
    if not line.strip():
        continue
    r = json.loads(line)
    vid = r.get("video_id")
    url = r.get("url") or (f"https://www.youtube.com/watch?v={vid}" if vid else None)
    category = r.get("category_pred") or r.get("category") or "uncategorized"
    if not vid or not url or vid in existing:
        continue
    print(f"{vid}\t{category}\t{url}")
PY

wc -l /tmp/missing_new_youtube_urls.tsv
head /tmp/missing_new_youtube_urls.tsv
```

批量补下载：

```bash
while IFS=$'\t' read -r vid category url; do
  echo "==== downloading $vid [$category] ===="
  mkdir -p "$CRAWL_RUN/media/$category"

  yt-dlp \
    --no-update \
    --no-playlist \
    --merge-output-format mp4 \
    -f "bv*+ba/best" \
    --extractor-args "youtube:player_client=android" \
    --socket-timeout 30 \
    --retries 5 \
    --fragment-retries 5 \
    -o "$CRAWL_RUN/media/$category/${vid}.%(ext)s" \
    "$url" || echo -e "$vid\t$url" >> "$CRAWL_RUN/manual_download_failed_android.tsv"
done < /tmp/missing_new_youtube_urls.tsv
```

---

## 3. 视频统计

```bash
cd /dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding

python - <<'PY'
from pathlib import Path
import json

data_root = Path("cluster_data")
crawl_root = Path("/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding/crawl_runs")
video_exts = {".mp4", ".webm", ".mkv", ".avi", ".mov"}

official = {p.stem: p for p in (data_root / "videos").rglob("*") if p.is_file() and p.suffix.lower() in video_exts}
crawl = {}
for media in crawl_root.glob("*_youtube/media"):
    for p in media.rglob("*"):
        if p.is_file() and p.suffix.lower() in video_exts:
            crawl.setdefault(p.stem, p)

all_unique = dict(official)
for vid, p in crawl.items():
    all_unique.setdefault(vid, p)

print("official videos in cluster_data/videos:", len(official))
print("downloaded videos in crawl_runs:", len(crawl))
print("unique total videos:", len(all_unique))

for run in sorted(crawl_root.glob("*_youtube")):
    n = len([p for p in (run / "media").rglob("*") if p.is_file() and p.suffix.lower() in video_exts])
    print(f"{run.name}: {n}")
PY
```

---

## 4. 构建 train/eval split + 转码

入口：

```text
scripts/data/select_crawl_videos.py
```

当前 full295 目标：

```text
train = 250
eval  = 45
```

运行建议使用 `tmux`，避免 SSH 断开：

```bash
tmux new -s select_full295
```

在 tmux 内运行：

```bash
cd /dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding
source /dss/dsshome1/04/ge75vid2/miniconda3/etc/profile.d/conda.sh
conda activate ultrasound

python scripts/data/select_crawl_videos.py \
  --crawl-root /dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding/crawl_runs \
  --skip-vlm-filter \
  --include-existing-videos \
  --data-root cluster_data \
  --dataset-name full295 \
  --train-count 250 \
  --eval-count 45 \
  --mode transcode \
  --no-scan-existing
```

该步骤会：

```text
1. 从 cluster_data/videos 和 crawl_runs/*/media 收集视频
2. 按 video_id 去重
3. 生成 train_full295 / eval_full295
4. 转成 Qwen3/OpenCV 兼容 mp4：H.264 + yuv420p + AAC
```

断开 tmux：

```text
Ctrl+b, 然后 d
```

重新进入：

```bash
tmux attach -t select_full295
```

检查：

```bash
find cluster_data/videos/train_full295 -name "*.mp4" | wc -l
find cluster_data/videos/eval_full295 -name "*.mp4" | wc -l
cat cluster_data/splits/full295_manifest.json
```

期望：

```text
250
45
```

输出：

```text
cluster_data/videos/train_full295/
cluster_data/videos/eval_full295/
cluster_data/splits/train_full295_videos.json
cluster_data/splits/eval_full295_videos.json
cluster_data/splits/full295_manifest.json
```

---

## 5. ASR + Clipping + 构建 Pretrain Samples

入口：

```text
scripts/slurm/prepare_extra_pretrain_data.slurm
QA/prepare/run_prepare.py
pretrain/build_samples.py
```

提交 job：

```bash
cd /dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding

DATASET_NAME=full295 \
WHISPER_MODEL=large-v3 \
LANGUAGE=en \
sbatch scripts/slurm/prepare_extra_pretrain_data.slurm
```

这个 job 会：

```text
1. 检查并修复视频 OpenCV 可读性
2. 对 train_full295 / eval_full295 跑 ASR
3. 跑 clipping
4. 构建 V4 mixedmask pretrain samples
```

输出：

```text
cluster_data/QA/train_full295/transcripts/
cluster_data/QA/train_full295/clips/
cluster_data/QA/eval_full295/transcripts/
cluster_data/QA/eval_full295/clips/

cluster_data/pretrain/train_full295_punct_sentence_mixedmask_samples.jsonl
cluster_data/pretrain/eval_full295_punct_sentence_mixedmask_samples.jsonl
```

监控：

```bash
squeue -u $USER
tail -f logs/prep_extra_pretrain_<JOBID>.out
tail -n 100 logs/prep_extra_pretrain_<JOBID>.err
```

检查完成：

```bash
find cluster_data/QA/train_full295/transcripts -name "*.json" | wc -l
find cluster_data/QA/eval_full295/transcripts -name "*.json" | wc -l

wc -l \
  cluster_data/pretrain/train_full295_punct_sentence_mixedmask_samples.jsonl \
  cluster_data/pretrain/eval_full295_punct_sentence_mixedmask_samples.jsonl
```

---

## 6. Pretrain 训练 + Eval

入口：

```text
scripts/slurm/pretrain_v4_mixedmask_train_eval_full.slurm
pretrain/train.py
pretrain/infer.py
pretrain/compare_predictions.py
```

提交 full295 训练：

```bash
RUN_NAME=pretrain_v4_mixedmask_qwen3vl_2b_full295 \
BUILD_SAMPLES=0 \
TRAIN_JSONL=cluster_data/pretrain/train_full295_punct_sentence_mixedmask_samples.jsonl \
EVAL_JSONL=cluster_data/pretrain/eval_full295_punct_sentence_mixedmask_samples.jsonl \
TRAIN_MAP=cluster_data/splits/train_full295_videos.json \
EVAL_MAP=cluster_data/splits/eval_full295_videos.json \
sbatch scripts/slurm/pretrain_v4_mixedmask_train_eval_full.slurm
```

该 job 会：

```text
1. 用 full295 train samples 训练 Qwen3-VL-2B LoRA
2. 对 eval_full295 跑 base inference
3. 对 eval_full295 跑 LoRA inference
4. 计算 direct metrics 对比
```

输出：

```text
cluster_data/experiments/pretrain/pretrain_v4_mixedmask_qwen3vl_2b_full295/
  checkpoints/adapter/
  eval/base_full_nomask.jsonl
  eval/lora_full_nomask.jsonl
  eval/compare_full_nomask_direct.jsonl
  eval/compare_full_nomask_direct_summary.json
  logs/
```

监控：

```bash
squeue -u $USER
tail -f logs/v4_mix_train_eval_<JOBID>.out
tail -n 100 logs/v4_mix_train_eval_<JOBID>.err
```

查看 summary：

```bash
cat cluster_data/experiments/pretrain/pretrain_v4_mixedmask_qwen3vl_2b_full295/eval/compare_full_nomask_direct_summary.json
```

---

## 7. 常用状态检查

查看 Slurm job：

```bash
squeue -u $USER
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,ExitCode
```

取消 job：

```bash
scancel <JOBID>
```

检查集群空间：

```bash
df -h /dss/mcmlscratch
du -sh cluster_data
```

检查 split：

```bash
find cluster_data/videos/train_full295 -name "*.mp4" | wc -l
find cluster_data/videos/eval_full295 -name "*.mp4" | wc -l
```
