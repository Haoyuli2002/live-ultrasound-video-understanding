# Live Ultrasound Video Understanding — 完整 Pipeline

本文件是当前项目端到端流程的权威说明。目标：从 YouTube/Bilibili 超声教学视频，构建数据并训练一个 answerability-aware 的实时超声视频理解模型（Qwen3-VL）。

---

## 0. 总览

```text
1. 视频爬取         UltrasoundCrawler_KeyCode_20260323_v2/
2. 视频过滤         src/video_filter.py / scripts/video_filter_vlm.py
3. ASR 转录         QA/prepare/asr.py         -> transcripts/{video_id}.json
        │
        ├── Pretrain 分支（学超声视觉-语言）
        │     4. build_samples   pretrain/build_samples.py  -> pretrain_samples.jsonl
        │     5. 预训练           pretrain/train.py          (Stage 1, LoRA)
        │
        └── QA / SFT 分支（学 WAIT/ANSWER 决策）
              6. Clipping        QA/prepare/clipping.py     -> clips/{video_id}_clips.json
              7. QA 生成         QA/generator.py / QA/offline_generator.py
              8. QA 校验/合并     QA/validator.py / QA/merger.py -> {video_id}_training_samples.jsonl
              9. SFT             QA/train/train.py          (Stage 2, LoRA)

10. 推理 / 评测     QA/eval/ , pretrain/infer.py
```

关键点：

- Pretrain 分支和 SFT 分支**共享前面的爬取/过滤/ASR**，之后分成两条独立数据流。
- 两条训练分支目前**完全解耦**（`pretrain/` 与 `QA/train/` 无 import 依赖）。是否用 Stage 1 的 adapter 作为 Stage 2 的起点，是可选实验。

---

## 1. 视频爬取

目录：`UltrasoundCrawler_KeyCode_20260323_v2/`

- 从 YouTube / Bilibili 爬取超声教学视频。
- 输出到 `output/.../media/<category>/{video_id}.mp4`。
- 有 Web UI（`webapp.py` / `run_ui.bat`）和 CLI（`cli.py`）。

产物：原始视频 `.mp4`。

---

## 2. 视频过滤

- `src/video_filter.py`
- `scripts/video_filter_vlm.py`
- `scripts/batch_filter.py`

用 VLM 判断视频是否为“有用的超声教学内容”，过滤掉无关视频（纯讲座、广告、非超声等）。

产物：过滤后的视频清单。

---

## 3. ASR 转录

入口：`QA/prepare/asr.py`（复用 `scripts/asr_pipeline.py`）

- ffmpeg 抽音频 → faster-whisper 转写 → 带时间戳的 JSON。
- 模型策略：默认 `medium`；**检测到 GPU 时自动升级 `large-v3`**；`--device auto` 自动选 cuda/cpu；GPU 用 `float16`，CPU 用 `int8`。
- 医学术语需要较大模型，避免污染下游文本。

命令：

```bash
python QA/prepare/asr.py --video path/to/video.mp4 --output-dir QA/results
# 手动：--model large-v3 --device cuda
```

产物：`{output_dir}/transcripts/{video_id}.json`

```json
{
  "video_id": "...",
  "duration_sec": 623.4,
  "segments": [{"start": 4.3, "end": 12.3, "text": "..."}, ...],
  "full_text": "..."
}
```

---

## 4. Pretrain 分支：构造预训练样本

入口：`pretrain/build_samples.py`

- 任务：整句 completion。对每个 ASR segment，看这句开始前最近 N 秒的帧，续写这句超声解说。
- `current_time = segment.start`，`video_window = [max(0, start - window_sec), start]`。
- 带 `prev_context`（前文解说，可截断，可 `--no-context` 关闭）。
- 过滤空句、过短句、`[music]/[applause]` 等。

命令：

```bash
python pretrain/build_samples.py \
  --transcripts QA/results/transcripts \
  --output pretrain/data/pretrain_samples.jsonl \
  --window-sec 8 --min-words 3 --context-max-chars 400
```

产物：`pretrain_samples.jsonl`（`sample_type=pretrain_caption`）。

详见 `pretrain/PRETRAIN_FORMAT.md`。

---

## 5. 预训练（Stage 1）

入口：`pretrain/train.py`

- LoRA 预训练，不加 special token，词表不变（adapter 小、快、省显存）。
- 早停（每 epoch 平均 loss，连续 N epoch 改善 < min_delta 停）+ TensorBoard。

命令：

```bash
python pretrain/train.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --train-jsonl pretrain/data/pretrain_samples.jsonl \
  --video-path-map pretrain/data/video_path_map.json \
  --output-dir /mnt/cache/qwenFT/pretrain_qwen3vl_bf16 \
  --window-size 4 --frame-size 224 \
  --num-train-epochs 3 \
  --per-device-train-batch-size 1 --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 --bf16 --gradient-checkpointing \
  --early-stop-patience 3 --early-stop-min-delta 0.001
```

产物：预训练 LoRA adapter。

---

## 6. Clipping（视频切片）

入口：`QA/prepare/clipping.py`（复用 `scripts/video_segmentation.py`）

超声友好、离线、无 LLM：

1. 视觉变化检测：固定时间网格（默认每 1.5s 一帧），SSIM 相似度（无 scikit-image 自动回退 framediff）；`similarity < scene_threshold(0.6)` 且相邻切点间隔 ≥ `min_scene_gap(3s)` 记为 scene change。
2. 句子边界：标点 `.?!` + 停顿 gap(0.8s) fallback + 全弱边界兜底。
3. 对齐：视觉切点找最近句边界（`tolerance=5s`），找不到就放弃该切点（保句子完整）。
4. 组装：`min_clip=30s`、`max_clip=240s`；短尾（< min_clip）合并到前一个 clip。
5. 超长（> max_clip）按句边界细分。

命令：

```bash
python QA/prepare/clipping.py \
  --video path/to/video.mp4 --output-dir QA/results \
  --visual-method ssim --min-clip 30 --max-clip 240
# 调阈值可加 --save-trace，输出每个采样点的相似度
```

产物：`{output_dir}/clips/{video_id}_clips.json`（含 method/params/coverage_pct/每 clip 的 start/end/duration/text/cut_reason）。

---

## 7. QA 生成

- `QA/generator.py`：streaming QA（next_action / next_observation，含 query_time / answer_time）。
- `QA/offline_generator.py`：offline QA（clip_summary，基于 clips）。
- `QA/run.py`：一体化入口。

产物：原始 QA（streaming + offline）。

---

## 8. QA 校验 / 合并

- `QA/validator.py`：校验 QA 质量/格式。
- `QA/merger.py`：合并 offline + streaming，`--expand-wait-answer` 展开成 WAIT/ANSWER 训练样本。

产物：`QA/results/{video_id}_training_samples.jsonl`

样本类型：

```text
offline_answer     -> <ANSWER> clip_summary
streaming_wait     -> <WAIT>   (query_time，证据不足)
streaming_answer   -> <ANSWER> (answer_time，证据充分)
```

详见 `QA/train/TRAINING_FORMAT.md` 和 `QA/schema.md`。

---

## 9. SFT（Stage 2）

入口：`QA/train/train.py`

- LoRA SFT，学 answerability：`<WAIT>` / `<ANSWER>` 决策。
- 新增 special token `<WAIT>`/`<ANSWER>`，并让 `embed_tokens` / `lm_head` 通过 `modules_to_save` 可训练。
- 冻结 vision encoder。
- 早停 + TensorBoard；T4 用 `--bf16`。

命令：

```bash
python QA/train/train.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --train-jsonl QA/results/{video_id}_training_samples.jsonl \
  --default-video-path path/to/video.mp4 \
  --output-dir /mnt/cache/qwenFT/qwen3vl_2b_lora_wait_answer \
  --window-size 8 --frame-size 336 \
  --num-train-epochs 100 \
  --per-device-train-batch-size 1 --gradient-accumulation-steps 4 \
  --learning-rate 2e-4 --bf16 --gradient-checkpointing \
  --early-stop-patience 3 --early-stop-min-delta 0.001
```

产物：SFT LoRA adapter。

详见 `QA/train/README.md`。

---

## 10. 推理 / 评测

- `QA/eval/infer_qwen.py`：base model raw 推理（answerability baseline）。
- `QA/eval/infer_qwen_lora.py`：base + LoRA adapter 推理，`skip_special_tokens=False` 保留 `<WAIT>`/`<ANSWER>`，统计 answerability accuracy。
- `QA/eval/analyze_predictions.py`：分析预测结果。
- `pretrain/infer.py`：预训练 adapter 的 caption 续写推理。
- `QA/test/check_collator_labels.py`：验证 collator 的 label mask 只监督 `<WAIT>`/`<ANSWER>` target。

命令示例（SFT adapter 推理）：

```bash
python QA/eval/infer_qwen_lora.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --adapter-path /mnt/cache/qwenFT/qwen3vl_2b_lora_wait_answer \
  --eval-jsonl QA/results/{video_id}_training_samples.jsonl \
  --default-video-path path/to/video.mp4 \
  --output /mnt/cache/qwenFT/predictions.jsonl \
  --window-size 8 --frame-size 336 --limit 20 --max-new-tokens 160 --bf16
```

---

## 11. 目录速查

```text
UltrasoundCrawler_KeyCode_20260323_v2/  1. 爬取
src/video_filter.py                     2. 过滤
scripts/video_filter_vlm.py             2. 过滤
QA/prepare/asr.py                       3. ASR
scripts/asr_pipeline.py                 3. ASR core
pretrain/                               4-5. 预训练分支
  build_samples.py / dataset.py / collator.py / train.py / infer.py
QA/prepare/clipping.py                  6. Clipping
scripts/video_segmentation.py           6. Clipping core (SSIM grid, no LLM)
QA/generator.py / offline_generator.py  7. QA 生成
QA/validator.py / QA/merger.py          8. QA 校验/合并
QA/train/                               9. SFT
  train.py / dataset.py / collator.py / video_sampling.py
QA/eval/                                10. 推理/评测
```

---

## 12. 环境要点

- GPU 训练环境：Azure T4（`azureml_py38`，torch 2.9.1+cu128，bf16 可用）。
- 大文件（模型下载 / checkpoint）放可写大盘，例如 `/mnt/cache`；根分区通常空间紧张。
- Transformers 在混合环境里可能误 import TensorFlow/Keras：训练脚本已在顶部
  `os.environ.setdefault("TRANSFORMERS_NO_TF", "1")` 等，或运行前 export。
- 依赖见 `requirements.txt`（含 `scikit-image` 用于 SSIM、`tensorboard`）。

---

## 13. 两阶段训练关系

```text
Stage 1 (pretrain/)  : ASR caption completion  -> 学超声视觉-语言知识
Stage 2 (QA/train/)  : WAIT/ANSWER answerability -> 学决策格式

两套代码解耦、可独立运行。
可选：Stage 2 从 Stage 1 的 adapter 继续，作为后续实验。
```
