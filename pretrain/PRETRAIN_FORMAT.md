# 预训练形式：Ultrasound ASR Caption Completion

本目录 `pretrain/` 是一个**独立**的预训练 pipeline，和 `QA/train`（answerability WAIT/ANSWER SFT）完全解耦：不共享代码、不共享 special token、不互相依赖。

目标：用 YouTube 超声视频 + ASR 字幕，让模型看视频帧去**续写/生成这句解说词**，从而学到超声领域的视觉-语言知识。这是 stage 1 预训练；之后可以再在 `QA/train` 上做 stage 2 的 WAIT/ANSWER SFT。

---

## 1. 任务定义（整句 completion）

对每个 ASR segment（一句解说）：

```text
current_time  = segment.start
visual_input  = current_time 之前最近 N 秒的帧
prev_context  = 前面已经说过的解说（可选，截断）
target        = segment.text（这一整句）
```

即：**看当前画面（+前文解说），生成/续写这句超声解说词。**

不使用 `<WAIT>` / `<ANSWER>` special token；词表不变；只训练 LoRA adapter（小、快、省显存）。

---

## 2. 数据流水线

```text
Step 1  YouTube 超声视频 (.mp4)
Step 2  ASR:
          python QA/prepare/asr.py --video xxx.mp4 --output-dir data/asr
        产出 data/asr/transcripts/{video_id}.json
Step 3  构造预训练样本:
          python pretrain/build_samples.py \
            --transcripts data/asr/transcripts \
            --output pretrain/data/pretrain_samples.jsonl \
            --window-sec 8 --min-words 3 --context-max-chars 400
Step 4  预训练:
          python pretrain/train.py --train-jsonl pretrain/data/pretrain_samples.jsonl ...
Step 5  推理检查:
          python pretrain/infer.py --adapter-path <ckpt> ...
```

---

## 3. ASR transcript 输入格式

来自 `QA/prepare/asr.py`：

```json
{
  "video_id": "TlckvYhqaFE",
  "duration_sec": 623.4,
  "segments": [
    {"start": 4.3, "end": 12.3, "text": "So the first thing we do is..."},
    ...
  ],
  "full_text": "..."
}
```

---

## 4. 预训练样本格式 `pretrain_samples.jsonl`

```json
{
  "sample_type": "pretrain_caption",
  "video_id": "TlckvYhqaFE",
  "video_window": [4.3, 12.3],
  "prev_context": "前面已经说过的解说（截断到 context-max-chars）",
  "target": "So the first thing we do is place the linear probe...",
  "meta": {"segment_idx": 5, "seg_start": 12.3, "seg_end": 15.8}
}
```

其中：

```text
video_window = [max(0, seg_start - window_sec), seg_start]
```

`build_samples.py` 会过滤：
- 空文本、词数 < `--min-words`
- 含 `[music]/[applause]/[laughter]/[inaudible]/[noise]` 的 segment

---

## 5. Prompt 与 Loss

System prompt（`pretrain/collator.py`）：

```text
You are an ultrasound teaching assistant.
You are given the most recent ultrasound video frames and, optionally, the
narration transcript so far. Continue the spoken narration, describing what is
happening in the ultrasound scan.
```

User：`[N frames] + (可选) "Narration so far: {prev_context}\nContinue..."`

Assistant：`{target}`

Loss：只对 assistant `target` 计算，其余（system / 图像 token / user / prev_context）都 `-100`。label mask 通过在完整 `input_ids` 中定位 target token 子序列实现，兼容 Qwen-VL image placeholder 展开。

---

## 6. 文件说明

```text
pretrain/
├── build_samples.py     # ASR transcripts -> pretrain_samples.jsonl
├── video_sampling.py    # 末尾 N 秒取 N 帧
├── dataset.py           # 读 jsonl -> frames（可用 --video-path-map 多视频）
├── collator.py          # multimodal chat + label mask（无 special token）
├── train.py             # LoRA 预训练（早停 + TensorBoard）
├── infer.py             # caption 续写推理
└── PRETRAIN_FORMAT.md   # 本文件
```

---

## 7. 视频路径映射

多视频预训练用 `--video-path-map`，JSON 形如：

```json
{
  "TlckvYhqaFE": "azure_data/videos/TlckvYhqaFE.mp4",
  "8V649L5Q368": "azure_data/videos/8V649L5Q368.mp4"
}
```

单视频也可以用 `--default-video-path`。

---

## 8. 推荐命令

### 8.1 构造样本

```bash
python pretrain/build_samples.py \
  --transcripts QA/results/transcripts \
  --output pretrain/data/pretrain_samples.jsonl \
  --window-sec 8 \
  --min-words 3 \
  --context-max-chars 400
```

### 8.2 预训练（T4 用 bf16；本项目 T4 环境 is_bf16_supported=True）

```bash
python pretrain/train.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --train-jsonl pretrain/data/pretrain_samples.jsonl \
  --video-path-map pretrain/data/video_path_map.json \
  --output-dir /mnt/cache/qwenFT/pretrain_qwen3vl_bf16 \
  --window-size 4 \
  --frame-size 224 \
  --num-train-epochs 3 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --bf16 \
  --gradient-checkpointing \
  --early-stop-patience 3 \
  --early-stop-min-delta 0.001
```

### 8.3 推理检查

```bash
python pretrain/infer.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --adapter-path /mnt/cache/qwenFT/pretrain_qwen3vl_bf16 \
  --eval-jsonl pretrain/data/pretrain_samples.jsonl \
  --video-path-map pretrain/data/video_path_map.json \
  --output /mnt/cache/qwenFT/pretrain_predictions_limit20.jsonl \
  --window-size 4 \
  --frame-size 224 \
  --limit 20 \
  --max-new-tokens 128 \
  --bf16
```

---

## 9. 与 stage 2 SFT 的关系

```text
Stage 1 (pretrain/)  : ASR caption completion -> 学超声视觉-语言
Stage 2 (QA/train/)  : WAIT/ANSWER answerability -> 学决策格式
```

两套完全独立。是否用 stage 1 的 adapter 作为 stage 2 起点，是后续可选实验，不影响当前 pipeline。

---

## 10. 当前限制

1. 第一版 batch size = 1（多样本 multimodal padding 兼容性差）。
2. 使用 image blocks（N 张图）而非 video block，兼容多版本 Qwen-VL processor。
3. 只做整句 completion；句中续写（prefix+补全）是后续 ablation。
4. `frame-size 224 / window-size 4` 是 T4 上的提速默认；可按显存调。