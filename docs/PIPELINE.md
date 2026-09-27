# Live Ultrasound Video Understanding — Final Pipeline

本文档是当前 repo 的**唯一权威实现文档**，集中说明完整 pipeline、各阶段实现逻辑、input / output / loss、脚本入口、训练与评估命令。

---

## 0. 总览

目标：构建面向实时超声视频的 streaming understanding 系统。模型持续接收超声视频流和可选 ASR narration；当用户提问时，判断当前证据是否足够，并输出：

```text
<WAIT> reason
```

或：

```text
<ANSWER> answer
```

完整路线：

```text
Raw ultrasound videos
↓
ASR transcript + video filtering / clipping
↓
Stage 1: Ultrasound visual-language pretraining
  video frames + ASR narration -> narration
↓
Stage 2: Two-level streaming memory compression
  every 1s: frames -> 1 short memory token
  every 60s: previous 60 long tokens + current 60 short tokens -> new 60 long tokens
↓
Stage 3: Streaming QA / WAIT-ANSWER SFT
  long memory + short memory + optional current frames + question
  -> <WAIT> reason / <ANSWER> answer
↓
Evaluation
```

关键点：历史视频不会在 query 到来时全部重新输入；历史由 short / long memory tokens 承载。

---

## 1. 数据准备

### 1.1 视频与 ASR

视频来自 YouTube / Bilibili 超声教学视频。ASR transcript 格式：

```json
{
  "video_id": "8V649L5Q368",
  "duration_sec": 1136.85,
  "segments": [
    {"start": 4.46, "end": 7.08, "text": "Today, we're going to be learning about lung ultrasound."}
  ],
  "full_text": "..."
}
```

ASR transcript 是 Stage 1 和 Stage 2 的语言监督来源。

### 1.2 过滤

过滤包括：

```text
1. ASR rule-based filtering
2. Teacher VLM video-type / anatomy / clinical-scenario filtering
```

当前推荐 teacher 设置：

```text
Primary open-source teacher:
  Qwen/Qwen3.5-35B-A3B via vLLM OpenAI-compatible local endpoint
  使用 video_url 直接输入整段视频。

Cross-validation teacher:
  Gemini 3 Pro via OpenRouter / Google OpenAI-compatible endpoint
```

推荐分类标签：

```text
hands_on_ultrasound_teaching
pure_ultrasound_scan
ultrasound_ppt_lecture
mixed_ultrasound_teaching
ultrasound_image_discussion
non_ultrasound_or_irrelevant
uncertain
```

同时输出：

```text
anatomy_regions
clinical_scenarios
scan_views_or_targets
spoken_language / language_evidence
has_realtime_ultrasound / has_probe_or_patient / has_ppt_or_slides / ...
keep_for_pretrain / keep_for_compression / keep_for_sft
```

分类脚本：

```bash
# Qwen3.5 open-source teacher, full classification
python scripts/data/classify_ultrasound_video_type_teacher.py \
  --video-map cluster_data/splits/train_full295_asr_keep_videos.json \
  --output cluster_data/splits/train_full295_qwen35_video_type.jsonl \
  --teacher qwen35 \
  --model Qwen/Qwen3.5-35B-A3B \
  --base-url http://localhost:8000/v1 \
  --api-key-env VLLM_API_KEY \
  --video-fps 1.0 \
  --resume

# Gemini 3 Pro cross-validation
python scripts/data/classify_ultrasound_video_type_teacher.py \
  --video-map cluster_data/splits/train_full295_asr_keep_videos.json \
  --output cluster_data/splits/train_full295_gemini3_video_type.jsonl \
  --teacher gemini3 \
  --model google/gemini-3-pro \
  --base-url https://openrouter.ai/api/v1 \
  --api-key-env OPENROUTER_API_KEY \
  --video-fps 1.0 \
  --resume

# Merge teacher outputs into final audit
python scripts/data/merge_video_type_teacher_labels.py \
  --primary-audit cluster_data/splits/train_full295_qwen35_video_type.jsonl \
  --validator-audit cluster_data/splits/train_full295_gemini3_video_type.jsonl \
  --output cluster_data/splits/train_full295_video_type_final.jsonl \
  --output-summary cluster_data/splits/train_full295_video_type_final_summary.json
```

推荐 keep policy：Stage 2 / Stage 3 默认保留 `hands_on_ultrasound_teaching` 和 `pure_ultrasound_scan`；`mixed_ultrasound_teaching` 标记 `needs_clipping`，默认不直接进入 compression / SFT；PPT / 静态图讨论主要用于 Stage 1 或 offline QA；无关和 uncertain 默认不进入 Stage 2 / Stage 3。

Qwen vLLM 推荐启动方式：

```bash
uv pip install vllm \
  --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/nightly

vllm serve Qwen/Qwen3.5-35B-A3B \
  --host 0.0.0.0 \
  --port 8000 \
  --media-io-kwargs '{"video":{"num_frames":-1}}'
```

如果视频是本地文件，vLLM / OpenRouter 的 `video_url` 需要可访问 URL。可在视频目录启动临时 HTTP server：

```bash
cd /path/to/videos
python -m http.server 9000
```

然后分类时加：

```bash
--video-url-base http://<node-hostname-or-ip>:9000
```

如果 classifier 和 vLLM 在同一节点，也可以使用 `file://` 绝对路径 fallback；如果 endpoint 不支持 `file://`，请使用 HTTP server。

全量批处理推荐使用 Slurm 脚本，它会在同一个 GPU 节点内自动启动视频 HTTP server、vLLM server、等待 ready、断点续跑分类并输出统计：

```bash
sbatch \
  --export=ALL,SPLIT=eval_full295,VIDEO_MAP=/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding/cluster_data/splits/eval_full295_asr_keep_videos.json,OUTPUT=/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding/cluster_data/splits/eval_full295_qwen35_video_type.jsonl \
  scripts/slurm/run_qwen35_video_type_labeling.sbatch
```

可选环境变量：

```text
MODEL=Qwen/Qwen3.5-35B-A3B
VIDEO_FPS=0.5
MAX_TOKENS=6000
ENABLE_THINKING=0     # default for stable batch JSON labeling; set 1 for difficult-case review
LIMIT=10              # smoke test only; omit for full run
HTTP_PORT=9000
VLLM_PORT=8000
```

---

## 2. Stage 1 — Ultrasound Visual-language Pretraining

### 2.1 目标

让 Qwen3-VL 学习超声视觉内容和 ASR narration 的对应关系。

```text
video frames + optional ASR context -> target narration
```

Stage 1 不涉及 memory token，也不涉及 `<WAIT>/<ANSWER>`。

### 2.2 Input / Output

Input：

```text
System Prompt
+ sampled ultrasound frames
+ optional previous ASR narration context
```

Output：

```text
target ASR narration sentence / chunk
```

### 2.3 Loss

普通 causal LM loss：

```text
loss_stage1 = CE(target_narration_tokens)
```

只监督 assistant target；system / user / image prompt tokens 置为 `-100`。

### 2.4 脚本

```text
pretrain/build_samples.py
pretrain/train.py
pretrain/infer.py
```

示例：

```bash
python pretrain/build_samples.py \
  --transcripts results/transcripts \
  --output pretrain/data/pretrain_samples.jsonl \
  --unit sentence --window-sec 8 --min-words 3

python pretrain/train.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --train-jsonl pretrain/data/pretrain_samples.jsonl \
  --video-path-map pretrain/data/video_path_map.json \
  --output-dir /mnt/cache/qwenFT/qwen3vl_stage1_pretrain \
  --window-size 4 --frame-size 224 \
  --num-train-epochs 3 \
  --per-device-train-batch-size 1 --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 --bf16
```

---

## 3. Stage 2 — Two-level Streaming Memory Compression

Stage 2 训练 query-agnostic memory compression。

### 3.1 Memory 定义

```text
Short-term memory:
  每秒新增 1 个 <SHORT_MEM> hidden-state token
  保存最近细粒度证据

Long-term memory:
  每 60s 全量更新 60 个 <LONG_MEM> hidden-state tokens
  保存更长历史
```

Decode / reconstruction 阶段只看 memory tokens，不看 raw visual tokens。

---

### 3.2 Short memory update

每秒处理一次视频流：

```text
x_t = frames in [t, t+1]
s_t = hidden_state(<SHORT_MEM> | x_t)
```

每秒得到一个 short memory token：

```text
s_0, s_1, s_2, ...
```

一个完整 60s block：

```text
S_k = [s_{60k}, s_{60k+1}, ..., s_{60k+59}]
```

---

### 3.3 Short memory supervision

ASR segment 用覆盖它的 short tokens 重建 narration。

例如：

```text
ASR: [4.46, 7.08]
Text: "Today, we're going to be learning about lung ultrasound."

short windows:
  [4,5], [5,6], [6,7], [7,8]

short tokens:
  s_4, s_5, s_6, s_7
```

Input：

```text
s_4, s_5, s_6, s_7
```

Output：

```text
ASR segment text
```

Loss：

```text
loss_short = CE(DecodeShortMemory(s_4, ..., s_7), ASR_segment_text)
```

---

### 3.4 Long memory update

每个完整 60s block：

```text
block k = [60k, 60k + 60]
S_k = [s_{60k}, ..., s_{60k+59}]  # 60 short tokens
L_{k-1} = [l_{k-1}^1, ..., l_{k-1}^{60}]  # 60 long tokens, first block empty
```

Input：

```text
previous long memory L_{k-1}
+ current short memory S_k
+ 60 learnable <LONG_MEM> query tokens
```

Output：

```text
L_k = hidden states at the final 60 <LONG_MEM> positions
```

形式化：

```text
L_k = Compress(L_{k-1}, S_k)
```

Long memory 是全量替换，不是 append：

```text
L_1, L_2, L_3, ... each has exactly 60 tokens
```

---

### 3.5 Long memory reconstruction prompts and losses

对同一个 `L_k` 做三个 decode tasks。

#### Current block reconstruction

Prompt：

```text
Reconstruct the narration for the past/current 60-second block only.
```

Target：

```text
current_block_target = ASR narration in [block_start, block_end]
```

Loss：

```text
loss_current = CE(DecodeCurrent(L_k), current_block_target)
```

#### Previous retention reconstruction

Prompt：

```text
Reconstruct the narration from the beginning of the video up to the start of the current block.
```

Target：

```text
previous_summary_target = ASR narration in [0, block_start], truncated
```

Loss：

```text
loss_previous = CE(DecodePrevious(L_k), previous_summary_target)
```

#### Accumulated reconstruction

Prompt：

```text
Reconstruct the narration from the beginning of the video up to the end of the current block.
```

Target：

```text
accumulated_summary_target = ASR narration in [0, block_end], truncated
```

Loss：

```text
loss_accumulated = CE(DecodeAccumulated(L_k), accumulated_summary_target)
```

Total loss：

```text
loss_long = (
    λ_all      * loss_accumulated
  + λ_current  * loss_current
  + λ_previous * loss_previous
) / (λ_all + λ_current + λ_previous)
```

默认：

```text
λ_all = 1.0, λ_current = 0.5, λ_previous = 0.3
```

---

### 3.6 Ground Truth 来源

当前全部从 ASR transcript 自动构造：

```text
current_block_target:
  ASR concat in [block_start, block_end]

previous_summary_target:
  ASR concat in [0, block_start], then truncated by --history-max-chars

accumulated_summary_target:
  ASR concat in [0, block_end], then truncated by --history-max-chars
```

默认：

```text
--history-max-chars 2400
```

当前截断是 tail truncation：超过长度时保留最后 `history_max_chars` 个字符。后续可用 rolling LLM summary 替换这些字段，训练代码无需改动。

---

### 3.7 Long block 尾部处理

Long memory compression 只构造完整 60s blocks：

```text
[0,60], [60,120], [120,180], ...
```

最后不足 60s 的尾巴直接丢弃，以保持固定结构：

```text
60 short tokens -> 60 long tokens
```

Short memory compression 仍覆盖所有 ASR segments，包括尾部。

---

### 3.8 Data Schema

Short sample：

```json
{
  "sample_type": "short_memory_compression",
  "video_id": "8V649L5Q368",
  "asr_window": [4.46, 7.08],
  "short_windows": [[4.0, 5.0], [5.0, 6.0], [6.0, 7.0], [7.0, 8.0]],
  "target": "Today, we're going to be learning about lung ultrasound.",
  "meta": {"source": "asr_segment", "step_sec": 1.0}
}
```

Long sample：

```json
{
  "sample_type": "long_memory_compression",
  "video_id": "8V649L5Q368",
  "block_idx": 1,
  "block_window": [60.0, 120.0],
  "history_window": [0.0, 120.0],
  "short_windows": [[60.0, 61.0], "...", [119.0, 120.0]],
  "previous_blocks": [
    {"block_idx": 0, "block_window": [0.0, 60.0], "short_windows": [[0.0, 1.0], "..."]}
  ],
  "num_long_tokens": 60,
  "current_block_target": "ASR narration in 60-120s",
  "previous_summary_target": "ASR narration in 0-60s, truncated",
  "accumulated_summary_target": "ASR narration in 0-120s, truncated",
  "target": "same as accumulated_summary_target",
  "meta": {"target_mode": "accumulated_summary_asr_truncated", "block_sec": 60, "step_sec": 1, "drop_last_incomplete": true}
}
```

---

### 3.9 Stage 2 scripts and commands

Files：

```text
pretrain/build_memory_compression_samples.py
pretrain/memory_dataset.py
pretrain/memory_collator.py
pretrain/train_memory_compression.py
pretrain/infer_memory_compression.py
pretrain/eval_memory_compression.py
```

Build：

```bash
python pretrain/build_memory_compression_samples.py \
  --transcripts results/transcripts \
  --output pretrain/data/memory_compression_samples.jsonl \
  --types short,long \
  --block-sec 60 \
  --step-sec 1 \
  --long-token-count 60 \
  --history-max-chars 2400 \
  --min-words 3
```

Train：

```bash
python pretrain/train_memory_compression.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --train-jsonl pretrain/data/memory_compression_samples.jsonl \
  --video-path-map pretrain/data/video_path_map.json \
  --output-dir /mnt/cache/qwenFT/qwen3vl_memory_compression \
  --short-frames 2 --frame-size 224 \
  --max-previous-blocks 1 \
  --lambda-all 1.0 --lambda-current 0.5 --lambda-previous 0.3 \
  --num-train-epochs 1 --learning-rate 1e-4 --bf16
```

Infer / Eval：

```bash
python pretrain/infer_memory_compression.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --adapter-path /mnt/cache/qwenFT/qwen3vl_memory_compression \
  --eval-jsonl pretrain/data/memory_compression_samples.jsonl \
  --output results/memory_compression_predictions.jsonl \
  --video-path-map pretrain/data/video_path_map.json \
  --short-frames 2 --frame-size 224 --bf16

python pretrain/eval_memory_compression.py \
  --pred-jsonl results/memory_compression_predictions.jsonl \
  --output results/memory_compression_eval.json
```

---

## 4. Stage 3 — Streaming QA / WAIT-ANSWER SFT

### 4.1 目标

使用 Stage 2 训练出的 memory 表示做 streaming QA。

Input：

```text
long memory L_t
+ recent short memory S_t
+ optional current visual frames
+ question Q
```

Output：

```text
<WAIT> reason
```

或：

```text
<ANSWER> answer
```

### 4.2 QA 数据核心字段

```text
query_time: 用户提问时刻
answer_time: 第一次证据足以回答问题的时刻
```

WAIT 样本：

```text
Input:  memory up to query_time + optional current frames + question
Output: <WAIT> reason
Loss:   CE(<WAIT> reason)
```

ANSWER 样本：

```text
Input:  memory up to answer_time + optional current frames + question
Output: <ANSWER> answer
Loss:   CE(<ANSWER> answer)
```

Answerability：

```text
p_WAIT   = P(<WAIT>   | L_t, S_t, V_t, Q)
p_ANSWER = P(<ANSWER> | L_t, S_t, V_t, Q)
```

Scripts：

```text
QA/run.py
QA/train/train_summary_decide.py
QA/eval/infer_summary_decide.py
```

Note：Stage 2 已经严格对齐 60-short / 60-long 设计；Stage 3 代码需要保持与该 memory semantics 同步。

---

## 5. Online Inference

```text
1. Initialize:
   short_memory = []
   long_memory = empty

2. Every second:
   receive current frames x_t
   s_t = EncodeShort(x_t, <SHORT_MEM>)
   append s_t to short_memory

3. Every 60 seconds:
   S_k = current 60 short tokens
   L_k = Compress(L_{k-1}, S_k)
   long_memory = L_k
   clear short_memory

4. If user asks Q:
   input = long_memory + current short_memory + optional latest frames + Q
   output = <WAIT> or <ANSWER>

5. If <WAIT>:
   continue streaming and updating memory
   re-evaluate with same Q later

6. If <ANSWER>:
   return answer
```

---

## 6. Evaluation

Stage 2 memory quality：

```text
short memory reconstruction:
  short tokens -> ASR segment narration

long memory reconstruction:
  long tokens -> current / previous / accumulated narration
```

Current automatic metrics：

```text
word overlap F1
ROUGE-L F1
medical term recall
```

Recommended：blind LLM judge for clinical correctness and temporal consistency.

Stage 3 QA quality：

```text
WAIT/ANSWER accuracy
premature answer rate
over-wait rate
answer delay
answer correctness
LLM judge clinical correctness
```

---

## 7. Current Limitations

```text
1. Stage 2 previous/accumulated targets 当前是 ASR concat + tail truncation，不是真正 summary。
2. Tail truncation 会偏向最近内容。
3. Long-memory recursive construction 会随 --max-previous-blocks 增大而变贵。
4. Stage 3 QA code 需要持续同步 strict 60-short / 60-long memory semantics。
```

---

## 8. Next Steps

```text
1. Add rolling LLM summary targets:
   summary_k = LLM(summary_{k-1}, current_block_asr_k)

2. Add blind LLM judge for Stage 2 memory reconstruction.

3. Fully align QA/train two-level memory path with strict 60-short / 60-long design.

4. Run ablations:
   - ASR truncation target vs rolling LLM summary target
   - max_previous_blocks = 0 / 1 / 2 / 4
   - short_frames = 1 / 2
   - with / without current visual frames in QA
```

---

## 9. One-line Summary

The full pipeline first teaches the model ultrasound visual-language alignment, then trains it to produce one short memory token per second and compress previous 60 long tokens plus current 60 short tokens into new 60 long tokens, and finally uses this memory for streaming QA with explicit `<WAIT>/<ANSWER>` answerability decisions.