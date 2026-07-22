# 当前训练形式：Answerability-aware Ultrasound QA SFT

本文档记录当前 `QA/` 数据对应的第一版训练形式。目标是训练一个实时超声视频理解模型，使其不仅会回答问题，还能判断当前视觉证据是否足够。

---

## 1. 训练目标

当前训练目标不是只让模型回答问题，而是让模型学会：

```text
当前视觉证据不足 → <WAIT>
当前视觉证据充分 → <ANSWER> answer
```

统一输入形式：

```text
System Prompt
+ 当前时间之前的最后 N 帧 visual tokens
+ Question
→ <WAIT> 或 <ANSWER>
```

第一版配置：

```text
MODEL = Qwen/Qwen3-VL-2B-Instruct
WINDOW_SIZE = 8 frames
FRAME_SAMPLING = last_n_frames
```

---

## 2. 数据来源

训练数据来自：

```text
QA/results/{video_id}_training_samples.jsonl
```

例如：

```text
QA/results/8V649L5Q368_training_samples.jsonl
```

该文件由：

```bash
python QA/run.py --video path/to/video.mp4 --expand-wait-answer
```

或：

```bash
python QA/merger.py ... --expand-wait-answer
```

生成。

每一行是一条训练样本。

---

## 3. 样本类型

当前训练样本有三类：

```text
offline_answer
streaming_wait
streaming_answer
```

---

### 3.1 `offline_answer`

用于完整 clip 理解。

示例：

```json
{
  "sample_type": "offline_answer",
  "video_id": "8V649L5Q368",
  "clip_idx": 1,
  "video_window": [87.45, 250.17],
  "question": "What does this clip demonstrate, and what should a learner take away from it?",
  "target": "<ANSWER> This clip demonstrates how to evaluate lung sliding at the pleural line...",
  "qa_type": "clip_summary",
  "meta": {
    "source_qa_id": "..."
  }
}
```

训练形式：

```text
完整 clip 内均匀采样 N 帧
+ question
→ <ANSWER> answer
```

Offline QA 永远是 `<ANSWER>`，没有 `<WAIT>`。

---

### 3.2 `streaming_wait`

来自 streaming QA 的 `query_time`。

示例：

```json
{
  "sample_type": "streaming_wait",
  "video_id": "8V649L5Q368",
  "clip_idx": 1,
  "video_window": [167.0, 197.0],
  "question": "Based on what we've seen so far, what should the learner look for next?",
  "target": "<WAIT> Not enough information yet. More video is needed.",
  "qa_type": "next_observation",
  "meta": {
    "source_qa_id": "...",
    "query_time": 197.0,
    "answer_time": 215.0
  }
}
```

训练形式：

```text
query_time 前最后 N 帧
+ question
→ <WAIT>
```

语义：

```text
在 query_time 时，模型看到的视觉证据还不够，
所以正确行为是等待更多视频。
```

---

### 3.3 `streaming_answer`

来自 streaming QA 的 `answer_time`。

示例：

```json
{
  "sample_type": "streaming_answer",
  "video_id": "8V649L5Q368",
  "clip_idx": 1,
  "video_window": [185.0, 215.0],
  "question": "Based on what we've seen so far, what should the learner look for next?",
  "target": "<ANSWER> Look for the bright pleural line between the rib shadows.",
  "qa_type": "next_observation",
  "meta": {
    "source_qa_id": "...",
    "query_time": 197.0,
    "answer_time": 215.0
  }
}
```

训练形式：

```text
answer_time 前最后 N 帧
+ question
→ <ANSWER> answer
```

语义：

```text
到 answer_time 时，关键视觉证据已经出现，
所以模型应该给出答案。
```

---

## 4. Visual Input 构造

`video_window` 是一个时间范围，不代表训练时把这个范围内的所有帧都送入模型。

当前策略：

```text
FRAME_SAMPLING = last_n_frames
WINDOW_SIZE = 8
```

也就是：

```python
frames = sample_last_n_frames(video, start, end, n=8)
```

其中：

```text
start, end = sample["video_window"]
```

---

### 4.1 Streaming 样本（末尾 N 秒取 N 帧）

对于 streaming 样本，采样偏向**当前时刻（video_window.end）**：

```text
current_time = video_window.end
recent_start = max(video_window.start, current_time - WINDOW_SIZE)
visual_input = 在 [recent_start, current_time] 内均匀取 WINDOW_SIZE 帧
```

即"末尾 N 秒取 N 帧"（WINDOW_SIZE=8 → 末尾 8 秒、约每秒 1 帧），使视觉输入聚焦最近上下文，而不是把整个 window 均匀采样。

```text
streaming_wait:
  current_time = query_time
  visual_input = query_time 前末尾 8 秒的 8 帧

streaming_answer:
  current_time = answer_time
  visual_input = answer_time 前末尾 8 秒的 8 帧
```

若窗口不足 8 秒，则在可用范围内均匀采样并复制末帧补齐到 8 帧。

对应实现：`sample_last_n_frames`（tail window 采样）。

---

### 4.2 Offline 样本（整段均匀采样）

对于 offline 样本：

```text
visual_input = 从完整 clip [start, end] 内均匀采样 8 帧
```

因为 offline QA 的目标是完整 clip 理解，而不是流式当前时刻判断。

对应实现：`sample_uniform_frames` / `sample_full_clip_frames`（与 streaming 的 tail 采样分开）。

---

### 4.3 为什么不把 `WINDOW_SIZE` 写进 JSON

训练数据里只保存：

```json
"video_window": [start, end]
```

不保存：

```json
"window_size": 8
```

原因是后续可以不重新生成 QA，直接做帧数消融：

```text
WINDOW_SIZE ∈ {8, 16, 32}
```

数据层只定义时间范围；训练脚本决定采多少帧。

---

## 5. Prompt 格式

### 5.1 System Prompt

当前训练使用统一 system prompt：

```text
You are a real-time ultrasound assistant.
You receive an ultrasound video window and a question.
Answer only if the current visual evidence is sufficient.
If the evidence is insufficient, output exactly:
<WAIT> Not enough information yet. More video is needed.
If the evidence is sufficient, output:
<ANSWER> followed by the answer.
```

---

### 5.2 User

User 输入包含视觉帧和问题：

```text
[8 visual frames]
Question: {question}
```

在当前实现中，8 帧作为 8 个 image blocks 输入给 Qwen-VL processor。

---

### 5.3 Assistant

Assistant target 来自样本里的 `target` 字段：

```text
{target}
```

可能是：

```text
<WAIT> Not enough information yet. More video is needed.
```

或：

```text
<ANSWER> {answer}
```

---

## 6. Loss 计算

训练使用标准 causal language modeling loss，但只对 Assistant target 计算 loss。

```text
System tokens    → label = -100
Visual tokens    → label = -100
Question tokens  → label = -100
Target tokens    → label = token ids
Padding tokens   → label = -100
```

也就是说，模型只被训练去生成：

```text
<WAIT> ...
```

或：

```text
<ANSWER> ...
```

不会被训练去复述 system prompt、视觉 token 或 question。

---

## 7. 是否显式区分 Offline / Streaming

训练输入里**不需要**额外告诉模型：

```text
this is offline
this is streaming
this is proactive
```

模型只需要根据当前视觉证据是否充分决定输出：

```text
证据不足 → <WAIT>
证据充分 → <ANSWER>
```

但 JSON metadata 中仍保留：

```json
"sample_type": "offline_answer" | "streaming_wait" | "streaming_answer"
"qa_type": "clip_summary" | "next_action" | "next_observation"
```

这些字段用于：

- 数据统计
- 采样平衡
- ablation
- evaluation 分组
- debug / 人工检查

它们不一定直接输入模型。

---

## 8. 当前 QA 类型

### 8.1 Offline

```text
clip_summary
```

完整 clip 的综合说明，覆盖：

- 扫查过程
- 关键视觉细节
- 相关医学知识

---

### 8.2 Streaming

```text
next_action
next_observation
```

含义：

```text
next_action      = 接下来该怎么操作
next_observation = 接下来该看什么视觉证据
```

更具体地说：

- `next_action`：探头移动、体位调整、呼吸配合、切换模式、加压等动作。
- `next_observation`：胸膜线、A-lines、B-lines、spine sign、curtain sign、回声模式、运动模式等画面证据。

---

## 9. 当前实现文件

```text
QA/train/video_sampling.py
```

负责从 `video_window` 中采样帧：

```python
sample_last_n_frames(...)     # streaming：末尾 N 秒取 N 帧（偏向 current_time）
sample_uniform_frames(...)    # offline：整段均匀采样 N 帧
sample_full_clip_frames(...)  # offline 别名，调用 sample_uniform_frames
```

---

```text
QA/train/dataset.py
```

负责读取 `training_samples.jsonl`，解析：

```text
video_window
question
target
qa_type
sample_type
```

并返回：

```python
{
  "frames": [PIL.Image, ...],
  "question": "...",
  "target": "...",
  ...
}
```

---

```text
QA/train/collator.py
```

负责构造 Qwen-VL multimodal chat 输入，并做 label mask：

```text
System + Images + Question + Assistant target
```

---

```text
QA/train/train.py
```

负责：

- 加载 `Qwen/Qwen3-VL-2B-Instruct`
- 添加 `<WAIT>` / `<ANSWER>` special tokens
- LoRA SFT
- 冻结 vision encoder
- 保存 LoRA adapter

---

## 10. 第一版训练配置

推荐第一版：

```text
Base model: Qwen/Qwen3-VL-2B-Instruct
Training: LoRA SFT
WINDOW_SIZE: 8 frames
FRAME_SAMPLING: last_n_frames
FRAME_SIZE: 336 or 448
Batch size: 1
Gradient accumulation: 8 or 16
Vision encoder: frozen
Target: assistant tokens only
```

后续消融：

```text
WINDOW_SIZE ∈ {8, 16, 32}
FRAME_SIZE ∈ {336, 448}
```

---

## 11. 推荐训练命令

### 11.1 Smoke train

先用 4 条样本测试 dataloader / processor / LoRA 是否能跑通：

```bash
python QA/train/train.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --train-jsonl QA/results/8V649L5Q368_training_samples.jsonl \
  --default-video-path UltrasoundCrawler_KeyCode_20260323_v2/output/20260520_162816_youtube/media/case_reasoning/8V649L5Q368.mp4 \
  --output-dir QA/checkpoints/smoke_qwen3vl \
  --window-size 8 \
  --frame-size 336 \
  --limit 4 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --learning-rate 1e-4 \
  --bf16
```

如果显卡不支持 bf16，改用：

```bash
--fp16
```

---

### 11.2 单视频训练

```bash
python QA/train/train.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --train-jsonl QA/results/8V649L5Q368_training_samples.jsonl \
  --default-video-path UltrasoundCrawler_KeyCode_20260323_v2/output/20260520_162816_youtube/media/case_reasoning/8V649L5Q368.mp4 \
  --output-dir QA/checkpoints/qwen3vl_2b_lora_wait_answer \
  --window-size 8 \
  --frame-size 448 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --bf16
```

显存紧张时：

```text
--frame-size 336
--gradient-checkpointing
--gradient-accumulation-steps 16
```

---

## 12. 总结

当前训练形式可以概括为：

```text
System Prompt
+ last 8 visual frames before current_time
+ Question
→ <WAIT> / <ANSWER>
```

其中：

```text
streaming_wait:
  current_time = query_time
  target = <WAIT>

streaming_answer:
  current_time = answer_time
  target = <ANSWER> answer

offline_answer:
  current_time = full clip context
  target = <ANSWER> clip_summary
```

这就是当前 answerability-aware ultrasound QA SFT 的训练格式。