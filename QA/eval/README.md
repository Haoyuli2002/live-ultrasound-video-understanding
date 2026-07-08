# QA Evaluation：Raw / Finetuned Qwen Answerability 评测

本目录用于评估模型在当前 QA 数据上的 **answerability-aware streaming QA** 能力。

核心问题：

```text
该 WAIT 的地方，模型有没有 WAIT？
该 ANSWER 的地方，模型有没有 ANSWER？
如果模型回答了，答案质量如何？
```

当前第一版只做 **WAIT / ANSWER 行为评测**，即 answerability 评测；答案内容质量先通过保存 prediction 供人工查看，后续可加 LLM-as-Judge。

---

## 1. 文件结构

```text
QA/eval/
├── infer_qwen.py              # 用 raw / finetuned Qwen 生成 prediction
├── analyze_predictions.py     # 统计 WAIT/ANSWER 指标
├── README.md                  # 本文件
└── results/                   # 输出目录
```

---

## 2. 输入数据

评测输入与训练输入相同：

```text
QA/results/{video_id}_training_samples.jsonl
```

例如：

```text
QA/results/8V649L5Q368_training_samples.jsonl
```

每一行是一条训练/评测样本：

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
    "query_time": 197.0,
    "answer_time": 215.0
  }
}
```

---

## 3. 评测输入形式

与训练一致：

```text
System Prompt
+ 当前时间之前的最后 N 帧 visual tokens
+ Question
→ model.generate()
```

当前默认：

```text
WINDOW_SIZE = 8 frames
FRAME_SIZE = 336
FRAME_SAMPLING = last_n_frames
```

注意：

```text
video_window 是采样时间范围；
实际送入模型的是 video_window 内最后 N 帧。
```

---

## 4. 评测目标

模型输出会被解析成三类：

```text
WAIT
ANSWER
OTHER
```

### 4.1 Ground truth label

从样本 `target` 判断：

```text
target starts with <WAIT>   → gt_label = WAIT
target starts with <ANSWER> → gt_label = ANSWER
```

### 4.2 Prediction label

从模型输出判断：

```text
输出以 <WAIT> 开头
或包含 "not enough information" / "need more context"
→ pred_label = WAIT

输出以 <ANSWER> 开头
或是一个实质性回答
→ pred_label = ANSWER

空输出 / 无法解析
→ pred_label = OTHER
```

Raw Qwen 没有经过 `<WAIT>/<ANSWER>` SFT，所以它可能不会严格输出这两个 token。第一版解析规则会把非 WAIT 的实质性回答视为 `ANSWER`。

---

## 5. 评测指标

### 5.1 Overall answerability accuracy

所有样本上：

```text
pred_label == gt_label 的比例
```

---

### 5.2 WAIT accuracy

只在 `streaming_wait` 样本上计算：

```text
WAIT accuracy =
  # streaming_wait 中 pred_label == WAIT
  /
  # streaming_wait 总数
```

越高越好。

---

### 5.3 Premature answer rate

只在 `streaming_wait` 样本上计算：

```text
premature_answer_rate =
  # streaming_wait 中 pred_label == ANSWER
  /
  # streaming_wait 总数
```

中文：**过早回答率**。

越低越好。

含义：

```text
本来证据不足，模型应该 WAIT，
但模型提前给了 ANSWER。
```

这是 raw Qwen baseline 预计会很高的指标。

---

### 5.4 ANSWER accuracy

在 `streaming_answer` 和 `offline_answer` 样本上计算：

```text
ANSWER accuracy =
  # ANSWER 样本中 pred_label == ANSWER
  /
  # ANSWER 样本总数
```

越高越好。

---

### 5.5 Over-wait rate

在 `streaming_answer` 和 `offline_answer` 样本上计算：

```text
over_wait_rate =
  # ANSWER 样本中 pred_label == WAIT
  /
  # ANSWER 样本总数
```

中文：**过度等待率**。

越低越好。

含义：

```text
证据已经足够，模型却还在 WAIT。
```

---

### 5.6 Other / Error rate

模型输出不属于 WAIT 或 ANSWER，或者生成时报错。

越低越好。

---

## 6. Raw Qwen baseline 推理

### 6.1 Azure T4 环境变量

AzureML 环境建议先设置：

```bash
export TRANSFORMERS_NO_TF=1
export USE_TF=0
export USE_FLAX=0
```

### 6.2 跑 10 条样本

```bash
python QA/eval/infer_qwen.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --eval-jsonl azure_data/QA/results/8V649L5Q368_training_samples.jsonl \
  --default-video-path azure_data/videos/8V649L5Q368.mp4 \
  --output QA/eval/results/qwen3vl_2b_raw_predictions_limit10.jsonl \
  --window-size 8 \
  --frame-size 336 \
  --limit 10 \
  --fp16
```

### 6.3 跑完整一个视频

```bash
python QA/eval/infer_qwen.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --eval-jsonl azure_data/QA/results/8V649L5Q368_training_samples.jsonl \
  --default-video-path azure_data/videos/8V649L5Q368.mp4 \
  --output QA/eval/results/qwen3vl_2b_raw_predictions_full.jsonl \
  --window-size 8 \
  --frame-size 336 \
  --fp16
```

---

## 7. 分析 prediction

### 7.1 控制台打印指标

```bash
python QA/eval/analyze_predictions.py \
  --predictions QA/eval/results/qwen3vl_2b_raw_predictions_limit10.jsonl
```

### 7.2 保存 JSON 指标

```bash
python QA/eval/analyze_predictions.py \
  --predictions QA/eval/results/qwen3vl_2b_raw_predictions_limit10.jsonl \
  --out QA/eval/results/qwen3vl_2b_raw_metrics_limit10.json
```

---

## 8. 输出文件格式

`infer_qwen.py` 输出 JSONL，每行一条 prediction：

```json
{
  "idx": 0,
  "sample_type": "streaming_wait",
  "qa_type": "next_observation",
  "video_id": "8V649L5Q368",
  "clip_idx": 1,
  "video_window": [167.0, 197.0],
  "question": "...",
  "target": "<WAIT> Not enough information yet. More video is needed.",
  "prediction": "<ANSWER> ...",
  "gt_label": "WAIT",
  "pred_label": "ANSWER",
  "correct_answerability": false
}
```

---

## 9. 如何解释 raw Qwen 结果

Raw Qwen 没有经过当前数据的 SFT，因此预期：

```text
WAIT accuracy 低
premature_answer_rate 高
```

原因：

```text
普通 instruction model 倾向于“用户问了就回答”，
不习惯在证据不足时输出 <WAIT>。
```

SFT 后希望看到：

```text
WAIT accuracy ↑
premature_answer_rate ↓
ANSWER accuracy 保持或提升
over_wait_rate 不显著上升
```

这就是评估 answerability-aware SFT 是否有效的核心证据。

---

## 10. 后续扩展

当前只做 answerability 行为评测。后续可以增加：

```text
QA/eval/llm_judge_answers.py
```

用于在 `pred_label == ANSWER` 的样本上比较：

```text
Question
Reference answer
Model answer
```

输出答案质量分数，例如：

```text
clinical correctness
visual grounding
completeness
safety