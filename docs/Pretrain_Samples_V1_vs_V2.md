# Pretrain Samples V1 vs V2 对比

本文档对比当前两版 pretrain sample 构建方式：

- **V1: ASR segment-level samples**
- **V2: sentence-level / sentence-like samples**

两版都来自同一批 ASR transcript，但样本切分粒度不同。

---

## 1. 背景

当前 pretrain 的直接任务是 narration continuation：

```text
video window + previous ASR context -> next narration unit
```

关键区别在于 `narration unit` 的定义：

- V1：unit 是 Whisper / faster-whisper 输出的 **ASR segment**。
- V2：unit 是在 ASR segments 上二次合并得到的 **sentence-like unit**。

需要明确：V1 和 V2 都仍然是 narration prediction。它们主要让模型适应 ultrasound narration、terminology 和 procedure flow，不等价于真正的 ultrasound visual domain knowledge learning。

---

## 2. V1 / V2 水平对比总览

| 对比维度 | V1: Segment-level | V2: Sentence-level / Sentence-like |
|---|---|---|
| 基本单位 | ASR segment | 由多个 ASR segments 合并得到的 sentence-like unit |
| 生成方式 | 每个合格 ASR segment 生成一个 sample | 顺序合并 ASR segments，直到遇到句末标点或 fallback 阈值 |
| `sample_type` | `pretrain_caption` | `pretrain_caption_sentence` |
| 训练目标 | next ASR segment prediction | next sentence-like unit prediction |
| 输入视频窗口 | `[segment_start - 8s, segment_start]` | `[sentence_start - 8s, sentence_start]` |
| `prev_context` 来源 | 前面的 ASR segments | 前面的 sentence-like units |
| `prev_context` 截断 | 最多 400 chars | 最多 400 chars，长上下文保留尾部 |
| target 粒度 | 短 segment，常是半句话 | 更长、更自然，但仍可能不是完美句子 |
| target 自然性 | 较弱，容易切断自然句 | 较强，可以合并被 ASR 切断的句子 |
| 时间对齐 | 更精细 | 更粗，因为一句可能跨多个 ASR segments |
| train samples | 7928 | 3776 |
| eval samples | 2199 | 尚未生成 |
| 当前训练状态 | 已训练完成：`pretrain_qwen3vl_train50` | 尚未训练 |
| 当前评估状态 | limit=200 quick eval 已完成，full eval 进行中 | 尚未评估 |
| 主要优点 | 样本多、target 短、时间对齐细 | target 更自然、减少断句问题 |
| 主要缺点 | 半句话多，语言不自然 | 样本少，时间对齐粗，受 ASR 标点质量影响 |
| 对 domain language 的帮助 | 强：学习 ultrasound narration 局部续写 | 强：学习更完整的 ultrasound narration 表达 |
| 对 visual domain knowledge 的帮助 | 间接、较弱 | 间接、较弱 |
| 推荐定位 | `Pretrain_Full_V1_segment` | `Pretrain_Full_V2_sentence` |

---

## 3. V1: Segment-level Samples

### 3.1 生成命令

```bash
python pretrain/build_samples.py \
  --transcripts cluster_data/QA/train/transcripts \
  --output cluster_data/pretrain/train_pretrain_samples.jsonl \
  --window-sec 8 \
  --min-words 3 \
  --context-max-chars 400 \
  --unit segment
```

`--unit segment` 是默认值。

### 3.2 样本定义

对每个合格 ASR segment：

```text
segment_start = 当前 ASR segment 开始时间
segment_end   = 当前 ASR segment 结束时间
target        = 当前 ASR segment 文本
```

生成：

```json
{
  "sample_type": "pretrain_caption",
  "video_id": "...",
  "video_window": [segment_start - 8, segment_start],
  "prev_context": "previous ASR segment text, max 400 chars",
  "target": "current ASR segment text"
}
```

### 3.3 当前数据量

```text
train videos: 50
train samples: 7928
eval videos: 20
eval samples: 2199
```

### 3.4 特点

优点：

- 样本数量多。
- target 较短。
- 时间对齐更精细。
- 与 ASR 原始 timestamp 直接对应。

缺点：

- target 不一定是完整句子。
- 自然句经常被 ASR segment 切断。
- 训练目标更偏局部 ASR continuation。
- 视觉 grounding 弱，模型可能更多依赖 `prev_context` 续写。

---

## 4. V2: Sentence-level / Sentence-like Samples

### 4.1 生成命令

```bash
python pretrain/build_samples.py \
  --transcripts cluster_data/QA/train/transcripts \
  --output cluster_data/pretrain/train_pretrain_sentence_samples.jsonl \
  --window-sec 8 \
  --min-words 3 \
  --context-max-chars 400 \
  --unit sentence \
  --sentence-max-words 40 \
  --sentence-max-duration 15
```

### 4.2 样本定义

V2 顺序读取 ASR segments，并合并相邻 segments，直到形成 sentence-like unit。

例如 V1 中可能出现：

```text
segment N:
... Massachusetts General

segment N+1:
Hospital. In my previous tutorial ...
```

V2 尝试合并为：

```text
... Massachusetts General Hospital.
```

V2 样本：

```json
{
  "sample_type": "pretrain_caption_sentence",
  "video_id": "...",
  "video_window": [sentence_start - 8, sentence_start],
  "prev_context": "previous sentence-like units, max 400 chars",
  "target": "current sentence-like unit",
  "meta": {
    "unit": "sentence",
    "sentence_idx": 0,
    "sentence_start": 1.88,
    "sentence_end": 15.90,
    "segment_start_idx": 0,
    "segment_end_idx": 2
  }
}
```

### 4.3 当前数据量

```text
train videos: 50
train samples: 3776
eval sentence-level samples: not built yet
```

对比 V1：

```text
V1 train samples: 7928
V2 train samples: 3776
V2 / V1 ≈ 47.6%
```

样本变少是正常的，因为多个 ASR segments 被合并成一个 sentence-like unit。

### 4.4 fallback 逻辑

ASR 可能没有稳定标点，所以 V2 不完全依赖句号切分。当前 fallback：

```text
--sentence-max-words 40
--sentence-max-duration 15
```

如果累计文本太长或时间跨度太长，即使没有完整句号，也会强制切分。

### 4.5 prev_context 优化

V2 中长句更多，因此 `prev_context` 做了 tail truncation：

```text
如果前文超过 context_max_chars，则保留最近的尾部上下文。
```

这样避免旧逻辑中“上一句太长导致 prev_context 直接为空”的问题。

---

## 5. 真实样本对比

以下示例来自同一个视频：

```text
video_id: -1i1i9sbjqE
```

---

### 5.1 V1 示例：自然句被 ASR segment 切断

#### V1 sample 3

```yaml
video_id: -1i1i9sbjqE
video_window: [3.18, 11.18]
prev_context: "Thanks for tuning in. In this module, I'm going to go over a little bit more in detail the cardiac ultrasound conventions that are commonly out there. As per usual, a number of the ultrasound"
target: "images and videos are courtesy of the Division of Emergency Ultrasound at Massachusetts General"
```

#### V1 sample 4

```yaml
video_id: -1i1i9sbjqE
video_window: [7.9, 15.9]
prev_context: "Thanks for tuning in. In this module, I'm going to go over a little bit more in detail the cardiac ultrasound conventions that are commonly out there. As per usual, a number of the ultrasound images and videos are courtesy of the Division of Emergency Ultrasound at Massachusetts General"
target: "Hospital. In my previous tutorial with respect to parasternal long axis cardiac imaging,"
```

这里：

```text
Massachusetts General Hospital
```

被切成：

```text
Massachusetts General
Hospital.
```

这说明 V1 是 ASR segment-level，不是 sentence-level。

---

### 5.2 V2 示例：合并成 sentence-like unit

#### V2 sample 1

```yaml
video_id: -1i1i9sbjqE
video_window: [0.0, 1.88]
meta:
  unit: sentence
  sentence_idx: 0
  sentence_start: 1.88
  sentence_end: 15.9
  segment_start_idx: 0
  segment_end_idx: 2
prev_context: ""
target: "Thanks for tuning in. In this module, I'm going to go over a little bit more in detail the cardiac ultrasound conventions that are commonly out there. As per usual, a number of the ultrasound images and videos are courtesy of the Division of Emergency Ultrasound at Massachusetts General"
```

#### V2 sample 2

```yaml
video_id: -1i1i9sbjqE
video_window: [7.9, 15.9]
meta:
  unit: sentence
  sentence_idx: 1
  sentence_start: 15.9
  sentence_end: 33.16
  segment_start_idx: 3
  segment_end_idx: 5
prev_context: "Thanks for tuning in. In this module, I'm going to go over a little bit more in detail the cardiac ultrasound conventions that are commonly out there. As per usual, a number of the ultrasound images and videos are courtesy of the Division of Emergency Ultrasound at Massachusetts General"
target: "Hospital. In my previous tutorial with respect to parasternal long axis cardiac imaging, I mentioned that you're going to be pointing the transducer marker either to the patient's right shoulder or to the patient's left hip. And more specifically, if your screen marker"
```

V2 缓解了部分切分问题，但仍不是完美自然句。原因是 ASR 标点质量有限，fallback 有时仍会在 phrase 中间切断。

---

## 6. 另一个视频对比

视频：

```text
video_id: 1E4NSR6yjMw
```

### 6.1 V1 segment-level

```yaml
V1-1:
target: "Hi, my name is Mike Avila, and today we will be talking about basic transthoracic echocardiography,"

V1-2:
target: "including the imaging windows and various pathologies. At the end, we will also talk"

V1-3:
target: "about IVC measurements. The probe of choice for this examination is the cardiac probe,"

V1-4:
target: "aka the phased array probe. This probe is great for fitting in between the rib spaces."
```

V1 把一个自然段拆成多个 ASR segment。

### 6.2 V2 sentence-level

```yaml
V2-1:
target: "Hi, my name is Mike Avila, and today we will be talking about basic transthoracic echocardiography, including the imaging windows and various pathologies. At the end, we will also talk about IVC measurements. The probe of choice for this examination is the cardiac probe, aka the phased array probe. This probe is great for fitting in between the rib spaces."
```

V2 将多个 ASR segments 合并成更完整的 sentence-like target。

---

## 7. 对 domain knowledge 的影响

V1 和 V2 都不是直接的 visual domain knowledge learning。

它们的监督目标仍然是：

```text
predict next narration unit
```

因此模型主要学习：

```text
ultrasound terminology
narration style
procedure flow
common educational explanations
```

而不是强监督学习：

```text
当前图像里有什么解剖结构
当前超声伪影是什么
当前 probe view 是什么
当前 finding 是否支持某个诊断
```

因此两版 pretrain 更准确的定位是：

```text
ultrasound narration / language-domain adaptation
```

而不是最终的 ultrasound visual domain knowledge training。

---

## 8. 后续建议

### 8.1 当前 V1

当前已经训练完成：

```text
cluster_data/checkpoints/pretrain_qwen3vl_train50
```

应继续完成：

```text
full eval20 base vs LoRA
```

以评估 V1 是否稳定提升 narration continuation。

### 8.2 V2

V2 可作为后续独立实验：

```text
Pretrain_Full_V2_sentence
```

需要：

1. 构建 eval sentence-level samples。
2. 用 `train_pretrain_sentence_samples.jsonl` 训练新的 LoRA adapter。
3. 用 `eval_pretrain_sentence_samples.jsonl` 做同粒度评估。
4. 和 V1 比较 word-F1 / semantic cosine / 人工样例质量。

### 8.3 真正 domain knowledge pretraining

如果目标是 ultrasound domain knowledge，建议新增更视觉相关任务：

```text
video clip -> structured ultrasound observation
video clip -> anatomy / artifact / view / finding labels
video clip + question -> visual-grounded answer
multiple-choice visual grounding
```

例如样本可以是：

```json
{
  "video_window": [120.0, 128.0],
  "target": {
    "anatomy": "pleural line",
    "artifact": "A-lines",
    "finding": "lung sliding present",
    "clinical_relevance": "pneumothorax less likely"
  }
}
```

这类任务会比 narration continuation 更直接训练 ultrasound visual knowledge。

---

## 9. 推荐命名

当前已经训练的 checkpoint：

```text
cluster_data/checkpoints/pretrain_qwen3vl_train50
```

建议在报告中标记为：

```text
Pretrain_Full_V1_segment
```

后续如果训练 sentence-level 版本，则另存为：

```text
cluster_data/checkpoints/pretrain_qwen3vl_train50_sentence
```

并标记为：

```text
Pretrain_Full_V2_sentence
```
