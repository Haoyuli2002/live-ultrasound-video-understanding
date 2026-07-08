# 创新点与方法设计

## 1. 核心创新点

本项目面向 **实时超声视频理解**，提出两个核心方向：

```text
1. 双模式超声问答：Offline QA + Streaming QA
2. 问题驱动的视觉记忆机制：Visual / Frame Memory Bank
```

目标是在有限视觉 token budget 下，实现可靠、可等待、可解释的流式超声视频理解。

---

## 2. 双模式超声 QA

我们将超声 QA 数据设计为两种模式：

```text
Offline QA
Streaming QA
```

---

## 2.1 Offline QA：完整片段理解

Offline QA 用于评估模型对完整超声片段的整体理解能力。

### 输入

```text
完整 ultrasound clip
+ 可选 ASR / narration
+ 问题
```

### 输出

```text
基于完整 clip 的回答
```

### 评估能力

Offline QA 主要评估：

```text
全局理解
关键发现总结
解剖结构识别
图像质量判断
医学知识解释
完整扫描过程描述
```

### 示例

```text
Question:
这个 clip 展示了什么超声发现？

Answer:
该片段展示了如何在肺部超声中观察胸膜线和 lung sliding，
并说明其在排除 pneumothorax 中的作用。
```

---

## 2.2 Streaming QA：实时可回答性建模

Streaming QA 用于评估模型在视频流中实时理解和判断的能力。

在某个时间点 `question_time` 提出问题，模型不一定需要立刻回答。  
如果当前已看到的视频不足以回答，模型应该继续观看，直到 `answer_time` 时证据足够，再给出答案。

核心思想是：

```text
模型不仅要会回答，还要知道什么时候不能回答、什么时候可以回答。
```

---

### 核心字段

```json
{
  "question_time": 110.0,  # 什么时候发问
  "answer_time": 145.0,    # 什么时候可以回答
  "answerable_at_question_time": false,
  "needs_more_context": true,
  "question": "...",
  "answer": "...",
  "evidence_window": [110.0, 145.0]
}
```

---

### Streaming QA 重点评估

```text
是否知道当前信息不足
是否能等待更多视频证据
是否能在证据足够时及时回答
是否避免使用未来信息
是否能给出医学上安全的回答
```

---

## 3. 视觉记忆机制：Visual / Frame Memory Bank

长时间流式视频会导致视觉 token 不断累积，带来显存、延迟和计算成本问题。

如果模型一直保留所有历史帧：

```text
video[clip_start, current_time]
```

会导致：

```text
visual tokens 持续增长
推理延迟增加
显存压力变大
长视频难以处理
```

因此，我们不直接保留所有历史视觉 token，而是使用：

```text
最近视频窗口 + 与当前问题最相关的历史视觉证据
```

---

## Frame-level Visual Bank

帧级别的视觉记忆库。

### 流程

```text
历史帧 → 视觉 / 文本 embedding → frame bank
当前问题 → 检索 top-k 相关历史帧
```

### 模型输入

```text
最近视频窗口
+ 检索到的历史关键帧
+ 可选历史 summary / ASR
+ 当前问题
```

### 优点

```text
不需要修改模型结构
能控制视觉 token 数量
能保留部分历史视觉信息
实现成本相对较低
```

---

## 4. 数据组织形式

建议将数据分为三层：

```text
Clip metadata
QA annotation
Training samples
```

---

## 4.1 Clip metadata

每个 clip 存储基础信息：

```json
{
  "video_id": "8V649L5Q368",
  "video_path": ".../8V649L5Q368.mp4",
  "clip_idx": 1,
  "clip_start": 87.45,
  "clip_end": 250.17,
  "duration": 162.72,
  "topic": "Diagnosis of pneumothorax using ultrasound",
  "asr_text": "...",
  "asr_available": true
}
```

---

## 4.2 QA annotation

每个 clip 对应一个 dual QA annotation：

```json
{
  "video_id": "8V649L5Q368",
  "video_path": ".../8V649L5Q368.mp4",
  "clip_idx": 1,
  "clip_start": 87.45,
  "clip_end": 250.17,
  "topic": "Diagnosis of pneumothorax using ultrasound",

  "offline_qa": [],
  "streaming_qa": [],

  "generation_meta": {
    "version": "dual_qa_v1",
    "generator_model": "google/gemini-2.5-pro"
  }
}
```

---

## 4.3 Offline QA 数据格式

Offline QA 使用完整 clip 作为输入。

```json
{
  "qa_id": "8V649L5Q368_c1_offline_001",
  "source": "offline",
  "qa_type": "scene_description",

  "input_video_start": 87.45,
  "input_video_end": 250.17,
  "input_policy": "full_clip",

  "asr_policy": "optional_full_asr",

  "question": "What is demonstrated in this ultrasound clip?",
  "answer": "This clip demonstrates how to evaluate lung sliding at the pleural line to help assess pneumothorax.",
  "evidence": "The clip shows probe placement, pleural line visualization, and interpretation of lung sliding."
}
```

训练时：

```text
video[clip_start, clip_end] + question → answer
```

---

## 4.4 Streaming QA 数据格式

Streaming QA 记录问题提出时间和可以回答的时间。

```json
{
  "qa_id": "8V649L5Q368_c1_stream_001",
  "source": "streaming",
  "qa_type": "visible_anatomy",

  "clip_start": 87.45,
  "clip_end": 250.17,

  "question_time": 110.0,
  "answer_time": 145.0,

  "answerable_at_question_time": false,
  "needs_more_context": true,

  "visual_context": {
    "type": "sliding_window",
    "start": 115.0,
    "end": 145.0,
    "duration_sec": 30,
    "anchor": "answer_time",
    "fps": 1,
    "max_frames": 32
  },

  "memory": {
    "type": "retrieved_frame_bank",
    "top_k": 4,
    "items": [
      {
        "time": 95.0,
        "type": "frame",
        "reason": "shows initial probe placement between ribs"
      }
    ]
  },

  "history_summary": "Earlier, the probe was placed between ribs to locate the pleural line.",
  "history_summary_source": "asr_or_visual_summary",

  "question": "What anatomical landmarks are visible at this point?",
  "answer": "The rib shadows and the bright pleural line between them are visible.",
  "evidence_window": [110.0, 145.0]
}
```

训练时：

```text
retrieved historical frames
+ recent video window
+ optional history summary
+ question
→ answer
```

---

## 5. Streaming QA 的输入策略

为了避免视觉 token 无限增长，Streaming QA 不直接输入全部历史视频：

```text
video[clip_start, answer_time]
```

而是使用：

```text
recent visual window
+ retrieved visual memory
+ optional text summary
```

推荐第一版参数：

```text
WINDOW_SEC = 30     # 保留最近30秒的视频帧
STEP_SEC = 5        # 每5秒，再调用一次模型
FPS = 2             # 每秒采样2帧
MAX_FRAMES = 50   
MEMORY_TOP_K = 10   # 从视觉库中再提取10帧
```

即：

```text
最近 30 秒视频
+ top-10 历史相关帧
+ 历史 summary
+ 当前问题
```

---

## 6. 训练样本构造

一条 Streaming QA 可以转成两类训练样本：

```text
WAIT sample
ANSWER sample
```

---

## 6.1 WAIT sample

如果在 `question_time` 时还不能回答：

```json
{
  "sample_type": "streaming_wait",
  "input_video_start": 80.0,
  "input_video_end": 110.0,
  "question": "What anatomical landmarks are visible at this point?",
  "target": "<WAIT> Not enough information to answer yet. More video is needed."
}
```

对应训练：

```text
video[visual_window_start, question_time]
+ optional memory / summary
+ question
→ <WAIT>
```

---

## 6.2 ANSWER sample

当到达 `answer_time`，证据足够：

```json
{
  "sample_type": "streaming_answer",
  "input_video_start": 115.0,
  "input_video_end": 145.0,
  "question": "What anatomical landmarks are visible at this point?",
  "target": "<ANSWER> The rib shadows and the bright pleural line between them are visible."
}
```

对应训练：

```text
video[visual_window_start, answer_time]
+ retrieved memory
+ optional summary
+ question
→ <ANSWER> answer
```

---

## 6.3 Offline sample

Offline QA 直接转成：

```json
{
  "sample_type": "offline_answer",
  "input_video_start": 87.45,
  "input_video_end": 250.17,
  "question": "What is demonstrated in this ultrasound clip?",
  "target": "<ANSWER> This clip demonstrates how to evaluate lung sliding..."
}
```

---

## 7. Loss 计算方式

不需要额外设计分类 head，也不需要单独的 answerability loss。

所有任务统一使用标准 causal language modeling loss。

---

## 7.1 输入和标签

对每个训练样本，输入形式是：

```text
User:
  video / retrieved frames / summary / question

Assistant:
  target text
```

例如 Streaming WAIT sample：

```text
User:
  recent video window
  history summary
  Question: What anatomical landmarks are visible?

Assistant:
  <WAIT> Not enough information to answer yet. More video is needed.
```

训练时：

```text
User 部分 label = -100
Video tokens label = -100
Assistant target tokens label = token ids
```

---

## 7.2 Loss 公式

统一使用 next-token prediction：

```text
L = - Σ log p(y_t | x, y_<t)
```

其中：

```text
x = video tokens + retrieved memory + text prompt
y = assistant target tokens
```

assistant target 可以是：

```text
<WAIT> Not enough information yet.
```

或者：

```text
<ANSWER> final answer.
```

因此：

```text
WAIT 和 ANSWER 都用同一个 causal LM loss
```

---

## 7.3 Offline QA loss

Offline QA：

```text
video[clip_start, clip_end] + question → <ANSWER> answer
```

loss：

```text
只计算 <ANSWER> answer 的 tokens
```

---

## 7.4 Streaming QA loss

Streaming QA 拆成 WAIT / ANSWER：

```text
video[window, question_time] + question → <WAIT> ...
video[window, answer_time] + question → <ANSWER> ...
```

loss：

```text
只计算 assistant 输出 tokens
```

这让模型同时学会：

```text
什么时候应该等待
什么时候可以回答
如何回答
```

---

## 8. 推理方式

推理时使用 progressive loop：

```text
seen_end = question_time

while seen_end <= clip_end:
    recent_window = video[seen_end - WINDOW_SEC, seen_end]
    memory = retrieve_visual_memory(question, seen_end)
    summary = history_summary_before_window

    output = model(recent_window + memory + summary + question)

    if output starts with <ANSWER>:
        return answer

    if output starts with <WAIT>:
        seen_end += STEP_SEC
```

---

## 9. 最终贡献总结

本方法包含两个核心创新：

### 1. 双模式 QA

```text
Offline QA:
  完整 clip 理解

Streaming QA:
  answerability-aware 实时理解
  显式建模 question_time / answer_time / 是否需要等待
```

### 2. 问题驱动的视觉记忆机制

```text
Visual Memory Bank:
  保留历史视觉证据
  根据当前问题检索相关帧 / token
  控制视觉 token 增长
```

最终目标：

```text
在有限视觉 token budget 下，实现可靠、可等待、可解释的实时超声视频理解。
```

---

## 10. 实现指引（当前已落地部分 → `QA/`）

本文档描述的**双时间戳 Streaming QA**（`query_time` + `answer_time`）以及配套的 answerability 建模，已经在独立的 `QA/` 目录里落地。老 pipeline（`scripts/*`）保留不动作为 baseline，新方法在 `QA/` 里独立演进。

### 10.1 目录组织

```
QA/
├── README.md              # 使用说明（中文）
├── schema.md              # 数据 schema 单一真相源（中文）
├── generator.py           # oracle 生成 QA，输出 query_time + answer_time
├── validator.py           # 三条硬约束逐条审
├── merger.py              # 合并 offline + streaming；可选展开 WAIT/ANSWER
├── run.py                 # 端到端 driver
└── _shared/               # 复用 scripts/_video_llm.py & _env_loader.py
```

### 10.2 已实现 vs 未实现

| 章节 | 状态 | 说明 |
|------|------|------|
| §2 双模式 QA（Offline / Streaming） | ✅ 已实现 | Offline 沿用老 pipeline 的 `scripts/qa_generation.py`；Streaming 由 `QA/generator.py` 输出双时间戳 |
| §4 数据格式 | ✅ 已实现 | 见 `QA/schema.md` |
| §5 Streaming 输入策略（最近窗口 + memory + summary） | ⚠️ 仅 schema 预留 | 目前不实现 memory bank；训练样本只输出最近窗口版本 |
| §6 WAIT / ANSWER 训练样本 | ✅ 已实现 | `QA/merger.py --expand-wait-answer` |
| §7 Loss 计算方式 | ⏳ 训练阶段处理 | 数据层保证 `<WAIT>` / `<ANSWER>` 字面串正确出现，模型层再决定是否加 special token |
| §8 推理 progressive loop | ⏳ 训练阶段处理 | 属于模型 / demo 层，不属于数据层 |
| §3 Visual / Frame Memory Bank | ⏳ 下一阶段 | 拟结合后续 VLM 的 vision encoder（可能是 Q-Former / cross-attn / MLP）一起做 |

### 10.3 三条硬约束（validator）

对每条 streaming QA，`QA/validator.py` 会检查（全部 `true` 才通过）：

1. **`question_no_leak`**：问题只用 `[clip_start, query_time]` 就能写出，不引用 future 独有内容。
2. **`not_answerable_at_query_time`**：在 `query_time` 时证据**不足以**回答（对应 §2.2 中"知道当前信息不足"）。
3. **`answerable_at_answer_time`**：到 `answer_time` 时证据**刚好充分**（对应 §2.2 中"证据足够时及时回答"）。

Verdict 完全由这三条 check 决定；validator 模型若给出与 check 不一致的 verdict，会被强制 override 为按 check 计算的结果。

### 10.4 使用入口

详见 [`QA/README.md`](../QA/README.md)：

```bash
python QA/run.py --video path/to/ID.mp4
```

会依次跑 generator → validator → merger，输出：

- `QA/results/{video_id}_streaming_qa.json`
- `QA/results/{video_id}_streaming_qa_validated.json`
- `QA/results/{video_id}.jsonl`
- `QA/results/{video_id}_training_samples.jsonl`（加 `--expand-wait-answer` 时）
