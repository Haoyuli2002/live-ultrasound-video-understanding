# QA 数据 Schema

## 1. 核心概念

### 1.1 Answerability（可回答性）

给定一条 streaming QA `(Q, A)` 和一个时刻 `T`，定义：

answerable(Q, T) = True，如果 video[clip_start, T] 里的证据（画面 + 讲解音频）已经足以给出正确答案 A。

否则 answerable(Q, T) = False

这是一个**随时间变化的判断**，它是流式模型必须显式学习的能力。

### 1.2 `query_time`

提出问题的时刻。视频流"当前时刻"。

### 1.3 `answer_time`


`answer_time` 是模型第一次看到足够证据来给出这个答案的时刻。

形式化：

也就是说，在`answer_time`之前，证据不足以让模型得出结论。

**约束**：`query_time < answer_time ≤ clip_end`。

- 如果 validator 判定问题在 `query_time` 时其实已经可答（不合我们要的"需要等待"语义），或者直到 `clip_end` 都答不了，这条 QA 会在 generator 阶段被丢弃。
- 也就是说：**新 pipeline 里保留下来的每一条 streaming QA 都必须是"query_time 时不可答、answer_time 时才可答"的**。

### 1.4 `evidence_window`

派生字段，`[query_time, answer_time]`。语义：这段视频里包含了让问题从"不可答"变成"可答"的关键证据。

---

## 2. Streaming QA 记录格式

Generator 输出的 `{video_id}_streaming_qa.json`：

```json
{
  "video_id": "8V649L5Q368",
  "video_path": "path/to/8V649L5Q368.mp4",
  "model": "google/gemini-2.5-flash",
  "qa_types": ["next_action", "next_observation"],
  "num_clips": 11,
  "time_ratios": [0.3, 0.6],
  "seen_window_sec": 240.0,
  "future_window_sec": null,
  "mode": "video_segments",
  "num_streaming_qa": 62,
  "num_skipped_bad_answer_time": 4,
  "generation_cost_usd": 0.4521,
  "generation_video_tokens_total": 4523100,
  "streaming_qa": [
    { ... 见 §2.1 ... }
  ]
}
```

### 2.1 单条 streaming QA 字段

```json
{
  "source": "streaming",
  "type": "next_observation",                // 或 "next_action"
  "video_id": "8V649L5Q368",
  "clip_idx": 1,
  "clip_start": 179.3,
  "clip_end": 250.2,
  "topic": "Lung sliding at the pleural line",

  "query_time": 197.0,                       // 提问时刻（视频秒数，绝对时间）
  "answer_time": 215.0,                      // 首次可答时刻
  "evidence_window": [197.0, 215.0],         // 派生：[query_time, answer_time]
  "ratio": 0.3,                              // 采样时用的 anchor ratio（调试用）

  "question": "Based on what we've seen so far, what should the learner look for next?",
  "answer": "Look for the bright pleural line between the rib shadows and observe whether it moves with respiration.",
  "evidence": "At ~213s the operator points out the pleural line, and the frames at 213–215s show the bright pleural interface between rib shadows. Before this point, the specific visual feature was not yet clearly demonstrated."
}
```

**说明**：

- `query_time` / `answer_time` 都是**视频里的绝对秒数**，不是相对 clip 起点的偏移。
- `evidence`（可选）是 oracle 给出的"为什么 answer_time 是这个数"的自然语言解释，用于 validator 参考和人工审核。
- `ratio` 保留是为了调试和统计（了解 anchor 分布），训练时可以忽略。

### 2.2 Validator 输出附加字段

`{video_id}_streaming_qa_validated.json` 在每条 QA 上加：

```json
{
  ...(§2.1 所有字段)...,

  "validation": {
    "verdict": "pass",                         // "pass" | "fail"
    "reason": "...一段解释...",                // 至少 2 句，引用具体证据
    "checks": {
      "question_no_leak": true,                // Q 只用 [clip_start, query_time] 就能问出
      "not_answerable_at_query_time": true,    // Q 在 query_time 时不能被回答
      "answerable_at_answer_time": true        // 到 answer_time 时才刚好可答
    },
    "validator_model": "google/gemini-2.5-flash"
  }
}
```

**verdict 规则**：`verdict = "pass"` 当且仅当 `checks` 三项全部为 `true`。任意一项为 `false` → `fail`。

顶层还会加：

```json
{
  ...,
  "validator_model": "google/gemini-2.5-flash",
  "validation_stats": {"pass": 48, "fail": 14, "error": 0},
  "validation_cost_usd": 0.30,
  "validation_video_tokens_total": 5230000,
  "num_after_validation": 48
}
```

默认丢弃 `fail`。用 `--keep-failed` 保留失败样本供人工审核。

### 2.3 辅助字段（非 schema 核心，但输出里会出现）

以下顶层字段不是 schema 的强制部分，但 generator / validator 会额外写出来供**调试与统计**使用，下游读取时应把它们视为可选：

| 字段 | 出现位置 | 含义 |
|---|---|---|
| `drop_log` | generator 输出 | 每次跳过 anchor / QA 时的记录，元素为 `{clip_idx, ratio, reason}`。`reason` 可能是 "FUTURE too short..."、"parse_failure"、"model-skipped: ..."、"answer_time X <= query_time Y" 或 exception 字串 |
| `validation_check_stats` | validator 输出 | 三项 check 各自 `{true, false}` 的计数，用于观察哪一项最常失败 |

如果只关心训练数据，忽略这两个字段即可。

---

## 2.4 Offline QA 记录格式

Offline generator（`QA/offline_generator.py`，Gemini 2.5 Flash + 完整 clip mp4 视频段）输出的 `{video_id}_offline_qa.json`：

```json
{
  "video_id": "8V649L5Q368",
  "video_path": "path/to/8V649L5Q368.mp4",
  "model": "google/gemini-2.5-flash",
  "qa_types": ["clip_summary"],
  "num_clips": 11,
  "clip_max_sec": 300.0,
  "mode": "video_segments",
  "num_offline_qa": 11,
  "num_errors": 0,
  "error_log": [],
  "generation_cost_usd": 0.32,
  "generation_video_tokens_total": 1234567,
  "qa_pairs": [
    { ... 见下面 ... }
  ]
}
```

单条 offline QA 字段：

```json
{
  "source": "offline",
  "type": "clip_summary",                    // 综合型：一条问答覆盖 scene / fine / knowledge 三方面
  "video_id": "8V649L5Q368",
  "clip_idx": 1,
  "clip_start": 179.3,
  "clip_end": 250.2,
  "topic": "Lung sliding at the pleural line",
  "question": "What does this clip demonstrate, and what should a learner take away from it?",
  "answer": "The operator first ... then ... finally ... . Along the way the pleural line is clearly seen at ... . Clinically, this pattern is used to ...（4-8 句自然融合 temporal / visual / knowledge 三方面）",
  "evidence": "The narration at ~40s explains ..., and the pleural line is visible from ~55s onward."
}
```

**约定**：
- 每个 clip 只生成 **1 条 clip_summary**（而非老 pipeline 的 3 类各 1 条）。
- 顶层 key 用 `qa_pairs`（复数）而非 `streaming_qa`，与老 pipeline `scripts/qa_generation.py` 输出兼容，`QA/merger.py` 读取时不区分来源。

---

## 3. 训练样本展开（Merger `--expand-wait-answer`）

一条通过验证的 streaming QA 会被展开成**两条**训练样本：

### 3.1 WAIT 样本

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

**语义**：在 `query_time` 附近的候选时间范围内采样当前时刻前最后 N 帧，模型的正确输出是"我还答不了，继续看"。

### 3.2 ANSWER 样本

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

**语义**：在 `answer_time` 附近观察 `WINDOW_SEC` 秒的视频，模型的正确输出是 `<ANSWER> {真答案}`。

### 3.3 `video_window`、`WINDOW_SEC` 与 `WINDOW_SIZE`

`video_window` 是一个**时间范围**，不是说训练时必须把这个时间段内的所有帧都输入模型。

当前第一版训练模式采用 **last-N-frame sampling**：

```text
System Prompt
+ 当前时间之前的最后 N 帧 visual tokens
+ Question
→ <WAIT> 或 <ANSWER>
```

其中：

```text
N = WINDOW_SIZE
```

推荐第一版训练配置：

```text
WINDOW_SIZE = 8 frames
FRAME_SAMPLING = last_n_frames
```

后续消融实验可以尝试：

```text
WINDOW_SIZE ∈ {8, 16, 32}
```

`WINDOW_SEC` 只用于给 dataloader 提供一个候选时间范围：

- 默认 `WINDOW_SEC = 30`，对齐 `docs/PROPOSED_METHOD.md` §5。
- Merger 里通过 `--window-sec` 覆盖。
- 如果 `answer_time - WINDOW_SEC < clip_start`，直接截断到 `clip_start`（不会跨 clip 边界取帧）。

训练时实际取帧策略：

```text
streaming_wait:
  current_time = query_time
  video_window = [query_time - WINDOW_SEC, query_time]
  visual_input = video_window 内 query_time 之前的最后 N 帧

streaming_answer:
  current_time = answer_time
  video_window = [answer_time - WINDOW_SEC, answer_time]
  visual_input = video_window 内 answer_time 之前的最后 N 帧

offline_answer:
  visual_input = 从完整 clip 内均匀采样若干帧
```

因此，`video_window` 是数据层给出的**采样范围**；`WINDOW_SIZE` 是训练脚本里的**帧数配置**。不要把 `WINDOW_SIZE` 写死进每条样本，方便后续做 `N=8/16/32/64` 的 ablation。

### 3.4 Offline QA 样本

Offline QA 直接展开为单条 ANSWER 样本（观测完整 clip）：

```json
{
  "sample_type": "offline_answer",
  "video_id": "8V649L5Q368",
  "clip_idx": 1,
  "video_window": [87.45, 250.17],
  "question": "What does this clip demonstrate, and what should a learner take away from it?",
  "target": "<ANSWER> This clip demonstrates how to evaluate lung sliding at the pleural line...",
  "qa_type": "clip_summary",
  "meta": { "source_qa_id": "..." }
}
```

---

## 4. Special Token

- `<WAIT>` — 让模型显式表达"当前证据不足，需要继续观察"。
- `<ANSWER>` — 让模型显式表达"我有把握，下面是答案"。

这两个 token 是**数据层的标记**，模型层怎么消费（作为字面字符串学 / 加进 tokenizer 作为 special token）下一阶段（训练脚本设计时）再决定。此阶段只保证数据里正确出现即可。

---

## 5. 文件命名和存放位置

```
QA/results/
├── {video_id}_streaming_qa.json               # generator 输出
├── {video_id}_streaming_qa_validated.json     # validator 输出
└── {video_id}_training_samples.jsonl          # merger 输出（含 --expand-wait-answer 时）
```

**输入**（来自老 pipeline，不改动）：
```
results/clips/{video_id}_clips.json            # Step 4 分段输出
results/transcripts/{video_id}.json            # Step 3 ASR 输出
results/qa/{video_id}_offline_qa.json          # Step 5a offline QA（可选合并）
```

---