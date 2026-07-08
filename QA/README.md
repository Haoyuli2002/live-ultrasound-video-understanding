# QA Pipeline（新版，含 `answer_time` 与 answerability）

本目录实现 `docs/PROPOSED_METHOD.md` 中提出的**双时间戳 Streaming QA**方案：每条问题除了标注 `query_time` 之外，还必须标注 `answer_time`——即视频里第一次让证据变充分、使问题可被回答的时刻。

> 与老 pipeline (`scripts/*`) 完全**独立**：
> - 老 pipeline 只有 `query_time`，answer 是 oracle 的真值，没有 answerability 概念。
> - 新 pipeline 显式建模 `query_time < answer_time`，并用三条硬约束的 validator 逐条审。
> - 老代码不动。新代码只通过 `import` 方式复用 `scripts/_video_llm.py` 与 `scripts/_env_loader.py`（在 `_shared/__init__.py` 里做了 re-export，源码只有一份）。

数据格式的正式定义见 [`schema.md`](schema.md)。

---

## 目录结构

```
QA/
├── README.md              # 本文件（中文）
├── schema.md              # 数据 schema（中文，single source of truth）
├── offline_generator.py   # Offline QA 生成器（每 clip 1 条 clip_summary）
├── generator.py           # Streaming QA 生成器（每 clip 2 anchor × 2 type = 4 条）
├── validator.py           # 三条硬约束 validator
├── merger.py              # 合并 offline + streaming；可选展开 WAIT/ANSWER
├── run.py                 # 端到端 driver：offline → streaming → validator → merger
├── results/               # 输出目录（脚本首次运行时自动创建，git-ignored）
└── _shared/
    └── __init__.py        # 复用 scripts/_video_llm.py & _env_loader.py
```

---

## 与老 pipeline 的接口

新 pipeline **消费**老 pipeline 的产物：

| 老 pipeline 产物 | 用途 |
|------------------|------|
| `results/transcripts/{video_id}.json` | ASR 转写（供 generator 的 prompt 拼接 & merger 的 text_stream） |
| `results/clips/{video_id}_clips.json` | 视频分段（供 generator 遍历） |
| `results/qa/{video_id}_offline_qa.json` | Offline QA（供 merger 合并到最终 record；可选） |

新 pipeline **不修改**任何 `scripts/` 下的文件。

---

## 三条硬约束（validator 的核心）

对每条 streaming QA，validator 要同时判断（全部为 `true` 才 `pass`）：

| 检查 | 语义 |
|---|---|
| `question_no_leak` | 问题只用 `[clip_start, query_time]` 的画面+讲解就能问出来，不引用 future 独有内容 |
| `not_answerable_at_query_time` | 在 `query_time` 时证据**不足以**回答（存在真实的信息缺口） |
| `answerable_at_answer_time` | 到 `answer_time` 时证据**刚好充分**（关键证据落在 `(query_time, answer_time]` 内） |

Validator 内部会把这三条 check 视为 verdict 的唯一依据；即使模型自己给的 verdict 与三项 check 不一致，我们会**以三项 check 为准**并在 `reason` 里加一条 override 说明。

---

## 快速开始（Smoke Test）

假设你已经用老 pipeline 跑完了 ASR（Step 3）和分段（Step 4），并且已经生成了 offline QA（Step 5a）：

```bash
# 单个视频完整 QA pipeline（offline + streaming + validator + merger）
python QA/run.py --video /path/to/8V649L5Q368.mp4

# 只跑 1 个 clip 的一个 anchor，做快速 smoke test（约 3-5 分钟）
python QA/run.py \
    --video /path/to/8V649L5Q368.mp4 \
    --single-clip 0 \
    --ratios 0.5

# 加上 WAIT/ANSWER 训练样本展开
python QA/run.py \
    --video /path/to/8V649L5Q368.mp4 \
    --expand-wait-answer --window-sec 30
```

各步骤都支持 `--skip-*` 开关重用已有产物：
- `--skip-offline`：跳过 offline QA 生成（若 `QA/results/{id}_offline_qa.json` 已存在）
- `--skip-generation`：跳过 streaming 生成
- `--skip-validation`：跳过 validator
- `--skip-merge`：跳过 merger

`.env` 里需要设置 `OPENROUTER_API_KEY`。会自动加载。

---

## 分步 CLI

如果你只想跑其中一步（例如只重新做 validator）：

```bash
# 1) Generator：产出未验证 streaming QA（带 query_time + answer_time）
python QA/generator.py \
    --video /path/to/ID.mp4 \
    --clips results/clips/ID_clips.json

# 输出：QA/results/ID_streaming_qa.json

# 2) Validator：对每条 QA 做三条硬约束校验
python QA/validator.py \
    --streaming-qa QA/results/ID_streaming_qa.json \
    --video /path/to/ID.mp4

# 输出：QA/results/ID_streaming_qa_validated.json

# 3a) Merger（默认模式）：一条 per-video record，含 clip metadata + text_stream + qa
python QA/merger.py \
    --video-id ID \
    --transcript    results/transcripts/ID.json \
    --clips         results/clips/ID_clips.json \
    --offline-qa    results/qa/ID_offline_qa.json \
    --streaming-qa  QA/results/ID_streaming_qa_validated.json \
    --out           QA/results/ID.jsonl --overwrite

# 3b) Merger（训练样本模式）：每条 streaming QA 展开为 WAIT + ANSWER 两条
python QA/merger.py \
    --video-id ID \
    --transcript    results/transcripts/ID.json \
    --clips         results/clips/ID_clips.json \
    --offline-qa    results/qa/ID_offline_qa.json \
    --streaming-qa  QA/results/ID_streaming_qa_validated.json \
    --expand-wait-answer --window-sec 30 \
    --out           QA/results/ID_training_samples.jsonl --overwrite
```

---

## 关键参数

### Generator

| 参数 | 默认 | 说明 |
|---|---|---|
| `--ratios` | `0.3,0.6` | 每个 clip 的 anchor 时间点比例（默认 2 个 anchor） |
| `--seen-window-sec` | `240` | SEEN 段最长保留最后多少秒（越大越贵） |
| `--future-window-sec` | `-1`（不设上限） | FUTURE 段上限。若设为正数会 clip 到 `query_time+N` |
| `--single-clip` | `None` | 只跑某个 clip_idx（debug） |

### Validator

| 参数 | 默认 | 说明 |
|---|---|---|
| `--before-window-sec` | `240` | BEFORE_QUERY 上限（keep latter portion） |
| `--evidence-window-sec` | `-1` | EVIDENCE_SPAN 上限；`-1 = uncapped`。EVIDENCE 通常 <60s，很少需要 cap |
| `--after-window-sec` | `10` | AFTER_ANSWER 短尾长度（用于 sanity check） |
| `--max-qa` | `None` | 只审前 N 条（smoke test） |
| `--keep-failed` | `False` | 保留 verdict='fail' 的 QA 供人工审核 |

### Merger

| 参数 | 默认 | 说明 |
|---|---|---|
| `--expand-wait-answer` | `False` | 展开为 WAIT + ANSWER 训练样本模式 |
| `--window-sec` | `30` | WAIT/ANSWER 样本的候选时间范围长度 |

---

## 当前训练输入格式

当前第一版训练模式不是把 `video_window` 内的所有帧都送入模型，而是：

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

### Streaming WAIT

```text
current_time = query_time
video_window = [query_time - WINDOW_SEC, query_time]
visual_input = video_window 内 query_time 之前的最后 N 帧
target = <WAIT> Not enough information yet. More video is needed.
```

### Streaming ANSWER

```text
current_time = answer_time
video_window = [answer_time - WINDOW_SEC, answer_time]
visual_input = video_window 内 answer_time 之前的最后 N 帧
target = <ANSWER> {answer}
```

### Offline ANSWER

```text
visual_input = 从完整 clip 内均匀采样若干帧
target = <ANSWER> {clip_summary_answer}
```

也就是说：

- `video_window` 是数据层提供的**采样时间范围**；
- `WINDOW_SIZE` 是训练脚本里的**帧数配置**；
- 不把 `WINDOW_SIZE` 写死进每条样本，方便后续做 `N=8/16/32/64` 的 ablation。

---

## 输出文件示例

### `QA/results/{video_id}_streaming_qa.json`

Generator 输出。每条 QA 长这样：

```json
{
  "source": "streaming",
  "type": "next_observation",
  "video_id": "8V649L5Q368",
  "clip_idx": 1,
  "clip_start": 179.3,
  "clip_end": 250.2,
  "topic": "...",
  "query_time": 197.0,
  "answer_time": 215.0,
  "evidence_window": [197.0, 215.0],
  "ratio": 0.3,
  "question": "Based on what we've seen so far, what should the learner look for next?",
  "answer": "Look for the bright pleural line between the rib shadows.",
  "evidence": "关键证据出现在 ~213s..."
}
```

### `QA/results/{video_id}_streaming_qa_validated.json`

Validator 输出。每条 QA 追加 `validation`：

```json
{
  "...(所有 generator 字段)...",
  "validation": {
    "verdict": "pass",
    "reason": "...",
    "checks": {
      "question_no_leak": true,
      "not_answerable_at_query_time": true,
      "answerable_at_answer_time": true
    },
    "validator_model": "google/gemini-2.5-flash"
  }
}
```

顶层还有 `validation_stats` / `validation_check_stats`（每条 check 的 true/false 计数）。

### `QA/results/{video_id}_training_samples.jsonl`（`--expand-wait-answer`）

每行一条训练样本。同一条 streaming QA 会产生 2 行（WAIT + ANSWER），每条 offline QA 会产生 1 行（offline ANSWER）：

```json
{"sample_type":"streaming_wait","video_id":"8V649L5Q368","clip_idx":1,"video_window":[167.0,197.0],"question":"...","target":"<WAIT> Not enough information yet. More video is needed.","qa_type":"next_observation","meta":{...}}
{"sample_type":"streaming_answer","video_id":"8V649L5Q368","clip_idx":1,"video_window":[185.0,215.0],"question":"...","target":"<ANSWER> ...","qa_type":"next_observation","meta":{...}}
{"sample_type":"offline_answer","video_id":"8V649L5Q368","clip_idx":1,"video_window":[87.45,250.17],"question":"...","target":"<ANSWER> ...","qa_type":"clip_summary","meta":{...}}
```

---

## 成本估算（单个视频，~10 clips，2 anchors/clip，2 types/anchor）

跑一个 ~19 分钟的教学视频，实测数据（demo 视频 `8V649L5Q368`）：

| 步骤 | 时间 | 成本 | API |
|---|---|---|---|
| Offline generator | ~15 min | ~$0.13 | OpenRouter / Gemini 2.5 Flash（完整 clip 视频段） |
| Streaming generator | ~15-20 min | ~$0.30 | OpenRouter / Gemini 2.5 Flash（SEEN+FUTURE） |
| Validator | ~20-30 min | ~$0.30 | OpenRouter / Gemini 2.5 Flash（3 段视频） |
| Merger | <1 s | 免费 | 本地 |
| **合计** | **~50-70 min** | **~$0.75** | |

产出数据密度：
- **10 条 offline QA**（每 clip 1 条 clip_summary）
- **~30-40 条 streaming QA**（10 clips × 2 anchors × 2 types = 40 上限，validator 通过率 ~80%）
- 加 `--expand-wait-answer` 展开成训练样本：~80-90 条

比老 pipeline 略贵（validator 多传一段 EVIDENCE_SPAN 视频），但换来了 answer_time 标注 + 三条硬约束的可信度。

---

## 已知限制 / 未来工作

1. **Generator 与 validator 同族**：都跑在 Gemini 2.5 Flash 上。虽然 prompt 和窗口都做了差异化，但仍不是严格 cross-family。未来可换成 Qwen2.5-VL 作为跨家族 validator。
2. **Memory Bank / retrieval 尚未实现**：schema 里预留了 evidence_window 字段，但当前 pipeline 不会为每条 QA 检索历史帧。这一层留给模型阶段（跟随 VLM encoder 一起做，可能是 Q-Former / cross-attn / MLP）。
3. **训练脚本尚未编写**：`<WAIT>` / `<ANSWER>` 目前只是字面 target，模型层如何消费（作为 special token 加进 tokenizer / 直接学字面字符串 / 是否单独加 answerability head）在下阶段（模型设计时）再决定。当前只保证数据里正确出现即可。
4. **Streaming type 当前固定为两类**：`next_action`（该怎么做）和 `next_observation`（该看什么）。这比旧版 `sonographer_intent` / `next_action_guidance` 更清晰，但仍需要在更多视频上观察是否存在重复、过泛或 C2 失败集中的问题。
5. **重跑成本**：generator + validator 都会重新走一次视频编码；ffmpeg 已经用了 stream-copy + 缓存文件名（含窗口大小），同一个 anchor 重跑不会重切视频，但会重新走 Gemini。

---

## 常见问题

**Q: 为什么 generator 会 skip 一些 anchor / QA？**
A: 三种情况：
- FUTURE 太短（`(clip_end - query_time) < 8s`）：这个 anchor 本身没意义，跳过。
- Gemini 判定"这个位置写不出符合 P1+P2+P3 的 QA"（例如问题在 query_time 已经完全可答）：返回 `skip:true`，我们尊重它。
- `answer_time` 越界或 <= `query_time`：字段格式违规，drop。

所有 skip 都会记在输出 JSON 的 `drop_log` 里。

**Q: Validator 通过率大概多少？**
A: 老 pipeline（二检查）通过率 ~80%。新 pipeline 加了第三条 (C3)，且第二条 (C2) 明显更严，预计通过率会降到 50–70% 之间。等第一批数据跑出来再看具体数字。

**Q: `<WAIT>` sample 会不会太"死板"（永远同一个 target 字符串）？**
A: 现在确实是。未来可以让 generator 也生成"待答理由"（例如"我看到了肋骨阴影但胸膜线还没进入视野，等一下"），把 WAIT target 换成模型自己写的话。当前先保持极简，让模型专注学"何时该说 WAIT"这一决策本身。
