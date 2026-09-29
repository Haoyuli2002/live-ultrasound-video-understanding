# 实时超声视频理解（Live Ultrasound Video Understanding）— 参考实现文档

> **状态：** 当前项目的唯一权威 pipeline 设计文档；后续实现与实验记录均以本文为准。
> **范围：** 数据准备、Stage 1 超声视觉-语言知识注入、Stage 2 摘要压缩学习、Stage 3 可回答性判断与 QA、在线推理、评估
> **核心原则：** 系统必须判断“当前已经看到的超声证据是否足以回答问题”。证据不足时继续等待并观察视频；证据足够时再生成答案。

---

## 0. 系统总览

### 0.1 目标

本项目面向的是 **实时超声视频理解（live ultrasound video understanding）**，而不是传统的离线 Video QA。

模型持续接收超声视频流。当用户提出问题后，系统需要根据“当前时刻之前已经观察到的证据”判断问题是否已经可回答。

- `WAIT`：当前视觉证据不足，系统继续观察后续视频；
- `ANSWER`：当前证据已经充分，可以开始生成最终答案。

因此，本项目的核心并不仅仅是答案生成，而是：

> **Streaming Evidence Sufficiency Estimation（流式证据充分性判断）**

即：模型不仅要“会回答”，还必须知道“什么时候还不能回答”。

---

### 0.2 完整 Pipeline

```text
原始超声视频
        ↓
ASR transcript + 视频过滤 / clipping
        ↓
Stage 1 — 超声视觉-语言知识注入
video frames → 对齐的 ASR narration
        ↓
Stage 2 — Summary Compression Stage
每 1 秒：
    frames → 1 个 short-memory token

每 60 秒：
    60 个 long-memory tokens
    + 当前 60 个 short-memory tokens
    → 新的 60 个 long-memory tokens

数据集构建：使用大模型离线生成视觉摘要标注
    video[T-60:T] → 局部总结标注
    video[0:T]    → 从视频开头到时刻 T 的累计总结标注

Memory Learning：
    short memory → 局部总结
    long memory  → 从视频开头到时刻 T 的总结。
        ↓
Stage 3 — Streaming Answerability + QA
long memory
+ 当前 short memory
+ 可选当前帧
+ question
+ <DECISION>
        ↓
answerability logit
        ↓
p_answer < threshold → WAIT
p_answer ≥ threshold → ANSWER + 生成答案
        ↓
Evaluation
```

---

### 0.3 各 Stage 的职责边界

| Stage | 模型输入 | 监督信号 | 学习目标 |
|---|---|---|---|
| Stage 1 | 超声视频帧 | 对齐的 ASR narration | 超声视觉-语言知识 |
| Stage 2 | 仅超声视频 | VLM 生成的局部 / 累积视觉摘要 | Streaming short / long memory |
| Stage 3 | Streaming memory + question + 可选当前帧 | answerability label + answer text | WAIT/ANSWER 判断与最终 QA |

整个系统有一个非常重要的约束：

> **ASR 只在 Stage 1 中作为监督信号使用。Stage 2、Stage 3 和在线推理阶段均不使用 ASR。**

因此，部署时系统只依赖视频及其内部维护的 streaming memory，不依赖实时语音或字幕。

---

### 0.4 核心时间结构

当前参考实现采用固定时间粒度：

```text
Short-memory 更新频率：      每 1 秒
每个 block 的 short 数量：  60 tokens
Short summary 标注粒度：    可密集到每 10 秒一个 local label
Long-memory 更新频率：       每 60 秒
Long-memory 容量：           固定 60 tokens
```

每一个完整 60 秒 block：

```text
当前 60 个 short-memory tokens
          +
上一轮 60 个 long-memory tokens
          ↓
       Compress
          ↓
新的 60 个 long-memory tokens
```

Long memory 是 **全量更新**，不是不断 append。

因此，无论视频持续 2 分钟、20 分钟还是更久，long-memory 长度始终固定为 60 tokens。

---

# 1. 数据准备

## 1.1 视频来源

训练数据主要来自超声教学类视频，例如：

```text
YouTube
Bilibili
```

数据选择应优先保证：

- 存在真实超声扫描画面；
- 视频具有时间连续性；
- 能看到探头移动、解剖结构变化、扫描视角变化或超声征象；
- 尽量减少纯 PPT、纯人像讲解等与实时视觉流关系较弱的内容。

---

## 1.2 ASR Transcript

对原始视频生成 ASR transcript。

示例：

```json
{
  "video_id": "8V649L5Q368",
  "duration_sec": 1136.85,
  "segments": [
    {
      "start": 4.46,
      "end": 7.08,
      "text": "Today, we're going to be learning about lung ultrasound."
    }
  ],
  "full_text": "..."
}
```

ASR 在整个 pipeline 中只承担两个作用：

1. 辅助前期视频过滤；
2. 作为 **Stage 1 的监督信号**，用于超声视觉-语言知识注入。

ASR **不会进入 Stage 2 memory learning，也不会进入 Stage 3 或 online inference**。

---

## 1.3 视频过滤

视频过滤分两层：

```text
1. 基于 ASR / metadata 的 rule-based filtering
2. VLM 打标 / anatomy / clinical scenario 分类
```

配置：

```text
Primary open-source VLM:
    Qwen/Qwen3.5-35B-A3B
    通过本地 vLLM OpenAI-compatible endpoint 提供服务

Cross-validation VLM:
    Gemini 3.1 Pro
    通过 OpenRouter / Google-compatible endpoint 调用
```

视频类型标签：

```text
hands_on_ultrasound_teaching
pure_ultrasound_scan
ultrasound_ppt_lecture
mixed_ultrasound_teaching
ultrasound_image_discussion
non_ultrasound_or_irrelevant
uncertain
```

同时输出辅助 metadata：

```text
anatomy_regions
clinical_scenarios
scan_views_or_targets
spoken_language
language_evidence

has_realtime_ultrasound
has_probe_or_patient
has_ppt_or_slides
ultrasound_fraction_estimate

keep_for_pretrain
keep_for_compression
keep_for_sft
```

---

## 1.4 各 Stage 的超声视频类型

### Stage 1

Stage 1 的目标是超声领域视觉-语言知识注入，因此可使用范围较广的超声相关视频：

```text
hands_on_ultrasound_teaching
pure_ultrasound_scan
mixed_ultrasound_teaching
ultrasound_ppt_lecture
ultrasound_image_discussion
```

---

### Stage 2

Stage 2 的目标是学习 **真正的流式视觉记忆**，因此应重点使用具有连续超声视觉信息的视频：

```text
pure_ultrasound_scan
hands_on_ultrasound_teaching
```

对于 `mixed_ultrasound_teaching`，只有在以下条件满足时才建议加入：

```text
has_realtime_ultrasound = true
```

并且：

- `ultrasound_fraction_estimate` 足够高；或
- 已经把实时超声部分单独 clipping 出来。

Stage 2 不推荐直接使用 discussion-only 视频，因为它们会让 memory 学到与真实 online ultrasound stream 不一致的内容。

---

### Stage 3

Streaming QA 数据优先来自：

```text
pure_ultrasound_scan
hands_on_ultrasound_teaching
高质量、已经 clipping 的 mixed_ultrasound_teaching
```

---

## 1.5 实现命令

具体 VLM 分类、Qwen3.5 vLLM serving、label merge 和 stage-specific keep/drop map 生成命令见 `docs/HOW_TO_RUN_PIPELINE.md`。

---

# 2. Stage 1 — 超声视觉-语言知识注入

## 2.1 目标

Stage 1 的目标是让基础 VLM 学会：

> 超声画面中的视觉信息与医学语言之间的对应关系。

训练任务：

```text
ultrasound video frames
        ↓
      model
        ↓
aligned ASR narration
```

核心原则：

> **ASR 是训练 target，而不是模型输入。**

这样可以避免模型依赖语言续写，而是尽可能通过超声视觉内容理解当前发生了什么。

---

## 2.2 Input / Target

Input：

```text
System Prompt
+ sampled ultrasound frames
```

Target：

```text
对应时间段的 ASR narration sentence / chunk
```

当前 reference design **不输入 previous ASR narration**。

---

## 2.3 Training Sample

对于 ASR segment：

```text
[start, end]
```

从对应的视频时间窗口采样 frames：

```text
video[start - context_left, end + context_right]
```

并将 aligned ASR text 作为 assistant target。

形式化：

```text
V[t0:t1] -> Y_ASR[t0:t1]
```

---

## 2.4 Loss

使用标准 causal LM loss：

```text
L_stage1 = CE(Y_hat_ASR, Y_ASR)
```

只监督 assistant target tokens。

以下 token label 置为：

```text
-100
```

包括：

```text
system prompt
## 2.5 实现命令

Stage 1 样本构建和训练命令见 `docs/HOW_TO_RUN_PIPELINE.md`。
  --video-path-map pretrain/data/video_path_map.json \
  --output-dir /mnt/cache/qwenFT/qwen3vl_stage1_pretrain \
  --window-size 4 \
  --frame-size 224 \
  --num-train-epochs 3 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --bf16
```

---

## 2.6 Stage 1 需要避免的 Shortcut

不希望模型学成：

```text
previous narration
        ↓
语言续写
        ↓
target narration
```

希望模型学习：

```text
ultrasound visual evidence
        ↓
visual-language understanding
        ↓
target narration
```

因此，reference implementation 中 previous ASR context 不作为 Stage 1 输入。

---

# 3. Stage 2 — Summary Compression Stage

## 3.1 目标

Stage 2 学习一个固定容量的、query-agnostic 的超声视频 streaming memory。

Stage 2 **完全不使用 ASR**。

系统包含两级 memory：

```text
Short-term memory:
    每秒 1 个 token
    保存最近的细粒度超声视觉证据

Long-term memory:
    固定 60 个 tokens
    递归压缩并保存更长时间范围内的重要历史证据
```

为了训练 memory 保留“重要的视觉信息”，使用更强的大模型 / VLM 离线生成视觉摘要标注，构建 Stage 2 的 supervised training dataset。

这里不是蒸馏大模型的 logits、hidden states 或推理轨迹；大模型只用于离线构建高质量训练数据。

---

# 3.2 VLM-Assisted Visual Summary Dataset Construction

## 3.2.1 视频选择

VLM 视觉摘要标注只优先生成在高质量 streaming ultrasound 视频上：

```text
pure_ultrasound_scan
hands_on_ultrasound_teaching
```

后续可以加入已经筛选 / clipping 的高质量 realtime mixed 视频。

---

## 3.2.2 时间定义

设一个 block 为 60 秒。

第 `k` 个 block 的结束时刻：

```text
T_k=60k
```

当前一分钟：

```text
V_k
=
V[T_{k-1}:T_k]
```

截至当前时刻已经观察到的完整历史：

```text
V_{<=k} = V[0:T_k]
```

对于每一个完整分钟，VLM 标注器离线生成两类 summary label：

```text
local summary labels:
    默认可按 10 秒子窗口生成，用于更密集地监督 short memory

global / cumulative summary label:
    每 60 秒生成一次，用于监督 long memory
```

因此，short memory 的**更新频率**是每 1 秒一次；short memory 的**local summary 监督频率**可以比 long memory 更密，例如每 10 秒一次。Long memory 仍然每 60 秒递归更新一次。

---

## 3.2.3 Local Summary

VLM 标注器只观察当前 60 秒：

```text
Y_k^{local}
=
VLMAnnotator(V[T_{k-1}:T_k])
```

它回答的问题是：

> **这一分钟发生了什么？**

摘要应重点覆盖：

```text
anatomy / organ
scan view
probe movement / acquisition action
visible finding
measurement
temporal change
clinically relevant visual evidence
```

示例：

```text
探头被放置在右上腹区域。画面中可见肝脏和右肾，
并对肝肾隐窝进行了扫查。本时间段内未见明显游离液体。
```

---

## 3.2.4 Global / Cumulative Summary

VLM 标注器输入从视频开头到当前时刻的全部视频：

```text
Y_k^{global}
=
VLMAnnotator(V[0:T_k])
```

它回答的问题是：

> **从视频开始到现在，已经出现过哪些重要的超声视觉证据？**

示例：

```text
目前检查已经覆盖右上腹和剑突下心脏切面。
已经观察到肝脏、右肾、肝肾隐窝以及心脏。
截至当前时刻尚未显示明显游离液体。
```

---

## 3.2.5 VLM 视觉摘要标注约束

VLM 生成的视觉摘要标注目标是：

> **描述视频中已经存在并能够被视觉证据支持的信息。**

而不是自由进行诊断推理。

推荐 VLM 标注 Prompt：

```text
仅总结当前指定超声视频时间范围内能够被视觉证据支持的信息。

重点描述：
- 可见的解剖结构和器官
- 扫描切面
- 探头移动或扫查动作
- 可见的超声征象
- 可见的测量结果
- 随时间发生的变化
- 对后续临床判断可能重要的视觉证据

不要推断视频中没有直接视觉支持的诊断。
不要加入指定时间范围之外的信息。
不要描述未来尚未发生的事件。
摘要应简洁、客观、基于可见证据。
```

用于标注的大模型可以使用医学知识识别结构和超声征象，但不能凭空加入视频未显示的临床结论。

---

# 3.3 Short-Term Memory

## 3.3.1 每秒更新

视频流每秒处理一次。

对于时间 `t`：

```text
x_t
=
frames in [t, t+1]
```

向视觉输入后加入特殊 query token：

```text
<SHORT_MEM>
```

取该位置 hidden state：

```text
s_t
=
hidden_state(<SHORT_MEM> | x_t)
```

因此：

```text
第 0 秒 → s_0
第 1 秒 → s_1
第 2 秒 → s_2
...
```

一个完整 60 秒 block 得到：

```text
S_k
=
[s_{60(k-1)},...,s_{60k-1}]
```

恰好 60 个 short-memory tokens。

---

## 3.3.2 Short Memory 的语义

Short memory 主要回答：

> **最近发生了什么？**

它保存最近时间窗口中较细粒度的超声视觉证据。

---

# 3.4 Long-Term Memory

## 3.4.1 每 60 秒更新

到达 60 秒边界时，模型输入：

```text
上一轮 long memory L_{k-1}
+ 当前 60 个 short-memory tokens S_k
+ 60 个 learnable <LONG_MEM> query tokens
```

最终 60 个 `<LONG_MEM>` 位置的 hidden states 定义为：

```text
L_k
=
[l_k^1, ..., l_k^60]
```

形式化：

```text
L_k
=
Compress(L_{k-1},S_k)
```

第一个 block：

```text
previous long memory = empty
```

---

## 3.4.2 Long Memory 是 Replace，而不是 Append

Long memory 始终固定长度：

```text
L_1: 60 tokens
L_2: 60 tokens
L_3: 60 tokens
...
```

每次：

```text
L_{k-1}
   +
S_k
   ↓
Compress
   ↓
L_k
```

新的 `L_k` 全量替换旧的 `L_{k-1}`。

---

## 3.4.3 训练时采用真实 Streaming Chronological Rollout

Stage 2 训练过程和在线推理采用相同的时间递归。

例如：

```text
0-1 s    → s_0
1-2 s    → s_1
...
59-60 s  → s_59

S_1 = [s_0 ... s_59]
L_1 = Compress(empty, S_1)

60-61 s   → s_60
...
119-120 s → s_119

S_2 = [s_60 ... s_119]
L_2 = Compress(L_1, S_2)

...

L_k = Compress(L_{k-1}, S_k)
```

因此，第 `k` 个 block 使用的：

```text
L_{k-1}
```

是前面所有视频按照真实时间顺序递归生成出来的 memory state。

并不是独立构造的单个 previous block。

所以：

> **Stage 2 的训练 recurrence 与 online inference 的 recurrence 保持一致。**

---

# 3.5 Memory Reconstruction Supervision

## 3.5.1 Short-Memory Reconstruction

当前一分钟的 60 个 short-memory tokens：

```text
S_k
```

需要重建 VLM 标注器为当前一分钟生成的 local summary label：

```text
Y_hat_local_k = DecodeShort(S_k)
```

Target：

```text
Y_k^{local}
=
VLMAnnotator(V[T_{k-1}:T_k])
```

Loss：

```text
L_short = CE(Y_hat_local_k, Y_local_k)
```

即：

```text
当前 60 秒视频
      ↓
60 个 short-memory tokens
      ↓
当前一分钟的视觉摘要
```

---

## 3.5.2 Long-Memory Reconstruction

递归更新得到的：

```text
L_k
```

需要重建从视频开始到当前时刻的累计视觉摘要：

```text
Y_hat_global_k = DecodeLong(L_k)
```

Target：

```text
Y_k^{global}
=
VLMAnnotator(V[0:T_k])
```

Loss：

```text
L_long = CE(Y_hat_global_k, Y_global_k)
```

即：

```text
从视频开始到当前时刻的全部历史视觉证据
                    ↓
         递归压缩后的 60 long tokens
                    ↓
          cumulative visual summary
```

---

## 3.5.3 Stage 2 总 Loss

最终：

```text
L_stage2 = lambda_short * L_short + lambda_long * L_long
```

初始建议：

```text
lambda_short = 1.0
lambda_long  = 1.0
```

这套 VLM-generated summary label 数据构建方式 **完全替代旧版 ASR reconstruction 主路线**：

```text
current ASR reconstruction
previous ASR reconstruction
accumulated ASR reconstruction
ASR concat + tail truncation
```

Stage 2 不再依赖 ASR。

---

# 3.6 Short / Long Memory 的职责

两种 memory 的监督目标有明确区分。

### Short Memory

它学习：

> **最近一分钟发生了什么？**

Target：

```text
VLM-generated local summary label
```

---

### Long Memory

它学习：

> **从视频开始到现在，已经出现过哪些值得长期保留的重要视觉证据？**

Target：

```text
VLM-generated cumulative summary label
```

因此：

```text
Short Memory
    = recent fine-grained evidence

Long Memory
    = compressed historical evidence
```

---

# 3.7 视频尾部处理

Long-summary compression 只在完整 60 秒 block 后发生：

```text
[0,60]
[60,120]
[120,180]
...
```

最后不足 60 秒时，不触发新的 long update。

但是 short memory 仍然继续每秒产生。

例如 143 秒的视频：

```text
0-60 s:
    产生 60 short tokens
    更新 L_1

60-120 s:
    产生 60 short tokens
    更新 L_2

120-143 s:
    产生 23 short tokens
    不更新新的 long memory
```

这与真实 online inference 保持一致。

---

# 3.8 实现命令与数据格式

Stage 2 的 JSONL 样本格式、dense local summary labels、训练、推理和评估命令见 docs/HOW_TO_RUN_PIPELINE.md。

---

# 4. Stage 3 — Streaming Answerability + QA

## 4.1 目标

Stage 3 学习两件事：

1. 当前证据是否已经足够回答；
2. 只有在证据足够时，才生成最终答案。

可回答性判断由一个特殊 token 完成：

```text
<DECISION>
```

需要强调：

> `<DECISION>` 不是需要模型生成出来的文本 token。

它是一个 **query / readout token**，我们读取它对应的 hidden state，并通过一个 binary classification head 判断当前是否可回答。

---

# 4.2 输入

在 streaming time `t`，输入：

```text
long memory L_t
+ 当前还未压缩的 short memory S_t
+ optional latest visual frames V_t
+ question Q
+ <DECISION>
```

取 `<DECISION>` 位置 hidden state：

```text
h_t^{dec}
```

---

# 4.3 Answerability Head

用一个 binary classification head：

```text
z_t
=
W h_t^{dec}+b
```

得到 scalar logit。

可回答概率：

```text
p_{answer}(t)
=
sigmoid(z_t)
```

Inference 时使用阈值 `tau`：

```text
p_answer(t) < tau  -> WAIT
```

```text
p_answer(t) >= tau -> ANSWER
```

阈值 `tau` 在 validation set 上确定，用于平衡：

```text
premature answer
over-wait
answer delay
```

---

# 4.4 Decision Supervision

每一个 decision sample：

```text
y_t=
0 if current evidence is insufficient
1 if current evidence is sufficient
```

Decision loss：

```text
L_decision = BCEWithLogitsLoss(z_t, y_t)
```

推荐直接使用：

```text
BCEWithLogitsLoss
```

而不是：

```text
sigmoid
+
BCELoss
```

因为前者数值稳定性更好。

---

# 4.5 Answer Generation

对于：

```text
y_t = 1
```

即 ANSWER sample，同时训练答案生成：

```text
L_answer = CE(A_hat, A)
```

对于：

```text
y_t = 0
```

即 WAIT sample：

```text
只训练 decision loss
不训练 answer generation
```

---

# 4.6 Stage 3 总 Loss

最终：

```text
L_stage3 = lambda_decision * L_decision + y_t * lambda_answer * L_answer
```

初始建议：

```text
lambda_decision = 1.0
lambda_answer   = 1.0
```

Decision loss 和 answer generation loss 应分别监控。

因为对于本项目来说：

> **什么时候回答，本身就是核心研究目标。**

---

# 4.7 QA 时间字段

每一个 QA 的关键 temporal annotations：

```text
query_time:
    问题开始生效的时间

answer_time:
    视频中第一次出现足够视觉证据、
    使该问题能够被可靠回答的时间
```

最简单 baseline：

```text
query_time  → y = 0
answer_time → y = 1
```

需要注意：

> 每个 sample 仍然只有一个 `<DECISION>` token。

后续增强版可以对同一个问题采样多个 decision time：

```text
t_1 < answer_time → y = 0
t_2 < answer_time → y = 0
t_3 < answer_time → y = 0
t_4 ≥ answer_time → y = 1
t_5 ≥ answer_time → y = 1
```

这是：

> **多个时间点对应多个 training samples**

而不是一个 sample 内放多个 `<DECISION>` tokens。

---

# 4.8 实现命令与数据格式

Stage 3 的 QA JSONL 格式、训练和评估命令见 docs/HOW_TO_RUN_PIPELINE.md。

---

# 5. Online Inference

## 5.1 初始化

```text
short_memory = []
long_memory = empty
active_questions = []
```

---

## 5.2 每秒更新

接收当前 frames：

```text
x_t
```

生成 short-memory token：

```text
s_t
=
EncodeShort(x_t, <SHORT_MEM>)
```

然后：

```text
short_memory.append(s_t)
```

---

## 5.3 每 60 秒更新 Long Memory

当 short memory 累积到 60 个：

```text
S_k
=
[s_{60(k-1)},...,s_{60k-1}]
```

更新：

```text
L_k
=
Compress(L_{k-1},S_k)
```

然后：

```text
long_memory = L_k
short_memory = []
```

long memory 始终固定 60 tokens。

---

## 5.4 用户提问

构造：

```text
long_memory
+ current short_memory
+ optional latest frames
+ question
+ <DECISION>
```

得到：

```text
h_t^{dec}
```

计算：

```text
p_{answer}
=
sigmoid(W h_dec_t + b)
```

如果：

```text
p_answer < tau
```

返回：

```text
WAIT
```

问题保持 active。

如果：

```text
p_answer >= tau
```

则开始生成最终答案并返回：

```text
ANSWER
```

---

## 5.5 WAIT 后继续观察

等待中的问题不会被删除。

流程：

```text
用户提问
   ↓
decision at t_0
   ↓
WAIT
   ↓
继续读取新视频
   ↓
更新 short / long memory
   ↓
decision at t_1
   ↓
WAIT / ANSWER
```

因此，系统能够随着视频继续播放不断更新判断，而不需要重新输入此前所有 raw video。

---

# 6. Evaluation

## 6.1 Stage 1 Evaluation

Stage 1 主要验证模型是否学到超声视觉和医学语言之间的对应关系。

可使用：

```text
language generation quality
medical-term recall
clinical concept coverage
blind VLM / LLM judge
```

但更重要的是验证：

> 生成的 narration 是否真正由对应时间段内的超声视觉信息支持。

---

# 6.2 Stage 2 Summary Compression Evaluation

## Short-memory Reconstruction

评估：

```text
S_k → local VLM-generated summary label
```

核心问题：

> Short memory 是否保留了最近一分钟内的重要视觉信息？

---

## Long-memory Reconstruction

评估：

```text
L_k → cumulative VLM-generated summary label
```

核心问题：

> 经过反复递归压缩后，long memory 是否仍然保留了从视频开始到当前时刻的重要临床视觉证据？

---

## 6.2.1 自动指标

可使用：

```text
ROUGE-L F1
token / word overlap F1
medical-term recall
clinical concept recall
```

但不能只依赖 lexical metrics，因为语义正确的摘要可能采用完全不同的语言表达。

---

## 6.2.2 Clinical / Visual Judge

建议 blind judge 评估：

```text
clinical correctness
visual grounding
hallucination
temporal consistency
important-evidence recall
local/global scope compliance
```

Judge 需要特别检查：

> reconstructed summary 是否加入了原视频时间窗口中不存在的证据。

---

## 6.2.3 Memory-Age Evaluation

对于 long memory，按照证据距当前时间的年龄分桶：

```text
0-1 min
1-3 min
3-5 min
5-10 min
>10 min
```

用于评估：

> 历史证据是否随着反复 compression 被逐渐遗忘。

---

# 6.3 Stage 3 Answerability Evaluation

Decision metrics：

```text
accuracy
precision
recall
F1
AUROC
AUPRC
Brier score
calibration error
```

Streaming-specific metrics：

```text
premature answer rate
over-wait rate
answer delay
time-to-answer
```

阈值 `tau` 应在 validation set 上调好，并在 test evaluation 前固定。

---

# 6.4 Final Answer Evaluation

对于 answerable samples：

```text
answer correctness
clinical correctness
visual grounding
hallucination rate
LLM / VLM judge
```

需要分别报告：

```text
Decision Quality
Answer Quality
```

因为：

> 一个答案本身生成得很好，但如果模型总是在证据不足时提前回答，它仍然不是成功的 streaming QA 系统。

---

# 6.5 60 秒边界评估

因为每 60 秒 short memory 会被压缩并清空，因此建议显式评估：

```text
t = 59 s
t = 60 s
t = 61 s
```

以及后续每一个对应边界。

目的是检查：

> long-summary compression 发生之后，模型能力是否出现瞬时性能下降。

如果确实存在明显 boundary drop，可以进一步做 overlap ablation，例如：

```text
compression 后全部清空 short memory

vs.

保留最近 5 秒 short memory

vs.

保留最近 10 秒 short memory
```

这属于后续 ablation，不属于当前 baseline 默认设计。

---

# 7. 当前正式设计决策

## 7.1 ASR 只属于 Stage 1

```text
Stage 1:
    video → ASR supervision

Stage 2:
    video only

Stage 3:
    video-derived memory only

Online:
    video-derived memory only
```

这样可以避免系统部署时依赖语音，同时把：

```text
领域知识注入
```

和：

```text
streaming memory learning
```

明确分开。

---

## 7.2 Memory 是 Query-Agnostic 的

Stage 2 不知道未来用户会问什么问题。

因此 memory 必须学习保存：

> 对各种可能问题都有潜在价值的通用超声视觉证据。

而不是只针对某一个提前已知的问题压缩视频。

---

## 7.3 VLM 视觉摘要标注只描述视觉证据

VLM 视觉摘要标注用于定义 memory 需要保留的信息：

> 什么信息值得保留。

而不是引导模型学习未经视觉支持的诊断推理。

因此 VLM 视觉摘要标注应尽可能接近：

```text
visible evidence
```

而不是：

```text
unsupported clinical inference
```

---

## 7.4 固定时间压缩作为 Baseline

当前默认设计保持：

```text
1 s short-memory step
60 s long-memory update
60 short tokens
60 long tokens
```

后续可以研究 event-driven / adaptive compression，但第一版使用固定时间机制，便于：

```text
实现
复现
控制变量
做 ablation
```

---

## 7.5 Long Memory 固定容量且递归更新

用户提问时不会重新输入完整历史 raw video。

当前 short-memory window 之前的历史信息，只通过：

```text
L_t
```

保存。

这是整个 streaming setting 的关键。

---

# 8. 当前局限

## 8.1 VLM 摘要标注成本

对于每一分钟都重新执行：

```text
VLMAnnotator(V[0:T])
```

会不断重复处理历史视频。

一个 `K` 分钟的视频，总输入规模近似：

```text
1 + 2 + ... + K = O(K^2)
```

例如：

```text
10 分钟：
1+2+...+10 = 55 分钟等价视频输入

20 分钟：
1+2+...+20 = 210 分钟等价视频输入
```

作为第一版高质量 baseline 是可以接受的，但大规模数据时成本会较高。

后续可尝试：

```text
Y_k^{global}
=
VLMAnnotator(
Y_{k-1}^{global},
V_k
)
```

即 rolling VLM summary label generation。

但 rolling summary 会存在 VLM annotation error accumulation，因此第一版仍建议优先使用：

```text
raw video[0:T] → VLM-generated global summary label
```

---

## 8.2 VLM 摘要标注质量依赖

Stage 2 训练数据质量高度依赖 VLM 摘要标注是否满足：

```text
visually grounded
temporally correct
clinically precise
consistent across blocks
```

因此需要：

```text
自动质量检测
+
抽样人工审核
```

---

## 8.3 固定 60 秒边界

一个完整临床扫查动作可能恰好跨越两个 block。

例如：

```text
55-65 s
```

会被分割到：

```text
[0,60]
[60,120]
```

第一版接受这个限制，以保持系统简单、稳定、容易做实验。

---

## 8.4 Long Memory 信息瓶颈

无论视频长度多长：

```text
long memory = 60 tokens
```

因此视频越长，长期 summary compression pressure 越大。

必须通过 memory-age evaluation 验证旧证据是否不断丢失。

---

## 8.5 Decision Calibration

只有一个 scalar answerability probability，结构简单且可解释。

但 threshold 会显著影响：

```text
premature answer
over-wait
answer delay
```

因此 threshold calibration 是正式 evaluation protocol 的一部分，而不是随意设置的部署参数。

---

# 9. 推荐 Ablation

## 9.1 Memory 时间粒度

```text
short step:
    0.5 s / 1 s / 2 s

long block:
    30 s / 60 s / 120 s
```

---

## 9.2 Memory 容量

```text
long-memory token count:
    32 / 60 / 128
```

---

## 9.3 Short Frame Sampling

```text
short_frames:
    1 / 2 / 4
```

---

## 9.4 VLM Summary Label Generation Strategy

比较：

```text
Direct global summary:
    VLMAnnotator(video[0:T])

vs.

Rolling global summary:
    VLMAnnotator(previous_global_summary, video[T-60:T])
```

---

## 9.5 Stage 3 是否使用当前 Raw Frames

比较：

```text
memory only

vs.

memory + latest raw visual frames
```

---

## 9.6 Decision Temporal Sampling

比较：

```text
只训练 query_time + answer_time

vs.

同一个问题采样多个 WAIT / ANSWER decision timestamps
```

每个 training sample 始终只包含一个 `<DECISION>` token。

---

## 9.7 Boundary Overlap

如果发现 long update 后存在明显性能下降，可以比较：

```text
compression 后 short memory 全部清空

vs.

保留最近 5 秒

vs.

保留最近 10 秒
```

---

# 10. 实现检查清单

详细命令和检查清单见 docs/HOW_TO_RUN_PIPELINE.md。

---

# 11. 一句话总结

> 整个系统首先通过“超声视频 → ASR narration”进行 Stage 1 视觉-语言知识注入；随后在 Stage 2 中完全脱离 ASR，每秒把当前超声画面压缩成一个 short-memory token，每 60 秒再将上一轮 60 个 long-memory tokens 与当前 60 个 short-memory tokens 递归压缩成新的 60 个 long-memory tokens，并使用离线 VLM 标注器为当前一分钟生成 local summary label、为 0→T 历史视频生成 cumulative summary label，构建 Stage 2 的 summary-compression 训练数据，使 short / long memory 学会保留关键视觉证据；最后在 Stage 3 中，通过单个 `<DECISION>` hidden state 和 `BCEWithLogitsLoss` 估计当前问题的可回答概率，证据不足时 WAIT，证据充分时才生成最终答案。
