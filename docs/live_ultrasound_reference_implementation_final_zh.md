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
三组视频／历史 ASR 条件 → 同一完整 ASR 目标句
        ↓
Stage 2 — Summary Compression Stage
每 1 秒：
    frames → 1 个 short-memory token

每 60 秒：
    60 个 long-memory tokens
    + 当前 60 个 short-memory tokens
    → 新的 60 个 long-memory tokens

数据集构建：使用大模型离线生成视觉摘要标注
    video[t-10:t] → 10 秒 local summary 标注
    video[T-60:T] → 60 秒 block summary 标注
    video[0:T]    → 从视频开头到时刻 T 的累计总结标注

Memory Learning：
    10 个 short tokens → 对应 10 秒 local summary
    60 个 short tokens → 当前 60 秒 block summary
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
| Stage 1 | `0→start`／`0→end` 视频与历史 ASR 保留／遮蔽条件 | 完整 ASR 讲解句子 | 超声领域知识注入与视觉依赖对照 |
| Stage 2 | 仅超声视频 | VLM 生成的局部 / 累积视觉摘要 | Streaming short / long memory |
| Stage 3 | Streaming memory + question + 可选当前帧 | answerability label + answer text | WAIT/ANSWER 判断与最终 QA |

整个系统有一个非常重要的约束：

> **ASR 仅用于数据准备及 Stage 1：目标句作为监督，目标句之前的历史 ASR 在两组输入条件中保留、第三组遮蔽。Stage 2、Stage 3 和在线推理阶段均不使用 ASR。**

因此，部署时系统只依赖视频及其内部维护的 streaming memory，不依赖实时语音或字幕。

---

### 0.4 核心时间结构

当前参考实现采用固定时间粒度：

```text
Short-memory 更新频率：      每 1 秒
每个 block 的 short 数量：  60 tokens
Local summary 监督频率：    每 10 秒，使用对应 10 个 short tokens
Block summary 监督频率：    每 60 秒，使用当前 60 个 short tokens
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

ASR 在整个 pipeline 中承担三个作用：

1. 辅助前期视频过滤；
2. 作为 **Stage 1 的目标句监督信号**；
3. 在 Stage 1 的两组对照条件中提供目标句之前的历史文本，第三组遮蔽该文本。

ASR **不会进入 Stage 2 memory learning，也不会进入 Stage 3 或 online inference**。

### Stage 1 的 LLM 辅助 ASR 清洗与轻度润色

当前实现使用 `stage1.clean_asr` 在构建 Stage 1 样本前处理 Whisper transcript。这是数据清理步骤，不是教师打标或知识蒸馏。输入为带 `start`、`end`、`text` 的 ASR segments、按视频 ID 选择样本的 pretrain keep-map，以及本地 vLLM 服务的 `Qwen/Qwen3.5-27B`。清洗模型**只读取 ASR 文本**；视频和音频不送入此步骤。默认每次提交连续 8 个 segments，附带前后各 2 个 segments 的文本作上下文。大模型需要按原编号逐段返回补标点、规范大小写及空格后的 `clean_text`，并为每个词语／术语修改提供 `from`、`to`、`reason`。

脚本验证返回数量与顺序，要求词语变化能由修改记录逐项解释，拒绝大幅改写；保留原 segment 边界、时间戳和其他元数据。每个视频写出新的清洗后 transcript：`segments` 与 `full_text` 为清洗文本，`raw_segments` 及原文件存在时的 `raw_full_text` 保留原文，`asr_cleaning.term_corrections` 和独立 audit JSONL 记录术语修改。原始 transcript 不覆盖；有标点和无标点的输入均处理。之后 `stage1.data` 从清洗后的 segments 切分完整句子并构建三组 Stage 1 样本。句子时间是 ASR 段内字符插值估计，不是音频强制对齐。由于清洗模型只看文本，术语修正仍是待对照音频核验的候选改动，不应直接视为真实语音标注。

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

> 从已有超声视频上下文预测完整的领域讲解句子，注入超声概念与表达方式。

训练任务：

```text
ultrasound video context + optional earlier ASR
        ↓
      model
        ↓
complete ASR narration sentence
```

核心原则：

目标句 ASR 是训练 target，绝不进入输入。目标句之前的历史 ASR 在两组条件中作为输入，第三组遮蔽。三组比较用于检查模型对视觉和历史文本的依赖；预测效果仍需通过画面打乱对照验证。

---

## 2.2 Input / Target

Input：

```text
System Prompt
+ sampled ultrasound frames
+ historical ASR 或 [ASR MASKED]
```

Target：

```text
一条完整的 ASR narration sentence
```

目标句及其后的 ASR 不作为输入。

---

## 2.3 Training Sample

对于完整讲解句子 `[start,end]`，生成三组配对样本，目标都为同一句：

| 条件 | 视频 | 目标句之前的 ASR |
|---|---|---|
| `before_with_asr` | `V[0:start)` | 输入 |
| `through_with_asr` | `V[0:end)` | 输入 |
| `before_mask_asr` | `V[0:start)` | 遮蔽 |

每个视频窗口最多按时间顺序采 120 帧；当窗口不超过 120 秒时约为 1 FPS，超过 120 秒时对整个窗口均匀降采样。三个条件共享完整目标句，训练／验证按视频分组。ASR 段内的句子时间戳目前通过字符位置插值估计，需抽样人工核验。

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
user / image prompt
padding
```

## 2.5 实现命令

新实现入口见 `stage1/README.md`；原 `pretrain/` 保留为历史实验。

---

## 2.6 Stage 1 需要避免的 Shortcut

历史 ASR 输入可能使模型只学语言续写，因此不能只看生成损失：

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

同一目标句比较 `before_with_asr` 与 `before_mask_asr`，再比较正常、黑屏和打乱视频的结果；这样才能判断历史文本和视觉画面的各自贡献。

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

设每个完整 block 为 60 秒，结束时刻为 `T_k = 60k`。第 k 个 block 为 `V[T_{k-1}:T_k]`。

第一版固定采用以下更新与监督机制：

1. 每秒从该秒视频生成一个 short-memory token。
2. 每 10 秒，用对应区间的 10 个 short tokens 重建该区间的 local summary。
3. 每 60 秒，用当前整分钟的 60 个 short tokens 重建这 60 秒的 block summary。
4. 同时，将这 60 个 short tokens 与上一轮 long memory 一起输入压缩器，生成新的固定 60-token long memory。
5. 更新后的 long memory 重建从视频开始截至当前时刻的 global summary。

因此共有三类摘要监督：

| 监督目标 | 视频标注范围 | 模型解码输入 | 频率 |
|---|---|---|---|
| Local summary | 当前 10 秒 | 对应的 10 个 short tokens | 每 10 秒 |
| Block summary | 当前完整 60 秒 | 对应的 60 个 short tokens | 每 60 秒 |
| Global summary | 视频开始至当前时刻 | 更新后的 60 个 long tokens | 每 60 秒 |

每个完整分钟边界 `T` 独立生成八条教师标注记录：`[T-60,T-50)`、`[T-50,T-40)`、`[T-40,T-30)`、`[T-30,T-20)`、`[T-20,T-10)`、`[T-10,T)` 六条 local，`[T-60,T)` 一条 block，以及 `[0,T)` 一条 global。`T=60` 时后两条的窗口都为 `0-60`，但类型和监督职责不同；`T=120` 时分别为 `60-120` 和 `0-120`。数据构建脚本核验八条记录后，将其合并成一个训练 block。10 秒边界只产生监督，不清空 short memory；60 秒时先完成 block 重建与 long-memory 更新，再清空当前 block 的 short memory。

---

## 3.2.3 Local / Block Summary

第 k 个 block 内，第 j 个 10 秒区间（`j = 1,...,6`）的标签：

```text
Y_local[k,j] = VLMAnnotator(V[T_{k-1}+10(j-1):T_{k-1}+10j])
```

整分钟标签：

```text
Y_block[k] = VLMAnnotator(V[T_{k-1}:T_k])
```

两类标签分别描述“这 10 秒发生了什么”和“这 60 秒发生了什么”，重点包括可见结构、扫描切面、扫查动作、可见征象、测量及时间变化。

Block summary 是当前整分钟的视觉总结，不是从视频开头开始的累计总结，也不要求简单拼接六条 local summaries。两类标签都必须仅依据各自时间范围内的视觉证据。

---

## 3.2.4 Global / Cumulative Summary

Global 标签的**目标范围**是从视频开头到当前时刻，不是只总结当前 60 秒：

```text
Y_k^{global}
=
VLMAnnotator(V[0:T_k])
```

上式表示期望覆盖的时间范围。当前 `stage2.annotate` 将相应时间窗裁成静音视频片段，直接作为教师输入：六个 10 秒片段、一个当前 60 秒片段，以及一个 `[0,T_k)` 累计片段。标注器不自行均匀抽取固定数量的图像帧；教师服务仍会按自身策略解码和采样视频，因此不能推断教师实际观察了每一帧。正式构建数据前，应审核短暂征象是否被遗漏，并检查长累计片段是否超出服务的大小、时长或上下文限制。

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
+ 60 个重复的 <LONG_MEM> query 位置（共享 token embedding）
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

## 3.5.1 Short-Memory Reconstruction：10 秒与 60 秒双尺度监督

令 `S_k` 为当前 block 的 60 个 short tokens，`S_{k,j}` 为其中第 j 个 10 秒区间对应的 10 个 tokens。

每 10 秒的局部重建：

```text
Y_hat_local[k,j] = DecodeShort(S_{k,j})
L_short[k,j] = CE(Y_hat_local[k,j], Y_local[k,j])
```

每 60 秒的整段重建：

```text
Y_hat_block[k] = DecodeShort(S_k)
L_local[k] = CE(Y_hat_block[k], Y_block[k])
```

Local 解码只能使用对应 10 秒的 tokens，不能读取后续 short tokens、整分钟记忆或 global label。Block 解码直接使用当前 60 个 short tokens，而不是更新后的 long memory。

这两种监督分别训练 short memory 保留局部细节和支持整分钟信息整合。

---

## 3.5.2 Long-Memory Update 与 Global Reconstruction

在同一个 60 秒边界，使用当前 short memory 和上一轮 long memory 更新：

```text
L_k = Compress(L_{k-1}, S_k)
```

`L_k` 固定为 60 个 tokens，全量替换 `L_{k-1}`。第一次更新时，上一轮 long memory 为空。

随后仅由更新后的 long memory 重建累计摘要：

```text
Y_hat_global[k] = DecodeLong(L_k)
L_global[k] = CE(Y_hat_global[k], Y_global[k])
```

Global target 描述 `V[0:T_k]`。Block 摘要重建与 long-memory 更新共享同一组 `S_k`；生成出的摘要文本不会作为 long-memory 更新的输入。

---

## 3.5.3 损失命名与更新时机

第一版固定三个权重为 `1:1:1`，损失名称统一为：

- `L_short`：当前 10 秒摘要，由对应 10 个 short tokens 重建。
- `L_local`：当前完整 60 秒摘要，由对应 60 个 short tokens 重建。
- `L_global`：视频开始至当前的累计摘要，由更新后的 long memory 重建。

每项 CE 对有效目标 tokens 取平均。每次监督后立即更新参数：

| 分钟内时刻 | 本次损失 | optimizer step |
|---|---|---|
| 10、20、30、40、50 秒 | 当前区间的 `L_short` | 每次各更新一次 |
| 60 秒 | `L_short[50:60] + L_local[0:60] + L_global[begin:current]` | 三项求和后更新一次 |

这里不是六个 local loss 平均后每分钟更新一次。每分钟共六次参数更新；三项权重均为 1，但 short 监督出现频率更高。

## 3.5.4 状态保留与梯度截断

每次 optimizer step 后，已经生成的 short states 保留数值并 detach，不重算：

```text
10 秒：生成 s_0...s_9 → L_short → backward + step → detach 并保留
20 秒：生成 s_10...s_19 → L_short → backward + step → detach 并保留
...
60 秒：前 50 秒已 detach 的 states + 最后 10 秒的新 states
       → L_short + L_local + L_global → backward + step
       → detach 新 long memory，清空当前 short memory
```

长期状态仍包含此前递归传递的信息：`L_k = Compress(detach(L_{k-1}), S_k)`。只保留最新 long memory，不同时输入 L_{k-2} 等历史快照。

分钟末损失可以训练当前压缩器、解码器和最后 10 秒的 short 编码过程，但不能回传到前 50 秒或上一分钟的编码计算。旧状态由更新前参数生成，不重算；这是第一版明确采用的近似。

同一视频按时间顺序处理，不打乱 blocks。切换视频或开始新 epoch 时清空 memory。梯度不跨分钟，并且在每个 10 秒更新点截断既有 short states。

## 3.5.5 模型与记忆表示

- 模型输入 FPS 固定为 1：每秒区间结束时取一帧，生成一个 short token。不能在区间结束之前使用该状态。
- Short 编码、long 压缩、摘要解码共用一个底座和一套 Stage 2 LoRA。特殊 token embedding 与输出层随 checkpoint 保存。
- Memory 是连续 hidden-state 向量；60 个 tokens 对应 `[batch, 60, hidden_dim]`，不是 60 个词。
- 将 memory 向量替换到下次输入的占位 token embedding，第一版不另加投影层。
- Long 更新输入依次包含旧 long、当前 short、60 个重复的 `<LONG_MEM>` 查询位置；取最后 60 个位置的 hidden states 作为新 long。查询共享 token embedding，由位置区分，不是 60 个独立可学习向量。
- 摘要文本仅用于监督，不作为后续 memory 更新的输入。Stage 2 不使用 ASR。

---

# 3.6 Short / Long Memory 的职责

两种 memory 的监督目标有明确区分。

### Short Memory

它学习：

> **最近一分钟发生了什么？**

Target：

```text
VLM-generated 10-second local summaries
+ 60-second block summary
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
    = recent fine-grained evidence + current-block integration

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

但是 short memory 仍然继续每秒产生。尾部完整的 10 秒区间继续进行 L_short 监督与参数更新；最后不足 10 秒不计算摘要损失。当前离线标注／训练入口只处理有完整 10 秒标签的区间，剩余秒数留给在线推理维护。

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

# 4.2 输入与因果注意力

第一版采用 memory-only 输入。固定模板使记忆位于问题之前，便于不同问题复用同一前缀：

```text
共享前缀                             每个问题的独立后缀
[固定 system / 模板] [Long memory L_t] [Short memory S_t] [Question Q_i] [<DECISION>]
                                                                  │
                                             h_dec[i,t] → Linear → z[i,t]
```

Long/Short 是连续 embedding，Question 和 `<DECISION>` 是文本／特殊 token embedding。模型采用因果注意力，因此 `<DECISION>` 可以读取当前全部记忆和完整问题，不能读取后续答案 tokens。Stage 2 构建的持久 memory 不依赖问题；Stage 3 问题相关的 hidden states 不写回持久 memory。当前原始帧留作后续 ablation。

## 4.2.1 `<DECISION>` 的 BCE 与 WAIT/ANSWER 文本 NTP

两种样本都训练可回答性判断，也都训练文本输出：

```text
y[i,t] = 0：当前视觉证据不足；目标文本 = <WAIT> + 当前缺少证据的具体原因
y[i,t] = 1：当前视觉证据充分；目标文本 = <ANSWER> + 有视觉依据的答案

L_decision = BCEWithLogitsLoss(z[i,t], y[i,t])
L_text     = NTP(目标文本)
L_stage3   = λ_decision · L_decision + λ_text · L_text
```

`<DECISION>` 是输入 readout token。目标文本位于它后面的 assistant 输出位置；NTP 监督输出的标记和原因／答案，固定前缀、记忆、问题、`<DECISION>` 与 assistant 模板均不计算文本损失。WAIT 原因只说明当前缺少什么证据，不泄漏未来画面或结论。因果注意力使决策位置看不到目标文本，因此 BCE 与 NTP 可以同一次前向计算。

推理先由 logit 与验证集阈值选择 WAIT/ANSWER，再根据当前记忆生成相应文本。可固定已选的 `<WAIT>`／`<ANSWER>` 输出前缀，保证生成分支与决策头一致；WAIT 问题保持 active，待新证据到来后重新判断。初始设 `λ_decision = λ_text = 1`。

## 4.2.2 多问题独立分支

```text
同一时刻的固定前缀 + L_t + S_t → 一次前缀前向 → 共享 memory KV
                                                    ├── Q1 + <DECISION> → z1
                                                    ├── Q2 + <DECISION> → z2
                                                    └── Q3 + <DECISION> → z3
```

每个问题有独立的 `<DECISION>`、logit 和可选答案后缀，不串接不同问题。共享前缀在因果注意力下不依赖后续问题；实现可复制各问题后缀 cache 或使用只读前缀分支，不能让一个问题的后缀修改其他问题的 KV。第一版每个决策时刻重建一次共享前缀，新 short memory 或 long 更新后让旧 Question KV 失效；具体生命周期见 §5.6。

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

# 4.5 WAIT Reason 与 Answer Generation

对于每个样本，NTP 目标取决于可回答性标签：

```text
y = 0 → <WAIT> + wait_reason
y = 1 → <ANSWER> + answer
```

`wait_reason` 应说明当前缺少的具体视觉证据，不应包含未来才可见的具体结论。ANSWER 文本应由当前已经可见的证据支持。

输出文本由 `<DECISION>` 后的 assistant 分支进行 next-token prediction。只对输出标记和后续文本计算交叉熵；`<DECISION>` 自身只作为二分类 readout。两种样本都进行 BCE 和 NTP 训练。

---

# 4.6 Stage 3 总 Loss

```text
L_stage3 = lambda_decision * L_decision + lambda_text * L_text
L_text   = NTP(<WAIT> + wait_reason)  if y = 0
         = NTP(<ANSWER> + answer)     if y = 1
```

第一版初始权重为 `lambda_decision = lambda_text = 1.0`。训练日志分别记录 decision loss、WAIT 文本损失和 ANSWER 文本损失，以检查判断与生成是否同步改善。

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

## 5.6 KV cache 生命周期（第一版）

这是已确认的实现规格，当前 Stage 3 独立决策头与多问题 cache 调度尚待落地。

区分两类状态：

- **Memory embeddings**：持久 long/short 向量，保存视频信息。
- **Stage 3 KV cache**：该记忆快照经问答模型前向计算后，各注意力层的 key/value，避免同一前缀重复计算。它不是 Stage 2 编码器内部 cache，也不能替代持久 memory。

第一版在每个决策时刻重建一次共享前缀，不跨时刻增量维护：

1. 接收已结束的一秒区间，完成 short 更新；若到达 60 秒边界，先更新 long 并清空 short。
2. 固定该时刻的 `(L_t, S_t)` 快照以及模型、模板与位置配置。
3. 对共享前缀执行一次 prefill，得到只读逻辑缓存。
4. 每个 active question 从该前缀独立分支，重新计算自己的 Question 和 `<DECISION>`，得到 logit。
5. WAIT 保持 active；新时刻有新证据时重新判断。旧 Question KV 不会自动获得新 memory 信息，必须失效。
6. ANSWER 标记为正在回答，使用触发时快照和独立分支生成答案；不要重复触发。后续视频可继续更新持久 memory，但不得原地修改该回答快照。完成后移出 active questions 并释放其分支。

| 事件 | 前缀 cache | Question／答案分支 |
|---|---|---|
| 同一 memory 状态下新增问题 | 复用已有前缀 | 创建独立分支 |
| 新决策时刻、short 更新 | 重建一次供所有问题共享 | WAIT 问题重新计算 |
| 60 秒边界 long 替换、short 清空 | 旧前缀失效并重建 | WAIT 分支失效；已触发答案保留自己的旧快照 |
| 切换视频／重置会话 | 清空 | 清空 |
| 权重、adapter、模板、位置配置变化 | 失效 | 对应分支失效 |

Cache 复用限定于固定参数、eval 模式的推理；不用于跨 optimizer step 复用训练计算图。评估延迟需包含前缀 prefill、各问题决策和生成的实际成本；问题分支 cache 的额外显存也应计入。

## 5.7 后续增量 cache 优化与验证

只有当旧 Long + Short 前缀完全不变、新 short token 真正追加在末尾，且 position IDs、mask、模板分隔符与模型设置保持兼容时，才可以增量追加 memory KV。若模板在 memory 后带有结束标记，也不能简单在标记之后插入新 short token而保持输入等价。第一版统一重建，不使用该优化。

验收至少包括：

- 同一输入的完整前向与缓存分支 logits 在指定数值容差内一致。
- 改变问题处理顺序，不改变各问题结果。
- 一个问题的生成不修改其他问题的前缀／后缀。
- 新证据到来后 WAIT 问题使用新 memory，不能沿用旧问题 KV。
- 60 秒边界及视频切换不会残留旧前缀。

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

> 在未见视频上，正常视频输入是否优于打乱视频输入；这有助于区分视觉利用与领域语言先验。

---

# 6.2 Stage 2 Summary Compression Evaluation

## Short-memory Reconstruction

评估：

```text
S_{k,j} → 对应 10 秒的 local summary
S_k → 当前 60 秒的 block summary
```

核心问题：

> Short memory 是否既保留每个 10 秒区间的细节，又能支持整分钟的信息整合？

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
    video + earlier ASR / masked earlier ASR → target-sentence supervision

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

当前入口每个完整分钟调用教师八次：六个 10 秒 local、一个当前 60 秒 block、一个从视频开始到当前分钟末的 global。标注器向教师发送静音视频片段。对于 `K` 分钟视频，调用次数约为 `8K`；global 视频片段的累计时长为 `60×(1+2+...+K)` 秒，随 `K` 近似二次增长。实际视觉 token 数还取决于教师服务的视频解码、采样和上下文限制，不能等同于原视频全部帧数。长累计片段可能超过接口限制，需在制数前实测并记录失败情况。

另一种研究用的 rolling 标注方式是：

```text
Y_k^{global}
=
VLMAnnotator(
Y_{k-1}^{global},
V_k
)
```

即用上一次 global 摘要和当前分钟更新标签。它可减少历史画面重复输入，但会累积标注错误；新 `stage2.annotate` 尚未实现这种模式。当前第一版直接输入累计视频片段，应审核视觉证据覆盖率并评估长视频的标注成本与服务限制。

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

> 整个系统首先通过三组视频／历史 ASR 保留与遮蔽条件预测同一讲解句子，进行 Stage 1 视觉-语言知识注入与视觉依赖对照；随后在 Stage 2 中完全脱离 ASR，每秒把当前超声画面压缩成一个 short-memory token，每 60 秒再将上一轮 60 个 long-memory tokens 与当前 60 个 short-memory tokens 递归压缩成新的 60 个 long-memory tokens，并使用离线 VLM 标注器为每 10 秒生成 local summary label、为当前一分钟生成 block summary label、为 0→T 历史视频生成 global summary label，构建 Stage 2 的 summary-compression 训练数据，使 short / long memory 学会保留关键视觉证据；最后在 Stage 3 中，通过单个 `<DECISION>` hidden state 和 `BCEWithLogitsLoss` 估计当前问题的可回答概率，证据不足时 WAIT，证据充分时才生成最终答案。
