# Live Ultrasound Video Understanding — Pipeline

> **范围：** 数据准备、Stage 1 超声视觉-语言知识注入、Stage 2 摘要压缩（long-term memory, short-term memory）学习、Stage 3 可回答性判断与 QA、在线推理、评估
> **核心原则：** 系统必须判断“当前已经看到的超声证据是否足以回答问题”。证据不足时继续等待并观察视频；证据足够时再生成答案。

---

## 0. 系统总览

### 0.1 目标

本项目面向的是 **实时超声视频理解（live ultrasound video understanding）**，而不是传统的离线 Video QA。

模型持续接收超声视频流（每秒更新，FPS=1）。当用户提出问题后，系统需要根据“当前时刻之前已经观察到的证据”判断问题是否已经可回答。

- `WAIT`：当前视觉证据不足，系统继续观察后续视频；
- `ANSWER`：当前证据已经充分，可以开始生成最终答案。

---

### 0.2 完整 Pipeline

完整系统从原始超声视频开始，依次完成数据标注、数据筛选、三阶段训练和最终评估。

**第一步：收集原始超声视频。**

输入是未经过模型训练处理的原始超声相关视频。这些视频可能包含纯超声机画面、床旁扫查教学、混合教学内容、PPT、静态图像讨论或无关片段。因此，原始视频不能直接全部用于所有训练阶段，必须先生成辅助标注并进行筛选。

**第二步：为原始视频生成 ASR transcript。**

ASR transcript 有三个用途。第一，它用于 ASR quality rule filter，例如检查语言、长度、超声关键词数量和重复文本比例，从而剔除明显错误或无价值的 transcript。第二，它为 clipping 提供 sentence-boundary signal，使视频切分边界尽量落在自然句子边界附近，而不是在一句话中间截断。第三，它是 Stage 1 的 narration supervision，用于训练模型建立超声视觉内容和医学语言之间的对应关系。

需要强调的是：ASR 只作为数据准备和 Stage 1 监督信号使用。Stage 2、Stage 3 和在线推理阶段不把 ASR transcript 作为输入。

**第三步：对原始视频进行 VLM classification。**

VLM classification 由 Teacher VLM 对完整视频进行分类和质量判断。它输出 video type、anatomy、clinical scenario、是否包含实时超声画面、是否需要 clipping，以及面向不同训练阶段的 keep flags：

- `keep_for_pretrain`：是否适合 Stage 1 domain pretraining；
- `keep_for_compression`：是否适合 Stage 2 Summary Compression；
- `keep_for_sft`：是否适合 Stage 3 QA / SFT。

这一步的作用是把“视频是否超声相关”和“视频适合哪个训练阶段”显式标注出来。不同阶段对视频质量的要求不同，因此不能只用一个统一的 keep/drop 决策。

**第四步：进行数据选择、视频过滤和 clipping。**

数据选择综合使用 ASR rule filter、VLM keep flags 和 clipping 结果。ASR rule filter 主要处理 transcript 质量问题；VLM keep flags 负责 stage-specific keep / drop；clipping 则结合 visual-change detection 和 ASR sentence-boundary alignment，把长视频或混合视频切成更适合训练的连续片段。

不同阶段使用不同的数据策略：

- Stage 1 使用范围最广，可以使用所有 ultrasound-related videos，因为目标是注入超声视觉-语言知识；
- Stage 2 优先使用 `pure_ultrasound_scan`、`hands_on_ultrasound_teaching`，以及高质量或已经 clipping 的 `mixed_ultrasound_teaching`，因为目标是学习真实流式视觉记忆；
- Stage 3 优先使用 pure / hands-on / clipped high-quality mixed 视频构造 QA，因为目标是训练模型在 streaming 条件下判断 WAIT 还是 ANSWER。

**第五步：Stage 1 — 超声视觉-语言知识注入。**

Stage 1 使用视频帧作为输入，使用对齐的 ASR narration 作为监督信号。模型在这一阶段学习超声画面、解剖结构、扫查动作和医学描述之间的基础对应关系。Stage 1 可以看作后续 Summary Compression 和 QA 的视觉-语言基础预训练。

**第六步：Stage 2 — Summary Compression Stage。**

Stage 2 不再使用 ASR，而是学习如何把连续超声视频压缩成 streaming memory。系统每 1 秒接收当前视频帧，并生成 1 个 short-memory token。每 60 秒，系统将上一轮 60 个 long-memory tokens 与当前 60 个 short-memory tokens 结合，递归更新为新的 60 个 long-memory tokens。

Stage 2 的主要监督信号来自 Teacher VLM 的视觉摘要：

- 对当前窗口 `video[T-60:T]`，Teacher 生成 `local visual summary`，用于监督 short memory；
- 对累计历史 `video[0:T]`，Teacher 生成 `cumulative visual summary`，用于监督 long memory。

Student 训练时需要从 short memory 重建 local summary，并从 long memory 重建 cumulative summary。这里的目标不是复述 ASR，而是让 memory token 学会保存可用于后续 QA 的视觉证据。

**第七步：Stage 3 — Streaming Answerability + QA。**

Stage 3 在用户提出问题后工作。模型输入包括 long memory、当前 short memory、question、`<DECISION>` token，以及可选的当前帧。模型首先通过 `<DECISION>` hidden state 产生 answerability logit，用于判断当前已经看到的证据是否足够回答问题。

如果 `p_answer < threshold`，系统输出 `WAIT`，继续观察后续视频；如果 `p_answer ≥ threshold`，系统输出 `ANSWER` 并生成最终答案。Stage 3 的核心不是单纯回答问题，而是在 streaming 场景下学习“什么时候应该等待，什么时候可以回答”。

**第八步：Evaluation。**

评估同时覆盖三个层面：Stage 1 是否学到超声视觉-语言对应关系，Stage 2 的 short / long memory 是否保留了关键视觉证据，Stage 3 是否能准确判断 answerability 并生成正确答案。最终系统必须同时满足两点：答案内容正确，以及回答时机正确。

---

### 0.3 各 Stage 的职责边界

| Stage | 模型输入 | 监督信号 | 学习目标 |
|---|---|---|---|
| Stage 1 | 超声视频帧 | 对齐的 ASR narration | 超声视觉-语言知识 |
| Stage 2 | 仅超声视频 | Teacher VLM 生成的局部 / 累积视觉摘要 | Streaming short / long memory |
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

Long memory 是 **全量替换（replace）**，不是不断 append。

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
2. Teacher VLM 的视频类型 / anatomy / clinical scenario 分类
```

推荐 Teacher 配置：

```text
Primary open-source teacher:
    Qwen/Qwen3.5-35B-A3B
    通过本地 vLLM OpenAI-compatible endpoint 提供服务

Cross-validation teacher:
    Gemini 3 Pro
    通过 OpenRouter / Google-compatible endpoint 调用
```

推荐视频类型标签：

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

## 1.4 各 Stage 的 Keep Policy

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

Stage 2 不推荐直接使用 slide-heavy / discussion-only 视频，因为它们会让 memory 学到与真实 online ultrasound stream 不一致的内容。

---

### Stage 3

Streaming QA 数据优先来自：

```text
pure_ultrasound_scan
hands_on_ultrasound_teaching
高质量、已经 clipping 的 mixed_ultrasound_teaching
```

---

## 1.5 Teacher 视频分类脚本

Qwen3.5：

```bash
python scripts/data/classify_ultrasound_video_type_teacher.py \
  --video-map cluster_data/splits/train_full295_asr_keep_videos.json \
  --output cluster_data/splits/train_full295_qwen35_video_type.jsonl \
  --teacher qwen35 \
  --model Qwen/Qwen3.5-35B-A3B \
  --base-url http://localhost:8000/v1 \
  --api-key-env VLLM_API_KEY \
  --video-fps 1.0 \
  --resume
```

Gemini cross-validation：

```bash
python scripts/data/classify_ultrasound_video_type_teacher.py \
  --video-map cluster_data/splits/train_full295_asr_keep_videos.json \
  --output cluster_data/splits/train_full295_gemini3_video_type.jsonl \
  --teacher gemini3 \
  --model google/gemini-3-pro \
  --base-url https://openrouter.ai/api/v1 \
  --video-fps 1.0 \
  --resume
```

合并两个 Teacher：

```bash
python scripts/data/merge_video_type_teacher_labels.py \
  --primary-audit cluster_data/splits/train_full295_qwen35_video_type.jsonl \
  --validator-audit cluster_data/splits/train_full295_gemini3_video_type.jsonl \
  --output cluster_data/splits/train_full295_video_type_final.jsonl \
  --output-summary cluster_data/splits/train_full295_video_type_final_summary.json
```

---

## 1.6 Qwen vLLM Serving

推荐启动方式：

```bash
uv pip install vllm \
  --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/nightly

vllm serve Qwen/Qwen3.5-35B-A3B \
  --host 0.0.0.0 \
  --port 8000 \
  --media-io-kwargs '{"video":{"num_frames":-1}}'
```

如果需要通过 HTTP 暴露本地视频：

```bash
cd /path/to/videos
python -m http.server 9000
```

然后：

```bash
--video-url-base http://<node-hostname-or-ip>:9000
```

集群全量运行：

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
ENABLE_THINKING=0
LIMIT=10
HTTP_PORT=9000
VLLM_PORT=8000
```

大规模 JSON 标注默认关闭 thinking，以提高格式稳定性；复杂样本人工复核时可单独开启。

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

\[
V_{t_0:t_1}
\rightarrow
Y^{ASR}_{t_0:t_1}
\]

---

## 2.4 Loss

使用标准 causal LM loss：

\[
\mathcal L_{\text{stage1}}
=
CE(\hat Y^{ASR},Y^{ASR})
\]

只监督 assistant target tokens。

以下 token label 置为：

```text
-100
```

包括：

```text
system prompt
user prompt
visual prompt / formatting tokens
```

---

## 2.5 Scripts

```text
pretrain/build_samples.py
pretrain/train.py
pretrain/infer.py
```

构造样本：

```bash
python pretrain/build_samples.py \
  --transcripts results/transcripts \
  --output pretrain/data/pretrain_samples.jsonl \
  --unit sentence \
  --window-sec 8 \
  --min-words 3
```

训练：

```bash
python pretrain/train.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --train-jsonl pretrain/data/pretrain_samples.jsonl \
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

目标主线中，Stage 2 **完全不使用 ASR**。当前 repo 已实现的 ASR reconstruction 版本仅作为 baseline / bootstrapping，用于验证 memory-token 训练链路。

系统包含两级 memory：

```text
Short-term memory:
    每秒 1 个 token
    保存最近的细粒度超声视觉证据

Long-term memory:
    固定 60 个 tokens
    递归压缩并保存更长时间范围内的重要历史证据
```

为了训练 memory 保留“重要的视觉信息”，使用更强的 Teacher VLM 离线生成视觉摘要作为 supervision。

---

# 3.2 Teacher Summary Supervision

## 3.2.1 视频选择

Teacher summary 只优先生成在高质量 streaming ultrasound 视频上：

```text
pure_ultrasound_scan
hands_on_ultrasound_teaching
```

后续可以加入已经筛选 / clipping 的高质量 realtime mixed 视频。

---

## 3.2.2 时间定义

设一个 block 为 60 秒。

第 \(k\) 个 block 的结束时刻：

\[
T_k=60k
\]

当前一分钟：

\[
V_k
=
V[T_{k-1}:T_k]
\]

截至当前时刻已经观察到的完整历史：

\[
V_{\le k}
=
V[0:T_k]
\]

对于每一个完整分钟，Teacher VLM 生成两个 target：

```text
local summary
global / cumulative summary
```

---

## 3.2.3 Local Summary

Teacher 只观察当前 60 秒：

\[
Y_k^{local}
=
Teacher(V[T_{k-1}:T_k])
\]

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

Teacher 输入从视频开头到当前时刻的全部视频：

\[
Y_k^{global}
=
Teacher(V[0:T_k])
\]

它回答的问题是：

> **从视频开始到现在，已经出现过哪些重要的超声视觉证据？**

示例：

```text
目前检查已经覆盖右上腹和剑突下心脏切面。
已经观察到肝脏、右肾、肝肾隐窝以及心脏。
截至当前时刻尚未显示明显游离液体。
```

---

## 3.2.5 Teacher Summary 约束

Teacher summary 的目标是：

> **描述视频中已经存在并能够被视觉证据支持的信息。**

而不是自由进行诊断推理。

推荐 Teacher Prompt：

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

Teacher 可以使用医学知识识别结构和超声征象，但不能凭空加入视频未显示的临床结论。

---

# 3.3 Short-Term Memory

## 3.3.1 每秒更新

视频流每秒处理一次。

对于时间 \(t\)：

\[
x_t
=
\text{frames in }[t,t+1]
\]

向视觉输入后加入特殊 query token：

```text
<SHORT_MEM>
```

取该位置 hidden state：

\[
s_t
=
hidden\_state(
\texttt{<SHORT\_MEM>}
\mid x_t
)
\]

因此：

```text
第 0 秒 → s_0
第 1 秒 → s_1
第 2 秒 → s_2
...
```

一个完整 60 秒 block 得到：

\[
S_k
=
[s_{60(k-1)},...,s_{60k-1}]
\]

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

\[
L_k
=
[l_k^1,\ldots,l_k^{60}]
\]

形式化：

\[
L_k
=
Compress(L_{k-1},S_k)
\]

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

新的 \(L_k\) 全量替换旧的 \(L_{k-1}\)。

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

因此，第 \(k\) 个 block 使用的：

\[
L_{k-1}
\]

是前面所有视频按照真实时间顺序递归生成出来的 memory state。

并不是独立构造的单个 previous block。

所以：

> **Stage 2 的训练 recurrence 与 online inference 的 recurrence 保持一致。**

---

# 3.5 Memory Reconstruction Supervision

## 3.5.1 Short-Memory Reconstruction

当前一分钟的 60 个 short-memory tokens：

\[
S_k
\]

需要重建 Teacher 为当前一分钟生成的 local summary：

\[
\hat Y_k^{local}
=
DecodeShort(S_k)
\]

Target：

\[
Y_k^{local}
=
Teacher(V[T_{k-1}:T_k])
\]

Loss：

\[
\boxed{
\mathcal L_{short}
=
CE(
\hat Y_k^{local},
Y_k^{local}
)
}
\]

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

\[
L_k
\]

需要重建从视频开始到当前时刻的累计视觉摘要：

\[
\hat Y_k^{global}
=
DecodeLong(L_k)
\]

Target：

\[
Y_k^{global}
=
Teacher(V[0:T_k])
\]

Loss：

\[
\boxed{
\mathcal L_{long}
=
CE(
\hat Y_k^{global},
Y_k^{global}
)
}
\]

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

\[
\boxed{
\mathcal L_{stage2}
=
\lambda_{short}\mathcal L_{short}
+
\lambda_{long}\mathcal L_{long}
}
\]

初始建议：

```text
lambda_short = 1.0
lambda_long  = 1.0
```

主线设计中，这套 supervision **替代旧版 ASR reconstruction baseline**：

```text
current ASR reconstruction
previous ASR reconstruction
accumulated ASR reconstruction
ASR concat + tail truncation
```

最终 Stage 2 主线不再依赖 ASR；如需快速验证工程链路，可继续运行当前已实现的 ASR reconstruction baseline。

---

# 3.6 Short / Long Memory 的职责

两种 memory 的监督目标有明确区分。

### Short Memory

它学习：

> **最近一分钟发生了什么？**

Target：

```text
Teacher local summary
```

---

### Long Memory

它学习：

> **从视频开始到现在，已经出现过哪些值得长期保留的重要视觉证据？**

Target：

```text
Teacher cumulative summary
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

# 3.8 Stage 2 Data Schema

## 3.8.1 Short-Memory Sample

```json
{
  "sample_type": "short_memory_summary",
  "video_id": "8V649L5Q368",
  "block_idx": 0,
  "block_window": [0.0, 60.0],
  "short_windows": [
    [0.0, 1.0],
    [1.0, 2.0],
    "...",
    [59.0, 60.0]
  ],
  "local_summary_target": "Teacher VLM 为当前 60 秒生成的摘要",
  "meta": {
    "source": "teacher_vlm",
    "step_sec": 1.0,
    "block_sec": 60.0
  }
}
```

---

## 3.8.2 Long-Memory Sample

```json
{
  "sample_type": "long_memory_summary",
  "video_id": "8V649L5Q368",
  "block_idx": 1,
  "block_window": [60.0, 120.0],
  "history_window": [0.0, 120.0],
  "short_windows": [
    [60.0, 61.0],
    "...",
    [119.0, 120.0]
  ],
  "num_long_tokens": 60,
  "local_summary_target": "Teacher summary for 60-120 s",
  "global_summary_target": "Teacher summary for 0-120 s",
  "meta": {
    "source": "teacher_vlm",
    "block_sec": 60.0,
    "step_sec": 1.0,
    "drop_last_incomplete": true,
    "chronological_rollout": true
  }
}
```

训练 DataLoader 必须保留同一视频内部的时间顺序，使 long-memory state 能够按时间递归传播。

---

# 3.9 Stage 2 Scripts

当前相关文件：

```text
pretrain/build_memory_compression_samples.py
pretrain/build_teacher_memory_summary_samples.py
pretrain/memory_dataset.py
pretrain/memory_collator.py
pretrain/train_memory_compression.py
pretrain/infer_memory_compression.py
pretrain/eval_memory_compression.py
```

Teacher-summary 主线 sample builder 读取预生成的：

```text
local_summary_target
global_summary_target
```

而不是 ASR concat target。

Teacher-summary 主线 build：

```bash
python pretrain/build_teacher_memory_summary_samples.py \
  --summaries-jsonl pretrain/data/teacher_visual_summaries.jsonl \
  --output pretrain/data/memory_compression_samples.jsonl \
  --block-sec 60 \
  --step-sec 1 \
  --long-token-count 60
```

当前已实现的 ASR reconstruction baseline 仍可用：

```bash
python pretrain/build_memory_compression_samples.py \
  --transcripts results/transcripts \
  --output pretrain/data/memory_compression_asr_baseline_samples.jsonl \
  --types short,long \
  --block-sec 60 \
  --step-sec 1 \
  --long-token-count 60
```

训练：

```bash
python pretrain/train_memory_compression.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --train-jsonl pretrain/data/memory_compression_samples.jsonl \
  --video-path-map pretrain/data/video_path_map.json \
  --output-dir /mnt/cache/qwenFT/qwen3vl_memory_compression \
  --short-frames 2 \
  --frame-size 224 \
  --lambda-all 1.0 \
  --lambda-current 1.0 \
  --lambda-previous 0.0 \
  --num-train-epochs 1 \
  --learning-rate 1e-4 \
  --bf16
```

Inference：

```bash
python pretrain/infer_memory_compression.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --adapter-path /mnt/cache/qwenFT/qwen3vl_memory_compression \
  --eval-jsonl pretrain/data/memory_compression_samples.jsonl \
  --output results/memory_compression_predictions.jsonl \
  --video-path-map pretrain/data/video_path_map.json \
  --short-frames 2 \
  --frame-size 224 \
  --bf16
```

Evaluation：

```bash
python pretrain/eval_memory_compression.py \
  --pred-jsonl results/memory_compression_predictions.jsonl \
  --output results/memory_compression_eval.json
```

> CLI 参数需要与最终代码实现同步。语义要求是：Stage 2 使用 Teacher visual summary supervision，并按真实时间顺序递归更新 memory。

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

在 streaming time \(t\)，输入：

```text
long memory L_t
+ 当前还未压缩的 short memory S_t
+ optional latest visual frames V_t
+ question Q
+ <DECISION>
```

取 `<DECISION>` 位置 hidden state：

\[
h_t^{dec}
\]

---

# 4.3 Answerability Head

用一个 binary classification head：

\[
z_t
=
W h_t^{dec}+b
\]

得到 scalar logit。

可回答概率：

\[
p_{answer}(t)
=
\sigma(z_t)
\]

Inference 时使用阈值 \(\tau\)：

\[
p_{answer}(t)<\tau
\Rightarrow
WAIT
\]

\[
p_{answer}(t)\geq\tau
\Rightarrow
ANSWER
\]

阈值 \(\tau\) 在 validation set 上确定，用于平衡：

```text
premature answer
over-wait
answer delay
```

---

# 4.4 Decision Supervision

每一个 decision sample：

\[
y_t=
\begin{cases}
0,&\text{当前证据不足}\\
1,&\text{当前证据充分}
\end{cases}
\]

Decision loss：

\[
\boxed{
\mathcal L_{decision}
=
BCEWithLogitsLoss(z_t,y_t)
}
\]

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

\[
\mathcal L_{answer}
=
CE(
\hat A,
A
)
\]

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

\[
\boxed{
\mathcal L_{stage3}
=
\lambda_{decision}\mathcal L_{decision}
+
y_t\lambda_{answer}\mathcal L_{answer}
}
\]

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

# 4.8 Stage 3 Data Schema

WAIT：

```json
{
  "video_id": "example_video",
  "question_id": "q1",
  "decision_time": 35.0,
  "query_time": 30.0,
  "answer_time": 50.0,
  "question": "肝肾隐窝中是否出现游离液体？",
  "decision_label": 0,
  "answer": null
}
```

ANSWER：

```json
{
  "video_id": "example_video",
  "question_id": "q1",
  "decision_time": 50.0,
  "query_time": 30.0,
  "answer_time": 50.0,
  "question": "肝肾隐窝中是否出现游离液体？",
  "decision_label": 1,
  "answer": "是，可以看到肝肾隐窝中出现无回声游离液体。"
}
```

---

# 4.9 Stage 3 Scripts

当前 QA 相关代码：

```text
QA/run.py
QA/train/train_summary_decide.py
QA/eval/infer_summary_decide.py
```

Stage 3 必须严格复用 Stage 2 的 memory semantics：

```text
每秒 1 个 short-memory token
完整 60 秒产生 60 short tokens
long memory 固定 60 tokens
每 60 秒递归更新并替换 long memory
```

Classifier 必须读取 `<DECISION>` 的 hidden state，并直接训练 binary answerability。

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

\[
x_t
\]

生成 short-memory token：

\[
s_t
=
EncodeShort(x_t,\texttt{<SHORT\_MEM>})
\]

然后：

```text
short_memory.append(s_t)
```

---

## 5.3 每 60 秒更新 Long Memory

当 short memory 累积到 60 个：

\[
S_k
=
[s_{60(k-1)},...,s_{60k-1}]
\]

更新：

\[
L_k
=
Compress(L_{k-1},S_k)
\]

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

\[
h_t^{dec}
\]

计算：

\[
p_{answer}
=
\sigma(W h_t^{dec}+b)
\]

如果：

\[
p_{answer}<\tau
\]

返回：

```text
WAIT
```

问题保持 active。

如果：

\[
p_{answer}\geq\tau
\]

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
S_k → local teacher summary
```

核心问题：

> Short memory 是否保留了最近一分钟内的重要视觉信息？

---

## Long-memory Reconstruction

评估：

```text
L_k → cumulative teacher summary
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

阈值 \(\tau\) 应在 validation set 上调好，并在 test evaluation 前固定。

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

## 7.3 Teacher Summary 只监督视觉证据

Teacher summary 用于指导 memory：

> 什么信息值得保留。

而不是引导模型学习未经视觉支持的诊断推理。

因此 Teacher summary 应尽可能接近：

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

\[
L_t
\]

保存。

这是整个 streaming setting 的关键。

---

# 8. 当前局限

## 8.1 Teacher Summary 成本

对于每一分钟都重新执行：

\[
Teacher(V[0:T])
\]

会不断重复处理历史视频。

一个 \(K\) 分钟的视频，总输入规模近似：

\[
1+2+\cdots+K
=
O(K^2)
\]

例如：

```text
10 分钟：
1+2+...+10 = 55 分钟等价视频输入

20 分钟：
1+2+...+20 = 210 分钟等价视频输入
```

作为第一版高质量 baseline 是可以接受的，但大规模数据时成本会较高。

后续可尝试：

\[
Y_k^{global}
=
Teacher(
Y_{k-1}^{global},
V_k
)
\]

即 rolling Teacher summary。

但 rolling summary 会存在 teacher error accumulation，因此第一版仍建议优先使用：

```text
raw video[0:T] → Teacher global summary
```

---

## 8.2 Teacher Summary 质量依赖

Stage 2 supervision 质量高度依赖 Teacher 输出是否满足：

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

## 9.4 Teacher Summary Strategy

比较：

```text
Direct global summary:
    Teacher(video[0:T])

vs.

Rolling global summary:
    Teacher(previous_global_summary, video[T-60:T])
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

# 10. Implementation Checklist

## Data

- [ ] 已生成 Stage 1 ASR supervision
- [ ] 已完成视频类型分类
- [ ] 已筛选 Stage 2 高质量视频
- [ ] 每个完整分钟已生成 Teacher local summary
- [ ] 每个完整分钟已生成 Teacher cumulative summary
- [ ] 已对 Teacher summary 做质量检查

---

## Stage 1

- [ ] semantic input 只有 video frames
- [ ] aligned ASR 只作为 target
- [ ] 不输入 previous ASR context
- [ ] 只监督 assistant target tokens

---

## Stage 2

- [ ] 每秒生成 1 个 short-memory token
- [ ] 完整 60 秒形成 60 short tokens
- [ ] long memory 始终为 60 tokens
- [ ] 每 60 秒全量替换 long memory
- [ ] 训练采用 chronological recursive rollout
- [ ] short memory 重建 local Teacher summary
- [ ] long memory 重建 cumulative Teacher summary
- [ ] Stage 2 完全不使用 ASR

---

## Stage 3

- [ ] 每个 sample 只有一个 `<DECISION>` readout token
- [ ] `<DECISION>` hidden state 输入 scalar classifier
- [ ] decision 使用 `BCEWithLogitsLoss`
- [ ] WAIT sample 不训练 answer text
- [ ] ANSWER sample 同时训练 decision + answer generation
- [ ] threshold 在 validation set 上 calibration
- [ ] memory semantics 与 Stage 2 完全一致

---

## Online Inference

- [ ] 每秒更新一次 short memory
- [ ] 每完整 60 秒更新一次 long memory
- [ ] WAIT 后问题继续保持 active
- [ ] 新证据到达后重新进行 decision
- [ ] 不重新输入历史 raw video

---

# 11. 一句话总结

> 整个系统首先通过“超声视频 → ASR narration”进行 Stage 1 视觉-语言知识注入；随后在 Stage 2 中完全脱离 ASR，每秒把当前超声画面压缩成一个 short-memory token，每 60 秒再将上一轮 60 个 long-memory tokens 与当前 60 个 short-memory tokens 递归压缩成新的 60 个 long-memory tokens，并分别使用 Teacher VLM 为当前一分钟生成的 local summary 和为 0→T 历史视频生成的 cumulative summary 监督 short / long memory；最后在 Stage 3 中，通过单个 `<DECISION>` hidden state 和 `BCEWithLogitsLoss` 估计当前问题的可回答概率，证据不足时 WAIT，证据充分时才生成最终答案。
