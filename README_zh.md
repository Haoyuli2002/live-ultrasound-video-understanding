# 实时超声视频理解

[English](README.md) | [简体中文](README_zh.md)

本项目研究**超声视频问答中的流式证据充分性**：持续观察视频，在视觉证据不足时等待，在证据充分时回答。

目标系统结合超声领域预训练、固定容量的视觉记忆和显式可回答性决策。仓库包含数据准备工具、训练与评估基线，以及持续演进的方法实现。目前尚不是已经完成验证的在线临床系统。

## 设计与实现状态

[参考设计](docs/live_ultrasound_reference_implementation_final_zh.md)定义目标三阶段系统。[运行命令](docs/HOW_TO_RUN_PIPELINE.md)中部分示例仍对应旧基线，复现时应检查 CLI 参数与以下状态。

| 模块 | 目标设计 | 当前实现 |
|---|---|---|
| 数据准备 | 获取视频、ASR 转写、文本清洗与裁剪 | 已有爬虫、ASR、视频裁剪与筛选，以及 Qwen3.5-27B 逐段 ASR 清洗脚本 |
| VLM 打标 | 视频分类、解剖部位与临床场景标注，按训练阶段筛选 | 已有 Qwen3.5 主教师、可选 Gemini 交叉校验、标签合并，以及 pretrain/compression/sft 保留标记筛选 |
| Stage 1：领域预训练 | `0→start`、`0→end` 与历史 ASR 遮蔽三组配对输入 → 同一完整讲解句子 | 新建独立 `stage1/` 样本、LoRA 训练和视觉对照评估代码；GPU 实跑待验证 |
| VLM 数据生成：local/block/global summary | 每个完整分钟生成六条 10 秒、一条整分钟和一条累计视觉摘要，为 Stage 2 提供监督 | 新 `stage2.annotate` 与 `stage2.build_data` 已实现八条标注及转换；符合新规格的数据尚未实际生成和审核，旧分钟级标注不可直接使用 |
| Stage 2：视觉记忆 | 每秒一个短期 token；每 60 秒全量更新为 60 个长期 token；使用 local/global summary 监督 | 新建独立 `stage2/` 时序训练和流式推理代码；GPU 实跑待验证 |
| QA 构建 | 标注 `query_time` 和最早证据充分的 `answer_time`，并验证 | 已有生成、验证和 WAIT/ANSWER 展开；自动时间戳是候选标注，不是专家真值 |
| Stage 3：可回答性与 QA | `<DECISION>` 隐状态 → BCE 二分类；WAIT 原因与 ANSWER 答案均用 NTP 监督 | 当前生成 `<WAIT>`/`<ANSWER>` 文本，尚未实现独立决策头 |
| Eval：评估 | 因果流式评估：可回答性、回答时机、答案质量、记忆保留与运行效率 | 已有预训练、记忆与 QA 基线评估脚本；完整在线循环和 benchmark 协议待实现与验证 |

## 模型流程

ASR 用于数据准备和 Stage 1 的训练目标；在 Stage 1 的两组条件中，目标句之前的历史 ASR 也作为输入。Stage 2、Stage 3 和在线推理不输入 ASR。离线标注器可以观察未来构建标签，但模型在决策时只能读取当时已经可见的证据。

### 各阶段接口与衔接

三个阶段共用**同一个 Qwen3-VL backbone**；每个阶段在上一阶段产物之上追加参数，而不是另起一个独立网络。下表是各阶段"产出什么、下一阶段消费什么、哪些衔接尚未实现"的唯一事实来源，文件行号对应当前代码。

| 阶段 | 新增可训练参数 | 产出 | 下一阶段消费 |
|---|---|---|---|
| Stage 1 | 在 `q/k/v/o_proj` + `gate/up/down_proj` 上的 LoRA（`stage1/model.py:65-68`） | `stage1_output/final` 的 LoRA adapter | Stage 2 的基座权重 |
| Stage 2 | 新的 LoRA，**外加**两个可学习 token embedding `<SHORT_MEM>` / `<LONG_MEM>`，经 PEFT `trainable_token_indices` 训练（`stage2/model.py:23-38`） | `stage2_output/final` 的 adapter + resize 后的 embedding；以及流式记忆 `.pt`（`stage2/stream.py:62-64`） | Stage 3 的记忆输入 |
| Stage 3 | 专用 `<DECISION>` 读出位 + BCE 头（仅设计） | WAIT/ANSWER 文本 + 可答性 logit | — |

**Stage 1 → Stage 2。** Stage 2 启动时先加载 Stage 1 adapter，并用 `PeftModel.from_pretrained(base, base_adapter).merge_and_unload()` 将其烤进 backbone（`stage2/model.py:21-22`），再在其上加自己的 LoRA 和两行新 token。因此 Stage 1 注入的领域知识在记忆训练开始前已固化进权重。合并所用的 adapter 路径记录在 `training_config.json`，供推理复现（`stage2/stream.py:46-50`）。

**这两个新 token 是什么。** `<SHORT_MEM>` 和 `<LONG_MEM>` 不是额外的网络模块，而是词嵌入表里新增的两行向量（`stage2/model.py:23-24`），用作可学习的**读出查询位**。每帧后接一个 `<SHORT_MEM>`，取其隐状态作为该秒的短记忆向量（`encode_frame`，`stage2/model.py:50-65`）；对 `[detach 的旧长记忆, 当前 60 个短向量]` 施加 60 个 `<LONG_MEM>` 查询，得到 60 个新长记忆向量，**整块替换**旧长记忆（`update_long`，`stage2/model.py:67-79`）。只有 LoRA 权重和这两行 embedding 接收梯度，其余词表全部冻结。

**Stage 2 → Stage 3。** 推理时 `stage2/stream.py` 以同样的因果 1-FPS 递归运行，但**不使用教师标签**，并保存一个 `.pt`，内含 `snapshots`（每分钟的长记忆快照）、`final_long`（至少满一分钟后形状为 `[1, 60, hidden]`）、`final_short`（形状 `[1, k, hidden]`，`k<60`，当前未满一分钟的残余）（`stage2/stream.py:14-31,62-64`）。这些是**记忆向量，不是文本**：local/block/global 文本摘要只在 Stage 2 训练时作为监督目标，用来逼这些向量"可被重建"（`reconstruction_loss`，`stage2/model.py:81-102`）；Stage 3 直接消费压缩后的向量，作为下文 Stage 3 输入图中的 `[Long memory] [Short memory]` 前缀。

**尚未实现（开放的衔接点）。** 目前没有任何 Stage 3 代码读取 Stage 2 的 `.pt`，专用 `<DECISION>` 头也未实现。`QA/train/train_summary_decide.py` 会追加长记忆 token 并生成 WAIT/ANSWER 文本，但尚未与 Stage 2 的固定 60-token 替换更新对齐，也没有 BCE 决策头。在出现 `.pt` reader 和决策头之前，请把 Stage 2 → Stage 3 的记忆接口视为规范说明，而非可运行的流水线。

#### 各阶段 input / target 与示例

下表逐阶段列出模型的输入、监督目标，以及一个具体示例。字段名对应真实代码。

**Stage 1 — 领域下一句预测**（`stage1/data.py:105-117`，`stage1/model.py:14-55`）

- **Input**：`[0, video_window_end]` 的因果视频帧（每条样本在该时间范围内最多 120 帧）+ 一段 `Earlier narration:`（目标句之前的历史 ASR，或在 mask 条件下写成 `[ASR MASKED]`）+ 指令句。每句生成**三条条件**样本：`before_with_asr`（视频到句首 + 历史 ASR）、`through_with_asr`（视频到句尾 + 历史 ASR）、`before_mask_asr`（视频到句首 + 遮蔽 ASR）。
- **Target**：**同一句**完整解说文本（`target`）。只有 target token 计 loss，system/帧/历史 ASR/指令全部 mask 掉（`model.py:49-51`）。

```text
真实示例（8V649L5Q368，before_with_asr）
  Input  = frames[0–4.46s] + "Earlier narration: Hi, I'm Dr. John Kugler ... series."
  Target = "Today, we're going to be learning about lung ultrasound."
  before_mask_asr：Input/Target 相同，仅 narration → "[ASR MASKED]"。
```

**Stage 2 — 时序视觉记忆**（`stage2/build_data.py`，`stage2/model.py`，`stage2/train.py`）

- **Input**：**静音**视频，每满 1 秒取 1 帧（无 ASR）。按时间顺序处理。
- **Target**（每满一分钟 `T` 由教师给出 8 条，构成一个 block）：6 条 10 秒局部摘要 `local_sub_summaries`、1 条 `[T-60,T)` 整分钟摘要 `block_summary_target`、1 条 `[0,T)` 累计摘要 `global_summary_target`。训练时用这些文本去重建对应的记忆向量（short/block/global），损失按 `1:1:1`。

```text
示例（第一分钟 block_window=[0,60]）
  Input = 每秒 1 帧，第 0..59 秒（静音）

  每条 loss = reconstruction_loss(记忆向量, 目标文本, kind)：
    只对目标 token 做 teacher-forced CE（记忆 + 提示被 mask 掉）。

  Lshort  （触发 6 次，每 10 秒末）：
    memory = 该 10 秒的 10 个短向量        例 [40,50) -> short[40..49]
    target = "右肝横切面，未见局灶性病变。"            (local[4])
    loss   = CE(target | 10 个短向量)

  Lblock  （触发 1 次，第 60 秒）：
    memory = 本分钟全部 60 个短向量
    target = "本分钟：右肝扫查完成，未见占位。"         (block)
    loss   = CE(target | 60 个短向量)

  Lglobal （触发 1 次，第 60 秒）：
    memory = update_long(旧长记忆, 60 短) 产出的 60 个“新”长向量
    target = "至今：已扫查右肝各切面，回声均匀。"       (global)
    loss   = CE(target | 60 个长向量)

  各步更新：
    第 10,20,30,40,50 秒 -> total = Lshort
    第 60 秒            -> total = Lshort + Lblock + Lglobal   (1:1:1)
```

**Stage 3 — 可答性决策 + QA**（设计见根 README 的 Stage 3 小节；`QA/train/train_summary_decide.py`，*决策头尚未实现*）

- **Input**：共享记忆前缀 `[系统/模板][Long memory][Short memory]` + 当前问题 + 读出位 `<DECISION>`。记忆来自 Stage 2 `stream.py` 产出的向量，不随问题改写。
- **Target**：① 可答性二分类标签 `y`（`y=0` 证据不足 / `y=1` 证据充分）用 BCE 监督 `<DECISION>` 的 logit；② 文本目标用 NTP 监督——`y=0` 时为 `<WAIT>` + 指出当前缺失证据的理由，`y=1` 时为 `<ANSWER>` + 有视觉支撑的答案。

```text
示例 — 问题："右肾有无结石？"
  时刻 A（尚未扫到右肾）：Input=[记忆]+问题+<DECISION>
    → y=0，"<WAIT> 尚未显示右肾，目前仅扫查肝脏。"
  时刻 B（已显示右肾、见强回声灶）：Input=[记忆]+问题+<DECISION>
    → y=1，"<ANSWER> 右肾下极见伴声影强回声灶，提示结石。"
```

### 视频爬取、ASR 转录与切片

使用 `UltrasoundCrawler_KeyCode_20260323_v2/` 获取 YouTube／Bilibili 超声视频。通过 `QA/prepare/run_prepare.py` 运行 ASR 转录和视频切片，输出带时间戳的 transcript 与 clips 信息，供后续训练样本和 QA 构建使用。

### VLM 打标

在训练数据构建前，使用 `scripts/data/classify_ultrasound_video_type_teacher.py` 对视频打标。主教师为通过本地 vLLM OpenAI-compatible 服务调用的 Qwen3.5，可选 Gemini 经 OpenRouter 交叉校验。这一步决定视频类型及适用阶段，与 Stage 2 的视觉摘要标注是不同任务。

| 视频类型标签 | 含义 | 合并脚本的保留策略 |
|---|---|---|
| `hands_on_ultrasound_teaching` | 实际探头操作与超声教学 | 三个阶段均保留 |
| `pure_ultrasound_scan` | 纯超声扫描画面／cine loop | 三个阶段均保留 |
| `ultrasound_ppt_lecture` | 超声相关幻灯片讲座 | 仅预训练 |
| `mixed_ultrasound_teaching` | 扫描、讲解、幻灯片等混合内容 | 仅预训练，并标记需要裁剪 |
| `ultrasound_image_discussion` | 静态超声图像讨论 | 仅预训练 |
| `non_ultrasound_or_irrelevant` | 非超声或无关内容 | 全部排除 |
| `uncertain` | 无法确定 | 合并后全部排除，需审核 |

原始标签还包含置信度、`anatomy_regions`、`clinical_scenarios`、`scan_views_or_targets`、语言及其依据、实时超声／探头／患者／幻灯片等内容标记、超声画面占比估计、教学价值、裁剪需求、视觉依据，以及 `keep_for_pretrain`、`keep_for_compression`、`keep_for_sft`。

合并脚本根据教师一致性和置信度差异选择最终标签，并输出 `needs_human_review` 等审核信息；随后按最终类别重新计算保留标记。原始教师可将 uncertain 保留用于预训练审核、也可为部分 mixed 视频给出其他标记，但合并后的策略以上表为准。教师一致不等于专家验证，筛选脚本也不会自动完成 mixed 视频裁剪。

### Stage 1：超声领域知识注入

**构建样本前的 ASR 清洗。** 这是用大模型清理与轻度润色 Whisper 文本的数据预处理步骤，独立于 VLM 视频打标和 Stage 2 的教师摘要监督。`stage1.clean_asr` 读取带时间戳的 Whisper transcript JSON 和预训练 keep-map，通过本地 vLLM 调用 `Qwen/Qwen3.5-27B`。每次默认处理 8 个 segment，前后各 2 个相邻 segment 的文本作上下文。模型收到 segment 的编号、起止时间和 ASR 文本；**不输入原视频或音频**。大模型逐段返回补标点、规范大小写及空格的 `clean_text`，如修正超声术语，还须列出 `from`／`to`／`reason`。脚本保留原分段和时间戳，拒绝缺失或乱序分段、未声明的词语改动和大幅改写；已有标点的 ASR 也会处理。

每个入选视频输出一份新的 transcript JSON：`segments` 和 `full_text` 保存清洗文本，`raw_segments` 和原文件中存在时的 `raw_full_text` 保存原文；`asr_cleaning.term_corrections` 与独立的 JSONL audit 记录术语修改。原 transcript 不会被覆盖。随后 `stage1.data` 从清洗后的 `segments` 推算句子及其时间，输出三条件 Stage 1 样本 JSONL。清洗模型只见文本，术语修正和段内时间估计需要抽样对照音频核验。详见 [Stage 1 数据流程](stage1/README.md)和[集群运行命令](docs/HOW_TO_RUN_PIPELINE.md)。

当前 train split 先只清洗原本无法生成样本的 7 个视频，再与已有 186 个视频的样本合并。合并脚本会核对是否完整覆盖筛选后的 193 个视频；Stage 1 训练另行提交 GPU 作业。

新 `stage1/` 为每句 `[start,end]` 构建三组配对样本：视频 `0→start` 加历史 ASR、视频 `0→end` 加历史 ASR，以及视频 `0→start` 且遮蔽历史 ASR。三组预测同一句完整讲解；目标句及之后的 ASR 不进入输入。每组在各自时间范围内最多取 120 帧。训练只对目标句计算损失，并按视频划分验证集；正常、黑屏和跨视频打乱画面用于视觉依赖性对照。详见 [stage1/README.md](stage1/README.md)；原 `pretrain/` 实验代码保留。

### Stage 2：视觉摘要监督与递归记忆

教师 VLM 根据对应静音视频片段生成三种监督目标：当前 10 秒摘要、当前 60 秒摘要，以及从视频开始至当前时刻的累计摘要。每个完整分钟边界 `T` 生成八条独立标注：当前分钟六条 10 秒摘要、`[T-60,T)` 一条整分钟摘要、`[0,T)` 一条累计摘要；构建脚本将其合并为一个训练 block。标注器直接发送各时间窗的视频片段，不再自行抽取固定数量的图像帧；教师服务仍会按自身视频解码策略采样，需审核短暂征象的覆盖情况。Stage 2 不输入 ASR，按以下流程训练：

- 每个视频从头按时间顺序处理，每秒生成一个 short-memory token。
- 每 10 秒，用对应的 10 个 short tokens 重建该区间摘要，计算 `Lshort` 并更新参数；随后将这些 short states detach，保留到当前分钟结束。
- 每 60 秒，用当前 60 个 short tokens 重建整分钟摘要，计算 `Llocal`；同时通过 `detach(L_previous) + S_current` 生成新的 60 个 long tokens，并重建累计摘要，计算 `Lglobal`。
- 在第 60 秒，将最后 10 秒的 `Lshort`、当前一分钟的 `Llocal` 和累计历史的 `Lglobal` 按 `1:1:1` 相加，统一更新一次。前 50 秒的 short states 参与计算，但不接收这次损失的梯度。
- 新 long memory detach 后传给下一分钟，当前 short memory 清空。始终只保留最新一份 long memory，递归承接此前累计历史。

训练样本包含 `local_sub_summaries`、独立 `block_summary_target` 和 `global_summary_target`。末尾不足一分钟时仅监督完整的 10 秒区间。CPU 调度和梯度测试已通过，真实 Qwen/LoRA 端到端训练仍待验证。

### Stage 3：可回答性决策与多问题 KV cache

以下是已确定、尚待完整实现的 Stage 3 设计。现有 `QA/train/train_summary_decide.py` 使用逐次追加长期 token 和生成 WAIT/ANSWER 文本的方式；仍需统一为 Stage 2 的固定 60-token 全量更新，并实现独立决策头及多问题 cache 管理。

#### 输入与因果注意力

```text
共享前缀                           每个问题的独立后缀
[固定 system / 模板] [Long memory] [Short memory] [Question] [<DECISION>]
                                                            │
                                          决策位置 hidden state h_dec
                                                            │
                                              Linear → logit z → sigmoid
                                                            │
                                                       WAIT / ANSWER
```

Long 和 Short 是连续 memory embeddings，模型以因果注意力处理整个序列。`<DECISION>` 放在问题之后，因此它的 hidden state 能读取当前记忆和完整问题。持久 memory 由视频持续构建，不依赖问题；不同问题各自读取同一份记忆，问题相关的 hidden states 不写回持久 memory。第一版只输入记忆和问题，当前原始帧留作对照实验。

#### `<DECISION>` 的 BCE 与 WAIT/ANSWER 文本 NTP

`<DECISION>` 是输入中的 readout token，不是要生成的文本。每个问题在每个决策时刻都有二分类标签：`y=0` 表示证据不足，`y=1` 表示证据充分。**WAIT 和 ANSWER 样本都计算 BCE，也都计算 NTP**，只是文本目标不同：

```text
输入：共享记忆前缀 + Question + <DECISION>
决策：z = Linear(h_dec)
      L_decision = BCEWithLogitsLoss(z, y)

WAIT   (y=0)：文本目标 = <WAIT>   + 当前缺少什么证据的具体原因
ANSWER (y=1)：文本目标 = <ANSWER> + 有视觉依据的答案

L_text = NTP(对应样本的文本目标)
L_stage3 = λ_decision · L_decision + λ_text · L_text
```

文本目标放在 `<DECISION>` 之后的 assistant 输出位置。NTP（next-token prediction）只监督该位置的输出 tokens，包括 `<WAIT>`／`<ANSWER>` 标记和后续原因／答案；固定前缀、记忆、问题、`<DECISION>` 及 assistant 模板部分均不计入文本损失。WAIT 原因描述当前**缺少的证据**，不能泄漏以后才出现的具体画面或结论。因果注意力保证 `<DECISION>` 看不到后面的监督文本，因此 BCE 和 NTP 可以在同一次训练前向中计算。

推理时先根据 BCE 头的 logit 与阈值决定 WAIT 或 ANSWER，再在该问题的记忆快照上生成相应文本：WAIT 输出标记和具体原因，保留问题以等待新证据；ANSWER 输出标记和答案。为了让文本分支与已选决策一致，可以固定对应的 `<WAIT>`／`<ANSWER>` 前缀，再生成后续内容。初始设 `λ_decision = λ_text = 1`。

#### 多个问题：共享前缀，独立决策

同一决策时刻只计算一次共享前缀，再为每个 active question 创建独立分支：

```text
固定前缀 + Long + Short → 共享前缀 KV cache
                          ├── Q1 + <DECISION> → logit1 → WAIT 原因 / ANSWER 答案1
                          ├── Q2 + <DECISION> → logit2 → WAIT 原因 / ANSWER 答案2
                          └── Q3 + <DECISION> → logit3 → WAIT 原因 / ANSWER 答案3
```

问题不串接；Q2 不应读取 Q1。每个问题分别有自己的 `<DECISION>`、可回答性 logit 和 WAIT 原因／ANSWER 答案的文本分支。第一版在每个新决策时刻完成 memory 更新后，重建一次共享前缀 cache，供当时所有问题使用。实现时需隔离可变的各问题后缀，不能让一个问题的 KV 污染其他问题。

#### KV cache 的更新与失效

- **新证据**：WAIT 问题基于新的记忆前缀重新计算 Question 和 `<DECISION>`；旧 Question KV 不包含新证据。
- **60 秒边界**：long memory 全量替换、short memory 清空后，旧前缀 cache 失效并重建。切换视频、模型权重、模板或位置配置时也需失效。
- **答案生成**：使用触发 ANSWER 时的记忆快照和该问题自己的分支；后续视频更新不修改正在生成的答案状态。

持久 memory embeddings 与各层 attention KV cache 是不同状态。此处 KV cache 是多问题推理的复用机制，不用于跨训练更新复用计算图。后续实现需验证缓存分支与完整前向的 logits 一致，且问题处理顺序不影响结果。

## 仓库结构

```text
stage1/                   新的三条件 Stage 1 实现
stage2/                   新的时序 Stage 2 实现
pretrain/                  先前 Stage 1/2 实验和基线
  build_samples.py         ASR 监督样本构建
  train.py / infer.py      领域预训练与推理
  build_teacher_memory_summary_samples.py
  memory_dataset.py / memory_collator.py
  train_memory_compression.py / infer_memory_compression.py
  eval_memory_compression.py
QA/                        时序 QA 构建和 SFT 基线
  prepare/                 ASR 与裁剪
  generator.py             双时间戳流式 QA
  offline_generator.py     离线 QA
  validator.py             时间与证据验证
  merger.py / run.py        导出、WAIT/ANSWER 展开与入口
  train/                   直接 QA 与记忆 SFT 基线
  eval/                    基线推理与预测分析
  schema.md                QA 与训练样本格式
scripts/                   共享工具和旧流程
  data/                    视频过滤与教师摘要
  slurm/                   集群实验启动脚本
src/                       早期视频过滤与输入模式实验
notebooks/                 交互实验与演示
docs/                      设计、命令与实验报告
results/                   本地输出及部分示例产物
livecc/                    LiveCC 参考代码
UltrasoundCrawler_KeyCode_20260323_v2/
                           YouTube／Bilibili 获取工具
```

本地数据、视频映射、检查点和输出不全部包含在仓库中。运行需提供当前机器上的有效路径。

## 环境配置

安装 `ffmpeg`，使用与所选 PyTorch/Transformers 兼容的 Python 环境。GPU 训练和本地教师服务面向合适的 CUDA 环境；不同集群实验可能有额外依赖。

```bash
pip install -r requirements.txt
```

按需从 `.env.example` 创建 `.env`，配置所用步骤的凭据：

- `OPENROUTER_API_KEY`：OpenRouter 视频标注和验证。
- `OPENAI_API_KEY`：旧版 OpenAI 流程。
- 本地教师：按脚本配置 base URL 和 API key 环境变量。

共享工具自动加载 `.env`。远程标注消耗 API 额度，凭据不应进入版本控制。

## 运行示例

在仓库根目录运行，将示例路径替换为本地路径。视频映射格式为 `{video_id: video_path}`；先按环境配置准备 API 凭据和本地教师服务。

### 0. VLM 打标、视频分类与分阶段筛选

依次运行主教师打标、可选 Gemini 打标、标签合并和筛选。只用主教师时，省略 Gemini 命令及 `--validator-audit`；`--stage` 可选 `pretrain`、`compression`、`sft`。

```bash
python scripts/data/classify_ultrasound_video_type_teacher.py \
  --video-map /path/to/video_path_map.json \
  --output results/video_types/qwen35.jsonl \
  --teacher qwen35 \
  --model Qwen/Qwen3.5-35B-A3B \
  --base-url http://localhost:8000/v1 \
  --api-key-env VLLM_API_KEY \
  --video-fps 0.5 \
  --disable-thinking \
  --resume

python scripts/data/classify_ultrasound_video_type_teacher.py \
  --video-map /path/to/video_path_map.json \
  --output results/video_types/gemini3.jsonl \
  --teacher gemini3 \
  --model google/gemini-3.1-pro-preview \
  --base-url https://openrouter.ai/api/v1 \
  --api-key-env OPENROUTER_API_KEY \
  --video-fps 0.5 \
  --resume

python scripts/data/merge_video_type_teacher_labels.py \
  --primary-audit results/video_types/qwen35.jsonl \
  --validator-audit results/video_types/gemini3.jsonl \
  --output results/video_types/final.jsonl \
  --output-summary results/video_types/summary.json

python scripts/data/filter_by_vlm_video_type.py \
  --video-map /path/to/video_path_map.json \
  --vlm-audit results/video_types/final.jsonl \
  --stage compression \
  --output-keep-map results/video_types/compression_keep.json \
  --output-drop-map results/video_types/compression_drop.json
```

### 1. 准备数据并生成时序 QA

先生成 transcript 和 clips，再将其路径传入 QA 入口。输出格式见 [QA schema](QA/schema.md)。

```bash
python QA/prepare/run_prepare.py \
  --video /path/to/VIDEO_ID.mp4 \
  --output-dir QA/results \
  --whisper-model base

python QA/run.py \
  --video /path/to/VIDEO_ID.mp4 \
  --transcript QA/results/transcripts/VIDEO_ID.json \
  --clips QA/results/clips/VIDEO_ID_clips.json \
  --out-dir QA/results \
  --validation-mode all \
  --expand-wait-answer
```

### 2. 构建 Stage 1 样本并训练

先启动服务 `Qwen/Qwen3.5-27B` 的本地 vLLM endpoint 并设置 `VLLM_API_KEY`；也可用会自行启动服务、逐个 split 构建样本的 [Slurm 脚本](scripts/slurm/run_stage1_qwen35_asr_clean.sbatch)。

```bash
python -m stage1.clean_asr \
  --transcripts results/transcripts \
  --video-map /path/to/pretrain_keep_videos.json \
  --output-dir results/transcripts_stage1_qwen35_clean \
  --audit-output results/stage1_qwen35_clean_audit.jsonl \
  --resume

python -m stage1.data \
  --transcripts results/transcripts_stage1_qwen35_clean \
  --video-map /path/to/pretrain_keep_videos.json \
  --output stage1/samples.jsonl

python -m stage1.train \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --train-jsonl stage1/samples.jsonl \
  --video-path-map /path/to/video_path_map.json \
  --output-dir /path/to/stage1_output \
  --frame-budget 120 \
  --frame-size 224 \
  --epochs 3 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4
```

### 3. Stage 2：视觉摘要生成与记忆训练

依次生成视觉摘要、构建样本并训练。按需添加 `--init-adapter /path/to/stage1_adapter` 从 Stage 1 初始化。

```bash
python -m stage2.annotate \
  --video /path/to/VIDEO_ID.mp4 \
  --video-id VIDEO_ID \
  --model google/gemini-3.1-pro-preview \
  --base-url https://openrouter.ai/api/v1 \
  --api-key-env OPENROUTER_API_KEY \
  --output results/teacher_visual_summaries/VIDEO_ID_visual_summaries.jsonl

python -m stage2.build_data \
  --teacher-jsonl results/teacher_visual_summaries/VIDEO_ID_visual_summaries.jsonl \
  --output results/teacher_visual_summaries/VIDEO_ID_memory_samples.jsonl
```

```bash
python -m stage2.train \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --train-jsonl results/teacher_visual_summaries/VIDEO_ID_memory_samples.jsonl \
  --video-path-map /path/to/video_path_map.json \
  --output-dir /path/to/stage2_output \
  --frame-size 224 \
  --epochs 1 \
  --learning-rate 1e-4
```

完整参数、推理和集群命令见[运行命令](docs/HOW_TO_RUN_PIPELINE.md)及 [LRZ 集群指南](docs/LRZ_CLUSTER_GUIDE.md)。

## 模型评估目标

评估应证明模型结构和训练目标的贡献。配套 benchmark 区分**是否回答、何时回答，以及答案是否有证据支持**。

- 决策：precision/recall、AUROC/AUPRC、校准、提前回答和漏答。
- 时机：相对证据充分时刻的回答延迟，以及视频结束仍不可答的情况。
- 答案：正确性、视觉依据、幻觉和专家评价。
- 记忆：不同年龄证据的保留能力、长历史 QA、压缩边界表现。
- 效率：流式吞吐、响应延迟、峰值显存和记忆更新开销。

以上是评估要求，并非已完成的结果。可信测试集需要专家验证自动标注。划分需避免原始视频重叠，在元数据允许时避免患者／来源泄漏。记忆方法比较应匹配视觉与计算预算。

## 文档与实验历史

- [参考设计](docs/live_ultrasound_reference_implementation_final_zh.md)：目标架构与语义。
- [运行命令](docs/HOW_TO_RUN_PIPELINE.md)：操作示例。
- [预训练 V1](docs/pretrain_V1.md)与[全量 V1](docs/Pretrain_Full_V1.md)：早期实验。
- [交错预训练](docs/Pretrain_Full_V3_Interleave.md)与[改进目标](docs/Pretrain_Improvement_Objectives.md)：历史／替代输入目标。
- [SFT V1](docs/sft_V1.md)：WAIT collapse 实验及分析。

历史文档可能采用不同阶段编号或输入策略，不能据此认为当前设计已完整实现。

## 参考与用途

- [LiveCC](https://github.com/showlab/livecc)：流式视频语言训练参考。
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)：当前骨干模型系列。
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)：ASR 实现。

仅供研究使用。第三方代码、模型和源视频遵循各自许可证与使用条款。
