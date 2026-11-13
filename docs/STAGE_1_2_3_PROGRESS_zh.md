# Stage 1、2、3 进展与验证清单

> 历史记录：下文保留 2026-10-20 的进展和命令，不代表当前数据、训练状态或默认入口。当前入口见 [主仓库说明](WORKSPACE.md)，Stage2 全量训练见 [FULL_TRAINING.md](../stage2/FULL_TRAINING.md)。请勿直接按旧作业编号取消或重提任务。

更新日期：2026-10-20。本文记录三阶段主线的**项目目标、各阶段作用、代码进展、下一步工作和完成判据**。这里的“已实现”表示仓库中有对应代码并通过本地检查，不表示已经在 SSH 集群上完成训练或取得模型效果。远端数据文件、作业日志和 checkpoint 尚未在本次整理中逐一核验；任何“全量完成”结论都应以这些产物为准。

三个阶段的主线 backbone 为 `Qwen/Qwen3.5-2B`。Stage 1 adapter 合并进 Stage 2 的基座；Stage 3 使用两个独立的 Stage 2 权重实例，一个冻结生成视觉记忆，一个训练可回答性与文本输出。Stage 2/3 的模型输入不含 ASR。详细设计见 `README_zh.md`，具体命令见 `docs/HOW_TO_RUN_PIPELINE.md`。

## 项目要解决什么

目标是在超声视频**逐秒到达**时回答用户问题：画面证据不足就等待并说明还缺什么；证据足够才回答，并尽量引用已经看到的视觉信息。核心研究问题不只是“最后答案对不对”，还包括**何时开始可答、是否过早回答、长视频中早期证据是否保留，以及推理的时间和显存成本**。“实时/流式”在当前设计中首先指**不能看未来帧的因果输入**。

训练数据准备时可以离线使用 ASR 和教师 VLM 制作监督标签。主线在线输入为视频画面与用户问题，**Stage 2、Stage 3 不输入 ASR**。这样才能检验模型是否真正依靠视觉证据，而不是从旁白中提前获知答案。

## 三个 Stage 分别解决什么

| 阶段 | 作用与训练信号 | 留给下一阶段的东西 |
|---|---|---|
| Stage 1：领域下一句预训练 | 三种配对输入都预测**当前 `[start,end]` 的同一句讲解**，学习超声画面与领域语言的对应关系 | Qwen3.5 的 Stage 1 LoRA；Stage 2 启动时将其合并进 backbone |
| Stage 2：视觉记忆 | 每秒读一帧生成 short 向量；每满一分钟用当前 60 个 short 和旧 long 递归生成**固定 60 个 long 向量**。每完整分钟的 6 个十秒 local、1 个六十秒 block、1 个从视频开头到现在的 global 摘要是训练目标 | 带 Stage 2 adapter 的视觉记忆模型；当前主线在线为 Stage 3 产生 short/long **向量**。是否额外使用摘要文字留作后续消融 |
| Stage 3：何时回答、回答什么 | 冻结 Stage 2 记忆模型；独立的决策模型读当前记忆和问题，以 BCE 学 WAIT/ANSWER，以文本损失学等待原因或答案 | 目标是流式可回答性判断和视觉问答输出；目前仅有模型及单样本损失接口，完整训练和在线推理尚未实现 |

衔接顺序是 **Stage 1 adapter → Stage 2 视觉记忆 → Stage 3 冻结记忆＋独立决策模型**。Stage 2 若不加载 Stage 1 adapter，是从基座开始的消融，不是完整三阶段主线。Stage 3 的两个模型都要恢复同一条 Stage 1→Stage 2 权重链；做决策的模型不能反向改写冻结的视觉记忆。三个 Stage 的模型版本和 adapter 来源必须一致，才能复现实验。

## 总览

| 阶段 | 数据集构建 | 训练 smoke | 全量训练 | 评测数据与评测 |
|---|---|---|---|---|
| Stage 1 | 三条件样本、ASR 清洗代码已实现；**主线采用 186 视频**（7 个 pilot 的 clean_asr 失败，已放弃补 193） | 用户确认 smoke 已运行；实际输出、步数及输入 JSONL 待按作业产物核验 | 训练、checkpoint、续训代码已实现；正式作业状态待核验 | 评测推理代码已实现；常规文本指标＋盲评 LLM Judge 待运行；视觉对照为可选诊断 |
| Stage 2 | **每个完整分钟**的 6 条十秒摘要＋1 条当前分钟摘要＋1 条从开头累计摘要：标注与校验代码已实现；真实标签待生成和审核 | 双卡 Qwen/LoRA/NCCL 作业已准备，CPU 双进程梯度同步测试通过；SSH smoke 待跑（smoke 标签为合成文本） | 8-rank DDP、验证和 epoch 边界续训已实现；正式训练未验证 | 重建损失和流式记忆导出代码已实现；独立评测标签、记忆质量评测和结果待做 |
| Stage 3 | QA 候选生成、验证、WAIT/ANSWER 展开已有；视觉专用数据审核与按时刻重放加载器待做 | 双模型加载及单样本损失有 CPU 测试；真实 GPU smoke 待设计和运行 | 训练 CLI 与循环尚未实现 | 视觉证据基准、在线推理与完整评测协议尚未实现 |

## Stage 1：领域预训练

### 目的与接口

- **目标**：让模型学习超声画面与领域讲解的对应关系；不是直接训练 WAIT/ANSWER。
- **输入与监督**：对每个当前句子 `[start,end)`，三条样本共用**同一句**目标文本：`before_with_asr` = `[0,start)` 画面＋历史 ASR；`through_with_asr` = `[0,end)` 画面＋同一段历史 ASR；`before_mask_asr` = `[0,start)` 画面＋遮蔽历史 ASR。目标句及之后的 **ASR 文本**不进入输入；`through_with_asr` 的画面覆盖目标句发生的时间。
- **阶段产物**：Stage 1 LoRA adapter，供 Stage 2 合并进同一个 `Qwen/Qwen3.5-2B` backbone。实现入口：`stage1/README.md`、`stage1/data.py`、`stage1/train.py`。

### 数据构建

- **已实现**：视频筛选、ASR 文本清洗、句子时间估计、三条件样本构建。训练/验证按视频划分。**主线决定采用 186 个视频**，训练 JSONL 为 `$DATA/cluster_data/pretrain/train_full295_stage1_samples.jsonl`（preflight `186 videos / 53244 rows / 17748 targets`）。
- **193 补齐已放弃（2026-10-20 决定）**：原计划用 7 个 pilot 视频（`OKtdZVXtZ0M OQJ2RhnuOQQ TlckvYhqaFE WhNxq98szE8 YYcTdkP7xhs d7IwDHAVcGQ vmmrjy1bhEc`）补成 193，但这 7 个 `clean_asr` 全部失败（作业 5832433，audit `cluster_data/QA/train_full295/stage1_seven_clean_audit.jsonl`，7/7 `status:failed`）。根因不是转录错误，而是清洗护栏正确地拒了 teacher（Qwen3.5-27B）的输出：臆造/多余/未声明的术语改写，以及 `WhNxq98szE8` 的响应被 `max_tokens` 截断。**决定不再补齐，主线以 186 视频为准**；`stage1.seven_video_pilot` 合并代码与 `stage1.verify_merged_193` 核验器保留，但不再是主线路径。
- **待完成（针对 186 主线）**：单独构建/核对 eval split，确认与 186 训练视频不重叠。句子时刻由 ASR 片段内插估计，需抽查；每条样本原设最多 120 帧，**当前正式作业已降帧预算（2B=48 / 4B=32）**，长前缀并非始终 1 FPS。
- **完成判据**：186 个 train 视频每句三条件及时间窗正确、视频路径可读、train/eval 来源不重叠；人工抽样对照原始音频检查术语与时间戳。结构核验不能代替内容审核。

### 训练：smoke → 全量

- **已实现/准备**：`scripts/slurm/run_stage1_pretrain_merged.sbatch` 支持两步 smoke、120 帧预算、周期 checkpoint 和 `--resume`；正式训练及续训代码已实现。2B/4B/9B 对比脚本也已准备。
- **提交前修正**：Slurm 的 `TRAIN_JSONL` 默认仍指向旧的 `train_full295_stage1_samples.jsonl`。193 视频主线必须显式指定 `train_full295_stage1_merged_193_samples.jsonl`，或修正默认值，并记录实际文件哈希及视频数。
- **smoke 已运行，待核验产物**：用户确认 sweep smoke 已跑完。输出路径可能受 `DATA`、`SWEEP_NAME` 等覆盖变量影响，必须读取实际作业的 `training_config.json`、`video_split.json`、`checkpoint-2/`、`final/` 和日志，确认模型、输入 JSONL、实际帧预算、有限 loss 与两步更新；不要仅凭脚本默认值认定 smoke 使用了 193 视频合并文件。随后核对全量作业的 `final/`、配置和续训状态。两步 smoke 不代表模型质量。

### 评测

- **已实现**：`stage1.evaluate` 生成预测；训练器保存按视频划分的内部 `validation_samples.jsonl`。`normal` 用原画面，`blank` 用黑画面，`shuffled` 用另一视频画面；后三者属于可选视觉依赖诊断，不是独立 eval 集。
- **待完成**：锁定独立 eval 视频及样本，在同样的帧预算与解码设置下生成 base/Stage 1 adapter 的预测；计算常规文本指标（例如 token F1、ROUGE-L），并做匿名、随机 A/B 顺序的 LLM Judge，关注医学术语、否定和无依据内容。现有 `pretrain/` 比较脚本使用旧 JSONL 字段，不能直接套在 `stage1.evaluate` 输出上，需做 Stage 1 格式适配。
- **完成判据**：train/eval 视频隔离；base 和 adapter 使用同一批样本；保存逐样本预测、聚合指标、Judge 原始判定和模型/提示配置。

## Stage 2：视觉摘要监督与递归记忆

### 目的与接口

- **目标**：把不断到来的画面压成可递归更新的视觉记忆，而不是把摘要文字直接交给问答模型。
- **输入与监督**：训练时只看静音视频，按 1 FPS 生成 short 向量；每个完整分钟，用当前 60 个 short 和旧 long 生成固定 60 个新 long 向量。教师 VLM 的 local/block/global 文字摘要用于重建损失监督，不是在线输入。
- **阶段产物**：Stage 2 adapter 与可学习的 memory-token embedding；在线为 Stage 3 产生 short/long **记忆向量**。额外使用摘要文字是后续消融，不属于当前主线。实现入口：`stage2/README.md`、`stage2/model.py`、`stage2/train.py`。

### 数据构建：每完整分钟八条教师标注

“八条”是**一个视频每经过一个完整的 60 秒块，就生成 8 条独立监督标签**，不是一个视频总共只有八条。设当前完整分钟的结束时刻为 `T`：

| 数量 | 标签 | 教师 VLM 看到的静音视频窗口 | 用途 |
|---:|---|---|---|
| 6 | `local_10s` | `[T-60,T-50)`、…、`[T-10,T)`，每条只看自己的 10 秒 | 监督该 10 秒的 short memory |
| 1 | `block_60s` | `[T-60,T)` | 独立概括当前整分钟，监督由本分钟 60 个 short 向量重建摘要 |
| 1 | `global_prefix` | `[0,T)` | 概括从视频开头到现在的累计视觉信息，监督 long memory |

例如 `T=60` 时，六个 local 窗口为 `0–10`、`10–20`、…、`50–60` 秒；block 和 global 都看 `0–60` 秒，**但提示词、摘要侧重点和训练目标不同，必须分别生成**。`T=120` 时，block 看 `60–120` 秒，global 看 `0–120` 秒。视频最后不足 60 秒时，只标注其中完整的 10 秒窗口；不会虚构缺少的整分钟与累计摘要。以上窗口均为左闭右开。

- **已实现**：`stage2.annotate` 对完整分钟生成 8 条事件；`stage2.build_data` 校验窗口、时序和 `stage2_facets_v1` 字段，并合成一个训练 block。教师输入为静音片段，**不含音频或 ASR**。旧版每分钟仅一条摘要不能直接使用。入口：`stage2/annotate.py`、`stage2/build_data.py`、`scripts/slurm/run_stage2_annotate_build.sbatch`。
- **待完成**：固定 train/eval 视频映射；先标注并人工审查一个真实视频，再批量生成。全量 train 标签、独立 eval 标签与人工审核结果目前尚未核实。批量前须处理脚本的 `VIDEO_IDS=${VIDEO_IDS:-...}` 空字符串回退，否则 `VIDEO_IDS=""` 仍只处理默认单视频；`README_zh.md` 当前的空字符串全量标注示例不能照用。
- **完成判据**：每完整分钟恰有 8 条类型/窗口正确的独立标签并生成一个 block；末尾不足一分钟仅有完整十秒 local；结构字段合法，train/eval 视频隔离；教师实际采帧设置、失败重试、静音输入及视觉质量抽查留档。

### 训练：smoke → 全量

- **已实现/本地验证**：时序和损失调度测试、双进程 CPU DDP 梯度平均与不等长 Join 测试通过；`stage2.train` 支持完整视频按 rank 分配的 8-rank DDP、跨 rank 验证汇总、rank 0 保存，以及同 world size 的完整 epoch 边界 `--resume`。`scripts/slurm/run_stage2_train_ddp.sbatch` 默认两节点各四卡。
- **待 SSH smoke**：`scripts/slurm/run_stage2_ddp_smoke.sbatch` 用两段真实视频、**合成摘要**运行 6 步对 1 步，检查 LoRA 梯度同步、60 秒 block/global 损失、Join、rank 日志和 checkpoint；成功日志应同时有 `[ddp-check] synchronized gradient` 与 `[smoke] PASS`。已有 Stage 1 adapter 时传 `STAGE1_ADAPTER` 验证完整加载链。合成标签不能用于正式训练或质量评测。
- **全量前后核验**：smoke 通过后先用正式标签做小规模训练，再确认 partition、跨节点通信、显存、吞吐和分片等待时间，最后提交 8-rank 全量。检查各 epoch 的 per-rank 日志、全局验证、`best/`、`final/` 和抢占续训。`global_step` 表示最长分片的同步优化轮次，不是各 rank 本地更新之和。

### 评测

- **已实现**：`stage2.evaluate` 计算 held-out 视频的 short/block/global teacher-forced 重建损失；`stage2.stream` 在无教师标签的情况下因果导出分钟 long 和记忆末尾 short。
- **待完成**：为独立视频制作并审核八标签 eval 数据；报告 short/block/global 重建损失，并在有生成摘要时计算常规文本指标及 LLM Judge。更细的长期记忆任务和消融可放在后续实验，不作为首轮 eval 的阻塞项。
- **完成判据**：独立评测视频未参与训练或 checkpoint 选择；推理保持 1 FPS 因果处理，Stage 1/2 adapter 链与训练一致；留存逐样本结果、指标、Judge 判定和 `.pt` 的 long/short 形状检查。

## Stage 3：可回答性决策与流式问答

### 目的与接口

- **目标**：给定当前视频记忆和问题，判断证据是否充分；不足时输出 WAIT 与原因，充分时输出 ANSWER 与答案。
- **输入与监督**：冻结的 Stage 2 memory 模型按时间生成 long/short 向量；独立的 decision 模型读取记忆、问题和 `<DECISION>`。BCE 监督 WAIT/ANSWER 二分类，文本 NTP 监督对应的等待原因或答案。Stage 3 不输入 ASR，也不能读取未来视频帧。
- **阶段产物**：目标是可回答性分数与条件化文本；目前只有双模型加载和单样本损失接口。实现入口：`stage3/README.md`、`stage3/model.py`。

### 数据构建

- **已实现**：`QA/generator.py`、`QA/validator.py`、`QA/merger.py` 可产生并检查候选 `query_time`/`answer_time`，展开 WAIT/ANSWER 样本。
- **待完成**：现有 QA 工具允许使用音频/讲解，故时间戳尚非“视觉首次充分”真值。须固定视频级 train/val/test，按视觉证据重审问题、答案、WAIT 缺失证据和最早可答时刻，并纳入视频结束仍不可答的负例。还需按决策时刻重放视频或预计算 long/short；`stage2.stream` 的分钟快照不足以恢复任意时刻的完整输入。
- **完成判据**：专家审核 `query_time` 时不可答、`answer_time` 时可答及其前后标签；问题在提出时刻已成立，数据加载不会泄漏未来帧、答案或 ASR。专家审核和自动验证分开记录。

### 训练：smoke → 全量

- **已实现/本地验证**：`stage3.model` 包含冻结 memory 模型、独立 decision 模型、注册的 BCE head、`<DECISION>` 输入顺序、单样本 BCE/NTP 前向及 stop-gradient 检查。旧 `QA/train/train_summary_decide.py` 是单模型基线，不是本阶段的训练器。
- **尚未实现**：完整训练 CLI、QA 时间调度加载器、优化器/调度器、checkpoint/续训和在线文本生成。因此现在不能提交 Stage 3 全量训练。
- **待 GPU smoke 与完成判据**：先实现加载器和循环，再在 SSH 加载真实 Stage 1/2 adapters，使两个模型同时驻留显存；核验 memory 无梯度、decision LoRA/新 token/BCE head 有梯度、WAIT/ANSWER 与 BCE 标签一致、损失有限且更新。之后验证保存/恢复、阈值标定和多问题 KV cache 的隔离与失效，才进入全量训练。

### 评测

- **已实现**：现有 `QA/eval/` 是旧基线工具；新 Stage 3 的完整在线评测器尚未实现。
- **待完成**：准备与训练视频隔离、经视觉证据复核的 QA eval 样本；固定逐秒输入及问题触发协议。先报告 WAIT/ANSWER 的常规分类指标、答案文本指标和盲评 LLM Judge；复杂的长时记忆分组与效率消融留作后续。
- **完成判据**：留存每个决策时刻的预测、真值、分类指标、答案及 Judge 判定；评测不一次性读完整视频，也不使用音频/ASR。`answer_time` 仍须先经视觉复核，否则这些指标的意义不可靠。

## 下一轮执行顺序

1. 在 SSH 上盘点 train/eval 视频映射、Stage 1 七视频 audit、193 视频合并样本及已有 checkpoint；将实际路径和作业 ID 记入实验记录。
2. 核对已跑 smoke 的真实输入、日志和产物；显式指定合并 JSONL 进行 Stage 1 正式训练，再做独立 eval 的文本指标＋LLM Judge，确认可用的 Stage 1 adapter。
3. 跑 Stage 2 双卡 DDP smoke；完成真实摘要单视频审核、全量 train/eval 标注与 block 构建；再运行 8-rank 训练和记忆评测。
4. 完成视觉专用 QA 数据审核和 Stage 3 时间调度/训练/生成；依次做双模型 GPU smoke、全量训练与锁定测试集上的因果在线评测。

每次进入下一阶段前保留数据清单、配置、代码版本、作业日志、checkpoint 路径和评测输出。只有通过对应的 SSH 运行和数据审核，才把上表的“待验证”改为“已验证”。
