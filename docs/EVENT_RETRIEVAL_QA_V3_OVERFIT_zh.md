# 事件检索流式超声问答：模型、QA v3 与十视频 overfit

更新于 2026-11-27。代码位于 `event_retrieval/`，实验使用独立的 scratch 目录。目标是对同一个问题逐秒判断：现有视觉证据够不够？证据不足时生成 WAIT 和原因，证据充分时生成 ANSWER 和答案，并保留可回看的事件与时间戳。

## N 帧短期摘要 + K 帧直通：全部已完成的 overfit

四次实验共用 QA v3 的 10 个视频、97 道题，在同集训练并逐秒回放。下表统一使用第 20 轮 checkpoint。仓库将 20+4 命名为 v2，将 28+2 命名为 v3；两者均采用下述三路输入。

**时刻 T 的输入。** Long Q-Former 对按问题检索的最多 5 个已发布历史 event 编码，每段均匀采样最多 24 帧，汇总成 128 个视觉 token。Short Q-Former 接收近期窗口中较早的 N 帧（T−K−N+1 至 T−K），输出 32 个 token。最近 K 帧（T−K+1 至 T）保留冻结 Qwen 视觉编码器输出的全部空间特征，每帧 64 个 token。Qwen3.5-2B 依次读取 Long、Short、最近帧、时间与证据来源、问题，生成 WAIT 或 ANSWER 以及相应文本。视频开头只使用已观察到的帧。历史与窗口均完整时，20+4 的视觉前缀有 416 个 token，28+2 有 288 个。

**训练。** 每组从预训练权重独立初始化，冻结视觉编码器，更新两支 Q-Former 的视觉 query 路径、投影、位置嵌入及完整语言模型。全部使用纯 NTP，246 个监督时点，20 轮。BS 是每次前向的微批大小，GA 是梯度累积次数。GA2 的两次微批按有效目标 token 总数归一化。

| 作业与原始回放指标 | N+K | BS / GA | 总更新数 | 最终 NTP | 即时准时 | 延迟准时 | 准时且精确匹配 | 准时且语义正确 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [5962151](../cluster_data/experiments/event_retrieval/v2_overfit_qa97_ntp_TIKRtc6G/replay/timing_metrics.json) | 20+4 | 4 / 1 | 1240 | 0.2527 | 81/81 | 14/15 | 68/96 | **81/96** |
| [5963020](../cluster_data/experiments/event_retrieval/v3_overfit_28plus2_ntp_DUn4h0bI/replay/timing_metrics.json) | 28+2 | 4 / 1 | 1240 | 0.0507 | 80/81 | 0/15 | 77/96 | **78/96** |
| [5963195](../cluster_data/experiments/event_retrieval/v3_overfit_28plus2_ntp_bs8_ig77ze5i/replay/timing_metrics.json) | 28+2 | 8 / 1 | 620 | 0.0240 | 81/81 | 12/15 | 87/96 | **90/96** |
| [5963544](../cluster_data/experiments/event_retrieval/v3_overfit_bs8_ga2_kmrfbeyn/replay/timing_metrics.json) | 28+2 | 8 / 2 | 320 | 0.0583 | 81/81 | 13/15 | 87/96 | **92/96** |

**时机与内容。** 20+4 的延迟题 14 道准时、1 道早 1 秒；该过渡秒被标为不确定并从二元训练监督中排除。唯一无标注答案的题在审核区间内保持 WAIT，审核期后触发。28+2 的三组均无早答、晚答，无答案题均等待至片尾；延迟题分别有 15、3、2 道等到片尾。BS4 还漏掉 1 道即时题。语义正确接受同义改写，部分正确不计通过；20+4 唯一提前的回答仅获部分正确，故 68 个精确匹配与 13 个通过复核的改写均为准时作答。[20+4 内容复核](../results/event_retrieval_qa97_loss/judge_inputs/v2_28_content_review_zh.md) · [28+2 研究记录](../research.MD)

**同集结果。** 20+4/BS4 的延迟准时题最多；若同时要求时机和内容，28+2/BS8/GA2 最高（92/96），包含即时 80/81、延迟 12/15。剩余 4 题为 2 题漏触发、2 题空间定位不完整。28+2 的 GA1 最终训练 loss 更低，逐题回放则是 GA2 通过更多。相同 BS4/GA1 时，改变 N/K 同时改变了近期窗口长度和视觉前缀长度，因此本组实验无法单独估计某一帧或某一分支的贡献。

![四组 N/K overfit 的逐轮训练 NTP，纵轴为对数刻度](../results/event_retrieval_qa97_loss/nk_overfit_comparison.png)

**逐题查看视频、GT 和生成答案。** [20+4 BS4](../results/event_retrieval_qa97_loss/judge_inputs/v2_ntp_epoch20.jsonl) · [28+2 BS4](../results/event_retrieval_overfit_research/5963020.jsonl) · [28+2 BS8/GA1](../results/event_retrieval_overfit_research/5963195.jsonl) · [28+2 BS8/GA2](../results/event_retrieval_overfit_research/5963544.jsonl)

## v2 实验结果（作业 5962151）

在参与训练的 10 个视频上，15 道延迟题中有 **14 道先持续 WAIT，在标注答案秒首次 ANSWER**，其中 **10 道准时且答案文字精确匹配**。81 道即时题全部在提问秒回答，其中 58 道文字精确匹配。作业已完成，以下生成指标取第 20 轮 checkpoint。

### 逐秒模型输入

冻结的 Qwen3.5-2B 视觉编码器每秒处理一张截至该秒已可见的 256 × 256 RGB 帧。每帧的 64 个空间 token 存入视频索引。时刻 T 收到同一个问题时，Qwen 按以下顺序读取视觉前缀。

| 输入顺序 | 可见内容与处理 | 输出到 Qwen |
|---|---|---:|
| 1 · 历史事件 | 从截至 T 已发布的历史事件中，排除完全落在最近 24 帧内的事件。按问题对代表帧打分，取 Top-5，每个事件均匀采样最多 24 帧，送入 Long Q-Former。同一问题在推理时可以复用检索结果和顺序都相同的长期前缀。 | 有历史时共 128 个 token |
| 2 · 较早的近期帧 | 将 T−23 至 T−4 的最多 20 帧按时间送入 Short Q-Former。 | 有这部分观测时共 32 个 token |
| 3 · 最新帧 | 将 T−3 至 T 的最多 4 帧按时间排列，直接拼接视觉编码器输出，每帧保留全部 64 个空间 token。 | 最多 256 个 token |

完整窗口且有历史时，视觉前缀共 **416 个 token**。视频开头不足 5 帧时，已有帧全部直通，第 5 帧起才有短期摘要。随后输入历史事件 ID 及区间、近期窗口范围、提问秒、当前秒和问题文本。Qwen 先在两个特殊 token 中选择动作，再生成 `<WAIT> 理由` 或 `<ANSWER> 答案`。首次 ANSWER 结束该题。标注的答案秒和证据窗只用于监督与核对。[视觉前缀实现](../cluster_data/experiments/event_retrieval/v2_overfit_qa97_ntp_TIKRtc6G/code/event_retrieval/qformer.py) · [解码与文本输入](../cluster_data/experiments/event_retrieval/v2_overfit_qa97_ntp_TIKRtc6G/code/event_retrieval/answer.py)

### 训练配置与 NTP 损失

| 项目 | v2 实际配置 |
|---|---|
| 数据 | QA v3，10 个 holdout 来源视频用于同视频 overfit。97 道题展开为 246 个时点，其中 96 个 ANSWER、150 个 WAIT。16 个 WAIT 有理由标注，另外 134 个只监督动作。 |
| 初始化与参数 | 从 Qwen3.5-2B 和 InstructBLIP-Vicuna-7B 预训练权重独立初始化。训练两支 Q-Former 的视觉 query 路径、投影、位置嵌入和完整语言模型；冻结视觉编码器。最新四帧直接拼接，无新增可训练参数。参数拆分见第 3 节，可训练合计 2,098,298,944。 |
| 优化 | AdamW，BF16，BS4，GA1。Q-Former 及投影学习率 1e-4，LLM 学习率 1e-5，seed 0。20 轮，每轮 62 步，共 1,240 次更新，每 2 轮保存一次。检索 BS64。 |
| 设备与回放 | 单张 H100。训练及最多 10 视频并行回放共耗时 1 小时 1 分 34 秒。全作业每秒采样，平均 GPU 利用率 49.5%，峰值显存 60,397 MiB（约 59.0 GiB）。 |

每个 batch 的目标为 $L=-\frac{1}{|S|}\sum_{(i,j)\in S}\log p_\theta(y_{ij}\mid x_i,y_{i,<j})$。$S$ 是有监督的输出位置集合，覆盖首动作、已标注的理由或答案正文，以及有正文时的 EOS。每个有效 token 系数相同，因此较长的回答贡献更多项。裸 WAIT 只贡献一个动作项。epoch loss 按监督 token 总数汇总。[运行配置](../cluster_data/experiments/event_retrieval/v2_overfit_qa97_ntp_TIKRtc6G/fit/config.json) · [冻结实验计划](../cluster_data/experiments/event_retrieval/v2_overfit_qa97_ntp_TIKRtc6G/plan.json) · [GPU 采样](../cluster_data/experiments/event_retrieval/v2_overfit_qa97_ntp_TIKRtc6G/logs/gpu-5962151.csv)

第 1 轮训练 NTP 为 **2.4813**，第 14 轮达到最低 **0.01225**，第 18 轮升至 **0.3428**，第 20 轮为 **0.2527**。第 14 轮的逐秒生成质量尚未评测。[训练日志](../cluster_data/experiments/event_retrieval/v2_overfit_qa97_ntp_TIKRtc6G/logs/slurm-5962151.log)

![v2 全部 20 轮训练 NTP 曲线，纵轴为对数刻度](../results/event_retrieval_qa97_loss/v2_ntp_loss.png)

### 第 20 轮逐秒回放与 v1 对照

从提问秒开始逐秒推理，取首次 ANSWER 与标注秒比较。表中“文字匹配”采用忽略大小写和合并空白后的精确匹配，未进行语义评分。分母以独立题计，只有最后一行按已观测的标注时点计。

| 指标 | v1 首动作 K 加权（5961170） | v2 纯 NTP（5962151） |
|---|---:|---:|
| 即时题准时 / 准时且文字匹配（81 题） | 81 / 80 | 81 / 58 |
| 延迟题提前 / 准时 / 晚答 / 等到结束（15 题） | 11 / 1 / 3 / 0 | **1 / 14 / 0 / 0** |
| 延迟题准时且文字匹配 | 1/15 | **10/15** |
| 可答题准时触发 | 82/96 | **95/96** |
| 可答题准时且文字匹配 | 81/96 | 68/96 |
| 可答题首次回答文字匹配，忽略时间 | 94/96 | 68/96 |
| 无标注答案的题一直 WAIT 到视频结束 | 1/1 | 0/1 |
| 实际走到的标注 WAIT 时点中保持 WAIT | 78/89 | 150/150 |

**怎样理解 150/150 与一次提前。** 心脏视频 `CW3TLabogko/Q004` 在 94–97 秒连续 WAIT，98 秒准时 ANSWER。同视频 `Q006` 在 203 秒回答，比首次明确可答的标注 204 秒早一秒；203 秒是预先排除二元监督的过渡帧。无标注答案的 `YllLJNPnpjY/Q009` 于 445 秒提问，人工审核覆盖至 459.24 秒，模型到 493 秒才输出 ANSWER。按整段回放计为触发，但 493 秒已超出人工审核区间。两次触发都不属于那 150 个有明确 WAIT 标签的时点。[审核记录](../cluster_data/experiments/event_retrieval/v2_overfit_qa97_ntp_TIKRtc6G/data/audit.json) · [逐秒生成](../cluster_data/experiments/event_retrieval/v2_overfit_qa97_ntp_TIKRtc6G/replay/stream_predictions.jsonl)

v2 的延迟触发时机更接近标注，答案文字的精确匹配率则下降，尤其是即时题。两次实验同时改变了最新帧输入、损失归约方式和批量大小（BS8 → BS4，620 → 1,240 步），现有结果无法单独估计四帧直通的贡献。指标来自同视频训练与回放，独立视频上的泛化仍待评测。[时机指标](../cluster_data/experiments/event_retrieval/v2_overfit_qa97_ntp_TIKRtc6G/replay/timing_metrics.json) · [两版同口径对照](../cluster_data/experiments/event_retrieval/v2_overfit_qa97_ntp_TIKRtc6G/replay/baseline_comparison.json) · [第 20 轮 checkpoint](../cluster_data/experiments/event_retrieval/v2_overfit_qa97_ntp_TIKRtc6G/fit/latest.pt)

## 1. v1 模型架构简介

每个决策时刻，LLM 接收按时间截断的视觉前缀（长期最多 128 个 token、近期 32 个 token）、问题和时间上下文。它先选择动作，再生成等待理由或答案。推理时首 token 只允许两个动作，选择原始 logit 较高的一项。

| 环节 | 实际实现 |
|---|---|
| 视频索引，与问题无关 | 每秒取 PTS 不晚于当前时刻的最后一帧，补边至 256 × 256。冻结的 Qwen3.5-2B 视觉编码器生成每帧 64 个空间 token。池化向量用于事件切分，完整空间特征和时间戳存入每视频的 SQLite 索引。 |
| 事件切分 | 当前帧与当前段中心的余弦距离连续两次达到 0.02、且满足最短 3 秒条件时确认边界。已结束事件按其首次可用时间入库；回放早期时刻只访问当时已发布的信息。 |
| 近期分支 | 最近最多 24 帧的空间 token 进入 Short Q-Former，输出 32 个视觉 token。同一视频同一时刻可跨问题共享。 |
| 长期分支 | 在已发布事件中剔除完全落入近期窗口的段。每个候选取最接近其中心的代表帧，由共享 Qwen 结合问题按 Yes/No logit 差打分，排序过程不回传梯度。取最多 5 段，每段均匀采样最多 24 帧的原始空间特征，由 Long Q-Former 联合压缩为 128 个视觉 token。 |
| 生成和缓存 | 长期前缀、近期前缀、问题、事件区间和当前时间进入 Qwen。首 token 在 `<WAIT>`、`<ANSWER>` 中贪心选择，再生成正文。WAIT 后推进到下一秒；首次 ANSWER 结束该题。推理时同题检索结果未变，可复用长期前缀。 |

“Query-agnostic”适用于视频编码、切分、建索引和近期视觉分支。问题参与历史排序和最终生成。每次决策记录被检索的事件、采样帧时间和近期窗口，便于回看；这些是候选来源，尚未验证生成答案对某一段的逐条引用。长期分支当前使用共享的 128 个 query；每事件 24 个局部 query 加 8 个跨段 query 是备选设计。

两个 Q-Former 从 InstructBLIP-Vicuna-7B 分别初始化。短期直接复制 32 个预训练 query；长期保留前 32 个，再复制三组并加入小幅独立噪声。训练更新两个 Q-Former、投影、位置参数、Qwen3.5-2B 全部语言层和输出层；视觉编码器冻结。[模型实现说明](../event_retrieval/README.md) · [检索](../event_retrieval/retrieve.py) · [解码与损失](../event_retrieval/answer.py)

## 2. QA v3 配置与生成规则

10 个 holdout 来源视频共 101.2 分钟，形成 **97 道英文候选 QA**。这次实验将这批视频同时用于拟合和回放。标注以有时间戳的原视频抽帧和触发附近的连续帧为依据，已有 ASR 辅助理解讲解，答案由静音画面支持。生成规则以每分钟约 4 道独立题为目标；本批按实际画面选题，密度为 0.96 道/分钟。标注尚未经独立临床专家盲评。

| 统计口径 | 分布 |
|---|---|
| 问题要求 | 当前观察 54、历史回忆 5、跨时比较 24、事件通知 14 |
| 提问时能否回答 | 即时可答 81、先 WAIT 后可答 15、已审核观察区间内未触发 1 |
| 展开的训练时点 | 246 个，其中 ANSWER 96、WAIT 150 |
| WAIT 文本监督 | 16 个提问时点有原因；后续 134 个 WAIT 时点只监督动作。当时没有独立标注的原因。 |

**题型与回答时机的交叉统计（单位：独立问题）**

| 题型 | 提问时可答 | 先 WAIT 后 ANSWER | 审核区间内始终 WAIT | 合计 |
|---|---:|---:|---:|---:|
| 当前观察 | 54 | 0 | 0 | 54 |
| 历史回忆 | 5 | 0 | 0 | 5 |
| 跨时比较 | 22 | 2 | 0 | 24 |
| 事件通知 | 0 | 13 | 1 | 14 |
| **总计** | **81** | **15** | **1** | **97** |

跨时比较需要两个观察时刻。22 道比较题在第二段画面出现后才提问，当场即可回看比较；另外 2 道在第二段出现前提问，需要等待。事件通知的 14 道都以 WAIT 开始，13 道随后触发，1 道在已审核区间没有触发。因此两类合计 38 道，其中 16 道提问时 WAIT，15 道有 WAIT → ANSWER 转换。

每题 11 个字段：`video_id`、`question_id`、`qa_type`、`query_time`、`question`、`decision_at_query`、`decision_reason_at_query`、`trigger_criterion`、`answer_time`、`reference_answer`、`evidence_windows`。时间沿用源视频秒数，证据窗为闭区间。触发条件、参考答案和证据窗用于标注与复核；推理输入只使用问题及当时可见的上下文。部分题的证据来自教学切镜或画中画，另一些来自连续扫查中的渐变。[数据与逐视频清单](../results/qa_v3_holdout10/README.md) · [生成规则](QA_GENERATION_RULES_zh.md) · [审核记录](../results/qa_v3_holdout10/audit.json)

### 怎样生成一道流式 QA

1. 按原视频时钟定位独立的视觉事件，区分连续扫查、历史回放、讲解切镜。可以由模型自主提问；题型由可见内容决定，每个视频没有四类配额。
2. 在提问时刻暂停，只用此前可见的画面写英文问题、当前决定和理由。若两段影像都已出现，可立即比较；若题目要求“再次出现”，需写清本次变化怎样才算发生，问句不能透露未来答案。
3. 对等待题逐秒向前核验，快速变化处回看连续原帧。把首次明确满足触发条件的采样时刻记作 `answer_time`；模糊过渡点单列审核。到审核区间结束仍无证据时，答案时间和参考答案均为 `null`。
4. 保存支持判断的最短证据窗；跨时比较保留前后两段。静音复核画面是否足以作答，检查字幕、箭头和教学布局是否泄露答案。QA v3 实际采用抽帧和带时间戳的 ASR，本批尚未逐题实施遮字消融。[标注与审核规则](QA_GENERATION_RULES_zh.md)

## 3. v1 Overfit 配置与共用模块参数量

共同设置：同一批 97 题，20 轮，每轮 31 步，共 620 次更新；实际 BS8，GA1，BF16，语言模型学习率 1e-5，两个分支及投影学习率 1e-4。从相同来源的预训练权重独立初始化。首轮基线使用单张 A100、检索批量 4；后两版使用单张 H100、检索批量 64。v1 作业 5961170 的 Slurm 运行时长（含训练与回放）为 53 分 9 秒。优化器为 AdamW，随机种子 0，每两轮保存一次 checkpoint。训练输入为静音画面；v1 首动作加权版评测并行推进最多 10 个视频，单次生成最多合批 10 道题。

### 作业 5961170 的模块参数量

| 模块 | 可训练参数 | 冻结参数 | 合计 |
|---|---:|---:|---:|
| Qwen3.5 语言层与共享词嵌入／输出头 | 1,881,825,088 | 0 | 1,881,825,088 |
| Short Q-Former（含 32 个 query 和输出投影） | 106,737,152 | 80,522,496 | 187,259,648 |
| Long Q-Former（含 128 个 query 和输出投影） | 106,810,880 | 80,522,496 | 187,333,376 |
| 共享视觉输入投影（2048 → 1408） | 2,884,992 | 0 | 2,884,992 |
| 帧位置嵌入（24 × 1408） | 33,792 | 0 | 33,792 |
| 事件位置嵌入（5 × 1408） | 7,040 | 0 | 7,040 |
| Qwen3.5 视觉编码器及其视觉合并层 | 0 | 331,416,576 | 331,416,576 |
| **整模去重合计** | **2,098,298,944** | **492,461,568** | **2,590,760,512** |

两个分支各自的 query 参数分别为 24,576 和 98,304；每个分支的输出投影含 1,574,912 个参数，均已计入分支行。两个编码器都保留了冻结的 BERT 词嵌入、文本位置嵌入与文本前馈层，视觉 query 使用自己的前馈层。Qwen 语言模型全参更新，输入嵌入与输出头绑定，两个动作 token 使用共享矩阵中的行。检索复用同一个 Qwen。

统计来自作业 5961170 的第 20 轮 checkpoint、冻结快照中的模块定义和原始 Qwen 视觉权重。按参数元素数统计，排除索引、位置 ID 缓冲区、梯度和 AdamW 动量。两组 AdamW 一阶矩覆盖的元素数分别为 **216,473,856**（双分支、投影与位置参数）和 **1,881,825,088**（语言模型），与可训练合计一致。[冻结模块定义](../cluster_data/experiments/event_retrieval/overfit_qa97_actionK_io6ld4bd/code/event_retrieval/qformer.py) · [优化器与训练入口](../cluster_data/experiments/event_retrieval/overfit_qa97_actionK_io6ld4bd/code/event_retrieval/train.py) · [第 20 轮 checkpoint](../cluster_data/experiments/event_retrieval/overfit_qa97_actionK_io6ld4bd/fit/latest.pt)

## 4. v1 Overfit 损失与逐轮曲线

令一个时点的动作 CE 为 $a_i$，正文及 EOS 的平均 CE 为 $b_i$。裸 WAIT 的 $b_i=0$。对延迟题的 ANSWER，原始权重 $w_i=K$，其中 K 为该题纳入训练的 WAIT 点数；其他时点取 1。归一化权重 $u_i=w_i/\bar w$，本批 $\bar w=366/246$。训练使用全词表下一 token 交叉熵。

| 作业 | 优化目标 |
|---|---|
| 5959357 · 等权 | 每个时点先对动作、正文及有效 EOS 的 CE 取 token 均值，再对时点等权平均。 |
| 5959690 · 整条 ANSWER 加权 | 将上述时点均值乘以 $u_i$。动作分到的系数随回答长度增加而减小。 |
| 5961170 · 仅首 token 加权 | 每批优化 $L=\frac{1}{B}\sum_i(u_i a_i+b_i)$。K 只作用于决定 WAIT/ANSWER 的首 token，正文逐时点单独平均。 |

v1 首动作加权版第 20 轮总 loss **0.5527**，其中加权动作 CE **0.5422**，正文 CE **0.0104**。三版目标的标度不同，横向比较以同一回放任务为准。

| Epoch | 加权决策 CE | 正文 CE | 总 loss |
|---:|---:|---:|---:|
| 1 | 2.1485 | 1.2418 | 3.3902 |
| 5 | 0.7227 | 0.0698 | 0.7925 |
| 10 | 0.5874 | 0.0056 | 0.5930 |
| 15 | 0.5167 | 0.0222 | 0.5390 |
| 20 | 0.5422 | 0.0104 | 0.5527 |

两张图都使用完整的 20 个训练轮次。首 token 加权版后期的主要损失来自动作选择，本次没有独立验证集 loss 曲线。

![前两版每轮训练损失](../results/event_retrieval_qa97_loss/loss_curves.png)

![v1 首动作加权版逐轮训练损失](../results/event_retrieval_qa97_loss/action_body_loss.png)

[前两版损失曲线](../results/event_retrieval_qa97_loss/loss_curves.png) · [v1 首动作加权版配置](../cluster_data/experiments/event_retrieval/overfit_qa97_actionK_io6ld4bd/plan.json) · [训练日志](../cluster_data/experiments/event_retrieval/overfit_qa97_actionK_io6ld4bd/logs/slurm-5961170.log)

## 5. v1 Overfit 后 checkpoint 的生成质量

三个模型均回放同一批视频。指标看每题**首次**输出 ANSWER：时间等于标注秒数才算准时；“答对”按忽略大小写、合并空白后的参考文本精确匹配。96 道可答题含 81 道即时题、15 道延迟题。另 1 道区间内未触发题在三版回放中都保持 WAIT。

| 指标 | 等权 | 整条 ANSWER 加权 | 仅首 token 加权 |
|---|---:|---:|---:|
| 即时题准时且答对 / 81 | 74 | 78 | 80 |
| 延迟题提前 / 准时 / 晚答 / 等到结束（共 15） | 0 / 0 / 0 / 15 | 0 / 1 / 2 / 12 | 11 / 1 / 3 / 0 |
| 准时且答对 / 96 | 74 | 79 | 81 |
| 首次回答文本精确匹配，忽略时间 / 96 | 74 | 81 | 94 |
| 实际回放决策数 | 3,734 | 3,296 | 692 |

整条加权版在 150 个实际看到的标注 WAIT 时点有 **0 次误触发**；仅首 token 加权版是 **11/89**。新模型提前回答后该题即停止，剩余 61 个原定 WAIT 时点没有被评估。固定标注点强制生成属于另一口径：等权基线的强制答案精确匹配为 89/96，QA demo 中展示的是该基线的固定时点结果。[等权逐秒评测](../cluster_data/experiments/event_retrieval/stream10_full97_5s81ummq/replay/timing_metrics.json) · [固定时点指标](../cluster_data/experiments/event_retrieval/overfit_qa_v3_10v_SprOW4No/fit/overfit_metrics.json) · [后两版逐秒对照](../cluster_data/experiments/event_retrieval/overfit_qa97_actionK_io6ld4bd/replay/baseline_comparison.json)

### WAIT 理由与 ANSWER 正文

v1 首动作加权版共生成 96 次 ANSWER、596 次 WAIT。96 道可答题里 **94 道的答案文字精确匹配**（97.9%），**81 道同时准时且文字匹配**（84.4%）。即时题贡献了其中 80 道，延迟题只贡献 1 道。

在提问时标为 WAIT 的 16 道题中，模型对 6 道输出 WAIT，这 6 条理由均与该时刻的参考理由精确匹配；另 10 道在提问当秒提前输出 ANSWER。全部 596 条 WAIT 都有非空理由，但按统一大小写和空白去重后只有 **6 种文本**。后续秒缺少独立参考理由，因此不能把“每次有文字”当作“每秒解释都正确”。[逐秒生成原文](../cluster_data/experiments/event_retrieval/overfit_qa97_actionK_io6ld4bd/replay/stream_predictions.jsonl)

### 提前回答的视觉证据

v1 首动作加权版的 11 道提前题里，10 道一提问就回答，提前幅度为 2–26 秒。10 道文本与参考答案精确匹配，另 1 道语义一致。复看作答时的原视频、近期 24 帧和检索历史后，**10 道在作答时仍应 WAIT，1 道触发边界有歧义**。例如心脏扇区在 94 秒刚变暗，模型却回答回声已经重现，98 秒画面才支持这句话。边界题在 269 秒已有较弱的亮带，271 秒才形成标注要求的明显亮柱。该复核是原画面抽帧对照，未独立逐帧重定所有题的最早可答时刻。

当前最明显的不足是对“本次目标是否已经出现”的判断。仅首 token 加权消除了延迟题一直等到结束的现象，同时引入了 11 次提前触发；15 道延迟题的准时数仍为 1。97 题 overfit 结果不能估计独立视频上的能力。[逐题视觉复核与截图](../results/event_retrieval_qa97_loss/early_answer_review/review.md) · [v1 首动作加权版逐秒预测](../cluster_data/experiments/event_retrieval/overfit_qa97_actionK_io6ld4bd/replay/stream_predictions.jsonl) · [v1 首动作加权版 checkpoint](../cluster_data/experiments/event_retrieval/overfit_qa97_actionK_io6ld4bd/fit)

## 6. v1 首动作加权版的回放耗时

单张 H100、最多十视频并行，训练索引可复用。用运行中两次观测到的预测文件更新时间对照记录行数：第 344 行（12:32:11）与第 505 行（12:33:42）都是同一视频的 Q004，分别在源视频 1801 秒和 1881 秒输出 WAIT。中间该题每个整数秒均有预测。约 91 秒墙钟时间推进了 80 个连续视频秒，平均约 **1.14 秒/次更新**，包含同批其他问题的工作。整个回放从约 12:29:35 到 12:36:01，产生 692 条决策，平均约 1.8 条决策/秒。以上为离线带正文生成的观测；单帧编码、事件更新、首 token 延迟和单请求 P95 未单独计时。
