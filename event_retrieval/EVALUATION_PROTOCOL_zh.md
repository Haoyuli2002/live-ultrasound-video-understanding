# Event Retrieval 完整评测流程与 LLM Judge 评分协议

版本：v2（2026-11-27 对话确认）。适用于 Event Retrieval 模型；与 StreamReady 的架构、checkpoint 和结果分开记录。

本协议包含三个独立部分：**WAIT／ANSWER 决策与 Timing、生成 ANSWER 的语义及视觉支持度、生成 WAIT 原因的语义**。Timing 由程序计算，内容由 Judge 评价；答案说得对，不代表回答及时，WAIT 动作正确也不代表等待理由正确。

这是确认后的目标协议，不表示所有字段已经接入评测代码。当前实现与待补项见第 9 节。旧版 [judge/GRADING.md](judge/GRADING.md) 的 0–2 分 ANSWER 标准由本文的 0–3 分标准替代；旧结果保留原版本，不能直接混算。

## 1. 数据、模型和评测记录

### 1.1 固定评测对象

评测前记录 checkpoint 路径、SHA256、训练轮次和 step、模型配置、数据版本及清单 SHA256、解码配置、采样策略、代码版本、Judge 模型及 revision、prompt 和 schema 版本。

当前全量任务的评测对象是开发 holdout：16 视频、464 QA、994 个标注决策点，其中 464 ANSWER、530 WAIT。训练集为 171 视频、5,236 QA。该 holdout 不是独立 Test；开发阶段曾用于 overfit 的视频必须标记，分别报告全部视频、开发使用过的视频和其余视频的结果。数据范围变更时重新从清单计算数量，不沿用这些常数。

保留以下两个层级的数据：

- **逐决策点**：video_id、task_id、sample_id（标注点适用）、question、qa_type、query_time、current_time、标注动作、自然预测动作、生成正文／WAIT 原因、实际可见帧时间、检索事件、finish_reason。
- **逐问题**：reference_answer、answer_time、first_answer_time、首次自然答案、Timing 类别、GT 点正常预测、GT 点诊断答案、是否真正强制生成、视频是否曾用于开发，以及 Judge 评分。

`answer_time` 是标注时刻，`first_answer_time` 是模型第一次自然输出 ANSWER 的时刻，不能相互覆盖。`current_time` 和这些时刻使用视频时间，不是运行耗时。

### 1.2 两种评测模式

| 模式 | 执行方式 | 用途 |
|---|---|---|
| 固定标注点独立评测 | 对每个标注点，使用截至该点的因果输入独立预测；即使自然回放已提前回答，也评估该点 | 完整 WAIT／ANSWER 分类及 GT 点答案诊断 |
| 自然逐秒回放 | 从 query_time 开始按秒预测，首次 ANSWER 后停止该问题；否则到指定评测终点 | 首次触发 Timing、真实回答行为和 WAIT 原因 |

两种模式均不得把标注动作或未来信息放入被评模型输入。强制 ANSWER 仅用于单独的诊断分支，不参与自然回放的动作或 Timing 计算。

评测终点必须记录。对于无答案题，只在标注确认无答案的时间范围内判定误触发；范围之外不能自动认定模型错误。当前可答题可以记录到片尾仍未回答，但必须注明实际回放终点。

## 2. WAIT／ANSWER 决策和 Timing

### 2.1 固定标注点的分类指标

设 ANSWER 为正类：TP 为标注 ANSWER 且预测 ANSWER；FN 为标注 ANSWER 但未预测 ANSWER；TN 为标注 WAIT 且预测 WAIT；FP 为标注 WAIT 且预测 ANSWER。

| 指标 | 分子／分母 |
|---|---|
| WAIT 正确率 | TN／全部标注 WAIT 点 |
| WAIT 误触发率 | FP／全部标注 WAIT 点 |
| ANSWER 点触发率 | TP／全部标注 ANSWER 点 |
| ANSWER 点漏触发率 | FN／全部标注 ANSWER 点 |
| Timing 分类准确率 | (TP + TN)／全部标注决策点 |
| ANSWER Precision | TP／全部预测 ANSWER 点 |
| Balanced accuracy（辅助） | WAIT 正确率和 ANSWER 点触发率的平均值 |

在合法动作只有 WAIT／ANSWER 时，WAIT 正确率与误触发率互补。解析失败或非法输出必须单列为 invalid，保留在相应标注点分母中，不能当成 WAIT 或直接删除；此时同时保留真实动作混淆矩阵，不再机械套用二分类互补关系。

这里的“Timing 分类准确率”特指**标注点的 WAIT／ANSWER 分类准确率**。不要另造一个把所有正常题都标成 on_time 的四分类 accuracy；自然首答时机使用下一节的分布和准时触发率。

### 2.2 自然首答的时机分类

令 t_first 为首次自然 ANSWER 时刻，t_GT 为标注 answer_time：

| Timing 标签 | 条件 |
|---|---|
| `on_time` | t_first = t_GT |
| `early` | t_first < t_GT |
| `late` | t_first > t_GT |
| `never` | 到评测终点仍未输出 ANSWER |

**ANSWER 准时触发率 = on_time 题数／全部可答题数。**提前率、迟答率、未触发率使用相同分母；四者相加为 100%。另外可报告失败题内部 early／late／never 的组成，但必须标明该分母是未准时触发题数。

未触发题的 first_answer_time 和自然答案为 null，不伪造回答时间。未触发题保留在题级准确率的分母中，不纳入有回答题的时间偏差平均值。

当前标注分辨率为 1 秒。主结果严格按标注时间判定；仅在预处理时统一时间单位和数值精度，不擅自四舍五入改变标签。辅助结果可给出 `0 <= t_first - t_GT <= 1/2/5 秒` 的迟答容忍窗口，不能用绝对值窗口把提前回答算对，也不改变主结果。

报告迟答的中位数和 P90、触发题的有符号时间偏差，以及各自分母。这里衡量视频时刻上的触发偏差，不是推理 wall-clock latency。

### 2.3 ANSWER 点触发率与准时触发率的区别

假设 t_GT = 10 秒：

| 模型行为 | ANSWER 点触发是否正确 | 自然首答是否准时 |
|---|---|---|
| 第 8 秒已回答；独立测试第 10 秒也回答 | 是 | 否，early |
| 第 10 秒首次回答 | 是 | 是 |
| 第 10 秒 WAIT，第 12 秒首次回答 | 否 | 否，late |
| 始终 WAIT | 否 | 否，never |

点触发率诊断“证据已到位时能否输出 ANSWER”；准时触发率评价整个实际交互过程。

### 2.4 分组和覆盖率

所有核心指标按以下维度拆分：

- 全部 QA；即时 QA（answer_time = query_time）；WAIT→ANSWER QA（answer_time > query_time）。
- current_observation、history_recall、event_notification、temporal_comparison。
- 每个视频；开发阶段使用过的视频／其余视频；总体 micro 和视频 macro。
- 有明确无答案标注的题单列，不放入可答题触发率分母。

即时 QA 组若没有标注 WAIT 点，WAIT 正确率为 N/A。任何指标分母为零均显示 N/A。

自然回放提前 ANSWER 后未再访问的 WAIT 点不能算正确。单独报告“实际访问的标注 WAIT 点／全部标注 WAIT 点”的覆盖率，以及覆盖点上的 WAIT 正确率；不要与固定标注点的完整分类结果混用。

## 3. ANSWER 生成与语义评估

### 3.1 自然答案与 GT 点强制诊断

**A. 自然准时答案**：以 on_time 自然首答为主语义评分组。提前、迟答答案保留，可以作为单独诊断评分，但不得混入“自然准时答案语义正确率”。自然漏答为 missing answer，不是一次已生成的空正文。

**B. GT 点诊断**：在 answer_time 使用固定点的因果输入进行正常预测：

1. 正常输出 ANSWER：保存其答案，`forced_used=false`。
2. 正常输出 WAIT：同一时刻强制 ANSWER，保存诊断答案，`forced_used=true`。
3. 自然回放提前结束、没有访问 GT 点：仍做上述独立 GT 点诊断，不拿提前答案冒充 GT 点答案。
4. 非法输出或生成错误：单列失败，不能伪装成正常 WAIT。

强制只改变输出动作约束，不补充未来帧、参考答案或标注证据文字。记录 GT 点正常动作、自然 Timing 类别和强制生成标记。

分别报告“全体 GT 点诊断答案”和“真正被强制的子集”。对子集再按自然 early／late／never 分组。正常 GT 点答案与自然准时答案只有在相同时刻、输入和解码一致且文本相同时才可复用评分。不能把所有 GT 点诊断答案都称作真正强制生成。

### 3.2 ANSWER 语义评分：0–3

| 分数 | 标准 |
|---|---|
| **0：错误** | 答案错误、答非所问、没有实质相关内容；输出 ANSWER 但正文为空也为 0。 |
| **1：部分正确，核心结论错误** | 有正确的局部信息，但核心结论错误，包括关键对象、方位、关系或变化方向判断错误。 |
| **2：基本正确** | 核心结论正确，没有改变结论的重大错误，但有必要信息、限定条件或比较细节遗漏。 |
| **3：正确完整** | 与参考答案语义一致，或者能够完整、正确地回答 question；接受同义改写、等价单位和合理的不同表述。 |

判分规则：

- 核心结论错误，最高 1 分，不能因正确关键词较多而给 2 分。
- 核心正确但遗漏题目要求的信息，给 2 分。简洁但完整的答案可以得 3 分。
- 不因与参考答案措辞不同而扣分，不因额外无关的专业术语而加分。
- 对比较题核对对象、前后关系、方向和变化；对数字核对数值与单位。
- 参考答案不是唯一合法措辞。候选答案与参考存在实质冲突、但画面可能支持候选时，标记人工复核，不能未经复核改写标注或把不确定判成已确认正确。

### 3.3 Visual support：模型答案是否有视觉依据

`visual_support` 检查的是：**训练后的模型所生成的具体答案，是否被回答时刻及之前的视觉证据支持。**它不是直接判断整道 QA 是否合理，也不是判断当前是否应该 WAIT 的替代指标。

| 标签 | 含义 |
|---|---|
| `supported` | 回答中的关键视觉事实有画面依据。 |
| `contradicted` | 画面明确显示与回答中的关键事实相反的内容。 |
| `insufficient` | 给 Judge 的画面不足以确认或否定回答，例如抽帧覆盖不足、模糊、目标不可见。 |

历史回忆和前后比较必须提供相关历史帧；不能只看当前帧。任何帧的真实 PTS 不得晚于该次回答时刻。`insufficient` 不等于答案错误，也不等于 WAIT 正确。

语义分和视觉支持度分别保存，不把“与参考文本一致”自动视为 supported。发现参考／时间标注与画面可能冲突时进入复核队列，原始 Timing 和评分保留。主协议不额外引入 `visual_answerability` 指标。

### 3.4 ANSWER Judge 返回示例

以下是拟采用的 v2 输出示例，不是当前某道题的实际结果：

```json
{
  "schema_version": "event_retrieval_answer_judge_v2",
  "grading_status": "scored",
  "semantic_score": 2,
  "core_conclusion": "correct",
  "correct_points": ["正确覆盖的关键事实"],
  "missing_points": ["遗漏的必要比较细节"],
  "incorrect_points": [],
  "visual_support": "supported",
  "needs_review": false,
  "review_reason": null,
  "reason": "核心结论正确，但遗漏了题目要求的比较细节。"
}
```

`semantic_score` 为整数 0–3；技术失败／无法可靠评分时为 null，并设置 `grading_status=ungraded`。`core_conclusion` 为 correct／incorrect／uncertain。空 ANSWER 正文是模型输出问题，评分为 0；Judge 超时或 JSON 解析失败是评测问题，不能评分为 0。

导出器在 Judge 结果外关联 task_id、answer_source（natural_on_time／gt_normal／gt_forced 等）、回答时刻、帧 PTS 和资源哈希，不让 Judge 自行生成这些标识。

### 3.5 ANSWER 指标和分母

| 指标 | 分母 |
|---|---|
| 自然准时答案 0／1／2／3 分分布、平均分 | 已成功评分的自然准时答案，并报告评分覆盖率 |
| 自然准时答案语义 ≥2 分／3 分比例 | 已成功评分的自然准时答案 |
| **准时且语义 ≥2 分的题级比例** | **全部可答题** |
| **准时且语义 3 分的题级比例** | **全部可答题** |
| 准时、语义合格且视觉 supported 的联合比例 | 全部可答题；分别保留 ≥2 和 3 分口径 |
| GT 点诊断语义分布和 ≥2／3 分比例 | GT 点诊断组，区分 gt_normal 和 gt_forced |
| 真正强制回答的语义分布和 ≥2／3 分比例 | forced_used=true 的诊断子集 |
| supported／contradicted／insufficient 分布 | 相应答案评分组 |

存在 Judge 技术失败时显示覆盖率和未评分数，不静默移除。全部可答题分母的最终主结果应在补齐评分后发布；若提前发布，只能明确标注部分结果及已确认通过率下界，不能把技术失败称作模型答错。

精确匹配和现有 normalized/numeric/short-phrase 文本匹配仅作辅助诊断，不能冒充 LLM 语义评分。不同评分版本的结果不直接比较。

## 4. WAIT 原因的语义评估

### 4.1 评估对象

对模型**自然输出 WAIT 时实际生成的原因正文**评分。无正文的裸 WAIT 记为未提供解释，不凭空补写原因，也不在主评测中追加“请解释原因”的提示诱导新输出。

如果以后需要额外追问解释，那是独立的 explanation probe，必须与自然 WAIT 原因分开报告。

正常 WAIT 阶段的理由与 GT 已到但仍 WAIT 的理由分组：前者检查合理等待，后者用于漏触发解释分析。正确的 WAIT 标签不能替代对理由内容的检查。

### 4.2 三个维度分别评分

各维度为 0／1／2 分；证据不足以判断则为 null，记录原因，不强行判错。

| 维度 | 0 分 | 1 分 | 2 分 |
|---|---|---|---|
| **事实正确性** | 对当前状态的关键描述与画面明确矛盾 | 部分正确，但存在不准确或过度推断 | 关键状态描述均有当前及必要历史画面支持 |
| **理由相关性** | 理由与回答问题所需证据无关 | 有关联，但没有明确指出缺少什么 | 明确指出与问题直接相关的缺失证据或未满足条件 |
| **等待充分性** | 条件已满足仍编造等待依据，或所述理由不能支持等待 | 等待可能合理，但所给解释不足 | 所缺证据确实阻止可靠回答，足以支持继续等待 |

例：问题为“目标消失时通知我”。“目标仍然清晰可见，尚未观察到消失”且画面支持，可三项均给 2 分；“需要更多信息”不能仅因动作正确就给满分；目标已明确消失却说未消失，事实和等待充分性应判错；无关地说“等待血流信号”，相关性不合格。

不能从一张无变化的抽帧直接断言整个历史都没有发生事件；证据覆盖不足时应标为无法确认。Judge 判断的是所展示证据中的解释合理性，不是保证视频全时段事实正确。

### 4.3 WAIT Judge 输入与返回

输入包括 question、query_time、current_time、截至 current_time 的当前及必要历史画面、生成的 reason。若使用 trigger_criterion，只允许独立于未来答案的条件定义；不得传入实际 answer_time、最终 reference_answer、未来 evidence 帧或事件发生结果。

不提供模型名称、实验组、程序 Timing 结果或自动匹配分数。Judge 不应看到“这是一个正确 WAIT”的提示。

```json
{
  "schema_version": "event_retrieval_wait_judge_v2",
  "grading_status": "scored",
  "reason_present": true,
  "factual_correctness": {
    "score": 2,
    "reason": "画面中目标仍然可见。"
  },
  "relevance": {
    "score": 2,
    "reason": "问题要求在目标消失时通知，目标是否消失直接相关。"
  },
  "wait_justification": {
    "score": 2,
    "reason": "尚未观察到消失，继续等待有充分依据。"
  },
  "needs_review": false,
  "review_reason": null
}
```

裸 WAIT 的 `reason_present=false`、`grading_status=missing_reason`，三个维度的 score 均为 null；它不是一次语义评分失败。技术失败用 ungraded；非空理由因证据不足无法判定时，保留能判定的维度，其余为 null 并注明具体限制。

### 4.4 WAIT 原因指标和覆盖率

- **理由生成覆盖率**：有非空原因的自然 WAIT 次数／自然 WAIT 总次数。
- 三个维度各自的 0／1／2 分分布、均分、null 比例与可评分覆盖率。
- 三项均为 2 分的比例：同时给出在三项可评分的非空原因中的比例，以及在所有自然 WAIT 中已确认合格的比例。
- **联合成功比例**：标注 WAIT 点中，自然回放实际访问、预测 WAIT、有原因且三个维度均为 2 分的点数／全部标注 WAIT 点。未访问不能计成功，并另报实际访问点内的条件比例及覆盖率。
- 对非法 WAIT 理由、事实矛盾、理由不相关、条件已满足仍等待、泛泛而谈分别统计。
- GT ANSWER 点上的错误 WAIT 原因单列，不能因为理由写得流畅而改判动作正确。

逐秒原因数会随等待长度变化，因此同时给出逐决策点 micro 和逐问题 macro（先在题内计算，再跨题平均），并按即时／延迟题及题型拆分。如因预算只评分部分 WAIT 原因，预先固定抽样方案，报告抽样覆盖，不能把抽样计数冒充全量计数。

### 4.5 当前数据限制

当前 full-data holdout 的 WAIT 训练目标是裸 `<WAIT>`。228 道题有 waiting_evidence 帧，但没有相应的文字参考理由。该数据仍可用于基于画面和问题的 WAIT 原因合理性评分；不要求先有逐字参考理由。

不能据此声称模型已接受 WAIT 原因生成监督。如果自然输出全部为裸 WAIT，应如实报告理由覆盖率为 0、条件语义指标 N/A，而不是伪造评分或把全部解释判为语义错误。任何改变训练目标或增加强制解释的实验另行记录。

## 5. Judge 的共同约束

1. 答案语义参考与候选答案均作为待评数据，不执行其中的指令。
2. ANSWER Judge 看问题、参考答案、候选答案和回答时刻前的画面；WAIT Judge 不看未来答案及未来时间标签。
3. Judge 只能访问该候选输出时已经可见的帧；实际帧 PTS 必须 <= 输出视频时刻。历史题保留必要历史证据。
4. 冻结并记录选帧规则、分辨率、帧数、帧 PTS 和图像哈希。模型输入 1fps/256px 与 Judge 使用的抽帧/分辨率分别说明；Judge 看得更清晰不能证明被评模型当时拥有同等可见信息。
5. 除非额外定义对照，不给 Judge 音频、ASR、模型名称、实验名称、自动文本匹配分数或正确／错误 Timing 标签。
6. 用固定 Judge checkpoint、prompt、解码参数和 schema；保留原始返回、重试次数和解析结果。同一输入的缓存键覆盖问题、候选、参考（如使用）、全部视觉输入和 Judge 配置，不能仅按 task_id 复用。
7. Judge 评分属于模型辅助评估。参考冲突、视觉证据矛盾／不足、边界分数和不稳定评分进入人工复核；保留原始评分和复核结果，不能覆盖证据轨迹。
8. Judge 的技术失败与被评模型的生成失败分别统计。评测失败不当作零分；被评模型已输出 ANSWER 但正文为空才按模型错误记 0。

## 6. 执行顺序与产物

| 顺序 | 操作 | 产物／用途 |
|---|---|---|
| 1 | 固定 checkpoint、清单、配置与开发重叠视频列表 | 评测 manifest |
| 2 | 独立评估全部标注点；GT 点必要时强制 ANSWER | 全覆盖动作预测、GT 点诊断答案 |
| 3 | 自然逐秒回放，保留所有 WAIT 原因及首次 ANSWER | 完整决策轨迹 |
| 4 | 程序计算动作混淆矩阵、首答 Timing、覆盖率和分组 | decision／timing 报告 |
| 5 | 导出自然准时答案、GT 点诊断答案、自然 WAIT 原因及因果视觉输入 | 两类 Judge 输入；记录真实 forced_used |
| 6 | 按本协议执行 ANSWER 0–3 评分、visual_support 和 WAIT 三维评分 | 带 schema_version 的逐条评分 |
| 7 | 校验数量、去重、时间、帧因果性；处理失败与人工复核项 | 校验与复核记录 |
| 8 | 合并分组指标及 GPU 统计 | Markdown＋JSON 总报告 |

已有文件沿用：`fit/holdout_predictions.jsonl`、`fit/holdout_metrics.json`、`replay/stream_predictions.jsonl`、`replay/judge_inputs.jsonl`、`replay/quality.json`。其中 quality.json 的自动文本匹配结果不是语义 Judge 结果。

v2 接入时建议新增独立版本目录 `judge_v2/`，存放 `manifest.json`、`answer_inputs.jsonl`、`answer_grades.jsonl`、`wait_inputs.jsonl`、`wait_grades.jsonl`、`review_queue.jsonl`、`summary.json`。这是目标产物约定，不声称这些文件已经生成。不要直接把新 0–3 分结果追加到旧版 0–2 分 graded.jsonl。

## 7. 最终报告结构

1. 实验与评测范围：模型、checkpoint、数据拆分、开发重叠、标注精度、评分版本及覆盖率。
2. 决策：WAIT 正确率、ANSWER 点触发率、Timing 分类准确率；即时／延迟 QA 分组。
3. 自然首答：准时、提前、迟答、未触发；失败组成和迟答分布。
4. ANSWER：自然准时与 GT 点正常／真正强制诊断的 0–3 分分布，语义 ≥2／3 分比例、题级时机＋内容联合指标、visual_support 分布。
5. WAIT 原因：理由覆盖率、事实／相关性／充分性三维分布、联合成功率及错误类型。
6. 分组结果：题型、视频、开发使用情况、micro／macro；所有比例给出分子和分母。
7. 资源：训练全程 GPU 和显存统计，索引／训练／生成评测分开。
8. 未完成评分、数据或视觉不确定性、技术失败和复核清单。

## 8. GPU 利用率与显存统计

- GPU 平均利用率按采样时间加权；每张卡分别报告，多卡同时报告聚合口径。
- 显存占用率为 memory.used／memory.total，报告平均、峰值和实际 GiB。不要把显存带宽忙碌率称作显存容量利用率。
- 训练作业全程统计包含初始化、训练、holdout loss、checkpoint 保存；不能称为纯 forward/backward 利用率。索引和生成评测作业另列。
- 报告采样间隔、有效覆盖秒数、作业起止时间、缺失／异常样本和长间隔。现有汇总使用梯形时间加权，超过 5 秒的采样间隔不插值，并报告缺口。
- 队列等待时间不进入 GPU 使用均值；没有完整遥测时明确标为部分统计。

## 9. 当前实现与待接入项

本次修改是文档整理，没有修改训练、Judge 代码、运行中作业或冻结的实验快照，也没有新提交训练或评分作业。

| 能力 | 已检查的实现／状态 |
|---|---|
| 固定标注点预测及 GT 点必要时强制回答 | [eval_overfit.py](eval_overfit.py) 已有；预测记录尚需显式导出 forced_used 和对应分组 |
| 自然逐秒回放 | [parallel_replay.py](parallel_replay.py) 已有；记录动作、答案／原因和实际时刻 |
| 自动匹配、时机和 GPU 汇总 | [report_training.py](report_training.py) 已有；文本匹配不是 LLM 语义评分 |
| ANSWER 视觉 Judge | [judge/grade_holdout.py](judge/grade_holdout.py) 当前实现为旧版 0–2 分；尚未迁移为本文 0–3 分 schema |
| WAIT 原因三维 Judge | 尚未在上述 Judge 中实现；需新增因果输入导出、评分和覆盖率统计 |
| v2 主指标和联合指标 | 待对接上述 v2 Judge 产物；不能将旧报告冒充已执行本协议 |
| invalid 动作／Judge 失败的完整统计 | 需对照本文统一；现有脚本中的断言或旧字段不等价于已完成 v2 统计 |

当前全量运行目录为：

```text
/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding/cluster_data/experiments/event_retrieval/full_28plus2_bs8_ga2_2ep_j7kvlz4d
```

该目录的冻结代码和历史报告维持其原评分口径。后续实现本协议时，应保存独立版本、验证 schema 和指标分母，再生成 v2 报告。
