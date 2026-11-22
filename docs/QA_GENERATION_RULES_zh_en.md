# 超声视频 QA 数据生成规则

# Ultrasound Video QA Generation Rules

版本 / Version: 1.2  
日期 / Date: 2026-11-20

本规则用于生成和审核超声视频 QA 训练候选数据。核心目标是理解超声影像、扫查操作及其动态变化，优先关注器官、解剖结构和切面。训练问题和答案使用英文，中文仅作审阅译文。

These rules govern the generation and review of candidate QA training data for ultrasound videos. The primary objective is to understand ultrasound images, scanning procedures, and changes over time, with priority given to organs, anatomical structures, and imaging views. Training questions and answers must be in English; Chinese translations are for review only.

## 中文规则

### 1. 内容范围

| 优先内容 | 考察重点 |
|---|---|
| 器官与解剖结构 | 当前出现了什么器官、结构在哪里、与周围组织有什么关系 |
| 超声影像特征 | 形态、边界、回声明暗、声影、可见的多普勒信号 |
| 切面理解 | 当前是什么可辨认的切面、切面如何随扫查变化 |
| 扫查过程 | 哪些结构先后出现、何时进入或离开视野、是否再次出现 |
| 探头操作 | 可见的移动、旋转、倾斜，以及对应的影像变化 |
| 介入操作 | 仅在相关视频中考察针体、针尖及其与目标结构的关系 |

超声图像本身可以构成充分证据，不要求同时显示设备、探头或操作者。纯超声片段中的器官、结构、切面及其出现、消失、再次显示和形态变化均可用于 QA。仅当题目考察探头或操作过程时，才需要相应操作画面；针尖不是必选目标。

不要求每个视频覆盖所有内容。无法可靠命名结构时，可以描述可见特征，也可以舍弃该题。描述操作和影像变化同时发生，不等于已经证明因果关系。

### 2. QA 类型

四类 QA 的目标比例为 1:1:1:1，各约占 25%。当前理解与历史回忆组成直接回答组（Immediate）；事件通知与时序比较组成 Proactive 组，两组目标比例为 1:1。示例仅说明题目方向，只有视频确实提供相应证据时才生成。

| 类型 | 要求 | 示例方向 |
|---|---|---|
| 当前理解 `current_observation` | 根据当前可见信息回答 | 当前超声画面中可以辨认出哪个器官？ |
| 历史回忆 `history_recall` | 必须用到此前的观察 | 在显示膀胱之前，视频扫描了哪个器官？ |
| 时序比较 `temporal_comparison` | 等待提问后的新证据，再与此前观察比较 | 当肾脏再次进入视野时，将它的截面形态与刚才的视图比较。 |
| 事件等待与通知 `event_notification` | 等待指定视觉条件满足 | 当超声画面中可以辨认出心脏时通知我。 |

历史题必须确实需要历史信息，不能仅在问题中添加“刚才”。直接回答组在提问时证据已经充分，`answer_time = query_time`。计入 Proactive 配额的题在提问时尚不可回答；已触发的题必须满足 `answer_time > query_time`。时序比较题必须等待新的视觉证据，并将其与此前观察比较。仅比较已经出现的两个时刻且可以立即回答的题，归入历史回忆，不计入 Proactive 配额。提问时目标已经出现的通知题也不能计入 Proactive 配额。

### 3. 事件通知的规则

优先选择以下触发目标：

- **器官或结构出现**：看到心脏、肾脏、膀胱或某个可辨认结构。
- **特定切面出现**：出现能够可靠辨认的目标切面。
- **目标再次出现**：扫查离开后重新回到某个器官。
- **影像状态变化**：目标离开视野、切面切换，或出现明确可见的变化。
- **介入相关事件**：仅在介入视频中使用针尖等目标，不将其作为默认目标。

必须明确区分以下条件：

| 条件 | 判定含义 |
|---|---|
| 可以辨认出器官 | 已有足够影像证据确认器官身份，不要求完整显示 |
| 器官完整显示 | 规定的完整显示范围已经满足，要求高于仅辨认身份；必须事先明确判据 |
| 目标切面出现 | 满足该切面的可见判据，不能仅凭字幕命名 |

屏幕写着“Heart”，不等于超声画面已经能够辨认出心脏。

### 4. WAIT 与 ANSWER 标注

- **提问时条件已经满足**：立即 `ANSWER`，不制造等待。
- **提问时未满足，之后满足**：先 `WAIT`，在首次确认满足条件时 `ANSWER`。
- **截至视频结束仍未出现**：只有完整检查了提问后的相关片段，才能标为“截至结束未触发”的负例；不编造肯定答案或触发时刻。此类负例不意味着目标在未来永远不会出现。
- **画面模糊、无法判定**：标为不确定并复核，不能直接当作“目标不存在”。

每道等待题必须记录：

1. 提问时刻 `query_time`。
2. 明确的视觉触发判据。
3. 已确认未满足条件的时刻及证据，包括首次触发前最后一个确认的负时刻。
4. 首次确认满足条件的时刻 `answer_time` 及证据；未触发时明确记录未触发状态，不虚构时间。
5. 核验范围、采样密度与时间精度。

`query_time` 和 `answer_time` 不需要位于同一分钟，可以跨越多分钟。每分钟的数量目标是整段视频的密度目标，不要求每分钟放一套四类题，也不要求在一分钟内完成问答。根据真实扫查过程覆盖较短与较长等待，避免所有问题都在事件前一两秒才提出；不得通过延后本来已经可以回答的时刻制造长等待。长间隔题必须检查等待区间内目标是否曾更早满足，不能只核对起点和终点。

为训练与评测“当前能否回答”，Proactive 题应保留经核验的早期等待、中间等待（适用时）、临近触发的负样本及首次可回答的正样本；不需要机械地每秒生成一行。只能观察截至当前时刻的视频，金标准 `answer_time`、未来证据和答案不得作为模型输入。未触发负例单独统计，不虚构正样本。

字幕提前提到目标，不算触发。只核验了每秒一帧，就只能报告一秒采样级的触发边界，不能声称找到了连续时间中的精确最早时刻。医学常识上能够猜出答案，不等于当前视频证据已经足够。

`WAIT` 与 `ANSWER` 是同一道 QA 在不同时刻的决策标签。决策训练行数与独立 QA 数必须分别统计。

### 5. 生成流程

1. **先通览视频**，识别主题、有效超声片段和扫查阶段。
2. **建立带时间的观察记录**，记录实际可见的器官、结构、特征及变化。
3. **根据观察生成问题**，不先套模板再勉强寻找证据。
4. **回看对应片段**，验证问题前提、答案和时刻。
5. **删除无关、重复和证据不足的题目**。

概览可以抽帧；器官出现、切面切换、组织运动等动态问题必须查看连续片段或更密集的帧。记录实际核验范围，不把导出了帧等同于已经检查了帧。

### 6. 不生成的题目

- Freeze 按钮颜色、菜单切换等纯界面题。
- 读取字幕、标签、箭头颜色即可回答的题。
- 不看视频也能回答的医学常识题。
- 仅凭讲解就断言画面存在某个器官、病变或操作结果的题。
- 从不充分的影像推断疾病、异常血流或操作成功的题。
- 同一个观察反复换措辞的重复题。

字幕和教学标注可以辅助核对，不能替代影像证据。筛选时可问：去掉超声影像及相关操作画面，只保留字幕和界面，这道题是否仍然能答？如果能，通常不应保留。

### 7. 问题与答案的语言

- 训练问题和答案均为英文，中文仅作审阅译文，两者含义保持一致。
- 每题聚焦明确的考察目标，答案简洁，只包含画面支持的判断。
- 区分画面方向与患者身体方向。
- 对识别题，避免在问题中泄露待识别结构的名称；对事件通知题，可以明确指定需要等待的目标器官或结构。
- 不确定的细粒度解剖名称、切面或病理判断，进入专业复核，不强行给出确定答案。

例如，“当可以辨认出心脏时通知我”是在指定检测目标，不构成答案泄露；“这颗心脏是什么器官”则泄露了识别题的答案。

### 8. 数量与质量验收

设原视频时长为 T 分钟（秒数 / 60），目标为约 4T 条独立 QA：当前理解、历史回忆、事件通知、时序比较各约 T 条。非整数分钟默认取每类目标 N = floor(T + 0.5)，合计 4N 条；不足 0.5 分钟的视频按有效证据单独确定数量。例：5 分钟的视频目标为 20 条，每类 5 条，直接回答与 Proactive 各 10 条。

这是争取达到的数量和比例目标，不是允许编造或重复的硬配额。证据不足时记录各类缺口及原因，优先补充有真实事件和变化的视频，不能用更多 observation 题掩盖 Proactive 不足。分别统计每视频及整体的四类数量、Immediate/Proactive 比例、含 Proactive 任务的视频数量、等待时长分布和未触发负例数；QA 比例 1:1 不代表 WAIT/ANSWER 决策行比例也为 1:1。

**4T 是期望目标，不是最低数量或强制配额。** 已充分审阅的视频，如果有效超声信息量不足以支持这么多有实质区别的问题，可以少于 4T，不需要强制补满。不得通过重复改写、拆碎同一事实、编造事件或加入无关问题凑数。四类各约 T 条、直接回答与 Proactive 约 1:1 仍是争取的分配目标，不能以牺牲证据质量来满足数量。

未达目标时，合理检查是否遗漏结构、回声特征、切面变化、历史对应关系或可核验的未触发条件；发现遗漏才补充，不要求为凑数反复审阅。分别报告目标数量、实际数量、各类数量和信息量不足的具体原因。已完成审阅和质量检查、且缺额有依据的视频，可以标记为“已完成，信息量受限”，不自动成为待回补任务；尚未充分审阅的才标记为待补充。结构通过、数量达标和审阅完成是不同状态。同一视觉事件可以支持实质不同的任务，但换一种说法不算新增内容。

训练与评测都应覆盖两组任务，并按视频划分，不能将同一视频的 QA 或同一问题的决策行拆到训练与评测两侧。分别评估能否回答的判断、提前回答、触发后的响应延迟与答案正确性；单独报告未触发负例的误触发情况。

每题进入训练候选集前，检查以下项目：

| 检查项 | 必须满足的要求 |
|---|---|
| 超声相关性 | 考察超声影像或相关操作理解 |
| 证据充分性 | 问题前提和答案都有可定位的画面依据 |
| 时间正确性 | 当前回答不依赖未来画面，触发时刻经过核验 |
| 任务必要性 | 历史题确实需要历史信息，时序题确实需要多个时刻 |
| 模型可见性 | 在训练实际的缩放、裁剪与抽帧后，关键证据仍可辨认 |
| 非重复性 | 与其他题有实质不同的考察目标 |
| 审核透明性 | 分别记录格式检查、视觉复核和专业审核状态 |

同一段证据可以支持不同能力的题，但必须有实质区别。同一视频的 QA 应归入同一数据划分；增加题数不能替代增加视频、受试者和场景的多样性。格式检查通过不代表视觉标注正确，也不代表已经通过医学专家审核。

## English Rules

### 1. Content Scope

| Priority content | What to assess |
|---|---|
| Organs and anatomical structures | Which organ is visible, where structures are located, and their spatial relationships |
| Ultrasound image features | Shape, boundaries, echogenicity, acoustic shadowing, and visible Doppler signals |
| Imaging views | Reliably identifiable views and how they change during scanning |
| Scanning sequence | The order in which structures appear, enter or leave the field of view, or reappear |
| Probe manipulation | Visible translation, rotation, or tilting and the corresponding image changes |
| Interventional procedures | Needle shaft, needle tip, and their relationship to target structures, only in relevant videos |

Ultrasound imagery alone can provide sufficient evidence; the device, probe, and operator do not need to be visible. Organs, structures, imaging views, their appearance, disappearance, reappearance, and morphological changes in ultrasound-only footage are valid QA subjects. Procedural footage is needed only for questions about probe handling or procedures; needle tips are optional targets.

A video does not need to cover every category. If a structure cannot be named reliably, describe its visible features or discard the question. Observing an action and an image change together does not by itself establish causation.

### 2. QA Types

Target a 1:1:1:1 ratio across the four QA types, approximately 25% each. Current understanding and history recall form the Immediate group; event notification and temporal comparison form the Proactive group. Target a 1:1 ratio between these groups. Examples indicate question directions only; generate them only when the video provides the necessary evidence.

| Type | Requirement | Example direction |
|---|---|---|
| Current understanding `current_observation` | Answer from currently observable information | Which organ can be identified in the current ultrasound image? |
| History recall `history_recall` | Require an earlier observation | Which organ was scanned before the bladder was shown? |
| Temporal comparison `temporal_comparison` | Wait for new evidence after the query, then compare it with an earlier observation | When the kidney reappears, compare its cross-sectional shape with the previous view. |
| Event waiting and notification `event_notification` | Wait until a specified visual condition is satisfied | Notify me when the heart can be identified in the ultrasound image. |

History questions must require earlier information; simply adding “earlier” is insufficient. Immediate tasks have sufficient evidence at query time, with `answer_time = query_time`. Tasks counted toward the Proactive quota must be unanswerable at query time; triggered tasks must satisfy `answer_time > query_time`. Temporal comparison tasks must wait for new visual evidence and compare it with an earlier observation. Comparisons of two already observed times that can be answered immediately belong to history recall, not the Proactive quota. Notification questions whose target is already visible at query time also do not count toward that quota.

### 3. Event Notification Rules

Prioritize these trigger targets:

- **An organ or structure appears**: the heart, kidney, bladder, or another identifiable structure becomes visible.
- **A specific imaging view appears**: the target view becomes reliably identifiable.
- **A target reappears**: scanning returns to an organ after leaving it.
- **An image state changes**: a target leaves the field of view, a view changes, or another clearly observable change occurs.
- **An intervention-related event occurs**: use targets such as needle tips only in interventional videos, not as the default target.

Distinguish these conditions explicitly:

| Condition | Meaning |
|---|---|
| The organ can be identified | There is enough image evidence to identify the organ; complete visualization is not required |
| The organ is fully visualized | A predefined completeness criterion is satisfied; this is stricter than identification and must be specified in advance |
| The target view appears | The view satisfies its visible criteria; a subtitle naming it is insufficient |

The word “Heart” on the screen does not establish that the heart is identifiable in the ultrasound image.

### 4. WAIT and ANSWER Annotation

- **The condition is already satisfied at query time**: label `ANSWER` immediately; do not create artificial waiting.
- **The condition is not yet satisfied but becomes satisfied later**: label `WAIT` first, then `ANSWER` when satisfaction is first confirmed.
- **The condition remains unmet through the end of the video**: label a negative example as “not triggered by the end” only after reviewing the full relevant interval following the query. Do not fabricate an affirmative answer or trigger time. This does not imply the target can never appear in the future.
- **The image is ambiguous or too unclear to judge**: mark it as uncertain and review it; do not equate uncertainty with target absence.

Every waiting task must record:

1. The query time, `query_time`.
2. An explicit visual trigger criterion.
3. Confirmed negative times and supporting evidence, including the last confirmed negative time before the first trigger.
4. The first confirmed positive time, `answer_time`, and supporting evidence. For an untriggered task, record that status explicitly without inventing a time.
5. The reviewed interval, sampling density, and temporal precision.

`query_time` and `answer_time` need not fall within the same minute and may span multiple minutes. The per-minute quantity target describes density over the whole video; it does not require one set of four types in every minute or completion within a minute. Include shorter and longer waits supported by the actual scanning sequence, rather than placing every query one or two seconds before its event. Do not manufacture a long wait by delaying an answer that was already supported. For long intervals, check whether the condition was met earlier within the interval; inspecting only the endpoints is insufficient.

To train and evaluate whether the model can answer now, retain verified early-wait, intermediate-wait (where applicable), near-trigger negative, and first-answerable positive samples for Proactive tasks. Generating a row at every second is unnecessary. The model may observe only the video up to the current time; gold `answer_time`, future evidence, and reference answers must not enter its inputs. Report untriggered negatives separately and do not fabricate positive samples.

A subtitle mentioning the target in advance is not a trigger. Reviewing one frame per second supports only a boundary at that sampling resolution, not an exact earliest time in the continuous video. An answer that can be guessed from medical knowledge is not necessarily supported by the currently available video evidence.

`WAIT` and `ANSWER` are decision labels for the same QA at different times. Report the number of decision training rows separately from the number of distinct QA tasks.

### 5. Generation Workflow

1. **Review the video overview** to identify its topic, useful ultrasound segments, and scanning stages.
2. **Create timestamped observation notes** describing organs, structures, features, and changes actually visible.
3. **Generate questions from those observations**, rather than forcing evidence to fit a predefined template.
4. **Revisit the relevant segments** to verify question premises, answers, and timestamps.
5. **Remove irrelevant, redundant, and insufficiently supported questions**.

Sampled frames may support an overview. Dynamic questions about organ appearance, view transitions, or tissue motion require continuous segments or denser frame inspection. Record what was actually reviewed; exporting frames is not the same as inspecting them.

### 6. Excluded Questions

Do not generate:

- Pure interface questions about Freeze-button color or menu changes.
- Questions answerable solely by reading subtitles, labels, or arrow colors.
- General medical knowledge questions that do not require the video.
- Claims that an organ, lesion, or procedural result is visible based only on narration.
- Diagnoses, abnormal-flow judgments, or procedural-success claims inferred from insufficient images.
- Paraphrases that repeatedly test the same observation without a substantive difference.

Subtitles and teaching overlays may assist verification, but cannot replace image evidence. A useful screening question is: if the ultrasound images and relevant procedure footage were removed, leaving only subtitles and the interface, could the question still be answered? If so, it should usually be excluded.

### 7. Question and Answer Language

- Training questions and answers must be in English. Chinese translations are for review only and must preserve the same meaning.
- Each question must have a clear assessment target. Answers must be concise and limited to visually supported judgments.
- Distinguish image coordinates from patient anatomical directions.
- In identification questions, do not reveal the structure to be identified. Event notification questions may explicitly name the organ or structure to wait for.
- Refer uncertain fine-grained anatomical labels, imaging views, or pathological judgments for expert review rather than forcing a definite answer.

For example, “Notify me when the heart can be identified” specifies a detection target and does not leak the answer. “Which organ is this heart?” leaks the answer to an identification question.

### 8. Quantity and Quality Acceptance

For an original video lasting T minutes (duration in seconds / 60), target approximately 4T distinct QA tasks: approximately T each for current understanding, history recall, event notification, and temporal comparison. For noninteger durations, the default per-type target is N = floor(T + 0.5), totaling 4N tasks. For videos shorter than 0.5 minutes, determine the quantity individually from valid evidence. For example, a 5-minute video targets 20 tasks: 5 per type, with 10 Immediate and 10 Proactive tasks.

These are quantity and balance targets, not hard quotas that permit fabrication or duplication. If evidence is insufficient, record each type's shortfall and the reason. Prefer adding videos with real events and changes; do not hide a Proactive shortfall by adding more observation questions. Report per-video and overall counts for all four types, the Immediate/Proactive ratio, the number of videos containing Proactive tasks, waiting-time distributions, and untriggered-negative counts. A 1:1 QA ratio does not imply a 1:1 WAIT/ANSWER decision-row ratio.

**4T is a preferred target, not a minimum or mandatory quota.** After adequate review, a video may yield fewer than 4T tasks if its useful ultrasound content cannot support that many substantively distinct questions. There is no requirement to fill the shortfall. Do not pad the dataset with paraphrases, fragmented versions of one fact, invented events, or irrelevant questions. Approximately T tasks per type and a 1:1 Immediate/Proactive ratio remain distribution targets, subordinate to evidence quality.

For a shortfall, reasonably check for missed structures, echo features, view transitions, historical relationships, or verifiable untriggered conditions. Add tasks when omissions are found; repeated review solely to fill the quota is not required. Report target and actual counts, per-type counts, and the specific content limitation. A video with completed review and quality checks and a justified shortfall may be marked "complete, content-limited" and is not automatically a supplementation backlog item. Reserve pending-supplementation status for insufficiently reviewed videos. Structural validity, quantity attainment, and review completion are separate states. A visual event may support substantively different tasks, but paraphrasing a task does not add new content.

Both training and evaluation should cover both groups, with splits made by video. Do not split QA from the same video, or decision rows from the same task, across training and evaluation. Evaluate answerability decisions, premature responses, response delay after the trigger, and answer correctness separately; report false triggers on untriggered negatives separately.

Check every question before including it in the candidate training set:

| Check | Required condition |
|---|---|
| Ultrasound relevance | Tests understanding of ultrasound images or related procedures |
| Sufficient evidence | Both the question premise and answer have locatable visual support |
| Temporal correctness | The answer at a given time does not use future frames, and trigger times are verified |
| Task necessity | History tasks require earlier information and temporal tasks require multiple times |
| Model visibility | Key evidence remains identifiable after the actual training resize, crop, and frame sampling |
| Nonredundancy | The assessment target differs substantively from those of other questions |
| Review transparency | Format validation, visual review, and expert review are recorded separately |

The same evidence may support different abilities only when the assessment targets differ substantively. Keep all QA from the same video in the same dataset split. More questions do not replace diversity in videos, subjects, or scenarios. Passing format validation establishes neither visual annotation correctness nor medical expert approval.
