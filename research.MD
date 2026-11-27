# 流式超声 QA 实验记录

## 2026-11-26 · v3，28 + 2 帧，BS4 / BS8

**研究问题**　在相同 v3 架构、数据和 20 轮训练预算下，增大实体 batch 后，首次回答时机与答案语义能否同时拟合？同轮比较时，BS8 的优化器更新次数约为 BS4 的一半。

| 项目 | 本轮共同配置 |
|---|---|
| 数据 | QA v3，10 个 holdout 来源视频用于同集训练与回放；97 题（81 即时、15 延迟、1 道审核区间内无答案）。246 个监督时点：96 ANSWER、150 WAIT，含 16 条已标注等待理由。 |
| 视觉输入 | 静音 RGB，1 fps，256×256。历史 Top-5 事件各取最多 24 帧 → Long Q-Former 输出 128 token；T−29 至 T−2 → Short Q-Former 输出 32 token；T−1、T 的 128 个空间 token 直通 Qwen。完整视觉前缀 288 token。 |
| 初始化与更新 | 两组均从 Qwen3.5-2B 和 InstructBLIP-Vicuna-7B 独立初始化。冻结视觉编码器；更新两个 Q-Former 的视觉 query 路径、投影、位置嵌入和完整语言模型。两份已提交作业各自使用运行目录中的视频索引。 |
| 目标与优化 | 纯 NTP，在 batch 内全部有效目标 token 上平均全词表 CE；裸 WAIT 监督首动作，有正文时监督动作、正文和 EOS。AdamW，Q-Former 等模块 LR 1e-4，LLM LR 1e-5，BF16，GA1，seed 0，检索 BS64。 |
| 回放与验收 | 第 20 轮回放 10 视频，逐秒推进至首次 ANSWER。96 道可答题逐题核对首次触发秒及语义正确性；无答案题单列审核覆盖范围。非精确匹配接受同义改写，复核关键视觉关系。 |

| 作业 | GPU | 训练 BS | 步/轮 × 轮数 | 已完成 | 最近一轮 NTP | 配置 |
|---|---|---:|---:|---:|---:|---|
| 5963020 | 1× A100 80 GiB | 4 | 62 × 20 | 20/20 | 0.0507 | [plan.json](cluster_data/experiments/event_retrieval/v3_overfit_28plus2_ntp_DUn4h0bI/plan.json) |
| 5963195 | 1× A100 80 GiB | 8 | 31 × 20 | 20/20 | 0.0240 | [plan.json](cluster_data/experiments/event_retrieval/v3_overfit_28plus2_ntp_bs8_ig77ze5i/plan.json) |

**参照**　v2（20 + 4，纯 NTP，BS4）延迟题 14/15 准时；96 道可答题中 68 道文字精确匹配，另 13 道经复核语义正确。[v2 逐题复核](results/event_retrieval_qa97_loss/judge_inputs/v2_28_content_review_zh.md)

**本轮状态**　两组训练均结束。另已保留 BS8 第 14 轮 checkpoint，供检查后期退化。本轮完整验收及下一轮候选见下。

**后续流程**　每轮在本文件追加验收结果、代表性失败题、原因假设、拟修改项与验证标准；向用户汇报，取得该轮明确批准后再提交重训，并补记作业号。

### 第 20 轮回放验收

- **BS8**　81/81 即时题、12/15 延迟题准时，余下 3 道延迟题等到片尾；无答案题也等到片尾。96 道可答题中，87 道准时且精确匹配。对照问题、参考答案与首次回答前的画面复核 6 道非精确生成，3 道保留核心描述，1 道部分正确（v9ar1mhpT-g/Q005 深度标记终点正确，暗区误称声影），2 道错误（同视频 Q001 把液性暗区描述成垂直声影；0Cboqf1Qnhc/Q007 混淆针体和血管）。按此口径，**90/96 准时且语义正确**。[逐题记录](results/event_retrieval_overfit_research/5963195.jsonl)
- **漏触发**　CW3TLabogko/Q004（98 s）、Q006（204 s）、YllLJNPnpjY/Q011（493 s）在标注答案秒均继续 WAIT。这些题涉及画面重新显现或切面转换后的跨时比较。
- **Loss 比较**　两组每轮均看 246 个监督时点，epoch loss 按全轮有效 token 汇总。BS4 / BS8 在第 14 轮为 0.0976 / 0.0094，第 20 轮为 0.0507 / 0.0240。按视频分组后打乱样本，每批按有效 token 平均并执行一步 AdamW。按实际 seed、视频分组和 tokenizer 重建 20 轮：纯裸 WAIT 批次 BS4 为 187/1240（15.1%），BS8 为 18/620（2.9%）。将每步中 WAIT 和 ANSWER 首动作各自的 token 数除以该步有效 token 数，再分别累加，两者系数之比从 6.22 降至 3.03。此值衡量 loss 的系数；实际梯度还依赖预测与参数，同一 seed 无法分离批次组成及更新次数的因果贡献。

- **BS16 离线采样对照**　相同 20 轮样本顺序下，BS16 的 320 个批次均含带正文的样本。逐批 WAIT 首动作 token 占比的总体标准差（BS4 / BS8 / BS16）为 33.8 / 17.1 / 6.3 个百分点，对应 WAIT/ANSWER 首动作系数比 6.22 / 3.03 / 2.10。BS16 只模拟了组批，尚无梯度、显存、训练或回放实测。

- **BS4**　81 道即时题准时 80 道，15 道延迟题全部等到片尾，另 1 道无答案题保持 WAIT。准时且精确 77/96。3 道非精确生成对照作答画面复核：针体长亮线 Q007 核心正确；lREUPXCpK8Y/Q002 给出方位却未说清两条静脉相对动脉的关系，YllLJNPnpjY/Q006 说对左移但添加无依据的壁接触，均计部分正确。**准时且语义正确 78/96**。[逐题记录](results/event_retrieval_overfit_research/5963020.jsonl)
- **判断与提案（待批准）**　两组均未达到同集 96/96 准时且语义正确。BS8 比 BS4 多通过 12 题，仍漏掉的 3 题在 v2（20+4）分别于 98、203、493 秒回答，其中第二题比标注早 1 秒。若获准，建议先固定 v3 的视觉结构、数据、初始化、LR、20 轮及纯 NTP，只把有效 BS 从 8 调至 16，先测显存，必要时用微批累计有效 token 的 CE 和再除以整批 token 数。检验漏答是否减少，同时记录提前回答、准时且语义正确数和无答案题的动作。若目标是直接检验梯度波动，可在相同权重上抽批比较梯度离散度。未提交新训练。
- **BS8 漏答的证据覆盖**　按首次应答标注秒对照 replay 的实际 Top-5 采样秒与近期帧。CW3TLabogko/Q004 的旧证据 94 s、YllLJNPnpjY/Q011 的旧证据 486 s 均进入 Short Q-Former，对应的新证据 97–98 s、492–493 s 均经最近两帧直通。CW3TLabogko/Q006 的旧证据 165 s 命中 E06 的实际采样帧，203–204 s 两帧直通。三题均覆盖全部标注窗口，可排除漏取标注帧这一解释；Q-Former 表征质量与 LLM 动作偏好仍未分离。
- **BS8 强制作答（评测 5963495）**　三道漏答题的应答秒与前一秒均复现 WAIT，检索事件与原回放一致。只在两个合法动作间归一化，P(ANSWER) 从前一秒到应答秒：CW3TLabogko/Q004 2.03% → 20.18%，Q006 0.52% → 0.76%，YllLJNPnpjY/Q011 0.46% → 6.75%。强制 ANSWER 后，后两题在两秒均与参考全文一致；Q004 在两秒均说对回声重新出现的位置，却把一小片组织回声说成垂直亮回声。三例中两例主要卡在动作选择，另一例兼有形态描述错误。同集强制生成不能单独证明视觉依据。[6 条原始分数和答案](results/event_retrieval_overfit_research/5963495_forced.jsonl)
- **BS4 → BS8 逐题配对**　在同一组 96 道可答题上，以准时且语义正确为通过：两组均通过 75 题，仅 BS8 通过 15 题（12 延迟、3 即时），仅 BS4 通过 3 题，两组均漏答 3 题。BS8 退化的 3 题均为即时题且均准时触发：0Cboqf1Qnhc/Q007 混淆针体与血管，v9ar1mhpT-g/Q001、Q005 将暗区称为声影。下一轮单列这 3 道内容退化题，与 3 道漏触发题分别验收。
- **BS8 统一动作阈值的限制**　若 P(ANSWER | 两个合法动作) ≥ τ 就回答，要让 Q006 在 204 s 回答需 τ ≤ 0.758%，要让 Q004 在 97 s 仍等待需 τ > 2.033%。两条件无交集。因此只改统一阈值无法让三道漏答题同时准时；6 个采样时点不足以估计其余题的误触发率。[原始分数](results/event_retrieval_overfit_research/5963495_forced.jsonl)
- **下一轮配置**　单卡微批 BS8、GA2、有效 BS16。累积有效 token 的 CE，按整次更新的实际 token 数归一化。246 时点每轮 16 次更新，20 轮共 320 次；同组 97 题逐秒回放，以 BS8/GA1 为基线。
- Slurm 5963544 已提交，提交后状态 PENDING (Priority)。[配置](/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding/cluster_data/experiments/event_retrieval/v3_overfit_bs8_ga2_kmrfbeyn/plan.json)

### BS8/GA2 第 20 轮验收（5963544）

320 次更新，最终 NTP 0.05831，Slurm 完成，训练含回放用时 1:10:50。792 次逐秒决策覆盖 97 题。即时 81/81 准时，延迟 13/15 准时，其余 2 题等到片尾；零早答、零晚答，1 道无答案题等到片尾。准时且精确 87/96。对照答前画面和参考答案，7 条非精确生成中 5 条保留核心事实，2 条空间定位不完整。严格口径准时且语义正确 92/96（GA1 为 90/96）。旧 3 道内容错误均恢复，YllLJNPnpjY/Q011 于 493 s 恢复；新增部分正确为 az6h_u_8nhA/Q010 和 lREUPXCpK8Y/Q006。CW3TLabogko/Q004、Q006 仍漏触发。下轮候选：在相同 BS8/GA2 下对延迟题 GT 秒和紧邻 WAIT 秒加同题动作排序项，保留正文 NTP；先诊断该 checkpoint 的首动作分数，重训前另行审批。[97 题及复核](results/event_retrieval_overfit_research/5963544.jsonl)

**GA2 漏答诊断**　对 15 道延迟题各检查 GT 前 1 秒与 GT 秒。30/30 个动作与原回放一致，15/15 题动作差值 z_ANSWER−z_WAIT 上升，13 题跨过零。漏答 Q004 的 P(ANSWER | 两动作) 26.9% → 40.7%，Q006 为 11.9% → 16.5%。两题两秒的强制作答均精确匹配；全部延迟题在 GT 前一秒已能强制精确生成 9/15。两漏答题之间不存在同时保住前一秒 WAIT 并找回 GT 的统一阈值。[动作与生成](results/event_retrieval_overfit_research/5964828_forced.jsonl)

**证据干预**　GA2 应答秒 14/15 延迟题的全部标注窗口有采样帧。仅 Q006 漏掉 165 s；固定 204 s 的近期帧、问题、历史前四个槽，以已可用且覆盖 165 s 的 E06 替换第五槽 E01 后，P(ANSWER) 从 16.5% 升至 18.2%，仍 WAIT。单例结果说明仅恢复这段历史不足以触发。[对照](results/event_retrieval_overfit_research/5964829_evidence.jsonl)

**下一轮提案，待批准**　保持 GA2、20 轮、数据与初始化一致。在纯 NTP 上加一项延迟题相邻决策的二分类损失。对每道延迟题取 T−1 的 WAIT 与 T 的 ANSWER，令 m=z_ANSWER−z_WAIT，配对损失为 (softplus(m前)+softplus(−m后))/2。这个目标分别锚定零点两侧，覆盖原先仅约束相对次序的提案。候选系数 0.1，按题对平均，不把长等待题重复放大；训练前核对 15 题的邻秒训练目标。验收 97 题全回放，对照 92/96、零提前和无答案题至片尾 WAIT。

## 2026-11-27 · v3 BS8 × GA2 有效 BS16 对照

用户批准提交。与 BS8/GA1 同组 10 视频、97 QA、246 时点，28+2 视觉输入，1 fps/256²，静音，Top-5；全新 Qwen3.5-2B + InstructBLIP 初始化并训练两个 Q-Former 和完整语言模型。物理 BS8，GA2，BF16，纯 NTP 合计有效 token 归一化，seed 0，LR 1e-4 / 1e-5，检索 BS64。20 轮，每轮 30 个完整微批和 1 个 6 样本微批，共 16 次更新，合计 320 步。1 张 A100 80 GiB 或 H100 94 GiB，4 小时上限，逐秒 GPU 监控，结尾 10 视频并行回放验收。scratch 冻结配置、源码、监控和日志位于 /dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding/cluster_data/experiments/event_retrieval/v3_overfit_bs8_ga2_ha1x25z9（plan.json、code/、logs/）；检查点在 fit/。输入校验、sbatch shell 语法通过；训练单测 6 项通过，含混合目标长度及尾批的累积梯度更新等价性。提交前状态：待提交。

已提交 Slurm **5966364**。提交后查询为 PENDING（Priority），待分配节点。日志 logs/slurm-5966364.log，GPU 监控 logs/gpu-5966364.csv。

更正：5963544 已完成同一 BS8 × GA2、20 轮、97 题的 overfit 及回放。重复作业 5966364 已在排队时撤销，Slurm 确认 CANCELLED、耗时 00:00:00。本对照采用 5963544 的结果。
