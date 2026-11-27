# 基于视频片段检索的流式问答

`event_retrieval/` 是这条新架构的源码入口。切分和建库只读取视频，提问时再按问题检索截至当前可用的片段。每个片段用视频 ID 和片段 ID 联合定位，可回放对应的原视频区间。

## 已实现

- 冻结的 Qwen3.5-2B 视觉编码器每秒提取一帧特征。[切分函数](segment.py) 按余弦距离检测边界。本次试验阈值为 0.02，连续两次采样超过或等于阈值时确认。
- Demo 的 10 个 holdout 视频共切出 295 段。[时间索引](../results/holdout10_visual_events/index.json) 和 [视觉向量索引](../results/holdout10_visual_events/event_vector_index.npz) 是这一版的实验产物。Demo 从「视频片段切分」栏目回放。
- 已实现逐秒 RGB 帧接入、增量切分、可恢复的 SQLite 索引、自然语言问题 Top-5 检索和长期视觉 prefix。回放路径见下文。

## 训练前共享视频索引

训练入口在 epoch 1 前准备 train 和 holdout 视频的完整 SQLite 索引。索引放在 scratch 的 `cluster_data/experiments/event_retrieval/shared_video_indexes/`，每次训练的 `index/<video_id>.sqlite` 链接到对应文件。文件键包含视频身份、源路径、大小和修改时间、视觉模型路径、预处理版本和事件切分设置。不同实验可共享相同键；同一视频并发构建受文件锁保护，构建中断后从已保存的帧继续。QA 训练打开只读索引，按当前时间选帧和过滤已发布事件，使用当次训练的权重检索、计算前缀并生成回答。10 个 QA v3 视频按 ID 命名的入口位于 `shared_video_indexes/qa_v3_holdout10/`。

## 从新帧到长期 prefix

[在线入口](replay.py) 每秒接收一张 RGB 帧和它在源视频中的 PTS。文件回放选用 PTS 不晚于该整秒的最新帧。冻结的 Qwen3.5 将每个 256×256 帧编码成 64 个空间 token。每帧只编码一次；平均向量供边界检测，空间 token 写入每个视频独立的 [SQLite 索引](index.py)。索引固定视频身份、特征来源、预处理和切分配置。每次事务一起写入帧、检测器状态和刚确认的 event，可关闭后从下一秒继续。部署时把数据库放在 scratch；2B 的 BF16 空间特征每秒约 256 KiB，另有索引与 SQLite 开销。

确认新边界时发布刚结束的片段。例如第 30、31 秒连续偏离时，旧片段的结束时间为 30 秒，首次可检索时间为 31 秒。区间左闭右开。正在形成的事件暂留在流中；只有显式收到 EOF 后，才以实际结束时间发布最后一段。查较早时间会重新按可用时间过滤，即使索引中已经有后续事件。

[问题检索](retrieve.py) 用每段中与归一化中心最接近的一帧作代表。用冻结的 Qwen3.5 读取问题、事件区间、代表帧时间和该帧的空间 token，以下一 token 的 Yes 与 No 的 logit 差打相关性分。分数仅用于排序；打平时按 event 编号排序。此版本是逐候选的多模态打分，第一次提问的开销随已发布事件数增长。打分前剔除被最近 24 个采样帧完整覆盖的片段。若片段跨过短期窗口左边界，则保留整段用于检索与回放。候选不足 5 段时全部返回。单张代表帧可能遗漏段内短暂变化，检索召回率需要单独评估。

每个问题维护一个 `QuestionSession`。每秒按当前窗口筛选历史片段，只给首次符合条件的候选打分；排序后的 Top-5 不变就复用长期 prefix。对命中事件，在完整区间内均匀选择最多 24 帧，从索引取回这些帧的原始 Qwen 空间特征。每个槽位携带 video_id、event_id、区间、确认时间、代表帧时间、采样秒和真实源 PTS。Python 调用 `session.at(t)` 返回 `prefix`、`mask` 和 `sources`，供之后的短期分支、LLM 和证据回放使用。该缓存适用于权重固定的推理会话。

命令行可对同一问题在多个时点执行完整前向，逐行输出 Top-5 来源、长期 prefix 形状及有效 token 数。示例在仓库根目录的 ultrasound 环境运行，先将现有模型路径设为 `QWEN_MODEL`、`INSTRUCTBLIP_CHECKPOINT`，并将 `EVENT_DB` 指向 scratch 上该视频的 SQLite 文件：

```bash
PYTHONPATH="results/streamready_e2e_setup/deps:.:StreamReady" python -m event_retrieval.replay \
  --model "$QWEN_MODEL" --qformer "$INSTRUCTBLIP_CHECKPOINT" \
  --video cluster_data/videos/train_full295/7G56DN38mz8.mp4 --index "$EVENT_DB" \
  --question 'What separates portions of the irregular bright bony contours?' --at 0 5 25 26
```

## Q-Former 的初始化和调用

[双分支视觉编码模块](qformer.py) 严格加载 InstructBLIP-Vicuna-7B 的两份独立 Q-Former 编码器权重。短期分支直接复制预训练的 32 个 query；长期的前 32 个逐元素复制，后 96 个用这 32 个 query 重复三组并加独立高斯噪声。初始噪声标准差取原始 query 全体元素标准差的 1%，由 `noise_seed` 控制。调用方用 `torch.manual_seed` 控制新投影层的初始化。共享输入投影将 Qwen 的 2048 维空间 token 映射到 1408 维，两个输出投影分别将 768 维 query 映射到 LLM 宽度。可训练的帧位置与历史 event 槽位嵌入从零初始化。

在项目的 ultrasound 环境中，将仓库根目录、`StreamReady` 和 `results/streamready_e2e_setup/deps` 加入 `PYTHONPATH`。调用示例：

```python
from event_retrieval.qformer import EventQFormers

model = EventQFormers.from_instructblip(
    "/dss/mcmlscratch/04/ge75vid2/torch_cache/hub/checkpoints/instruct_blip_vicuna7b_trimmed.pth",
    noise_seed=0,
)
# recent [B,T,P,2048]；events [B,E,T,P,2048]。
# recent_mask [B,T]；event_mask [B,E,T]。True 表示有观测，时间轴右侧填充。
prefix, prefix_mask = model(recent, events, recent_mask=recent_mask, event_mask=event_mask)
```

`event_frame_indices(n)` 返回按时间排列的均匀采样索引，同时用于选取空间特征和原视频时间戳。模型消费已选好的近期帧和历史 event；每个有效帧传入完整的空间 patch。输出为长期在前、短期在后的 `[B,160,llm_width]` 和布尔 mask。无历史可用时传入 `[B,0,0,P,2048]`，输出的前 128 个位置全部屏蔽。LLM 拼接问题时须一并传递这个 mask。在推理模式下可分别调用 `encode_short`、`encode_long` 并按下面的更新条件缓存结果。

## 双分支流式问答与 Qwen 生成

每个问题维护一次从提问到回答的会话。LLM 的输入按 `长期视觉 prefix + 短期视觉 prefix + Question` 排列。两段 prefix 是可学习 query 读取画面后的输出。LLM 自己生成首个动作和后续文本，输出形式为 `<WAIT> 等待原因` 或 `<ANSWER> 回答`。

| 分支 | 读取范围 | 视觉 prefix |
|---|---|---|
| 短期 | 截至当前的最近 24 个 1 FPS 采样帧，包含当前帧；视频开始时使用已有帧。每帧保留 64 个空间 token。 | 32 个 token |
| 长期 | 按问题在同一视频已可检索的历史 event 中取 Top-5，依据源视频时间戳读取帧级空间 token。每段最多取 24 帧，超过 24 秒时在整段可用采样帧上均匀选取，保留时间顺序；不足 24 帧时全部保留。最多共 120 帧。 | 5 段合计 128 个 token |

### v2：最近四帧直接输入 Qwen

训练入口 `python -m event_retrieval.train` 支持 `--raw-recent-frames 4`。时刻 T 的视觉前缀依次为历史检索结果（Long Q-Former，最多 128 个 token）、T−23 至 T−4 的观测（Short Q-Former，32 个 token）、T−3 至 T 的原始 Qwen 视觉编码器输出（逐帧按时间拼接，每帧 64 个空间 token）。在 256×256 输入下，完整视觉前缀共 416 个 token，后面接事件区间、提问时间、当前时间和问题文本。

v2 的纯 NTP overfit 使用 `--raw-recent-frames 4 --loss-objective ntp`。在每个 batch 的全部有效目标 token 上取全词表交叉熵均值，动作、已有标注的正文及其 EOS 各计一次。epoch loss 按实际监督 token 数汇总。

开头只观测到 1–4 帧时，所有已观测帧直接进入 Qwen，第 5 帧开始才有短期摘要。此选项复用同一份逐帧索引。训练配置将布局存入 checkpoint，续训、单视频回放和多视频并行评测会自动恢复。省略训练参数时沿用原来的 24 帧压缩模式。

### 逐秒生成与训练监督

[问答解码器](answer.py) 在 Qwen 自带的聊天模板中插入长期、短期视觉 embedding，然后输入检索到的 event ID 与时间区间、近期帧区间、提问时间、当前时间和问题。屏蔽的视觉槽直接从序列移除。给 Qwen 词表注册独立特殊 token `<WAIT>` 和 `<ANSWER>`。首个生成位置只在这两个 ID 之间比较模型 logits。后续正文由语言模型生成。

在上面的回放命令中将 `--at 0 5 25 26` 换成 `--ask-at 25 --until 35 --max-new-tokens 64`，即可从第 25 秒起每秒生成一次，遇到 ANSWER 就停止。省略 `--until` 时持续至视频结束。每行 JSON 返回 `at`、`action`、`reason` 或 `answer`、`finish_reason`、近期帧区间及检索候选事件。`finish_reason=length` 表示用完生成预算，正文可能截断。检索来源可回放，是否支持答案仍需核对。

训练时用当前参数执行两支 Q-Former 的前向，直接将输出传给同一解码器：

```python
from event_retrieval.answer import AnswerDecoder, DecisionContext

decoder = AnswerDecoder(backbone)
visual, mask = model(recent, events, recent_mask=recent_mask, event_mask=event_mask)
# 每个样本提供提问时间、决策时间、已采样的短期区间以及按检索顺序排列的历史来源。
contexts = [DecisionContext(question, ask_s, now_s, first_s, last_s, tuple(sources))]
loss = decoder.alignment_loss(visual, mask, contexts, ["<ANSWER> " + answer])
loss.backward()
```

损失是 assistant 回复上的逐 token 平均交叉熵，问题、视觉 prefix 和 padding 的标签均屏蔽。带理由的 `<WAIT> reason` 和 `<ANSWER> answer` 监督动作、正文和 EOS；只有 `<WAIT>` 标签的样本仅监督动作。联合训练更新两支 Q-Former、视觉投影以及 Qwen 的完整语言模型，包括输入 embedding 和输出头。视觉编码器冻结。推理通过 `--checkpoint RUN/latest.pt` 加载全部已训练权重；历史上仅包含 Q-Former 的旧权重继续使用 `--aligned-qformers` 和原分词方式。只提供 InstructBLIP 初始化时，输入和输出投影仍是新初始化的。

### 联合训练入口

在 ultrasound 环境设置上文的 `PYTHONPATH`，`DATA` 指向含 `samples.jsonl`、`tasks.jsonl`、`video_map.json` 及对应 `holdout/` 的 Stage3 分割目录。可直接使用已有全量数据清单 `cluster_data/experiments/streamready/full_train171_holdout16_bs96_block8_2gpu2ep_20261125T1403Z`。`RUN` 应为 scratch 上 `cluster_data/experiments/event_retrieval/` 下新的运行目录。

```bash
python -m event_retrieval.train --model "$QWEN_MODEL" --qformer "$INSTRUCTBLIP_CHECKPOINT" \
  --data "$DATA" --run "$RUN" --epochs 1 --lr 1e-4 --llm-lr 1e-5
```

单设备训练，`--batch-size N` 控制每步一起前向和反传的决策样本数，默认 1。`--bf16` 对两支 Q-Former 和语言模型的训练前向启用 BF16 autocast；可训练参数与 AdamW 状态保留 FP32。不同长度的视觉输入按本批次补齐并传入 mask。每个决策点先对自己的监督 token 取平均，再对样本取平均；末批按实际样本数计算。程序按视频和样本洗牌，按需从源视频建立因果索引，使用当前 Qwen 给历史候选打分，并用当前 Q-Former 权重计算前缀。直接可回答题和延迟回答题使用各决策点的 GT。现有全量清单中 WAIT 的 GT 只有动作。每轮在独立 holdout 上报告样本平均 loss，同时保存两个 Q-Former、Qwen 语言参数、动作 ID、优化器与随机状态到 `latest.pt`。词表写入 `tokenizer/`。用原命令加 `--resume --epochs N` 从已完成的一轮继续至第 N 轮。清单哈希和初始化路径随检查点校验。

### 哪些计算可以跨问题共享

这个版本把“与问题无关”限定在视频侧。每秒新增一帧时，编码、event 切分、索引维护以及当秒的短期视觉表示都只依赖已观察到的画面。设计上两个 Q-Former 只用可学习 query 对视觉 token 做 cross-attention。问题文本用于检索历史 event，并作为文本输入交给 LLM。当前版本通过多模态相关性打分来连接问题与视觉证据。

按每帧 64 个空间 token 计算，短期最多读取 1,536 个视觉 token；长期最多读取 7,680 个，保留每段的归属及帧的时间顺序。event 中心向量用于检索，命中的 event 按时间戳取回帧级空间特征用于长期压缩。两支输出经各自的投影对齐到 LLM 的输入维度，依次形成最多 128 + 32 = 160 个视觉 prefix token，再接问题文本。推理时，同一秒的短期表示可以供多个问题共用；长期表示只有在选中的 event、采样帧和排列一致时才可共享。

训练和回放可用 `--retrieval-batch-size 16` 让 Qwen 一次给至多 16 个候选 event 打分，默认值为 4。训练的 `--batch-size` 单独控制每次参数更新使用的决策样本数。overfit 评测从训练配置读取检索批量。

### 同一问题的更新时机

1. **问题到达**。只检索已发布且至少有一帧早于当前短期窗口的 event，计算首份长期 prefix，同时计算短期 prefix，调用 LLM。历史为空时屏蔽长期输入，仍可仅凭近期画面作答。
2. **生成 WAIT**。向用户返回等待原因，问题继续挂起。每过一秒，用新窗口更新短期 prefix，再调用 LLM。每秒检查窗口滑动后是否有旧片段首次成为长期候选。
3. **等待期间有新 event 可用**。片段发布或短期窗口滑动均可能使长期候选增加。按原问题重新检索历史。若选中的 event、排列和采样帧均相同，直接复用长期 prefix；发生变化时重新计算，再结合当秒的短期 prefix 做决策。正在形成的片段只使用当时已观察到的帧。
4. **生成 ANSWER**。向用户返回回答，结束这个问题的等待循环。新的问题从自己的首次检索开始。缓存复用的是长期分支的输出表示，可学习 query 的参数在推理期间保持固定。

### 证据来源与备选分配

每个检索结果携带 `video_id`、`event_id` 和所取帧的时间戳，供回放和后续引用评测使用。长期 128 个 query 的一种**备选、待验证**分配是每个 event 绑定 24 个局部 query，另设 8 个跨 event 的 query，即 `5 × 24 + 8 = 128`。局部 query 只读取对应 event，跨段 query 可用于比较历史片段，不足 5 段时屏蔽空槽。该分配留作后续对照实验。

## 实验归属

此架构的配置、训练入口和评测归入 event_retrieval；新作业的训练产物写入 scratch 的 cluster_data/experiments/event_retrieval/ 下的独立运行目录。StreamReady 复现使用原有的 StreamReady/、endtoend/ 和 streamready 实验目录。两条路线可以使用相同的原始视频和数据划分进行对照，各自记录初始化和结果。

### 十视频逐秒回放

用 `python -m event_retrieval.parallel_replay --run <fit目录> --output <新的scratch目录>` 同时回放最多 10 个视频。每个视频按自己的时间前进，每秒补入一帧；已索引的帧直接从对应视频的 SQLite 读取。历史索引只读，超过已索引范围时在输出目录创建副本并继续写入。问题在标注的 query_time 激活，WAIT 后在下一个整秒再问，直到首次 ANSWER 或视频结尾（包含不足一秒的尾段）。同一视频、同一时刻的问题共享短期表示；每个待答问题独立缓存检索分数和长期表示。

`--videos 10 --generation-batch-size 10 --retrieval-batch-size 64` 分别设置并行视频数、单次生成的问题数和每个问题的一批候选数。视频在不同的本地时间到达推理点也可以拼批。每次决策立即写入 `stream_predictions.jsonl`，`summary.json` 在所有问题回放结束后生成。逐秒回放不会使用标注的 answer_time 停止推理。固定标注点的强制答案对照评测仍由 `event_retrieval.eval_overfit` 提供。


整条样本加权版（作业 5959690）训练时，每道延迟题在首次可答时刻的整条样本（动作、答案正文及 EOS）权重等于该题实际纳入训练的 WAIT 决策点数 K。即时 ANSWER 和所有 WAIT 的权重为 1。先对每条样本的有效目标 token 取平均 CE，再按样本权重加权。归一化使用整个训练集的平均权重，避免一个 batch 中的其他问题或单样本尾批改变 K 的相对系数。训练配置记录 `answer_weighting=wait_count`；续训按原 checkpoint 的配置处理。

新建 `--overfit-qa` 默认使用 `action_body`：按上述归一化权重只加权首个决策 token 的 CE，正文及 EOS 按时点独立取平均；裸 WAIT 的正文项为零。最新作业为 5961170。参见[模型、QA v3 与三版 overfit 汇总](../docs/EVENT_RETRIEVAL_QA_V3_OVERFIT_zh.md)。


## 评测协议

[完整评测流程与 LLM Judge 评分协议](EVALUATION_PROTOCOL_zh.md)：WAIT／ANSWER 决策与 Timing、ANSWER 0–3 分语义及视觉支持度、WAIT 原因三维评分、分组和统计分母、GPU 报告，以及当前实现与待补项。

## 实时训练进度（后续新运行）

`event_retrieval.train` 默认在 stdout／Slurm 日志输出持久化进度行；不用等待整轮结束。示例仅说明格式，不是某个作业的实际结果：

```text
[train] epoch 1/2 | optimizer steps 120/1550 (7.7%) | epoch steps 120/775 | 15.00s/update | ETA updates 05:57:30 | ETA job unknown
```

- 每次成功执行 `optimizer.step()` 后推进一步。GA microbatch 不单独算 step，多卡同步更新只由 rank 0 记录一次；最后不足一个有效 batch 的更新也计入。总更新数为 `epochs * ceil(train_endpoints / (batch_size * world_size * GA))`。
- `--log-every-steps 10` 和 `--log-every-seconds 30` 控制 stdout／JSONL 刷新；达到任一条件后在下一次更新／holdout batch 完成时输出。轮末和阶段切换立即输出。这不是中断长时间单个 kernel 或检索操作的后台心跳。
- `<run>/progress.json` 在每次 optimizer 更新后原子替换；holdout 每批刷新。它记录当前／总 step、epoch、处理端点数、吞吐、ETA、更新时间和 phase，方便 dashboard／脚本读取。`<run>/progress.jsonl` 保留节流后的时间序列。
- 阶段包括 initializing、indexing、distributed_setup、train、holdout、checkpoint、epoch_complete、completed、failed。进度对象在模型与优化器初始化／恢复后启用；此前仍由原始初始化日志说明状态。
- `ETA updates` 根据最近最多 50 个完成更新的墙钟耗时估计，包含对应的检索、数据准备和训练计算；不把 holdout、保存时间混入每步速度。开始时没有观测就显示 unknown。
- `ETA job` 在验证／保存开销均有观测后，加入剩余这些阶段的预计耗时；首轮可能显示 unknown。ETA 不包括独立索引作业、排队和训练之后的生成评测。它是估计，视频长度／候选数变化会改变速度。
- `--resume` 根据恢复的完整 epoch 初始化 global step，本次会话的速度重新测量，既不把以前的更新重复计数，也不把离线间隔算入吞吐。
- 更新进度达到 100% 时可能还在 holdout／保存，必须结合 phase；所有训练阶段成功结束才标 completed。异常退出记录最后已完成 step 和错误；进程被强杀时以最后更新时间及 Slurm 状态判断，不能把旧快照当实时心跳。

读取示例：

```bash
tail -n 5 /path/to/scratch/run/fit/progress.jsonl
cat /path/to/scratch/run/fit/progress.json
```

此功能只影响包含新代码的后续运行。已提交作业使用各自的冻结代码，不会因修改工作区自动获得逐步日志；无需为此重启现有训练。
