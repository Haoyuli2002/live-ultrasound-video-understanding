# StreamReady 训练与评测数据集

核对日期：2026-11-19。论文版本：arXiv 2603.08620v1。

StreamReady 使用 **4 个来源、共 62,000 条视频 QA 训练**，在 **9 个 benchmark 上评测**。下表区分论文中的用途、论文使用数量，以及目前可访问的 Hugging Face 数据入口。HF 镜像或转换版不等于作者实际抽取的训练子集。

论文：[StreamReady: Learning What to Answer and When in Long Streaming Videos](https://arxiv.org/abs/2603.08620)。代码：[官方 GitHub](https://github.com/sacrcv/StreamReady)，本地副本为 [StreamReady](../StreamReady/)，本次核对版本为 `a5af28b44e3ac92095980dface1be0d58d442eab`。

## 训练数据

下列数量为 StreamReady 论文附录 D 表 11 的使用量，不是原数据集总量。[论文训练数据表](https://arxiv.org/html/2603.08620v1#A4)

| 数据集 | StreamReady 使用 QA 数 | Hugging Face 入口 | 版本与获取说明 |
|---|---:|---|---|
| MSRVTT-QA | 10,000 | [MMInstruction/M3IT 中的 msrvtt-qa](https://huggingface.co/datasets/MMInstruction/M3IT/tree/main/data/vqa/msrvtt-qa) | M3IT 转换子集；每个视频随机选一个 QA，训练仅 6,513 行，不能直接当作论文的 10,000 条训练集。完整原始 QA 应从下方作者仓库获取。 |
| MSVD-QA | 10,000 | [morpheushoc/msvd-qa](https://huggingface.co/datasets/morpheushoc/msvd-qa) | 第三方镜像；包含 QA 字段及单独的 `MSVD-QA.zip`，须核对原始划分。 |
| ActivityNet-QA | 32,000 | [NoahMartinezXiang/ActivityNet-QA](https://huggingface.co/datasets/NoahMartinezXiang/ActivityNet-QA) | 第三方镜像；顶层可见 `test_q.json`、`test_a.json`，不能据此认定训练标注齐全。训练标注从下方作者仓库获取。 |
| MovieChat-1k | 10,000 | [Enxin/MovieChat-1K_train](https://huggingface.co/datasets/Enxin/MovieChat-1K_train) | StreamReady 官方 README 指向的训练数据入口；当前页面要求登录并同意访问条件。 |
| **合计** | **62,000** | — | 作者选用的训练量。 |

原始标注来源：

- **MSRVTT-QA / MSVD-QA**：[xudejing/video-question-answering](https://github.com/xudejing/video-question-answering)，也是 StreamReady README 指定的来源。HF 上的 [morpheushoc/msrvtt](https://huggingface.co/datasets/morpheushoc/msrvtt/blob/main/README.md) 是 caption/帧数据，其字段没有 QA，不能替代 MSRVTT-QA。
- **ActivityNet-QA**：[MILVLG/activitynet-qa](https://github.com/MILVLG/activitynet-qa)。原始划分为训练 32,000、验证 18,000、测试 8,000 条 QA；复现应使用训练划分。
- **MSVD-QA 原始格式的额外 HF 镜像**：[csuhan/OneLLM_Eval 的 MSVD-QA 目录](https://huggingface.co/datasets/csuhan/OneLLM_Eval/tree/main/video/MSVD/MSVD-QA)，含 `train_qa.json`、`val_qa.json`、`test_qa.json`；这是其他项目保存的镜像，不是 StreamReady 发布的子集。

这些 QA 用于训练推理模块与 readiness 机制，视觉编码器和语言解码器冻结。readiness 使用伪证据时间监督，不要求人工提供训练样本的可回答时间。论文中的评测集应与训练数据保持分离。[论文第 3.4 节](https://arxiv.org/html/2603.08620v1#S3.SS4)

## 普通 QA 如何生成 readiness 训练监督

原始样本只有视频、问题和参考答案。StreamReady 在训练前向过程中生成伪正负时间点，用来学习回答时机；它没有先将 62,000 条 QA 导出为带人工 `answer_time` 或逐秒 WAIT/ANSWER 标签的新数据集。论文描述的是弱监督方法，下述分位数、池化方式及损失计算细节以本地克隆版本的实现为准。[论文第 3.4 节](https://arxiv.org/html/2603.08620v1#S3.SS4)

### 处理步骤

1. **用参考答案训练问答推理。** 输入视频和问题，以参考答案计算生成交叉熵损失，视觉编码器和语言解码器冻结。论文要求 readiness loss 仅更新 `<RDY>` token 和 readiness head，并未明确要求先后执行两个训练阶段。本地代码提供可选的 `reasoning` / `readiness` 分阶段路径，也支持 `joint` 联合训练并隔离 readiness 梯度；我们当前使用后者，并以已有 WAIT/ANSWER 标注的 BCE 替代论文的伪时间损失。[训练实现](../StreamReady/lavis/models/blip2_models/streamready.py) · [当前端到端监督](../endtoend/model.py)
2. **逐时刻记录问题相关表示。** 对本次输入的采样视频序列按顺序更新记忆、检索和推理。时刻 `t` 的推理只读取当前已观察到的前缀；将长期分支输出的 query tokens 取均值，得到 `z(t)`，收集为 `z_l_seq`。训练不启用 readiness 阈值提前停止。[流式前向实现](../StreamReady/lavis/models/streamready/module.py)
3. **用序列结束后的记忆生成匹配分数。** 前向结束后，将每个 `z(t)` 经 `query_proj.w_cent` 投影，并与最终视觉记忆中的有效聚类中心计算余弦相似度。聚类中心先在 slot-token 维度取均值，每个时刻取最高相似度作为 `s(t)`。伪标签计算在 `no_grad` 下完成，输入推理表示也已 detach。[标签调用入口](../StreamReady/lavis/models/blip2_models/streamready.py)
4. **按每条输入序列的分位数选择正负时间点。** 默认将分数不低于 70% 分位数的时刻选为伪正区域，不高于 30% 分位数的选为伪负区域；中间区域不参加正负配对，但仍参加后面的时间平滑项。[伪证据区域实现](../StreamReady/lavis/models/streamready/readiness.py)
5. **训练相对 readiness 排序。** 从正负区域抽取时间点对，使正例时刻的 readiness logit 高于负例；同时约束相邻时刻的 sigmoid 分数变化，减少抖动。这里没有逐帧生成答案并判断答对与否，也没有直接使用参考答案计算伪时间标签。[损失实现](../StreamReady/lavis/models/streamready/readiness.py)

```text
视频 V + 问题 Q + 参考答案 A
            │
            ├─ A 监督问答生成，学习推理表示
            │
            └─ 视频按时间进入模型，记录 z(t)
                         │
                   与最终聚类记忆比较
                         │
                    匹配分数 s(t)
                         │
              按分位数选择正、负时间点
                         │
                排序损失 + 时间平滑
                         │
                  学习 readiness
```

### 代码中的计算规则

令 `c_j` 为有效聚类中心的池化表示，`W` 为维度投影，`r(t)` 为 readiness head 输出的原始 logit：

```text
s(t) = max_j cosine(W z(t), c_j)

P = {t | s(t) >= quantile(s, 0.7)}
N = {t | s(t) <= quantile(s, 0.3)}

L_pair   = mean[-log sigmoid(r(t_positive) - r(t_negative))]
L_smooth = mean[abs(sigmoid(r(t+1)) - sigmoid(r(t)))]
L_ready  = L_pair + 0.1 * L_smooth
```

70%/30% 是用于选择伪标签的**相似度分位数**，不是推理时触发回答的阈值。实现中配对项使用 logit 差，平滑项使用 sigmoid 后的分数；中间区域没有被赋予确定的可回答类别。相同分数可能导致分位数边界并列，甚至正负区域重叠，因此不能保证正负例各严格占 30%。[实现细节](../StreamReady/lavis/models/streamready/readiness.py)

### 推理方式与监督的含义

推理时不再计算上述伪标签，而是由当前前缀产生 `<RDY>` 隐状态，经 head 和 sigmoid 得到分数。默认阈值为 **0.35**；首次达到阈值时保留该时刻的回答表示并生成答案。当前代码若始终未触发，会回退到序列末尾回答。[默认配置](../StreamReady/lavis/projects/streamready/train.yaml) · [触发逻辑](../StreamReady/lavis/models/streamready/module.py)

这种方法把“推理表示与记忆匹配程度高”当作证据充分的代理信号，**不是已经验证的可回答性真值**。它不保证得到单一、连续、单调的“之前 WAIT、之后 ANSWER”边界。训练标签可以利用整个输入序列结束后的记忆；推理决策仍只能使用当前已观察到的前缀。

与我们的 Stage 3 相比，现有数据显式提供 `query_time`、`answer_time` 和端点 WAIT/ANSWER 标签，可直接做可回答性 BCE 监督；StreamReady 的普通 QA 通过上述流程产生相对时间排序监督。要在我们的代码中采用该方法，需要实现并验证伪标签生成和对应损失，不能只把普通 QA 转成现有 JSONL 字段就视为完成。

## 评测数据

论文第 5 节和表 2–4 明确列出以下 9 个评测集。ProReady-QA 评估正确性和回答时机，其余 8 个按各自协议报告准确率；离线评测关闭 readiness 与 contextual reasoning。[论文实验设置](https://arxiv.org/html/2603.08620v1#S5)

| 类别 | 数据集 | Hugging Face 入口 | 版本或使用说明 |
|---|---|---|---|
| 回答时机 | **ProReady-QA** | [shehreen3764/ProReady-QA](https://huggingface.co/datasets/shehreen3764/ProReady-QA) | 作者发布；32 个视频，当前 HF 为 **4,507 条 QA**，论文写约 5,000。仅用于评测。 |
| 流式视频 | **StreamingBench** | [mjuicem/StreamingBench](https://huggingface.co/datasets/mjuicem/StreamingBench) | 项目发布入口；900 个视频、4,500 条 QA。 |
| 流式视频 | **OVO-Bench** | [JoeLeelyf/OVO-Bench](https://huggingface.co/datasets/JoeLeelyf/OVO-Bench) | 作者发布；包括回溯、实时感知、主动响应。数据卡说明当前版本相较初版有修改。 |
| 流式视频 | **OVBench** | [MCG-NJU/OVBench](https://huggingface.co/datasets/MCG-NJU/OVBench) | VideoChat-Online 项目的 benchmark，**与 OVO-Bench 不同**；一行内可包含多个问题。 |
| 流式视频 | **VStream-QA** | [IVGSZ/VStream-QA](https://huggingface.co/datasets/IVGSZ/VStream-QA) | Flash-VStream 项目发布；区分流式 RVS 与 VS-Ego/VS-Movie 等子集，按评测协议选择。 |
| 离线视频 | **Video-MME** | [lmms-eval/Video-MME](https://huggingface.co/datasets/lmms-eval/Video-MME) | LMMs-Eval 托管入口；test 为 2,700 条 QA。 |
| 离线视频 | **MLVU** | [MLVU/MVLU](https://huggingface.co/datasets/MLVU/MVLU) | 作者组织入口；仓库名实际拼作 **MVLU**。视频访问需同意条件，标注入口见数据卡。 |
| 离线视频 | **MVBench** | [OpenGVLab/MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench) | 作者组织入口；包含 20 个任务配置，不能把单个配置视为完整评测。 |
| 离线视频 | **EgoSchema** | [lmms-eval/egoschema](https://huggingface.co/datasets/lmms-eval/egoschema) | LMMs-Eval 转换入口；含多个配置。完整测试与公开子集不可混用，原始协议见 [EgoSchema 官方仓库](https://github.com/egoschema/EgoSchema)。 |

## 复现时需要区分的内容

1. **论文使用量与 HF 行数不同。** 视频、QA、对话和转换后的数据行并非同一个计数单位。训练表的 10k/10k/32k/10k 不能通过任意镜像的行数直接复现；仍需匹配原始划分和抽样规则。
2. **HF 的 `train` 名称不决定实验用途。** 例如 ProReady-QA 页面自动展示 `train`，但其数据卡明确规定 evaluation-only；OVBench 和 MVBench 页面也会出现 `train` 标签。在 StreamReady 实验中，它们属于评测集。
3. **ProReady-QA 的媒体需另行准备。** HF 提供标注与清单；按数据卡获取 Ego4D/MovieNet 来源视频并裁剪。它的证据起止窗口用于 ARS，不应当作训练标签混入训练集。[ProReady-QA 数据卡](https://huggingface.co/datasets/shehreen3764/ProReady-QA)
4. **论文评测范围大于当前 README 的直接运行入口。** 克隆版本 README 明确提供 ProReady-QA 与 OVO-Bench 的评测命令；不能仅凭论文列出了 9 个 benchmark，就认定 9 个均可通过同一命令直接运行。[官方代码说明](https://github.com/sacrcv/StreamReady#evaluation)
5. **原始数据仍需转换。** 官方代码要求训练标注整理为统一 QA schema，并准备 1 FPS 帧；下载上述数据不等于完成了 StreamReady 输入准备。[官方数据说明](https://github.com/sacrcv/StreamReady#datasets)

本文为数据来源索引；链接核对不代表已下载、校验媒体或复现训练。现阶段只保存本文和已有代码副本。
