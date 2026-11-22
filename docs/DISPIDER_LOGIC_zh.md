# Dispider 完整逻辑：论文设计、当前实现与历史训练代码

本文依据本地官方仓库的实际代码整理，服务于超声流式视频理解项目的设计比较。

- 官方仓库：https://github.com/Mark12Ding/Dispider
- 当前核查版本：`803f6b4634b3232293e89a38bcce9f5815c33ed4`，提交说明为 `Reorganize Dispider as an inference-only runtime`。
- 历史训练代码核查版本：`8b03eda`。
- 论文：https://arxiv.org/html/2501.03218v1
- 核查方式：静态阅读代码和 Git 历史；未下载模型权重、未进行 GPU 推理。现有 Python 环境缺少 pytest，相关测试未执行。

## 1. 总体结构

Dispider 将持续感知、回答时机判断和答案生成分工：

| 组件 | 主要模型 | 职责 |
| --- | --- | --- |
| Perception | CLIP 视觉塔 | 提取帧级 patch 特征 |
| Perception–Decision | Qwen2-1.5B | 压缩片段、汇总历史、判断回答时机、计算检索相关性 |
| Reaction | Qwen2-7B | 读取检索到的详细证据与全局记忆，生成答案 |

压缩与决策复用同一个小模型，并非另有一个独立 Q-Former。

```text
视频帧 → 片段采样 → 视觉编码 → 片段 memory 与 time 状态
                                      ↓
                           历史上下文＋问题＋TODO
                                      ↓
                                二分类决策头
                              ↙             ↘
                           等待              触发
                            ↓                 ↓
                       继续下一片段       历史片段检索
                                              ↓
                              详细视觉特征＋压缩特征＋全局记忆
                                              ↓
                                      Reaction 生成答案
```

这是职责层面的流程。当前非缓存入口可能重算历史；不能把图理解为每个张量只计算一次。

## 2. 片段如何定义，何时做一次判断

### 2.1 当前流式入口

[StreamingSession](https://github.com/Mark12Ding/Dispider/blob/803f6b4634b3232293e89a38bcce9f5815c33ed4/dispider/streaming/session.py) 默认设置：

- `window_seconds=16.0`；
- `frames_per_window=16`；
- 视频按不重叠的固定窗口划分。

```text
C1 = [0,16) 秒，采 16 帧
C2 = [16,32) 秒，采 16 帧
C3 = [32,48) 秒，采 16 帧
```

完整窗口内的理想采样点为窗口起点加 `0.5, 1.5, …, 15.5` 秒，选择最近的实际输入帧。若输入稀疏，同一帧可能被多个采样位置使用；不能保证始终是 16 张不同画面。视频结束时 `flush()` 处理残余窗口。

通常在下一帧跨过窗口边界时结束前一个窗口并执行判断。因此当前默认判断粒度约为每 16 秒一次，不是每秒一次。事件记录的证据时间戳也不等于计算完成、用户收到答案的墙钟时间。

### 2.2 与论文的区别

论文描述使用 SigLIP 特征相似度检测场景变化、构建不等长片段。当前 `StreamingSession` 则使用固定窗口；不能说当前入口已实现论文的场景切分。

## 3. 图像尺寸与视觉编码

[适配器](https://github.com/Mark12Ding/Dispider/blob/803f6b4634b3232293e89a38bcce9f5815c33ed4/dispider/streaming/dispider_adapter.py) 的 `_to_square_pil_frames()` 对非正方形帧先插值为 `min(H,W) × min(H,W)`。这是直接改变宽高比的缩放，不是等比例缩放加 padding，也不是中心裁剪。随后调用 checkpoint 的图像处理器。

论文报告 CLIP-L/14、224×224 输入。以下形状按这一配置计算；运行其他 checkpoint 时仍应核对其处理器配置。

1. 每帧产生 `16×16=256` 个空间 patch 特征。
2. [perception.py](https://github.com/Mark12Ding/Dispider/blob/803f6b4634b3232293e89a38bcce9f5815c33ed4/dispider/model/perception.py) 从 `hidden_states[select_layer]` 读取配置指定层；在 `patch` 模式下移除 CLS。
3. 视觉塔设置 `requires_grad_(False)`，前向使用 `torch.no_grad()`。
4. 不同画面产生不同特征；“冻结”指权重不更新，不表示输出向量恒定。

当前代码的冻结设置不能单独证明原始训练第一阶段中视觉塔的全部更新策略。

## 4. 从 patches 到片段记忆

### 4.1 相邻 2×2 patch 拼接

[decision_inputs.py](https://github.com/Mark12Ding/Dispider/blob/803f6b4634b3232293e89a38bcce9f5815c33ed4/dispider/model/decision_inputs.py) 的 `_mix_spatial_tokens()`：

```text
每帧 [256,C] → 相邻 2×2 空间拼接 → [64,4C]
16 帧 → [1024,4C]
```

这里是通道拼接，不是平均池化。将四个 patch 的信息装进一个更宽的 token。

### 4.2 添加池化位置，再投影

[PoolProjector](https://github.com/Mark12Ding/Dispider/blob/803f6b4634b3232293e89a38bcce9f5815c33ed4/dispider/model/projectors.py) 保留详细特征，并追加两类池化特征：

| 部分 | 生成方式 | token 数 |
| --- | --- | --- |
| 详细视觉特征 | 2×2 拼接后的 tokens | `T×R` |
| 帧内池化特征 | 每帧按空间区域平均 | `T×P` |
| 片段全局池化特征 | 整个片段所有详细 tokens 平均 | `1` |

`T` 是帧数，`R` 是合并后每帧 token 数，`P=pool_num` 是配置项。

三部分按照以下顺序拼接，再经过 `Linear → GELU → Linear`，映射到小模型隐藏维度 `D`：

```text
[详细视觉 tokens][帧内池化 tokens][片段全局池化 token]
```

若 `T=16, R=64, P=1`，视觉部分有 `1024+16+1=1041` 个位置，另有时间描述等文本位置。`P=1` 是条件示例，不能在未读取 checkpoint 配置时把它当作所有版本的固定值。

### 4.3 经小模型后，按位置取 hidden states

输入组装器用角色标记指出需要读出的向量：

- 帧内池化位置：`MEMORY_TOKEN=100`；
- 全局池化位置：`TIME_TOKEN=200`。

这些数值是 `indicators` 中的位置角色，不是词表 token ID。

经过 Qwen2-1.5B 后，[extract_perception_memory](https://github.com/Mark12Ding/Dispider/blob/803f6b4634b3232293e89a38bcce9f5815c33ed4/dispider/model/decision_backbone/memory.py) 执行：

```python
clip = hidden[indicators == MEMORY_TOKEN]
time = hidden[indicators == TIME_TOKEN]
```

因此每个片段得到：

- `T×P` 个片段 memory 向量；
- 一个 time/片段标识向量。

池化向量是这些读出位置的初始输入。经过 Transformer 后，输出融合了可访问的前序视觉上下文，不再只是原始平均值。

**time 向量不是一个纯时间戳，也不是只有秒数信息。** 它是全局池化位置的上下文化状态。片段开始/结束时间也以文字进入输入，因此它可以包含时间与内容信息。

## 5. 历史汇总与存储

[summarize_memory / summarize_range](https://github.com/Mark12Ding/Dispider/blob/803f6b4634b3232293e89a38bcce9f5815c33ed4/dispider/model/decision_backbone/memory.py) 将多个片段的 memory 状态与 time 状态拼接，再过小模型，取末尾 time 位置的输出作为汇总状态。

```text
[多个片段的 memory][这些片段的 time 状态]
                         ↓ 小模型
                 每个片段对应一个汇总状态
```

提问和历史触发位置决定哪些区间被汇总，最近未汇总部分可保持更细的片段 memory。

这不是把全部历史递归覆盖为固定 60 个向量。当前适配器的 `_clips` 会持续积累片段，直到 reset。虽然调度器有长度受限的 `_history` 队列，它并不限制模型适配器的完整历史。

还要区分概念上的特征库与实际存储：适配器保留片段预处理图像及时间输入；未启用缓存的路径可能重新编码历史。不能声称所有历史特征都只提取一次。

## 6. 决策输入、分类头与触发

简化的决策序列为：

```text
[历史汇总][问题][新片段 memory][TODO]
```

实际代码包含 ANS 分隔、问题重复等模板细节，见 [streaming.py](https://github.com/Mark12Ding/Dispider/blob/803f6b4634b3232293e89a38bcce9f5815c33ed4/dispider/model/decision_backbone/streaming.py) 的 `_conversation_query()` 和 `infer_trigger_sequence()`；上述仅用于表达语义。

[decision.py](https://github.com/Mark12Ding/Dispider/blob/803f6b4634b3232293e89a38bcce9f5815c33ed4/dispider/model/decision.py) 定义：

```python
self.silent_head = nn.Linear(config.hidden_size, 1, bias=False)
```

取 TODO 查询位置的最终状态 `h`，计算：

```text
z = wᵀh
z > 0 → 触发回答
z ≤ 0 → 等待
```

等价地，sigmoid 后与 0.5 比较。`silent_head` 是历史命名，代码中正分数表示触发，不表示沉默。

分类头只给出一个标量，不负责生成答案。ANS 记录触发动作，不是整段回答文本。

## 7. 检索如何执行

### 7.1 查询与候选

查询来自结合问题和已观察历史的小模型查询状态；流式路径对应 TODO 读出状态。候选 key 是各片段的 time 向量。不是直接用孤立问题 embedding 匹配图片。

[similarity()](https://github.com/Mark12Ding/Dispider/blob/803f6b4634b3232293e89a38bcce9f5815c33ed4/dispider/model/decision_backbone/memory.py) 进行 L2 归一化、点积，并使用温度 `0.1`：

```text
s_i = cosine(query, time_i)
p_i = softmax(s_i / 0.1)
```

### 7.2 选择与返回

- 流式路径：`k=min(4, 已观察片段数)`。
- 离线路径：依据原始相似度大于 `0.2` 的数量动态选择，目标数量限定在 4～28，再受实际片段数限制。
- top-k 选出后按片段索引重新排序，恢复时间顺序。
- 返回选中片段的详细视觉特征和压缩 memory 特征。

检索的是片段，不是单帧，也不是只返回 4 个向量。流式默认选 4 个完整的 16 秒采样片段，不能把它理解为取回这些时段原视频的所有帧。

当前 `react()` 调用 Reaction 的多模态生成；其 `encode_sequences()` 再调用小模型选择片段。实现不等于直接复用先前决策调用中缓存好的检索结果。

## 8. Reaction 如何生成答案

[reaction.py](https://github.com/Mark12Ding/Dispider/blob/803f6b4634b3232293e89a38bcce9f5815c33ed4/dispider/model/reaction.py) 使用不同投影器映射三类特征：

1. 选中片段的详细视觉特征 → `clip_projector`；
2. 选中片段的压缩 memory → `compress_projector`；
3. 全局 memory → `compress_projector`。

拼接这些特征，放入 Reaction 的多模态提示位置，再由 Qwen2-7B 自回归生成。

```text
问题文本＋检索片段的详细视觉证据＋压缩证据＋全局历史
                            ↓
                       Qwen2-7B
                            ↓
                          回答
```

`clips_large` 虽出现在接口中，但当前 `PerceptionDecision.forward()` 用 `_clips_large` 接收并未使用；不能据此宣称还有独立的高清视觉分支。

论文描述 Reaction 使用历史回答，并训练 SILENT 输出。当前适配器 `_prompt_token_ids()` 构造的是图像占位符与当前问题，没有直接拼接历史回答文本；历史触发位置仍传入决策/记忆路径。当前 `react()` 解码并 strip 输出，调度器过滤空字符串，不能据此认定实现了完整的 SILENT 回退协议。

## 9. 当前运行时是否异步

**当前默认 StreamingSession 是串行的。**

`_finish_window()` 依次调用：

```python
perception = perceive(window)
triggered = should_respond(perception)
if triggered:
    answer = respond(perception)
```

适配器还会在存在未回答触发时拒绝新的 perceive 调用。因此：

- 论文：感知/决策与 Reaction 异步解耦；
- 当前代码：模块职责分离，但默认调度阻塞等待回答。

锁不等于并行调度。若要真正异步，需要另行设计任务队列、快照所有权、GPU 资源安排和回答提交协议。

## 10. Decision KV cache

当前入口支持 `auto/on/off` 策略。缓存分支只编码新片段，并维护决策上下文；它要求片段逐个到达、问题和触发历史保持一致。

`DecisionCache` 与 `DecisionKVBackend` 提供缓存执行及完整计算 oracle 核对，接近阈值时可以回退到 oracle 分数。相关文件：

- [decision_cache.py](https://github.com/Mark12Ding/Dispider/blob/803f6b4634b3232293e89a38bcce9f5815c33ed4/dispider/streaming/decision_cache.py)
- [kv_backend.py](https://github.com/Mark12Ding/Dispider/blob/803f6b4634b3232293e89a38bcce9f5815c33ed4/dispider/streaming/kv_backend.py)
- [dispider_adapter.py](https://github.com/Mark12Ding/Dispider/blob/803f6b4634b3232293e89a38bcce9f5815c33ed4/dispider/streaming/dispider_adapter.py)

缓存优化的是 Decision 路径，不等于整个系统不再重算视觉特征，也不等于回答已经异步。

## 11. 训练监督：论文和历史实现

### 11.1 论文中的两个阶段

论文 §4.1 描述：先训练流式处理与决策，再冻结视觉编码器和小模型，训练 Reaction。第一阶段使用 GroundVQA 对应数据、ET-Instruct 和补充的带时间标签 QA；第二阶段构建流式 QA 训练回答。

论文没有充分说明全部样本的人工审核来源、时间区间到 clip 标签的转换规则，也不能仅从当前推理代码推断原始训练的全部参数冻结与 LoRA 设置。

### 11.2 检索目标分布

论文将相关片段集合记为 R，给相关片段均分概率：

```text
相关片段：第 2、4 个
目标 y：[0, 0.5, 0, 0.5, 0]
预测 p：softmax(查询与各片段的相似度)
```

用 KL(y || p) 让预测靠近标签。目标分布通常由时间标注自动转换；不需要人手填写概率。但时间标注是否正确，仍取决于上游标注来源和审核。

### 11.3 Git 历史里的真实损失

当前 HEAD 移除了训练目标，但 `8b03eda` 的以下文件仍有训练逻辑：

`dispider/model/language_model/qwen_model/modeling_qwen.py`

可只读查看：

```bash
git show 8b03eda:dispider/model/language_model/qwen_model/modeling_qwen.py
```

核查到的流式 grounding 分支：

- `silent_label==0` 时跳过检索损失；正标签对应触发。
- 截断 `time_labels` 和候选到当前已观察前缀。
- 将截断后的标签除以其总和，得到目标分布 y。
- 使用 `softmax(similarity/0.1)` 得到 p。
- 时间损失包含以下两项，epsilon 为 `1e-4`：

```text
L_time = -Σ y_i log(p_i + epsilon)
         -0.01 Σ p_i log(p_i + epsilon)
```

第一项是目标分布交叉熵；固定 y 时，与 KL 的梯度等价（忽略数值 epsilon 的近似）。第二项是正的熵惩罚；最小化它倾向于更集中的预测，不能误说成鼓励均匀分布。随后按正触发数量归一化。

BCE 对正触发样本按负/正样本数量比加权，再归一化权重：

```text
L = weighted_BCE(trigger_logit, trigger_label) + L_time
```

历史分支直接使用上游传入的 `time_labels`；没有在此验证它一定为人工标注的二值 mask。若是二值标签，归一化才得到相关片段均分概率。代码也假设正触发前缀有非零证据标签。

目前没有据此核实出完整可复现的训练数据处理与 trainer 管线。发现损失代码，不等于完整训练已开源。

## 12. 和我们的超声系统对照

| 项目 | Dispider 当前实现 | 我们的秒级 Q-Former |
| --- | --- | --- |
| 流式单位 | 默认 16 秒、16 帧片段 | 每秒一帧 |
| 每帧详细视觉 tokens | 论文 CLIP 配置下，256 patches 拼接成 64 tokens | Qwen 256×256、patch 16、merge 2 后为 64 tokens |
| 压缩方式 | 池化读出位置经过小语言模型 | 512 维 learned query，通过 cross-attention 读取视觉特征 |
| 压缩输出 | 每片段 T×P 个 memory 状态及一个 time 状态 | 每秒一个 512 维内部状态，上投影为 2048 维 Short |
| 历史形式 | 片段历史与分段汇总，随历史增长 | 每分钟更新固定 60 个 Long，另保留当前分钟 Short |
| 决策 | TODO 状态＋线性二分类头 | DECISION 状态＋二分类头 |
| 检索 | 查询与片段 time 向量匹配 | 当前尚未增加历史片段检索 |
| 回答证据 | 检索细节＋压缩特征＋全局记忆 | 当前以 Short/Long 向量为主 |

两者每帧合并后 token 数相同，不表示视觉表示质量、信息量或算力相同。

我们已有决策头，借鉴重点是历史证据保留、检索监督和回答输入。若新增检索，应明确额外存储预算，并保持证据只来自当前已观察前缀；不能继续把整个系统描述为只有固定 60 个历史向量。

## 13. 阅读顺序

1. [stream_inference.py](https://github.com/Mark12Ding/Dispider/blob/803f6b4634b3232293e89a38bcce9f5815c33ed4/stream_inference.py)：流式入口。
2. [session.py](https://github.com/Mark12Ding/Dispider/blob/803f6b4634b3232293e89a38bcce9f5815c33ed4/dispider/streaming/session.py)：窗口采样和串行调度。
3. [dispider_adapter.py](https://github.com/Mark12Ding/Dispider/blob/803f6b4634b3232293e89a38bcce9f5815c33ed4/dispider/streaming/dispider_adapter.py)：预处理、决策、缓存与回答桥接。
4. [perception.py](https://github.com/Mark12Ding/Dispider/blob/803f6b4634b3232293e89a38bcce9f5815c33ed4/dispider/model/perception.py)：视觉特征提取。
5. [decision_inputs.py](https://github.com/Mark12Ding/Dispider/blob/803f6b4634b3232293e89a38bcce9f5815c33ed4/dispider/model/decision_inputs.py)、[projectors.py](https://github.com/Mark12Ding/Dispider/blob/803f6b4634b3232293e89a38bcce9f5815c33ed4/dispider/model/projectors.py)：空间拼接、池化位置和投影。
6. [memory.py](https://github.com/Mark12Ding/Dispider/blob/803f6b4634b3232293e89a38bcce9f5815c33ed4/dispider/model/decision_backbone/memory.py)：隐状态提取、汇总与检索基础函数。
7. [streaming.py](https://github.com/Mark12Ding/Dispider/blob/803f6b4634b3232293e89a38bcce9f5815c33ed4/dispider/model/decision_backbone/streaming.py)：决策序列和流式/离线检索差异。
8. [reaction.py](https://github.com/Mark12Ding/Dispider/blob/803f6b4634b3232293e89a38bcce9f5815c33ed4/dispider/model/reaction.py)：回答特征组装及生成。
9. Git 历史 `8b03eda`：查训练损失，避免把当前推理包装器误作 trainer。
