# Stage 3 — 可回答性决策与回复

数据目录与跨机器使用说明见 [Stage3 releases](releases/README.md)。

> 项目级大纲（目标、三阶段状态、接口衔接、运行命令）见根目录
> [`README.md`](../README.md) / [`README_zh.md`](../README_zh.md)，那里是权威入口。
> 本文件只保留 Stage 3 的具体细节。

Stage 3 使用两个相互独立的 `Qwen/Qwen3.5-2B` 实例。二者都严格恢复 Stage 2 实际使用的权重链：仅当该 Stage 2 训练使用过 Stage 1 时，先 merge 对应 Stage 1 adapter，再恢复
`<SHORT_MEM>` / `<LONG_MEM>` 两行 embedding，再加载 Stage 2 adapter。

| 模型 | 状态 | 职责 |
|---|---|---|
| 冻结 `MemoryModel` | `eval()`、全部参数 `requires_grad=False`，调用时使用 `torch.no_grad()` | 以因果 1-FPS 递归产生 long/short memory |
| `DecisionModel` | 在独立的 Stage 2 权重副本上增加新 LoRA、`<DECISION>` / `<WAIT>` / `<ANSWER>` token 与 BCE head | 根据 memory 和问题判断 WAIT/ANSWER，并学习原因或答案文本 |

`DecisionModel` 继承 `torch.nn.Module`，因此一次 `decision.to("cuda")` 会同时
移动语言模型和 BCE head，避免 head 留在 CPU。冻结模型输出的 memory 在进入
决策模型前必须 `detach()`；`forward_answerability` 会强制检查这个不变量。

## Stage 2 → Stage 3 的正确加载顺序

```python
import torch
from stage3.model import DecisionModel, load_frozen_memory_model

memory_model = load_frozen_memory_model(
    "Qwen/Qwen3.5-2B",
    stage1_adapter="/path/to/stage1_output/final",
    stage2_adapter="/path/to/stage2_output/final",
)
decision_model = DecisionModel(
    "Qwen/Qwen3.5-2B",
    stage1_adapter="/path/to/stage1_output/final",
    stage2_adapter="/path/to/stage2_output/final",
).to("cuda")
```

恢复已经保存的 Stage 3 adapter 和 BCE head 时，额外传入
`decision_adapter="/path/to/stage3_output/final"`；构造器会同时加载
`decision_head.pt`，不会随机重新初始化分类头。

不能把 Qwen3-VL adapter 与 Qwen3.5 backbone 混用，也不能只把 Stage 2 adapter
直接加载到未经 Stage 1 merge 和 memory-token resize 的原始模型。

## 输入与损失

实际代码中的因果序列为：

```text
[固定 system/template] [Long memory] [Short memory]
[Question] [<DECISION>] [assistant template] [<WAIT>/<ANSWER> + 原因/答案]
                              │
                              └── hidden state → Linear → answerability logit
```

- long 与 short 是连续向量，调用方按此顺序拼成一个 `[1, M, hidden]` tensor。
- `<DECISION>` 是问题后的最后一个决策前缀 token。
- assistant template 和目标文本位于 `<DECISION>` 之后，因果注意力保证决策位
  看不到监督答案。
- `L_decision = BCEWithLogits(logit, y)`；`<WAIT>` 对应 `y=0`，`<ANSWER>`
  对应 `y=1`。代码会拒绝标签与 target marker 不一致的样本。
- `L_text` 只监督 `<WAIT>/<ANSWER>`、原因/答案及 EOS；system、memory、问题、
  `<DECISION>` 和 assistant template 均为 `-100`。
- 总损失由训练循环组合为
  `L = λ_decision·L_decision + λ_text·L_text`，第一版建议两个权重均为 1。

## 单卡训练入口

H100 / A100 80GB 的统一提交方式见 [GPU_TRAINING.md](../docs/GPU_TRAINING.md)。
`python -m stage3.train` 读取现有 QA JSONL，冻结 memory 模型并优化决策模型的 BCE + NTP。

## 当前真实进展

已经实现：

1. 冻结 Stage 2 memory 模型的正确加载和参数冻结。
2. 决策模型对完整 Stage 2 权重链的恢复与 merge。
3. device-safe 的已注册 BCE head。
4. 与上图一致的 system → memory → question → decision 输入顺序。
5. 单样本 BCE + NTP 前向、stop-gradient 检查及 WAIT/ANSWER 标签一致性检查。
6. Stage 3 adapter、processor、BCE head 和来源配置的保存与恢复。
7. QA JSONL 读取、决策时刻因果视频重放和单 GPU 双模型训练循环。

尚未实现或验证：

1. Stage 3 优化器状态断点续训。
2. Stage 3 DDP 多 GPU 训练。
3. 真实 Qwen 的阈值决策与自由生成 GPU 验证（代码已在 stage3.inference 实现，CPU mock 已覆盖）。
4. 多问题共享前缀 KV cache、分支隔离及 cache 失效逻辑。
5. Qwen3.5 双模型 GPU 端到端 smoke test。

本地 laptop 只执行语法、schema 和 CPU 单元测试。上述 GPU smoke test、真实
Stage 1/2 adapter 加载以及显存评估必须在保存模型权重的 SSH 训练环境完成。

当前已提供单 GPU 训练 CLI 和 CPU 因果性/梯度回归测试；真实 Qwen GPU smoke、
优化器状态恢复和在线部署验证仍未完成。

## 关于 memory 快照

`stage2.stream` 的 `.pt` 只保存每分钟 long-memory 快照和视频结束时的残余
short memory，不足以直接重建任意 `query_time` / `answer_time` 的输入。完整训练
应优先用冻结 memory 模型按时间顺序重放视频；如果要离线预计算，则必须额外保存
所有 QA 决策时刻的 long/short 快照。

## 关于多视角 summary

多视角 summary（固定 7 类字段标签：View／Anatomy／Probe／Finding／
Dynamic／Measure／Change）现已是 Stage 2 的**正式**教师数据逻辑，由
`stage2/annotate.py` 的默认提示生成并做首次格式校验，再由
`stage2/build_data.py` 二次校验并合并。
Stage 3 消费的仍是 Stage 2 产出的 memory 向量（非文本），这些结构化 summary
只在 Stage 2 训练阶段作为重建监督目标；此处仅说明其来源，不改变 Stage 3 接口。


## 两视频 overfit / 重建 / 评测

见 [OVERFIT.md](OVERFIT.md)。新增 stage3.overfit 入口支持显式 mock / qwen 后端、冻结 memory 缓存、立即回答和 null 未触发题、阈值决策后的自由生成、首次回答时序指标与零 memory 对照。当前准备的 P3+Cz 子集为12条英文QA；CPU mock 是接口验证，不代表真实 Stage3 已训练成功。之后接真实 Stage2 checkpoint 时须验证其初始化链。

## 512 维视觉证据检索（可选新路径）

旧的 `stage3.train` / 不带检索头的检查点仍使用记忆直接回答。
新入口是 `stage3.retrieval_training` 与 `stage3.retrieval_inference`，不依赖 Dispider。

```text
当前帧视觉 tokens ── 均值池化 ── Linear(D,512) + normalize ── key
       └─ 原有 Q-Former ── Short/Long + question ── DECISION
                                                   ├─ 可答性 BCE
                                                   └─ Linear(D,512) + normalize ── query
query/key 点积 / 0.1 ── 历史证据窗口 ── 原始采样帧视觉 tokens
                                     └─ evidence-only prefix
                                           ├─ 证据充分性 BCE
                                           └─ 带 [E1] 等引用的回答
```

Key 不经过 Q-Former，也不融合历史；query 通过 Decision 状态读取 Long/Short。
每个视频使用独立 `EvidenceBank`。索引保存冻结的当前帧均值、原视频路径、
帧索引、近似 PTS 和可用时刻；512 维 key 在检索时计算，训练更新投影后不会
读取过期 key。当前实现为精确扫描，不是 ANN，也不是固定总内存的系统。
索引占用随视频长度增长，但不持久缓存所有 patch tokens。

默认取最高分的 16 个候选秒点，最多返回 4 个窗口、8 个采样帧；先选分散的
窗口，再补充相邻命中，每个窗口最长 4 秒，并按时间排序。窗口是采样帧的回放
上下文，不代表逐帧审核过的完整支持区间。只允许 `available_at <= current_time`。
回答时重新编码所选原帧，沿用 Stage2 的帧尺寸和 `VideoReader` 预处理。
Q-Former 回放和建库共享一次视觉编码；检索后的重编码不更新 Short 状态。

首版回答前缀只读问题、可观察时间和证据视觉 tokens，不读 Long/Short、参考答案、
人工证据描述或 ASR。可答性不足返回 `next_action=observe`；视频已可答但检索
证据不足返回 `next_action=retrieve_again`。目前只返回重检索动作，不自动循环扩大
搜索。引用缺失或引用不存在的 ID 返回 `next_action=regenerate`，保留 draft 供调试。
引用 ID 合法性检查不等于语义支持验证，输出明确标记 `citation_support_verified=false`。

### 数据与训练

输入 bundle 使用单个训练视频的 `tasks.jsonl`、`evidence.jsonl`、
`decision_view.jsonl`、`retrieval_view.jsonl`、`answer_view.jsonl`，以及绑定原视频
SHA256 的 `media_manifest.json`。字段示例可见 `results/stage3_evidence_pilot_P3_v1/`。
训练输入由白名单字段构造；标注只用于损失。默认拒绝 `training_ready=false`，
明确做草稿接线实验时才可添加 `--allow-draft`。不确定决策行被屏蔽。

组损失为每个必需组的 `-log(sum p(positive alternatives))`，其 softmax 候选
只包含本组正例和显式负例；其他组的正例和未标注帧不视为负例，再对必需组求平均。
没有显式负例的行没有对比学习信号，检索单独训练时跳过；不能凭四个草稿 QA
宣称检索已学会。答案训练使用标注完整证据和人工构造的缺失证据；实际检索错误
分布的训练需要后续收集 retriever 输出。

```bash
# 阶段一：Stage2 和 Decision backbone 均冻结，仅学习 query/key 投影。
python -m stage3.retrieval_training \
  --dataset-dir /path/to/reviewed_bundle \
  --video-path-map /path/to/video_map.json \
  --stage2-adapter /path/to/stage2/training/final \
  --phase retrieval --epochs 5 --output-dir /path/to/retrieval_run

# 阶段二：Stage2 保持冻结，训练 Stage3 LoRA、两种分类头和检索投影。
python -m stage3.retrieval_training \
  --dataset-dir /path/to/reviewed_bundle \
  --video-path-map /path/to/video_map.json \
  --stage2-adapter /path/to/stage2/training/final \
  --init-decision-adapter /path/to/retrieval_run/final \
  --phase joint --epochs 5 --output-dir /path/to/evidence_run

python -m stage3.retrieval_inference \
  --video /path/to/video.mp4 --question 'Describe the earlier transition.' \
  --query-time 21 --times 21 27 33 35 40 \
  --decision-adapter /path/to/evidence_run/final \
  --output /path/to/predictions.jsonl
```

如 Stage2 依赖 Stage1，两次训练都需传入同一个 `--stage1-adapter`。
帧尺寸从 Stage2 `training_config.json` 读取，缺失时拒绝猜测。
检索与充分性头保存在 `retrieval_heads.pt`，配置写入 `stage3_config.json`。
只有检索训练、尚未训练 evidence 回答的检查点不能用于完整回答推理。
训练时也可以用 `--init-decision-adapter` 从旧 Stage3 检查点初始化，新检索头随机初始化；推理不允许这样补随机头。
不带检索头的旧检查点继续兼容；带检索头的检查点必须提供 evidence bank，
不会静默退回记忆直接回答。

当前训练入口是单视频 bundle、单 GPU、无 optimizer resume/DDP 的首版实现。
CPU 测试检查接线、因果性、梯度与检查点；真实 Qwen GPU 训练和语义检索效果
需要使用完成的 Stage2 检查点另行验证。
