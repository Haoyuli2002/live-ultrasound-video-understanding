# Stage 3 — 可回答性决策（双模型实现骨架）

Stage 3 使用两个相互独立的 `Qwen/Qwen3.5-2B` 实例。二者都严格恢复
Stage 1 → Stage 2 权重链：先把 Stage 1 adapter merge 进 backbone，恢复
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

## 当前真实进展

已经实现：

1. 冻结 Stage 2 memory 模型的正确加载和参数冻结。
2. 决策模型对完整 Stage 2 权重链的恢复与 merge。
3. device-safe 的已注册 BCE head。
4. 与上图一致的 system → memory → question → decision 输入顺序。
5. 单样本 BCE + NTP 前向、stop-gradient 检查及 WAIT/ANSWER 标签一致性检查。
6. Stage 3 adapter、processor、BCE head 和来源配置的保存与恢复。

尚未实现或验证：

1. 从 QA JSONL 按 `query_time` / `answer_time` 调度视频流的数据加载器。
2. 同时常驻冻结 memory 模型和 decision 模型的完整训练 CLI。
3. BCE 阈值决策后固定 `<WAIT>` / `<ANSWER>` 前缀的文本生成。
4. 多问题共享前缀 KV cache、分支隔离及 cache 失效逻辑。
5. Qwen3.5 双模型 GPU 端到端 smoke test。

本地 laptop 只执行语法、schema 和 CPU 单元测试。上述 GPU smoke test、真实
Stage 1/2 adapter 加载以及显存评估必须在保存模型权重的 SSH 训练环境完成。

因此当前 Stage 3 是“模型与单样本损失接口正确，但训练和在线推理流水线未完成”，
不能描述成已经可训练或可部署。

## 关于 memory 快照

`stage2.stream` 的 `.pt` 只保存每分钟 long-memory 快照和视频结束时的残余
short memory，不足以直接重建任意 `query_time` / `answer_time` 的输入。完整训练
应优先用冻结 memory 模型按时间顺序重放视频；如果要离线预计算，则必须额外保存
所有 QA 决策时刻的 long/short 快照。

## 暂未落地的研究项

多视角 summary（解剖、探头运动、视觉发现等）仍是待定义的消融方向，不属于
当前 Stage 2 数据 schema，也不应在主 README 中描述为已经实现。
