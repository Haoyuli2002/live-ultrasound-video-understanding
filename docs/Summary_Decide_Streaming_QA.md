# Summary Bank Streaming QA 设计

本设计用于第三阶段 QA/SFT，目标是让模型在实时超声视频流中判断当前证据是否足以回答用户问题。

统一三阶段路线：

```text
Stage 1: Pretrain / Ultrasound Knowledge Injection
Stage 2: Information Compression / Learn Summary Bank
Stage 3: SFT / QA + Instruction Tuning
```

核心状态定义：

```text
B_t = [S_{t-(K-1)Δ}, ..., S_t]
```

`S_t` 本身就是截止时间 `t` 的 summary token；`K` 是同一时刻 bank 内 summary token 的数量。比如 `K=20` 表示：

```text
B_t = [S_{t-190}, S_{t-180}, ..., S_t]  # K=20, Δ=10s
```

---

## 1. End-to-end Mechanism

整体机制不是生成自然语言 summary，而是维护 sliding temporal hidden-state summary bank。QA 阶段不引入额外分类模块；模型直接通过生成首 token `<WAIT>` 或 `<ANSWER>` 完成 answerability decision。

```text
Summary bank:
B_t = [S_{t-(K-1)Δ}, ..., S_t]

Summary update:
S_{t+1} = hidden_state(<SUMMARY> | B_t, V_{t+1}, optional T_{t+1})
B_{t+1} = append(B_t, S_{t+1})
if len(B_{t+1}) > K: B_{t+1} = keep_last_K(B_{t+1})

QA / Answerability:
p_WAIT   = P(<WAIT>   | S_{t+1}, V_{t+1}, Q)
p_ANSWER = P(<ANSWER> | S_{t+1}, V_{t+1}, Q)

if p_WAIT > p_ANSWER:
    generate <WAIT> reason
    continue streaming

if p_ANSWER >= p_WAIT:
    generate <ANSWER> answer
    stop or return answer
```

其中：

```text
S_t = query-agnostic summary bank / hidden memory state
V_t = current video chunk / current visual evidence
T_t = optional ASR narration for the chunk
Q   = user query
K   = number of summary tokens in S_t
```

---

## 2. Query-agnostic Summary Bank

视频流按 chunk 输入。每来一个新 chunk，模型更新一次 summary bank：

```text
S_t = hidden_state(<SUMMARY> | B_{t-Δ}, V_t, optional T_t)
B_t = append(B_{t-Δ}, S_t)
if len(B_t) > K: B_t = keep_last_K(B_t)
```

实现上可以使用两组 special tokens：

```text
<MEM_1> ... <MEM_K>          # 输入旧 summary bank 的占位符
<SUMMARY_1> ... <SUMMARY_K>  # 读取新 summary bank 的输出锚点
```

update pass 中：

```text
embedding(<MEM_i>) = s_{t-1}^i
s_t^i = hidden_state(<SUMMARY_i>)
```

`S_t` 是 query-agnostic，只压缩视频和可选 narration 历史。

---

## 3. Information Compression 阶段

第二阶段目标是：

```text
past context -> S_t -> reconstruction / prediction
```

训练时递归更新 summary bank：

```text
S_1 = Update(S_0, C_1)
S_2 = Update(S_1, C_2)
...
S_t = Update(S_{t-1}, C_t)
```

其中：

```text
C_t = V_t + optional T_t
```

然后用 `S_t` 重建被压缩的信息或预测未来 narration：

```text
S_t -> reconstruct masked / historical narration
S_t + V_{t+1} -> predict T_{t+1}
S_t -> reconstruct selected clinical facts
```

关键是 reconstruction / prediction pass 不能再看到完整 raw history，否则模型会绕过 summary bank。

---

## 4. QA / Answerability by `<WAIT>/<ANSWER>`

当用户在时间 `t` 提问 `Q` 时，模型使用最新 summary bank 做 answerability 判断并生成回答：

```text
Input:
S_t + current visual + Q

Output:
<WAIT> reason
```

或者：

```text
Input:
S_t + current visual + Q

Output:
<ANSWER> answer
```

answerability 判断直接来自首 token 概率：

```text
p_WAIT   = P(<WAIT>   | S_t, V_t, Q)
p_ANSWER = P(<ANSWER> | S_t, V_t, Q)
```

推理时可以直接 greedy generate；更稳定的实现是先显式比较 `<WAIT>` 和 `<ANSWER>` 的 next-token logits，再生成对应文本。

---

## 5. SFT 训练目标

SFT 阶段只使用普通 causal LM loss：

```text
loss = CE(target tokens)
```

WAIT 样本：

```text
Input:
S_t + current visual + Question

Target:
<WAIT> specific wait reason
```

ANSWER 样本：

```text
Input:
S_t + current visual + Question

Target:
<ANSWER> answer text
```

因为 target 第一个 token 就是 `<WAIT>` 或 `<ANSWER>`，模型会通过标准 next-token prediction 学会当前 evidence 是否足够。

---

## 6. WAIT reason 是否写回 Summary Bank

不写回。

`S_t` 是 query-agnostic video memory，只压缩视频和可选 narration 历史：

```text
S_{t+1} = Update(S_t, V_{t+1}, optional T_{t+1})
```

previous `<WAIT>` reason 属于 query-specific interaction history。它可以在下一次 QA prompt 中作为可选上下文输入：

```text
P(<WAIT>/<ANSWER> | S_{t+1}, V_{t+1}, Q, optional previous WAIT reasons)
```

但不要写入 `S_{t+1}`，否则 summary bank 会被某个具体 query 污染。

---

## 7. 推理流程

```text
1. 初始化 summary bank B_0 = []
2. 视频流每 Δ 秒到来一个 chunk，例如 Δ=10s
3. 使用 S_t、V_{t+1} 和 optional T_{t+1} 更新得到 S_{t+1}
4. 如果用户问题 Q 尚未出现，只持续更新 query-agnostic summary bank
5. 如果用户问题 Q 已出现，构造 S_{t+1} + 当前视觉 + Q
6. 比较 next-token logits: P(<WAIT>) vs P(<ANSWER>)，或直接生成首 token
7. 如果首 token 是 <WAIT>：生成 wait reason，继续 streaming
8. 如果首 token 是 <ANSWER>：生成 answer，返回答案
```

---

## 8. 最小 special tokens

概念上需要：

```text
<MEM_1> ... <MEM_K>
<SUMMARY_1> ... <SUMMARY_K>
<WAIT>
<ANSWER>
```

工程上第一版每个 chunk 先生成一个 `<SUMMARY>` hidden state 作为新的 `S_t`，再 append 到 bank；bank 超过 `K` 后移出最旧 token。

---

## 9. 一句话总结

`B_t = [S_{t-(K-1)Δ}, ..., S_t]` 是 sliding temporal summary bank；每个 chunk 生成一个新的 `S_t` 并 append 到 bank，超过 `K` 后移出最旧 summary token。QA 阶段直接让 LM 以 `<WAIT>` 或 `<ANSWER>` 作为第一个生成 token 来完成 answerability decision。