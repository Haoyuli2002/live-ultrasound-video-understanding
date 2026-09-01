# Summary + Decide Streaming QA 设计

这个设计用于 QA/SFT 阶段，目标是让模型在实时超声视频流中判断：当前看到的证据是否足以回答用户问题。

核心思想是把系统拆成两个状态：

```text
<SUMMARY> = query-agnostic video memory
<DECIDE>  = query-aware answerability decision token
```

`<SUMMARY>` 不依赖具体问题，只负责总结到当前时刻为止的视频和 narration 历史；`<DECIDE>` 在用户问题出现后，结合 summary、当前视觉证据和 query，判断现在应该 `<WAIT>` 还是 `<ANSWER>`。

---

## 1. Query-agnostic Summary

视频流按 chunk 输入。每来一个新 chunk，模型更新一次 summary state：

```text
S_{t-1} + V_t + optional T_t + <SUMMARY> -> S_t
```

其中：

```text
S_t = hidden_state(<SUMMARY>)
```

`S_t` 应该包含当前为止的超声视频状态，例如：

```text
扫查目标
已见解剖结构
已见超声征象
当前操作阶段
已经出现的证据
```

它不针对某个具体问题，因此叫 query-agnostic memory。

---

## 2. Query-aware Decide

当用户在时间 `t` 提问 `Q` 时，模型使用当前 summary state 做 answerability 判断：

```text
S_t + current visual + Q + <DECIDE>
```

在 `<DECIDE>` 位置，模型产生 WAIT/ANSWER 的 token logits：

```text
answerability_logit = logit(<ANSWER>) - logit(<WAIT>)
```

如果 logit 偏向 `<WAIT>`，说明当前证据不足；如果偏向 `<ANSWER>`，说明当前证据足以回答。

---

## 3. 训练目标


训练样本仍然使用普通 causal LM 监督，不需要额外 binary head。

WAIT 样本：

```text
Input:
S_t + current visual + Question + <DECIDE>

Target:
<WAIT> specific wait reason
```

ANSWER 样本：

```text
Input:
S_t + current visual + Question + <DECIDE>

Target:
<ANSWER> answer text
```

因为 target 第一个 token 是 `<WAIT>` 或 `<ANSWER>`，模型会自然学到 answerability decision。推理时可以直接比较 `<WAIT>` 和 `<ANSWER>` 的 logits。

---

## 4. 和 Pretrain 的关系

Pretrain 阶段先训练 `<SUMMARY>`：

```text
S_t + V_{t+1} -> T_{t+1}
```

也就是让 summary state 学会承载过去视频和 narration 信息，并用于预测后续 narration。

QA/SFT 阶段再训练 `<DECIDE>`：

```text
S_t + current visual + Q + <DECIDE> -> <WAIT>/<ANSWER>
```

这样模型先学会“看视频并维护状态”，再学会“根据问题判断证据是否充分”。

---

## 5. 为什么不先加 binary head

可以加独立分类头：

```text
concat(S_t, Q) -> answerability logit
```

但第一版更推荐使用 LM token logits：

```text
logit(<ANSWER>) - logit(<WAIT>)
```

优点是：

```text
不改模型结构
兼容现有 SFT 格式
可同时训练 decision 和自然语言 wait_reason / answer
```

---

## 6. 推理流程

实时推理时：

```text
1. 视频流不断更新 S_t
2. 用户问题 Q 到来
3. 构造 S_t + 当前视觉 + Q + <DECIDE>
4. 读取 <WAIT>/<ANSWER> logits
5. 如果 WAIT：生成 wait_reason
6. 如果 ANSWER：生成 answer
```

---

## 7. 最小 special tokens

需要：

```text
<SUMMARY>
<DECIDE>
<WAIT>
<ANSWER>
```

其中 `<WAIT>` 和 `<ANSWER>` 已在 QA/SFT 中使用；新增的是：

```text
<SUMMARY>
<DECIDE>
```

---

## 8. 一句话总结

`<SUMMARY>` 负责不依赖 query 地维护视频历史状态；`<DECIDE>` 负责在给定 query 后做 answerability 判断。第一版不加额外分类头，而是用 `<ANSWER>` 和 `<WAIT>` 的 LM logits 作为是否可回答的分数。
