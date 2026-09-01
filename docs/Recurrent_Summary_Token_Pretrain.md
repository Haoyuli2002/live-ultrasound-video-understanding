# Recurrent Summary Token Pretrain 设计

新的 pretrain 方向：让模型在观看超声视频流时，学会把已经看过的视频和 narration 压缩成一个可以递推更新的 summary state。这个方向不是为了生成自然语言 summary，而是为了训练一个隐藏状态形式的 memory token，使它能够承载 past context，并帮助模型预测后续 narration 以及之后的 QA SFT。

---

## 1. 动机

当前 V4 mixedmask pretrain 的形式大致是：

```text
历史 visual-text chunks + 当前 visual chunk -> 当前 narration
```

也就是模型直接从完整 history 中读取信息，再预测当前句子。这种方式有效，但它仍然依赖把历史上下文显式放进输入序列里。对于 streaming ultrasound video，这不是最理想的形式，因为真实场景中视频是持续到来的，模型应该维护一个不断更新的历史状态，而不是每次重新读完整历史。

因此我们希望引入一个 recurrent summary state：

```text
S_t = 到时间 t 为止的历史视频和文本状态
```

每来一个新 chunk，模型更新这个状态：

```text
S_t + C_{t+1} -> S_{t+1}
```

然后用新的 summary state 和当前视觉证据预测后续 narration。

---

## 2. 核心思想

这里的 summary token 不是文本摘要，也不需要人工标注。它是一个 special token 对应的 hidden state。

我们只引入一个 special token：

```text
<SUMMARY>
```

它在两个地方复用：

- 在 update pass 中，`<SUMMARY>` 是输出锚点。模型看到当前 context 后，在这个 token 位置形成新的 summary hidden state。
- 在下一步 prediction pass 中，`<SUMMARY>` 是输入占位符。它的普通 token embedding 会被替换成上一步得到的 summary hidden state。

也就是说，词表里只有一个 `<SUMMARY>`；不同步骤里它承担“读入旧状态”和“写出新状态”两个角色。这样设计更简单，也避免引入多个 summary token 语义。

这样，summary state 的监督不是来自人工 summary，而是来自后续 narration prediction 的 next-token loss。

---

## 3. Two-pass 训练流程

### 3.1 Summary Update Pass

假设当前已有历史状态 `S_{t-1}`，新的 chunk 是：

```text
C_t = V_t + T_t
```

其中 `V_t` 是当前 chunk 的视频帧，`T_t` 是当前 chunk 的 ASR narration。

Update pass 的输入是：

```text
S_{t-1} + V_t + T_t + <SUMMARY>
```

模型 forward 后，取最后 `<SUMMARY>` 位置的 hidden state：

```text
S_t = hidden_state(<SUMMARY>)
```

这个 `S_t` 就是融合了旧 summary 和当前 context 的新 summary state。

---

### 3.2 Narration Prediction Pass

接下来用新的 summary state 来预测下一段 narration。

Prediction pass 的输入是：

```text
<SUMMARY> + V_{t+1}
```

其中开头这个 `<SUMMARY>` 的 token embedding 会被替换成上一步得到的 `S_t`。

模型需要输出：

```text
T_{t+1}
```

训练 loss 是普通 causal LM loss：

```text
CE(T_{t+1})
```

也就是说，如果 `S_t` 没有携带足够的历史信息，模型就无法很好地预测 `T_{t+1}`。这样 summary state 会被迫学习压缩 past context。

---

## 4. 多步展开

一个训练样本可以展开为：

```text
S0 + C1 -> S1
S1 + C2 -> S2
S2 + C3 -> S3
S3 + V4 -> T4
```

其中：

```text
C_i = V_i + T_i
```

`S0` 可以用 `<SUMMARY>` 的初始 embedding，或者一个 learnable initial summary state。

第一版不建议展开太长。可以先用：

```text
unroll_chunks = 2 or 3
frames_per_chunk = 2 to 4
frame_size = 224
```

先验证 forward 和 backward 能跑通，再逐步扩大。

---

## 5. 数据单位

这个设计更适合固定时间 chunk，而不是 sentence chunk。

建议第一版：

```text
chunk_sec = 10
fps = 1
frames_per_chunk = 2~4 for smoke, later 10
```

每个 chunk 包含：

```json
{
  "chunk_idx": 0,
  "video_window": [0, 10],
  "text": "ASR narration inside this chunk"
}
```

训练 target 是下一 chunk 的 ASR narration。

---

## 6. 为什么这是真正的 memory bottleneck

普通 interleave pretrain 是：

```text
history V+T + current V -> current T
```

target tokens 可以直接 attend 到全部 raw history，所以模型不一定需要压缩历史。

而 recurrent summary 方案中，prediction pass 只看到：

```text
S_t + V_{t+1}
```

它看不到原始的 `V_1, T_1, ..., V_t, T_t`。历史信息必须经过 `S_t` 传递。因此 `S_t` 才是真正的历史瓶颈。

---

## 7. 实现挑战

这个方案需要自定义训练逻辑，不能直接复用当前普通 `Trainer + collator`。

主要难点包括：

1. **提取 summary hidden state**  
   update pass 需要 `output_hidden_states=True`，并定位 `<SUMMARY>` token 的位置。

2. **替换 prediction pass 的 embedding**  
   prediction pass 需要先正常 tokenize `<SUMMARY>`，再用 `S_t` 替换该 token 的 embedding。

3. **Qwen3-VL multimodal 输入兼容性**  
   需要确认 Qwen3-VL 在传入 `inputs_embeds` 的同时，是否还能正确处理 image/video inputs、`pixel_values`、`image_grid_thw` 等多模态字段。

4. **显存开销**  
   一个样本需要多个 update forward 加一个 prediction forward。完整反传会比当前 V4 贵很多。

5. **是否 detach summary state**  
   如果不 detach，梯度能穿过所有 update steps，但显存高。  
   如果 detach，训练更稳更省显存，但 summary update 学习信号会弱一些。

---

## 8. 推荐落地路线

不要直接替换当前稳定的 V4 mixedmask。建议新增一条实验线：

```text
Pretrain V5-RecurrentSummary
```

### Step 1: smoke feasibility

先写一个最小 smoke 脚本，只测试：

```text
1 个样本
2 个 chunks
每个 chunk 2 帧
能否提取 <SUMMARY> hidden state
能否替换 prediction pass 中 <SUMMARY> 的 embedding
能否 forward + backward
```

### Step 2: 小规模训练

如果 smoke 成功，再做：

```text
limit = 16 or 64 samples
unroll_chunks = 2
frames_per_chunk = 2
```

### Step 3: full295_asr_keep 训练

确认稳定后，再用过滤后的 full295 ASR 数据扩大训练。

---

## 9. 和后续 QA/SFT 的关系

这个 recurrent summary pretrain 不是直接训练 WAIT/ANSWER，而是训练一个历史压缩机制。

后续 QA/SFT 可以使用同样的 summary state：

```text
S_t + current visual + question -> <WAIT> or <ANSWER>
```

这样模型在回答实时问题时，不需要重新读取完整历史，而是利用 pretrain 学到的 streaming memory。

---

## 10. 一句话总结

Recurrent Summary Token Pretrain 的核心是：

```text
用 update pass 把 S_{t-1} 和当前视频/文本 chunk 融合成 S_t，
再用 S_t 和下一段视觉证据预测下一句 narration。
```

它不需要人工 summary 标签，监督来自 ASR next narration 的 causal LM loss。这个方案工程上比当前 V4 复杂，但它是真正让 summary token 承载 past context 的建模方式。
