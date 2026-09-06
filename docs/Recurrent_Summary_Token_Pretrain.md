# Information Compression / Recurrent Summary Bank Pretrain 设计

这个文档描述第二阶段训练：**Information Compression / Learn Summary Bank**。

三阶段训练路线为：

```text
Stage 1: Pretrain / Ultrasound Knowledge Injection
Stage 2: Information Compression / Learn Summary Bank
Stage 3: SFT / QA + Instruction Tuning
```

Stage 1 的 V4 mixedmask pretrain 目标是 ultrasound-domain knowledge injection；Stage 2 才真正训练 summary memory；Stage 3 再训练 streaming QA、answerability decision 和 instruction following。

---

## 1. 动机

当前 V4 mixedmask pretrain 的形式大致是：

```text
历史 visual-text chunks + 当前 visual chunk -> 当前 / 下一段 narration
```

这种方式有助于超声视频-文本对齐和领域知识注入，但它仍然依赖把历史上下文显式放进输入序列里。对于 streaming ultrasound video，更理想的形式是维护一个 sliding temporal summary bank，而不是每次重新读取完整历史。

因此 Stage 2 引入 recurrent summary bank：

```text
B_t = [S_{t-(K-1)Δ}, ..., S_t]
```

其中 `S_t` 本身就是时间 `t` 的 summary bank，表示到时间 `t` 为止的历史视频和可选 narration 状态。

如果 `K=20`：

```text
B_t = [S_{t-190}, S_{t-180}, ..., S_t]  # K=20, Δ=10s
```

这里 `K=20` 表示同一时刻 bank 里有 20 个 summary token hidden states；它不表示保存 `S10, S20, ..., S200` 这些历史快照。如果 `S200` 是 recurrent update 得到的，那么 `S200` 已经压缩了 `0-200s` 的历史。

---

## 2. 核心思想

每来一个新 chunk，模型用上一轮 summary bank 和当前 context 更新出新的 summary bank：

```text
S_t = Update(S_{t-Δ}, C_t)
```

其中：

```text
Δ   = chunk size，例如 10s
C_t = current chunk = V_t + optional T_t
V_t = 当前 chunk 的超声视频帧
T_t = 当前 chunk 的 ASR narration，可选
```

更具体地写成 hidden-state 形式：

```text
S_t = hidden_states(
    <SUMMARY_1>, ..., <SUMMARY_K>
    | S_{t-Δ}, V_t, optional T_t
)
```

`S_t` 不是自然语言 summary，而是一组 latent hidden vectors。训练目标不是人工 summary 标签，而是让 `S_t` 能够重建被压缩的信息或帮助预测未来 narration。

---

## 3. Special Tokens 和实现语义

概念上使用两组 memory tokens：

```text
<MEM_1> ... <MEM_K>          # 输入上一轮 summary bank 的占位符
<SUMMARY_1> ... <SUMMARY_K>  # 输出 / 读取新 summary bank 的锚点
```

Update pass 中：

```text
embedding(<MEM_i>) = s_{t-Δ}^i
s_t^i = hidden_state(<SUMMARY_i>)
```

也就是说：

```text
旧 summary bank S_{t-Δ} 注入到 <MEM_1>...<MEM_K>
当前 chunk C_t 作为新证据输入
新 summary bank S_t 从 <SUMMARY_1>...<SUMMARY_K> 位置读取
```

工程上第一版也可以先用重复的 `<SUMMARY>` token 实现多个位置，但文档语义统一为：`S_t` 是由 `K` 个 summary token hidden states 组成的 summary bank。

---

## 4. Two-pass 训练流程

### 4.1 Compression / Update Pass

假设当前已有历史 summary bank：

```text
S_{t-Δ} = [s_{t-Δ}^1, ..., s_{t-Δ}^K]
```

新的 chunk 是：

```text
C_t = V_t + optional T_t
```

Update pass 的输入是：

```text
Previous memory:
<MEM_1> ... <MEM_K>      # embeddings replaced by S_{t-Δ}

Current chunk:
V_t + optional T_t

Updated memory:
<SUMMARY_1> ... <SUMMARY_K>
```

模型 forward 后读取：

```text
S_t = [hidden_state(<SUMMARY_1>), ..., hidden_state(<SUMMARY_K>)]
```

这个 `S_t` 就是融合了旧 summary bank 和当前 context 的新 summary bank。

---

### 4.2 Reconstruction / Prediction Pass

接下来用新的 summary bank 来重建被压缩的信息或预测下一段 narration。

输入只给 summary bank 和必要的当前/未来视觉证据，不能再给完整 raw history：

```text
S_t + reconstruction prompt -> masked / historical narration
S_t + V_{t+Δ} -> T_{t+Δ}
S_t -> selected clinical facts
```

训练 loss 是普通 causal LM loss：

```text
CE(target text)
```

也就是说，如果 `S_t` 没有携带足够的历史信息，模型就无法很好地重建或预测目标文本。这样 summary bank 会被迫学习压缩 past context。

---

## 5. 多步展开

一个训练样本可以展开为：

```text
S_0 = initial summary bank
S_10 = Update(S_0, C_10)
S_20 = Update(S_10, C_20)
S_30 = Update(S_20, C_30)
S_30 + V_40 -> T_40
```

这里 `S_30` 已经压缩了 `0-30s` 的历史，因此 prediction / reconstruction pass 不再输入 `C_10, C_20, C_30` 的 raw history。

`S_0` 可以用 `<MEM_1>...<MEM_K>` 的初始 embeddings，或者一个 learnable initial summary bank。

第一版不建议展开太长。可以先用：

```text
summary_bank_size K = 8 or 20
unroll_chunks = 2 or 3
chunk_sec = 10
frames_per_chunk = 2 to 4
frame_size = 224
```

先验证 forward 和 backward 能跑通，再逐步扩大。

---

## 6. 数据单位

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

训练 target 可以是：

```text
1. 下一 chunk 的 ASR narration
2. masked historical narration
3. selected clinical facts / synthetic compression questions
```

第一版可以优先使用 next narration prediction，因为已有 ASR 数据可以直接构造监督。

---

## 7. 为什么这是真正的 memory bottleneck

普通 interleave pretrain 是：

```text
history V+T + current V -> current T
```

target tokens 可以直接 attend 到全部 raw history，所以模型不一定需要压缩历史。

而 recurrent summary bank 方案中，reconstruction / prediction pass 只看到：

```text
S_t + limited current/future evidence
```

它看不到原始的 `V_1, T_1, ..., V_t, T_t`。历史信息必须经过 sliding temporal `B_t = [S_{t-(K-1)Δ}, ..., S_t]` 传递。因此 `B_t` 才是真正的历史瓶颈。

---

## 8. 实现挑战

这个方案需要自定义训练逻辑，不能直接复用当前普通 `Trainer + collator`。

主要难点包括：

1. **提取 summary bank hidden states**  
   update pass 需要 `output_hidden_states=True`，并定位 `<SUMMARY_1>...<SUMMARY_K>` token 的位置。

2. **替换 reconstruction / prediction pass 的 embeddings**  
   prediction pass 需要先正常 tokenize `<MEM_1>...<MEM_K>`，再用 `S_t` 替换这些 token 的 embeddings。

3. **Qwen3-VL multimodal 输入兼容性**  
   需要确认 Qwen3-VL 在传入 `inputs_embeds` 的同时，是否还能正确处理 image/video inputs、`pixel_values`、`image_grid_thw` 等多模态字段。

4. **显存开销**  
   一个样本需要多个 update forward 加一个 reconstruction / prediction forward。完整反传会比当前 V4 贵很多。

5. **是否 detach summary bank**  
   如果不 detach，梯度能穿过所有 update steps，但显存高。  
   如果 detach，训练更稳更省显存，但长期 credit assignment 会弱一些。

---

## 9. 推荐落地路线

不要直接替换当前稳定的 V4 mixedmask。建议新增一条实验线：

```text
Pretrain V5-InformationCompression-SummaryBank
```

### Step 1: smoke feasibility

先写一个最小 smoke 脚本，只测试：

```text
1 个样本
2 个 chunks
每个 chunk 2 帧
K = 4 or 8
能否提取 <SUMMARY_1...K> hidden states
能否替换 prediction pass 中 <MEM_1...K> 的 embeddings
能否 forward + backward
```

### Step 2: 小规模训练

如果 smoke 成功，再做：

```text
limit = 16 or 64 samples
unroll_chunks = 2
frames_per_chunk = 2
K = 8 or 20
```

### Step 3: full295_asr_keep 训练

确认稳定后，再用过滤后的 full295 ASR 数据扩大训练。

---

## 10. 和后续 QA/SFT 的关系

这个 Information Compression pretrain 不是直接训练 WAIT/ANSWER，而是训练一个历史压缩机制。

后续 QA/SFT 使用同样的 summary bank：

```text
S_t + current visual + question -> <WAIT> or <ANSWER>
```

这样模型在回答实时问题时，不需要重新读取完整历史，而是利用 Stage 2 学到的 streaming memory。

---

## 11. 一句话总结

Information Compression / Recurrent Summary Bank Pretrain 的核心是：

```text
B_t = [S_{t-(K-1)Δ}, ..., S_t]
S_t = Update(S_{t-Δ}, current video/optional narration chunk)
S_t -> reconstruct compressed information or predict future narration
```

它不需要人工 summary 标签，监督来自 reconstruction / ASR next narration 的 causal LM loss。这个方案工程上比当前 V4 复杂，但它是真正让 sliding temporal summary bank 承载 past context 的建模方式。