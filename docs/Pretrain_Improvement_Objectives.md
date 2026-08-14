# Pretrain 改进目标

本文档总结当前 V3 interleave pretrain 之后的下一步改进方向。最新设计将训练目标统一为：

```text
Next Sentence Prediction with Mixed Visual & Textual Masking
```

核心思想是：给定历史 ultrasound visual-text chunks，以及当前 chunk 的部分模态信息，预测当前 chunk 的 narration text。通过不同 masking mode，让模型同时学习：

```text
visual grounding
future narration prediction
cross-modal robustness
```

---

## 1. 动机

当前 V3 主要做 Future Narration：

```text
past visual-text history
→ next narration
```

它能学习 ultrasound teaching video 中的时间顺序、操作流程和讲解逻辑，但仍有两个不足：

1. 没有显式使用当前 chunk 的 visual evidence 来生成当前 narration；
2. 模型可能过度依赖 text continuation，而不是学习 visual-text correspondence。

因此，下一版 pretrain 不再把 Grounded Narration 和 Future Narration 分成两个完全独立的任务，而是统一成一个 masked next sentence prediction objective。

---

## 2. V4: Next Sentence Prediction with Mixed Visual & Textual Masking

假设使用 `history_units = 3`，目标是预测 `chunk4 text`。

完整输入形式是：

```text
chunk1 visual + chunk1 text
chunk2 visual + chunk2 text
chunk3 visual + chunk3 text
chunk4 visual
→ chunk4 text
```

其中：

- 历史 chunks 同时包含 visual 和 text；
- 当前 chunk 只输入 visual；
- assistant 输出当前 chunk 的 narration text。

记：

- $S_t$：当前 chunk 的 narration text，即 target；
- $V_t$：当前 chunk 的 ultrasound visual segment；
- $H_{<t}$：历史 visual-text chunks；
- $\widetilde{V}_t$：可能被 mask 的当前 visual；
- $\widetilde{H}_{<t}$：可能被 mask 的历史 visual-text context。

统一训练目标为：

$$
\mathcal{L}_{\mathrm{NSP\text{-}Mask}}
=
-\log P(S_t \mid \widetilde{H}_{<t}, \widetilde{V}_t)
$$

这里仍然是标准 causal language modeling loss，不需要自定义 loss。关键变化在 sample / collator 阶段进行 mixed masking。

---

## 3. Mixed Masking Policy

每条训练样本随机选择一种 masking mode。

### 3.1 Mask current visual: 33%

```text
chunk1 visual + chunk1 text
chunk2 visual + chunk2 text
chunk3 visual + chunk3 text
chunk4 [VISUAL MASKED]
→ chunk4 text
```

对应目标：

```text
past visual-text history
→ current / next narration
```

该 mode 退化为 Future Narration，主要学习：

```text
temporal progression
procedure progression
narrative progression
```

它回答：

```text
What is likely to come next?
```

---

### 3.2 No mask: 33%

```text
chunk1 visual + chunk1 text
chunk2 visual + chunk2 text
chunk3 visual + chunk3 text
chunk4 visual
→ chunk4 text
```

对应目标：

```text
past visual-text history + current visual
→ current narration
```

该 mode 对应 Grounded Narration，主要学习：

```text
ultrasound visual evidence ↔ ultrasound language
```

它回答：

```text
What is visible now?
```

---

### 3.3 Random unit-level modality mask: 34%

随机选择一个 unit，并 mask 掉该 unit 的 visual 或 text。

例 1：mask 历史 visual：

```text
chunk1 visual + chunk1 text
chunk2 [VISUAL MASKED] + chunk2 text
chunk3 visual + chunk3 text
chunk4 visual
→ chunk4 text
```

例 2：mask 历史 text：

```text
chunk1 visual + chunk1 text
chunk2 visual + [TEXT MASKED]
chunk3 visual + chunk3 text
chunk4 visual
→ chunk4 text
```

也可以 mask 当前 chunk visual，使其退化为 future prediction。

该 mode 主要学习：

```text
cross-modal robustness
visual-text dependency learning
modality dropout
```

它强迫模型不能只依赖文本续写，也不能只依赖视频，而要学会在某个模态缺失时利用另一个模态补偿。

---

## 4. 统一后的能力覆盖

| Masking mode | Input | Target | 学到的能力 |
|---|---|---|---|
| Mask current visual | $H_{<t}$ | $S_t$ | Future narration / temporal anticipation |
| No mask | $H_{<t}, V_t$ | $S_t$ | Grounded narration / visual-language grounding |
| Random unit modality mask | partially masked $H_{<t}, V_t$ | $S_t$ | Cross-modal robustness / modality dependency |

因此 V4 可以统一覆盖之前讨论的两个 AR objective：

```text
Grounded Narration
Future Narration
```

并额外加入 modality masking 带来的跨模态约束。

---

## 5. 与旧版 Two-Objective 设计的关系

旧设计可以写成：

$$
\mathcal{L}_{AR}
=
\mathcal{L}_{ground}
+
\lambda_f \mathcal{L}_{future}
$$

其中：

$$
\mathcal{L}_{ground}
=
-\log P(S_t \mid H_{<t}, V_t)
$$

$$
\mathcal{L}_{future}
=
-\log P(S_t \mid H_{<t})
$$

现在 V4 将二者统一为：

$$
\mathcal{L}_{\mathrm{NSP\text{-}Mask}}
=
-\log P(S_t \mid \mathrm{MaskedInterleave}(H_{<t}, V_t))
$$

不同 masking mode 决定当前样本更偏向 grounded narration、future narration，还是 modality robustness。

这样做的优势是：

```text
一个数据格式
一个训练目标
一个 causal LM loss
通过 masking policy 控制学习信号
```

---

## 6. 推荐 sample schema

V4 样本应保留完整 unmasked 信息，由 dataset / collator 在训练时动态采样 mask。

```json
{
  "sample_type": "pretrain_next_sentence_mixedmask",
  "video_id": "...",
  "history": [
    {
      "sentence_idx": 0,
      "text": "chunk1 text",
      "video_window": [start, end],
      "num_frames": 3
    },
    {
      "sentence_idx": 1,
      "text": "chunk2 text",
      "video_window": [start, end],
      "num_frames": 3
    },
    {
      "sentence_idx": 2,
      "text": "chunk3 text",
      "video_window": [start, end],
      "num_frames": 3
    }
  ],
  "current_visual": {
    "sentence_idx": 3,
    "video_window": [start_t, end_t],
    "num_frames": 3
  },
  "target": "chunk4 text",
  "mask_policy": {
    "mask_current_visual_prob": 0.33,
    "no_mask_prob": 0.33,
    "random_unit_modality_mask_prob": 0.34
  }
}
```

注意：

```text
build_samples.py 只生成完整 unmasked sample；
dataset / collator 在训练时动态决定 mask。
```

这样同一条样本在不同 epoch 可能看到不同 mask，起到数据增强作用。

---

## 7. Prompt 设计

System prompt：

```text
You are an ultrasound teaching assistant.
You are given a sequence of ultrasound video frames and narration chunks.
Some visual or textual parts may be masked.
Use the available visual and textual context to produce the target narration chunk.
Output only the target narration chunk.
```

未 mask 的 unit：

```text
[frames...]
Narration: chunk text
```

mask visual 的 unit：

```text
[VISUAL MASKED]
Narration: chunk text
```

mask text 的 unit：

```text
[frames...]
Narration: [TEXT MASKED]
```

当前 chunk no-mask：

```text
[current frames...]
Narration:
```

当前 chunk visual masked：

```text
[VISUAL MASKED]
Narration:
```

assistant 输出：

```text
target chunk text
```

---

## 8. Contrastive Video-Text Alignment

在 V4 稳定后，可以继续加入 contrastive objective，用来显式对齐当前 ultrasound visual segment 和对应 narration sentence。

正样本对为：

$$
V_t \leftrightarrow S_t
$$

定义：

$$
z_t^V = f(V_t)
$$

$$
z_t^T = g(S_t)
$$

其中 $z_t^V$ 和 $z_t^T$ 是归一化后的视觉与文本表示。

Video-to-text InfoNCE：

$$
\mathcal{L}_{V \rightarrow T}
=
-\frac{1}{N}
\sum_{i=1}^{N}
\log
\frac{
\exp(\operatorname{sim}(z_i^V, z_i^T) / \tau)
}{
\sum_{j=1}^{N}
\exp(\operatorname{sim}(z_i^V, z_j^T) / \tau)
}
$$

Text-to-video InfoNCE：

$$
\mathcal{L}_{T \rightarrow V}
=
-\frac{1}{N}
\sum_{i=1}^{N}
\log
\frac{
\exp(\operatorname{sim}(z_i^T, z_i^V) / \tau)
}{
\sum_{j=1}^{N}
\exp(\operatorname{sim}(z_i^T, z_j^V) / \tau)
}
$$

Symmetric contrastive loss：

$$
\mathcal{L}_{contrast}
=
\frac{1}{2}
\left(
\mathcal{L}_{V \rightarrow T}
+
\mathcal{L}_{T \rightarrow V}
\right)
$$

最终可扩展为：

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{NSP\text{-}Mask}}
+
\lambda_c \mathcal{L}_{contrast}
$$

负样本可以包括：

```text
easy negatives: batch 中其他 video-text pairs
hard negatives: 同一视频中相邻但不匹配的 chunks
```

---

## 9. 推荐实现顺序

### V4: Mixed Masked Interleave Pretraining

优先实现：

```text
Next Sentence Prediction with Mixed Visual & Textual Masking
```

需要改动：

```text
pretrain/build_samples.py
pretrain/dataset.py
pretrain/collator.py
pretrain/infer.py
```

其中：

- sample builder 生成 history + current_visual + target；
- dataset 负责采样 history frames 和 current frames；
- collator 动态采样 masking mode；
- training loss 仍然是 causal LM loss；
- inference 默认使用 no-mask setting，即 history + current visual -> current text。

### V5: Contrastive-enhanced Pretraining

等 V4 稳定后再加入：

```text
Mixed masked NSP + symmetric InfoNCE
```

需要额外实现：

```text
video/text representation extraction
projection head
custom Trainer loss
positive/negative pair construction
```

---

## 10. Eval 设计

V4 eval 应分三种 setting：

### 10.1 No-mask eval

```text
history visual-text + current visual
→ current text
```

衡量 grounded narration。

### 10.2 Current-visual-masked eval

```text
history visual-text + [current visual masked]
→ current text
```

衡量 future narration / temporal anticipation。

### 10.3 Random modality mask eval

```text
partially masked history + current visual
→ current text
```

衡量 modality robustness。

每种 setting 都可以计算：

```text
word F1
BLEU-1/2/4
ROUGE-L
prefix@1/3/5
length ratio
medical concept precision / recall / F1
hallucination rate
```

---

## 11. 总结

| Component | Input | Target | Main ability |
|---|---|---|---|
| Mask current visual | $H_{<t}$ | $S_t$ | Future narration / temporal anticipation |
| No mask | $H_{<t}, V_t$ | $S_t$ | Visual-language grounding |
| Random modality mask | partially masked $H_{<t}, V_t$ | $S_t$ | Cross-modal robustness |
| Contrastive alignment | $V_t, S_t$ | matched pair | Fine-grained video-text correspondence |

最终模型不只是学习“继续说下一句”，而是同时学习：

```text
当前画面如何被描述
不看当前视频时如何预测未来讲解
缺失某个模态时如何利用另一个模态补偿
哪段视频和哪句 narration 匹配
```

这会让 pretraining story 从单纯的 next narration prediction，升级为：

```text
Next Sentence Prediction with Mixed Visual & Textual Masking
```
