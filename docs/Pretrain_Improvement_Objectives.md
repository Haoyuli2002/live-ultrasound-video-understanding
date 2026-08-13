# Pretrain 改进目标

本文档总结当前 V3 pretrain 之后的下一步改进方向。核心思想是：不要只让模型预测下一句 narration，而是同时学习 **当前视觉 grounding**、**未来 narration 预测** 和 **视频-文本对比对齐**。

## 1. 动机

当前 V3 主要做 Future Narration：

```text
past visual-text history
→ next narration
```

它有助于学习 ultrasound teaching video 中的时间顺序、操作流程和讲解逻辑，但它没有显式要求模型把当前 ultrasound 画面和当前 narration 对齐。

因此，下一版 pretrain 应该包含三个互补目标：

1. **Grounded Narration**：学习当前 ultrasound visual evidence 和当前 narration 的对应关系。
2. **Future Narration**：学习 temporal / procedural / narrative progression。
3. **Contrastive Video-Text Alignment**：进一步强化 ultrasound clip 与 narration sentence 的细粒度匹配。

整体故事可以概括为：

```text
Visual grounding
+ Temporal anticipation
+ Video-text contrastive alignment
= Ultrasound visual-text temporal pretraining
```

## 2. Objective A: Grounded Narration

### 目标

让模型根据当前 ultrasound 视觉片段，生成当前 narration。

### 输入与输出

```text
past visual-text history + current visual
→ current narration
```

记：

- $S_t$：当前 narration sentence / chunk。
- $V_t$：当前 ultrasound visual segment。
- $H_{<t}$：当前时刻之前的 visual-text history。

Grounded Narration 的损失为：

$$
\mathcal{L}_{ground}
=
-\log P(S_t \mid H_{<t}, V_t)
$$

### 学到的能力

这个目标主要学习：

```text
ultrasound visual evidence ↔ ultrasound language
```

它回答的问题是：

```text
What is visible now?
```

也就是：当前画面里能看到什么，应该如何用 ultrasound 教学语言表达。

## 3. Objective B: Future Narration

### 目标

让模型根据过去的 ultrasound visual-text history，预测下一句 narration。

### 输入与输出

```text
past visual-text history
→ next narration
```

Future Narration 的损失为：

$$
\mathcal{L}_{future}
=
-\log P(S_t \mid H_{<t})
$$

### 学到的能力

这个目标主要学习：

```text
temporal progression
procedure progression
narrative progression
```

它回答的问题是：

```text
What is likely to come next?
```

也就是：根据之前的讲解和画面，接下来最可能讲什么。

## 4. Mixed AR Training

两个 autoregressive objective 可以组合为：

$$
\mathcal{L}_{AR}
=
\mathcal{L}_{ground}
+
\lambda_f \mathcal{L}_{future}
$$

其中 $\lambda_f$ 控制 Future Narration 的权重。

工程上，第一版不需要改 Trainer loss。可以先做 **data-level mixing**：

```text
70% grounded current narration samples
30% future narration samples
```

这样仍然使用标准 causal language modeling 训练流程，只是在数据层面混合两类任务。

后续可以做 ablation：

```text
100% grounded
100% future
70% grounded / 30% future
50% grounded / 50% future
```

## 5. Contrastive Video-Text Alignment

### 目标

除了 next-token prediction，还显式学习当前 ultrasound visual segment 和对应 narration sentence 的匹配关系。

正样本对为：

$$
V_t \leftrightarrow S_t
$$

其中 $V_t$ 是当前 ultrasound visual segment，$S_t$ 是对应 narration sentence。

### 表征

定义 video embedding 和 text embedding：

$$
z_t^V = f(V_t)
$$

$$
z_t^T = g(S_t)
$$

其中 $z_t^V$ 和 $z_t^T$ 是归一化后的视觉与文本表示。

### Video-to-Text InfoNCE

对于 batch 中的 $N$ 个 video-text pair，video-to-text loss 为：

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

### Text-to-Video InfoNCE

对称的 text-to-video loss 为：

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

### Symmetric Contrastive Loss

最终对比学习损失为：

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

### 负样本设计

可以使用两类负样本：

```text
easy negatives: batch 中其他 video-text pairs
hard negatives: 同一视频中相邻但不匹配的 chunks
```

hard negatives 对 ultrasound teaching video 尤其重要，因为相邻 clip 往往视觉和语义都很接近，能迫使模型学习更细粒度的 video-text correspondence。

## 6. 最终目标

完整 pretraining objective 可以写成：

$$
\mathcal{L}
=
\mathcal{L}_{AR}
+
\lambda_c \mathcal{L}_{contrast}
$$

也就是：

$$
\mathcal{L}
=
\mathcal{L}_{ground}
+
\lambda_f \mathcal{L}_{future}
+
\lambda_c \mathcal{L}_{contrast}
$$

其中：

- $\lambda_f$ 控制 Future Narration 的权重。
- $\lambda_c$ 控制 Contrastive Alignment 的权重。

## 7. 推荐实现顺序

### V4: Mixed AR Pretraining

优先实现：

```text
Grounded Narration + Future Narration
```

建议先使用：

```text
70% grounded
30% future
```

这一版只需要改 sample builder 和 prompt / collator，不需要自定义 loss，实现风险低。

### V5: Contrastive-enhanced Pretraining

等 V4 稳定后再实现：

```text
Mixed AR + symmetric InfoNCE
```

这一版需要：

```text
video/text representation extraction
projection head
custom Trainer loss
positive/negative pair construction
```

实现成本更高，但能进一步增强 ultrasound visual-text alignment。

## 8. 总结

| Component | Input | Target | Main ability |
|---|---|---|---|
| Grounded Narration | $H_{<t}, V_t$ | $S_t$ | Visual-language grounding |
| Future Narration | $H_{<t}$ | $S_t$ | Temporal / procedural anticipation |
| Contrastive Alignment | $V_t, S_t$ | matched pair | Fine-grained video-text correspondence |

最终模型不只是学习“继续说下一句”，而是同时学习：

```text
当前画面如何被描述
接下来可能讲什么
哪段视频和哪句 narration 匹配
```

这会让 pretraining story 从单纯的 next narration prediction，升级为：

```text
Ultrasound visual-text temporal pretraining
```
