# 当前 Interleave Pretrain 设计总结

## 1. 实验目标

当前 Interleave Pretrain 的目标是：

```text
历史 sentence chunk 的超声视频帧 + 历史 sentence chunk 的 narration 文本
→ 预测下一个 sentence chunk
```

它不是直接做 anatomy / finding supervision，而是做一种更强的 **visual-text temporal narration pretraining**。

---

## 2. 训练样本构建方案

当前推荐使用：

```text
punctuation-aware sentence chunk interleave
```

流程：

```text
ASR segments
→ 拼接成完整 transcript
→ 按主要标点切分 sentence chunk
→ 将 sentence chunk 映射回 ASR segment 时间戳
→ 每个历史 sentence chunk 均匀采 3 帧
→ 图文交错输入
→ 预测下一个 chunk
```

推荐切分标点：

```text
. ? ! ; :
```

默认不按逗号切分，避免 chunk 过碎。

---

## 3. 三种分割版本对比

| 版本 | 切分方式 | 输入形式 | 预测目标 | 优点 | 缺点 | 当前定位 |
|---|---|---|---|---|---|---|
| V1 Segment | 每个合格 ASR segment 一个样本 | 当前窗口 frames + 纯文本 `prev_context` | next ASR segment | 简单；样本多；时间对齐细 | 太碎；半句话多；容易学成局部 ASR continuation | 已完成 baseline |
| V2 Segment-merge | 顺序累积 ASR segments，遇到句号或 fallback 切分 | 当前窗口 frames + 纯文本前文句子 | next sentence-like chunk | 比 V1 更自然；减少部分断句 | fallback 可能在 phrase 中间切断；仍然不是严格句子 | 中间版本 / 对照 |
| V3 Punctuation Interleave | 先拼接完整 ASR，再按标点切 sentence chunk，并映射回时间戳 | 历史 sentence chunk 的 3 帧 + chunk 文本交错 | next punctuation-aware sentence chunk | 切分更干净；图文时序关系更强；避免 `Massachusetts General / Hospital` 被切开 | 仍依赖 ASR 标点；不是显式 visual finding supervision | 当前推荐 pretrain 方案 |

---

## 4. V3 样本形式

对于 target sentence at chunk `k`，输入最近 `N` 个历史 sentence chunks：

```text
sentence at chunk k-3 的 3 帧
Narration: sentence at chunk k-3

sentence at chunk k-2 的 3 帧
Narration: sentence at chunk k-2

sentence at chunk k-1 的 3 帧
Narration: sentence at chunk k-1
```

assistant 直接输出：

```text
sentence at chunk k
```

默认：

```text
history_units = 3
frames_per_sentence_chunk = 3
```

因此每条样本最多包含：

```text
3 个历史 sentence chunks × 每个 chunk 3 帧 = 9 帧
```

---

## 5. Prompt 设计

任务说明放在 system prompt：

```text
You are an ultrasound teaching assistant.
You are given a sequence of ultrasound video frames interleaved with prior narration sentences.
Each group of frames corresponds to the narration sentence immediately following it.
Continue the sequence by directly producing the next narration sentence.
Output only the next narration sentence.
```

user content 只包含历史图文，不再包含额外指令：

```text
[sentence chunk 1 frame 1]
[sentence chunk 1 frame 2]
[sentence chunk 1 frame 3]
Narration: sentence at chunk 1

[sentence chunk 2 frame 1]
[sentence chunk 2 frame 2]
[sentence chunk 2 frame 3]
Narration: sentence at chunk 2
```

assistant target：

```text
sentence at chunk 3
```

---

## 6. 样本 schema

```json
{
  "sample_type": "pretrain_caption_sentence_interleave",
  "video_id": "...",
  "history": [
    {
      "sentence_idx": 0,
      "text": "previous sentence chunk text",
      "video_window": [start, end],
      "num_frames": 3,
      "segment_start_idx": 0,
      "segment_end_idx": 1
    }
  ],
  "video_window": [target_start, target_end],
  "prev_context": "previous sentence chunk text",
  "target": "next sentence chunk text",
  "meta": {
    "unit": "sentence",
    "format": "interleave",
    "target_sentence_idx": 1,
    "target_sentence_start": 10.0,
    "target_sentence_end": 18.0,
    "history_units": 1,
    "frames_per_sentence": 3
  }
}
```

`video_window` 对 target sentence chunk 只作为 metadata，不作为输入帧使用。输入帧来自 `history[*].video_window`。

---

## 7. 生成命令

```bash
python pretrain/build_samples.py \
  --transcripts cluster_data/QA/train/transcripts \
  --output cluster_data/pretrain/train_pretrain_punct_sentence_interleave_samples.jsonl \
  --unit sentence \
  --sentence-mode punctuation \
  --format interleave \
  --history-units 3 \
  --frames-per-sentence 3 \
  --min-words 3 \
  --sentence-max-words 80 \
  --split-punctuation ".?!;:"
```

---

## 8. 真实样本检查结论

以 `-1i1i9sbjqE` 为例，punctuation split 已经能把原本跨 ASR segment 的内容合并：

```text
As per usual, a number of the ultrasound images and videos are courtesy of the Division of Emergency Ultrasound at Massachusetts General Hospital.
```

避免了旧版中：

```text
Massachusetts General
Hospital.
```

被拆开的情况。

同时 interleave history 正确滑动：

```text
sample 1: chunk 0 -> target chunk 1
sample 2: chunk 0 + chunk 1 -> target chunk 2
sample 3: chunk 0 + chunk 1 + chunk 2 -> target chunk 3
sample 4: chunk 1 + chunk 2 + chunk 3 -> target chunk 4
```

---

## 9. 实现文件

```text
pretrain/build_samples.py
pretrain/video_sampling.py
pretrain/dataset.py
pretrain/collator.py
pretrain/infer.py
```

关键新增能力：

- `--sentence-mode punctuation`
- `--format interleave`
- `--history-units`
- `--frames-per-sentence`
- 每个历史 chunk 内均匀采帧
- interleaved multimodal prompt
- interleave inference 支持

---

## 10. 评估指标

当前 Interleave Pretrain 的评估不应只看生成文本是否“像不像”，而应分层看：

| 评估方向 | 指标 | 目的 |
|---|---|---|
| Language prediction | NLL / PPL | 衡量模型对真实 next sentence chunk 的条件概率，即 teacher-forcing 下是否更会预测下一段语言 |
| Next narration generation | BLEU / ROUGE / word-F1 | 衡量自由生成文本与目标 sentence chunk 的 n-gram / overlap 相似度 |
| Semantic similarity | embedding cosine | 衡量 prediction 与 target 的语义接近度，补充 BLEU/ROUGE 对同义表达不敏感的问题 |
| Medical concept evaluation | concept precision / recall / F1 | 衡量 prediction 是否覆盖关键 ultrasound anatomy、view、artifact、finding、procedure concepts |
| Medical hallucination | hallucinated concept rate | 衡量 prediction 是否生成 target/history 中没有依据的医学概念 |

### 10.1 Language prediction: NLL / PPL

最直接的 pretrain 指标是 teacher-forcing 下的 target token negative log-likelihood。

给定：

- \(x\)：interleaved history
- \(y=(y_1,\dots,y_T)\)：target sentence chunk tokens
- \(T\)：target token length

token-level NLL：

$$
\mathrm{NLL}(x, y)
=
-\frac{1}{T}
\sum_{t=1}^{T}
\log p(y_t \mid x, y_{<t})
$$

Perplexity：

$$
\mathrm{PPL}(x, y)
=
\exp\left(\mathrm{NLL}(x, y)\right)
$$

dataset-level 平均：

$$
\mathrm{NLL}_{\mathrm{dataset}}
=
\frac{\sum_i T_i \cdot \mathrm{NLL}_i}{\sum_i T_i}
$$

$$
\mathrm{PPL}_{\mathrm{dataset}}
=
\exp\left(\mathrm{NLL}_{\mathrm{dataset}}\right)
$$

这对应训练目标本身。  
如果 LoRA adapter 有效，应该在 eval set 上比 base model 有更低的 NLL / PPL。

### 10.2 Next narration generation: BLEU / ROUGE

自由生成时可以看 BLEU、ROUGE-L 和 word-overlap F1。这些指标衡量 prediction 和 target sentence chunk 的字面相似度。

### BLEU

BLEU 使用 modified n-gram precision 和 brevity penalty：

$$
\mathrm{BLEU}
=
\mathrm{BP}
\cdot
\exp\left(
\sum_{n=1}^{N} w_n \log p_n
\right)
$$

其中：

- $p_n$：modified n-gram precision
- $w_n$：n-gram 权重，通常 $w_n = \frac{1}{N}$
- $\mathrm{BP}$：brevity penalty

brevity penalty：

$$
\mathrm{BP}
=
\begin{cases}
1, & \text{if } c > r \\
\exp\left(1 - \frac{r}{c}\right), & \text{if } c \le r
\end{cases}
$$

其中：

- $c=\mathrm{len}(\mathrm{pred})$
- $r=\mathrm{len}(\mathrm{ref})$

### ROUGE-L

ROUGE-L 基于 longest common subsequence，即 LCS：

$$
R_{\mathrm{LCS}}
=
\frac{\mathrm{LCS}(\mathrm{pred}, \mathrm{ref})}{\mathrm{len}(\mathrm{ref})}
$$

$$
P_{\mathrm{LCS}}
=
\frac{\mathrm{LCS}(\mathrm{pred}, \mathrm{ref})}{\mathrm{len}(\mathrm{pred})}
$$

$$
F_{\mathrm{LCS}}
=
\frac{(1+\beta^2) R_{\mathrm{LCS}} P_{\mathrm{LCS}}}
{R_{\mathrm{LCS}} + \beta^2 P_{\mathrm{LCS}}}
$$

常用时可以取：

$$
\beta = 1
$$

此时 $F_{\mathrm{LCS}}$ 近似为 LCS-F1。

### Word-overlap F1

把 prediction 和 reference target 都转成词集合或词袋：

$$
\mathrm{overlap}
=
\mathrm{words}(\mathrm{pred})
\cap
\mathrm{words}(\mathrm{ref})
$$

$$
\mathrm{Precision}
=
\frac{|\mathrm{overlap}|}{|\mathrm{words}(\mathrm{pred})|}
$$

$$
\mathrm{Recall}
=
\frac{|\mathrm{overlap}|}{|\mathrm{words}(\mathrm{ref})|}
$$

$$
\mathrm{WordF1}
=
\frac{2 \cdot \mathrm{Precision} \cdot \mathrm{Recall}}
{\mathrm{Precision} + \mathrm{Recall}}
$$

适合回答：

```text
模型是否更会生成真实 next narration？
```

但局限是：医学同义表达可能被低估，长解释也可能被误判。

### 10.3 Medical concept evaluation

为了评估 ultrasound domain adaptation，需要抽取医学/超声概念：

```text
anatomy: pleural line, left ventricle, gallbladder, IVC
view: parasternal long axis, apical four chamber, RUQ FAST
artifact: A-lines, B-lines, mirror artifact, lung sliding
finding: pneumothorax, pleural effusion, pulmonary edema, free fluid
procedure/action: probe marker, transducer orientation, scan mid-axillary line
```

构建概念词表：

```text
docs/ultrasound_concepts.json
```

每个 concept 包含：

```json
{
  "canonical": "lung_sliding",
  "category": "artifact",
  "aliases": ["lung sliding", "sliding lung"]
}
```

对 history、target 和 prediction 分别抽取 concept set：

```text
H = history_concepts
T = target_concepts
P = predicted_concepts
```

其中：

- $H$：历史 interleaved narration 中已经出现过的概念；
- $T$：目标 sentence chunk 中应当预测到的概念；
- $P$：模型生成文本中出现的概念。

概念匹配：

$$
\mathrm{TP} = P \cap T
$$

$$
\mathrm{FP} = P - T - H
$$

$$
\mathrm{FN} = T - P
$$

这里从 FP 中减去 $H$，是因为模型复述历史中已经出现过的概念不一定是 hallucination。

concept precision：

$$
\mathrm{Precision}_{concept}
=
\frac{|\mathrm{TP}|}{|\mathrm{TP}| + |\mathrm{FP}|}
$$

concept recall：

$$
\mathrm{Recall}_{concept}
=
\frac{|\mathrm{TP}|}{|\mathrm{TP}| + |\mathrm{FN}|}
$$

concept F1：

$$
\mathrm{F1}_{concept}
=
\frac{
2 \cdot \mathrm{Precision}_{concept} \cdot \mathrm{Recall}_{concept}
}{
\mathrm{Precision}_{concept} + \mathrm{Recall}_{concept}
}
$$

hallucinated concept rate：

$$
\mathrm{HallucinationRate}_{concept}
=
\frac{|\mathrm{FP}|}{|P|}
$$

可以进一步按类别分别统计：

```text
anatomy F1
view F1
artifact F1
finding F1
procedure/action F1
```

并单独维护 critical concepts：

```text
pneumothorax
lung sliding
A-lines
B-lines
pleural effusion
Morrison's pouch
parasternal long axis
apical four chamber
IVC
free fluid
```

critical concept recall：

$$
\mathrm{Recall}_{critical}
=
\frac{|P_{critical} \cap T_{critical}|}{|T_{critical}|}
$$

这个指标更接近我们关心的 domain knowledge，而不是单纯 narration style。

---

## 11. 推荐命名

样本：

```text
cluster_data/pretrain/train_pretrain_punct_sentence_interleave_samples.jsonl
cluster_data/pretrain/eval_pretrain_punct_sentence_interleave_samples.jsonl
```

checkpoint：

```text
cluster_data/checkpoints/pretrain_qwen3vl_train50_punct_sentence_interleave
```

实验名：

```text
Pretrain_Interleave_Punctuation
```
