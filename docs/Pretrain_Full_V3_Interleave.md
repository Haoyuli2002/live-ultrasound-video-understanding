# Pretrain Full V3: Sentence Interleave 设计总结

## 1. 目标

Pretrain Full V3 的目标是把 pretrain 从普通 narration continuation 改成更明确的 **图文交错自回归预测**：

```text
前面句子的视觉帧 + 前面句子的 narration 文本
→ 预测下一句 narration
```

相比 V1/V2，V3 更强调：

```text
visual progression + narration progression
```

让模型学习句子文本和对应超声画面之间的时序关系。

---

## 2. 设计核心

V3 不再按 ASR segment 作为基本单位，而是按 **sentence / sentence-like unit** 作为基本单位。

每个句子有：

```text
sentence_start
sentence_end
sentence_text
```

每个句子在自己的时间区间内均匀采样 3 帧：

```text
frames_i = uniform_sample(video, [sentence_start_i, sentence_end_i], n=3)
```

训练样本采用自回归形式：

```text
句子1的3帧 + 句子1
→ 预测句子2

句子1的3帧 + 句子1
+ 句子2的3帧 + 句子2
→ 预测句子3

句子1的3帧 + 句子1
+ 句子2的3帧 + 句子2
+ 句子3的3帧 + 句子3
→ 预测句子4
```

为了控制输入长度，使用 sliding history window：

```text
--history-units 3
```

即最多只保留最近 3 个历史句子。

---

## 3. 和 V1 / V2 的区别

| 版本 | 基本单位 | 输入 | 目标 |
|---|---|---|---|
| V1 | ASR segment | 当前 segment 前的视频窗口 + 前文 ASR | 当前 ASR segment |
| V2 | sentence-like unit | 当前句子前的视频窗口 + 前文句子 | 当前 sentence-like unit |
| V3 | sentence-like unit | 历史句子的帧 + 历史句子的文本交错输入 | 下一句 sentence-like unit |

V3 的关键变化：

```text
不再把 prev_context 作为纯文本块
而是把每个历史句子和它对应的视频帧绑定起来
```

---

## 4. 样本 schema

V3 sample type：

```text
pretrain_caption_sentence_interleave
```

示例：

```json
{
  "sample_type": "pretrain_caption_sentence_interleave",
  "video_id": "demo",
  "history": [
    {
      "sentence_idx": 0,
      "text": "Sentence one shows the probe being placed on the chest.",
      "video_window": [0.0, 2.0],
      "num_frames": 3,
      "segment_start_idx": 0,
      "segment_end_idx": 0
    },
    {
      "sentence_idx": 1,
      "text": "Sentence two explains that lung sliding is visible.",
      "video_window": [2.0, 4.0],
      "num_frames": 3,
      "segment_start_idx": 1,
      "segment_end_idx": 1
    }
  ],
  "video_window": [4.0, 6.0],
  "prev_context": "Sentence one ... Sentence two ...",
  "target": "Sentence three discusses A-lines and pleural artifacts.",
  "meta": {
    "unit": "sentence",
    "format": "interleave",
    "target_sentence_idx": 2,
    "target_sentence_start": 4.0,
    "target_sentence_end": 6.0,
    "history_units": 2,
    "frames_per_sentence": 3
  }
}
```

注意：

- `history` 中每个句子都有自己的 `video_window`。
- 每个历史句子均匀采 3 帧。
- target 句子本身不提供帧，避免泄漏答案。
- `video_window` 记录 target 句子的时间范围，主要用于 metadata / 分析。

---

## 5. Prompt 形式

Collator 会把 V3 sample 构造成如下 interleaved prompt：

```text
[history sentence 1 frame 1]
[history sentence 1 frame 2]
[history sentence 1 frame 3]
Narration: sentence 1

[history sentence 2 frame 1]
[history sentence 2 frame 2]
[history sentence 2 frame 3]
Narration: sentence 2

[history sentence 3 frame 1]
[history sentence 3 frame 2]
[history sentence 3 frame 3]
Narration: sentence 3

Continue the narration:
```

assistant target：

```text
sentence 4
```

loss 仍然只计算 assistant target 部分。

---

## 6. 句子切分模式

V3 支持两种 sentence unit 构建方式。

### 6.1 `segment_merge`

旧的 sentence-like 构建方式：

```text
逐个 ASR segment 累积
遇到句末标点或 word/time fallback 后切分
```

命令参数：

```bash
--sentence-mode segment_merge
```

优点是简单、时间对齐直接；缺点是 fallback 可能在自然句中间切开。

### 6.2 `punctuation`

新的推荐方式：

```text
先拼接整个 ASR transcript
再按标点符号切分
最后把切分出来的文本 span 映射回原 ASR segment 时间范围
```

命令参数：

```bash
--sentence-mode punctuation
--split-punctuation ".?!;:"
```

这个模式可以修复类似：

```text
Massachusetts General
Hospital.
```

被 ASR segment 切开的情况。

如果需要更细粒度，也可以打开逗号切分：

```bash
--include-comma-split
```

但默认不建议按逗号切，因为会让单位过碎。

---

## 7. 已实现代码改动

### `pretrain/build_samples.py`

新增参数：

```bash
--format standard|interleave
--history-units 3
--frames-per-sentence 3
```

V3 生成命令：

```bash
python pretrain/build_samples.py \
  --transcripts cluster_data/QA/train/transcripts \
  --output cluster_data/pretrain/train_pretrain_sentence_interleave_samples.jsonl \
  --unit sentence \
  --sentence-mode punctuation \
  --format interleave \
  --history-units 3 \
  --frames-per-sentence 3 \
  --min-words 3 \
  --sentence-max-words 80 \
  --split-punctuation ".?!;:"
```

### `pretrain/video_sampling.py`

新增：

```python
sample_uniform_n_frames(...)
```

用于在每个句子的 `[sentence_start, sentence_end]` 内均匀采样 3 帧。

### `pretrain/dataset.py`

新增对 V3 sample 的支持：

```text
history_frames = [
  sentence_1 的 3 帧,
  sentence_2 的 3 帧,
  ...
]
```

### `pretrain/collator.py`

新增：

```python
build_interleave_messages(...)
```

用于构造图像帧和 narration 交错的 multimodal prompt。

### `pretrain/infer.py`

新增对 interleave samples 的 inference 支持。

---

## 8. Smoke Test 结果

### 8.1 Interleave smoke test

toy transcript：

```text
sentence 1
sentence 2
sentence 3
sentence 4
```

生成 3 条 V3 samples：

```text
sample 1:
sentence 1 的 3 帧 + sentence 1 -> target sentence 2

sample 2:
sentence 1 的 3 帧 + sentence 1
sentence 2 的 3 帧 + sentence 2
-> target sentence 3

sample 3:
sentence 1 的 3 帧 + sentence 1
sentence 2 的 3 帧 + sentence 2
sentence 3 的 3 帧 + sentence 3
-> target sentence 4
```

这符合预期设计。

### 8.2 Punctuation split smoke test

toy ASR segments：

```text
segment 0: This is Massachusetts General
segment 1: Hospital. Next sentence starts
segment 2: here and continues.
```

使用：

```bash
--sentence-mode punctuation
--format interleave
```

生成结果会把跨 segment 的短语合并为：

```text
This is Massachusetts General Hospital.
```

并预测下一句：

```text
Next sentence starts here and continues.
```

这验证了 punctuation 模式可以避免在 `Massachusetts General / Hospital` 中间断开。

---

## 9. 推荐命名

样本文件：

```text
cluster_data/pretrain/train_pretrain_sentence_interleave_samples.jsonl
cluster_data/pretrain/eval_pretrain_sentence_interleave_samples.jsonl
```

checkpoint：

```text
cluster_data/checkpoints/pretrain_qwen3vl_train50_sentence_interleave
```

实验名：

```text
Pretrain_Full_V3_sentence_interleave
```

---

## 10. 设计定位

V3 相比 V1/V2 更接近图文时序建模：

```text
past visual-text sequence -> next narration sentence
```

但它仍然不是最终的 ultrasound visual domain knowledge supervision。

更准确的定位是：

```text
ultrasound visual-text temporal narration pretraining
```

后续如果目标是 domain knowledge，还需要更显式的任务，例如：

```text
video clip -> anatomy / view / artifact / finding
video clip + question -> grounded answer
multiple-choice visual grounding
structured ultrasound observation generation
```
