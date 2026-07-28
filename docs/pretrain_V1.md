# Pretrain V1 — Base vs Pretrained Adapter 对比报告

本文记录第一次 Stage 1 pretrain（ASR caption completion）完成后的推理对比结果。目标是判断：相对于未训练的 `Qwen/Qwen3-VL-2B-Instruct` base model，预训练 LoRA adapter 是否让模型更像“实时超声教学解说员”，学习超生的domain knowledge。

---

## 1. 实验目的

Stage 1 pretrain 的任务不是回答 QA，也不是学习 `<WAIT>` / `<ANSWER>` 决策，而是：

```text
给定当前时间之前最近 N 帧 ultrasound video frames
+ 可选 prev_context（此前 ASR narration）
→ 续写当前 ASR segment 的 narration
```

因此本次评估关注：

1. 未训练 base model 和 pretrained adapter 在同一批 pretrain samples 上的输出差异。
2. pretrained adapter 是否更贴近 target ASR narration。
3. pretrained adapter 是否减少 base model 的“长篇泛化 / 教科书式发散”，转向更短、更口语化、更像现场解说的输出。

---

## 2. 实验配置

| 项目 | 配置 |
|---|---|
| Base model | `Qwen/Qwen3-VL-2B-Instruct` |
| Pretrain adapter | `azure_data/checkpoints/pretrain_qwen3vl_bf16` |
| Eval samples | `pretrain/data/pretrain_samples.jsonl` |
| Video | `8V649L5Q368` |
| Window size | `4` frames |
| Frame size | `224` |
| Eval limit | `20` samples |
| Precision | `bf16` |
| 输出目录 | `azure_data/checkpoints/` |

---

## 3. 推理命令

### 3.1 Base model（未训练）推理

```bash
cd /home/azureuser/live-ultrasound-video-understanding
conda activate azureml_py38

python pretrain/infer.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --no-adapter \
  --eval-jsonl pretrain/data/pretrain_samples.jsonl \
  --video-path-map pretrain/data/video_path_map.json \
  --output azure_data/checkpoints/pretrain_pred_BASE_limit20.jsonl \
  --window-size 4 \
  --frame-size 224 \
  --limit 20 \
  --max-new-tokens 128 \
  --bf16
```

### 3.2 Pretrained adapter 推理

```bash
python pretrain/infer.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --adapter-path azure_data/checkpoints/pretrain_qwen3vl_bf16 \
  --eval-jsonl pretrain/data/pretrain_samples.jsonl \
  --video-path-map pretrain/data/video_path_map.json \
  --output azure_data/checkpoints/pretrain_predictions_limit20.jsonl \
  --window-size 4 \
  --frame-size 224 \
  --limit 20 \
  --max-new-tokens 128 \
  --bf16
```

### 3.3 对比脚本

```bash
python pretrain/compare_predictions.py \
  --base azure_data/checkpoints/pretrain_pred_BASE_limit20.jsonl \
  --lora azure_data/checkpoints/pretrain_predictions_limit20.jsonl \
  --output azure_data/checkpoints/pretrain_compare_limit20.jsonl \
  --embed-device cpu
```

> 语义余弦使用 `sentence-transformers/all-MiniLM-L6-v2`。由于云端 `torchcodec` 与当前 PyTorch/CUDA 组合存在兼容问题，运行前曾临时处理环境，使 sentence-transformers 可以正常加载文本 embedding 模型。

---

## 4. 定量结果

### 4.1 Word-overlap F1

词重叠 F1 是一个简单指标：比较 prediction 和 target 的词集合重叠，范围 0~1。它不理解语义，只看词是否重合。

```text
BASE (untrained) : 0.138
PRETRAIN adapter : 0.198
delta            : +0.060
```

相对提升：

```text
0.060 / 0.138 ≈ 43%
```

说明 pretrained adapter 的输出在用词层面更接近真实 ASR narration。

### 4.2 Semantic cosine similarity

语义余弦使用句向量模型计算 prediction 与 target 的语义相似度。

```text
BASE (untrained) : 0.441
PRETRAIN adapter : 0.449
delta            : +0.008
```

语义余弦提升较小，说明 base model 本身已有较强的医学 / lung ultrasound 语义知识；pretraining 的主要收益不完全体现在“是否知道医学概念”，而更多体现在输出风格、长度和局部 narration progression 上。

---

## 5. 典型样本分析

### 5.1 明显正例：idx=7

Target:

```text
To get started, we'll start with pneumothorax.
```

Base:

```text
Now, let's talk about the key features of the ultrasound image...
```

Pretrain:

```text
So let's start with pneumothorax.
```

指标：

```text
BASE     f1=0.033  cos=0.336
PRETRAIN f1=0.429  cos=0.931
```

解读：pretrained adapter 几乎准确学到了当前教学 narration 的推进方式，而 base model 直接发散到影像特征描述。

---

### 5.2 正例：idx=10

Target:

```text
But it can be useful, especially after performing a procedure where pneumothorax is possible.
```

Base:

```text
However, when a patient presents with a sudden onset of chest pain...
```

Pretrain:

```text
But pneumothorax is a condition where air leaks into the pleural space.
```

指标：

```text
BASE     f1=0.137  cos=0.561
PRETRAIN f1=0.385  cos=0.679
```

解读：pretrained adapter 更贴近当前 pneumothorax 叙述主题，尽管没有逐字复现 target。

---

### 5.3 正例：idx=11

Target:

```text
Now using ultrasound to look for pneumothorax is really nice because it's quick
```

Base:

```text
The ultrasound is a great tool for detecting pneumothorax...
```

Pretrain:

```text
So today we're going to look at how to use ultrasound to look for pneumothorax.
```

指标：

```text
BASE     f1=0.211  cos=0.713
PRETRAIN f1=0.370  cos=0.872
```

解读：pretrained adapter 更像口语化教学过渡句。

---

### 5.4 正例：idx=12

Target:

```text
and it has a very, very high negative predictive value.
```

Base:

```text
and it's non-invasive...
```

Pretrain:

```text
and it's very sensitive.
```

指标：

```text
BASE     f1=0.115  cos=0.096
PRETRAIN f1=0.429  cos=0.335
```

解读：pretrained adapter 学到了这段 narration 在讲 test performance / sensitivity / predictive value，而 base model 发散到更一般的超声优点。

---

### 5.5 正例：idx=19

Target:

```text
though to be honest with you, you could do this scan with many different probes and it works.
```

Base:

```text
...using a linear probe, which is ideal for assessing the heart...
```

Pretrain:

```text
but you can use a curved probe,
```

指标：

```text
BASE     f1=0.167  cos=0.486
PRETRAIN f1=0.087  cos=0.601
```

解读：词重叠 F1 不一定更高，但语义余弦显示 pretrain 更接近“不同 probe 都可用”的 target 语义。

---

## 6. 失败与局限

### 6.1 早期帧信息不足导致 topic guess

idx=0 / idx=1：

- Target 是 Stanford 25 / lung ultrasound introduction。
- Pretrain 预测为 liver / abdomen。
- 原因可能是视频最开始窗口非常短（如 `[0.0, 0.7]`），视觉信息和上下文都不足，模型依赖先验猜测。

示例：

```text
TARGET   : Today we're going to be learning about lung ultrasound.
PRETRAIN : Today we're going to be looking at the abdomen.
```

### 6.2 Pretrain 有时仍然输出泛化句

例如：

```text
It's very, very easy to do.
It's very, very safe.
and that's really important.
```

这些句子风格上像教学 narration，但信息量不足，说明模型学到了“语气”，但还没有稳定学到精确的下一句内容。

### 6.3 Semantic cosine 提升小

语义余弦只从 0.441 提升到 0.449，原因包括：

1. Base model 本身已经具备较强医学知识。
2. Many base predictions are semantically related to pneumothorax/lung ultrasound, even when overly verbose.
3. Sentence embedding model 对冗长文本可能给较高语义分，即使它不适合做逐句 narration continuation。
4. 本次只评估 20 条，样本量小。

---

## 7. 结论

Pretrain V1 是有效的。

主要证据：

1. **Word-overlap F1 明显提升**：
   ```text
   0.138 → 0.198 (+0.060, +43% relative)
   ```

2. **输出风格明显改善**：
   - Base model 倾向生成长篇、泛化、教科书式段落。
   - Pretrained adapter 倾向生成更短、更口语化、更接近教学视频 narration 的句子。

3. **局部 narration progression 更好**：
   - 在 "start with pneumothorax"、"look for pneumothorax"、"probe choice" 等局部上下文中，pretrain 更容易接上当前教学流程。

4. **语义知识不是主要瓶颈**：
   - Base 已经知道很多 lung ultrasound / pneumothorax 概念。
   - Pretrain 的收益更像是 domain narration style alignment 和 local continuation alignment。

---

## 8. 下一步

### 8.1 Stage 2 SFT

使用该 pretrain adapter 作为 SFT 起点：

```bash
python QA/train/train.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --init-adapter azure_data/checkpoints/pretrain_qwen3vl_bf16 \
  --train-jsonl azure_data/QA/results/8V649L5Q368_training_samples.jsonl \
  --default-video-path azure_data/videos/8V649L5Q368.mp4 \
  --output-dir azure_data/checkpoints/qwen3vl_2b_sft_wait_answer \
  --window-size 8 --frame-size 336 \
  --num-train-epochs 3 \
  --per-device-train-batch-size 1 --gradient-accumulation-steps 4 \
  --learning-rate 2e-4 --bf16 --gradient-checkpointing \
  --early-stop-patience 3 --early-stop-min-delta 0.001
```

### 8.2 改进 Pretrain

后续可考虑：

1. 增加更多视频，避免单视频过拟合。
2. 过滤最早期信息不足的 samples（如 window < 2s）。
3. 增大上下文稳定性，减少 topic guess。
4. 做不同 window/frame 配置消融。
5. 用更可靠的 semantic metric 或人工评分评估 narration quality。

---

## 9. 产物路径

```text
Base predictions:
azure_data/checkpoints/pretrain_pred_BASE_limit20.jsonl

Pretrain adapter predictions:
azure_data/checkpoints/pretrain_predictions_limit20.jsonl

Merged comparison:
azure_data/checkpoints/pretrain_compare_limit20.jsonl

Pretrain adapter:
azure_data/checkpoints/pretrain_qwen3vl_bf16
```
