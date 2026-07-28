# SFT V1 — Qwen3-VL-2B-Instruct vs Pretrain→SFT Adapter Answerability 对比报告

本文记录第一次 Stage 2 SFT（WAIT/ANSWER answerability）完成后的推理结果。目标是判断：在 QA training samples 上，SFT 后的模型是否学会根据当前视频证据输出 `<WAIT>` 或 `<ANSWER>`。

本次实验结论比较明确：**SFT V1 发生了 WAIT collapse（全部输出 WAIT），效果比未 SFT 的Qwen3-VL-2B-Instruct更差。**

---

## 1. 实验背景

### 1.1 Stage 2 SFT 目标

SFT 的训练目标是 answerability-aware QA：

```text
当前视觉证据不足 → <WAIT>
当前视觉证据充分 → <ANSWER> answer
```

训练样本来自：

```text
azure_data/QA/results/8V649L5Q368_training_samples.jsonl
```

模型应学习：

- 对 `streaming_wait`：在 `query_time` 前证据不足，应输出 `<WAIT>`
- 对 `streaming_answer`：在 `answer_time` 前证据充分，应输出 `<ANSWER>`
- 对 `offline_answer`：完整 clip summary，始终应输出 `<ANSWER>`

---

## 2. 实验配置

| 项目 | 配置 |
|---|---|
| Base model | `Qwen/Qwen3-VL-2B-Instruct` |
| Pretrain adapter | `azure_data/checkpoints/pretrain_qwen3vl_bf16` |
| SFT adapter | `azure_data/checkpoints/qwen3vl_2b_sft_wait_answer` |
| Eval data | `azure_data/QA/results/8V649L5Q368_training_samples.jsonl` |
| Video | `azure_data/videos/8V649L5Q368.mp4` |
| Eval limit | `20` samples |
| Window size | `8` |
| Frame size | `336` |
| Precision | `bf16` |

---

## 3. 推理命令

### 3.1 SFT 后模型推理

由于 SFT 训练时使用了 `--init-adapter`，推理时也必须先 merge 同一个 pretrain adapter，再加载 SFT adapter：

```bash
python QA/eval/infer_qwen_lora.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --init-adapter azure_data/checkpoints/pretrain_qwen3vl_bf16 \
  --adapter-path azure_data/checkpoints/qwen3vl_2b_sft_wait_answer \
  --eval-jsonl azure_data/QA/results/8V649L5Q368_training_samples.jsonl \
  --default-video-path azure_data/videos/8V649L5Q368.mp4 \
  --output azure_data/checkpoints/sft_predictions_limit20.jsonl \
  --window-size 8 \
  --frame-size 336 \
  --limit 20 \
  --max-new-tokens 160 \
  --bf16
```

加载顺序：

```text
1. Qwen/Qwen3-VL-2B-Instruct base
2. merge pretrain adapter: azure_data/checkpoints/pretrain_qwen3vl_bf16
3. resize tokenizer / embeddings to include <WAIT>/<ANSWER>
4. load SFT adapter: azure_data/checkpoints/qwen3vl_2b_sft_wait_answer
5. run QA inference
```

---

### 3.2 未Pretrain & SFT Qwen3-VL-2B-Instruct推理

作为对照，也在同一批 20 条样本上运行未 SFT 的 Qwen3-VL-2B-Instruct：

```bash
python QA/eval/infer_qwen.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --eval-jsonl azure_data/QA/results/8V649L5Q368_training_samples.jsonl \
  --default-video-path azure_data/videos/8V649L5Q368.mp4 \
  --output azure_data/checkpoints/base_qwen_predictions_limit20.jsonl \
  --window-size 8 \
  --frame-size 336 \
  --limit 20 \
  --max-new-tokens 160 \
  --bf16
```

---

## 4. 数据构成

前 20 条样本包含：

| 范围 | 类型 | Ground truth |
|---|---|---|
| idx 0–9 | `offline_answer` / `clip_summary` | `ANSWER` |
| idx 10,12,14,16,18 | `streaming_wait` | `WAIT` |
| idx 11,13,15,17,19 | `streaming_answer` | `ANSWER` |

因此这 20 条中：

```text
ANSWER: 15
WAIT:   5
```

---

## 5. 定量结果

### 5.1 SFT V1 结果

SFT 后模型输出统计：

```text
answerability accuracy: 5/20 = 0.2500
pred label counts: {'WAIT': 20}
```

也就是说，SFT V1 对所有 20 条样本全部预测为：

```text
<WAIT> Not enough information yet. More video is needed.
```

结果：

| GT | 数量 | SFT prediction | 正确数 |
|---|---:|---|---:|
| ANSWER | 15 | 全部 WAIT | 0 |
| WAIT | 5 | 全部 WAIT | 5 |
| **总计** | **20** | — | **5/20** |

---

### 5.2 Qwen3-VL-2B-Instruct 结果

从 Qwen3-VL-2B-Instruct 推理输出可见，base 并非全 WAIT。前 20 条中大致为：

```text
pred WAIT   : 15
pred ANSWER : 5
accuracy    : 10/20 = 0.50
```

Base 对前 10 条 `offline_answer` 中部分样本能输出 `ANSWER`，也能正确预测一些 streaming WAIT。

| 模型 | Accuracy | Prediction distribution |
|---|---:|---|
| Qwen3-VL-2B-Instruct | ~10/20 = 0.50 | WAIT 15 / ANSWER 5 |
| SFT V1 | 5/20 = 0.25 | WAIT 20 / ANSWER 0 |

---

## 6. 核心结论

### 6.1 SFT V1 发生 WAIT collapse

SFT 后模型不是学会了更好的 answerability，而是坍塌成：

```text
无论输入是什么，都输出固定 WAIT
```

典型输出：

```text
<WAIT> Not enough information yet. More video is needed.<|im_end|>
```

这说明模型学会了 `<WAIT>` special token 和固定 WAIT 句式，但没有学会在证据充分时切换到 `<ANSWER>`。

---

### 6.2 SFT V1 比 base 更差

Qwen3-VL-2B-Instruct 在没有 SFT 的情况下仍能对部分 `offline_answer` 输出 `<ANSWER>` 类型内容，而 SFT V1 将这些样本也全部预测成 WAIT。

因此本次 SFT V1 的效果是：

```text
Qwen3-VL-2B-Instruct > SFT V1
```

这不是模型能力不足，而是训练数据和任务设计导致的 collapse。

---

## 7. 具体例子

### 7.1 Offline answer 被错误预测为 WAIT：idx=0

Question:

```text
What does this clip demonstrate, and what should a learner take away from it?
```

Ground truth:

```text
<ANSWER> The clip introduces the utility of lung ultrasound, particularly for diagnosing pneumothorax. The instructor first highlights that while emergency department colleagues often diagnose pneumothorax, it's crucial for hospitalists, especially after procedures where pneumothorax is a risk. He emphasizes that lung ultrasound is quick and boasts a very high negative predictive value, outperforming chest X-rays and approaching CT scans in its ability to confidently rule out pneumothorax. Finally, he prepares to demonstrate the scan, mentioning that he will use a linear probe, though other probes can also be effective for this assessment.
```

SFT prediction:

```text
<WAIT> Not enough information yet. More video is needed.<|im_end|>
```

判断：

```text
gt=ANSWER
pred=WAIT
correct=False
```

问题：offline clip summary 本应始终可回答，但 SFT V1 仍输出 WAIT。

---

### 7.2 Offline answer 被错误预测为 WAIT：idx=1

Question:

```text
What does this clip demonstrate, and what should a learner take away from it?
```

Ground truth:

```text
<ANSWER> This clip demonstrates how to diagnose a pneumothorax using lung ultrasound. First, the operator explains the importance of patient positioning, recommending a flat supine position to allow air to rise, making it easier to visualize the pleural interface. Then, he places a linear probe across two rib spaces on the patient's chest, showing a real-time ultrasound image where the bright white pleural interface is visible between the ribs. He highlights the normal "lung sliding" movement, which he likens to "ants on a log," indicating the absence of pneumothorax. Finally, the clip presents a side-by-side comparison of normal lung sliding with a static pleural interface, explaining that the absence of sliding suggests a pneumothorax due to air between the visceral and parietal pleura, although other conditions can also cause this absence, necessitating further investigation.
```

SFT prediction:

```text
<WAIT> Not enough information yet. More video is needed.<|im_end|>
```

判断：

```text
gt=ANSWER
pred=WAIT
correct=False
```

对比 Qwen3-VL-2B-Instruct：base 在同一条样本上能输出较合理的 `<ANSWER>` 内容：

```text
<ANSWER> This clip demonstrates the use of a transducer to perform a pleural ultrasound, specifically to assess the pleural interface. The key takeaway for a learner is that the pleural interface should show normal sliding, which is a sign of a healthy pleural space. The video shows two different states of the pleural interface: one with no sliding (which may indicate a problem) and one with normal sliding (which is normal). This helps learners understand the normal appearance of the pleural interface and how to interpret ultrasound images.<|im_end|>
```

说明 SFT 后反而丢失了 base 已有的回答倾向。

---

### 7.3 Streaming wait 被预测正确：idx=10

Question:

```text
At this point, what should the operator do next to begin the ultrasound scan?
```

Ground truth:

```text
<WAIT> Not enough information yet. More video is needed.
```

SFT prediction:

```text
<WAIT> Not enough information yet. More video is needed.<|im_end|>
```

判断：

```text
gt=WAIT
pred=WAIT
correct=True
```

这类样本被正确预测，但不是因为模型真正判断了证据不足，而是因为模型全部输出 WAIT。

---

### 7.4 Streaming answer 被错误预测为 WAIT：idx=11

Question:

```text
At this point, what should the operator do next to begin the ultrasound scan?
```

Ground truth:

```text
<ANSWER> The operator should grab a probe to begin the scan.
```

SFT prediction:

```text
<WAIT> Not enough information yet. More video is needed.<|im_end|>
```

判断：

```text
gt=ANSWER
pred=WAIT
correct=False
```

问题：同一个 question，在 `answer_time` 的视觉窗口中证据已经充分，模型仍然输出 WAIT，说明没有学到 query_time vs answer_time 的差异。

---

### 7.5 Streaming answer 被错误预测为 WAIT：idx=13

Question:

```text
Based on what we've seen so far, what type of probe will the operator use for this scan?
```

Ground truth:

```text
<ANSWER> The operator will use a linear probe for the scan.
```

SFT prediction:

```text
<WAIT> Not enough information yet. More video is needed.<|im_end|>
```

判断：

```text
gt=ANSWER
pred=WAIT
correct=False
```

这再次说明模型没有学到从 WAIT 切换到 ANSWER 的条件。

---

## 8. 原因分析

### 8.1 WAIT target 完全固定

当前训练数据中所有 `streaming_wait` target 都是同一句：

```text
<WAIT> Not enough information yet. More video is needed.
```

这使得模型非常容易记住固定输出，而不是学习具体证据缺失原因。

---

### 8.2 Offline answer target 过长且任务太难

`offline_answer` 的 target 是长 summary，通常数百 token。但输入只包含少量视频帧，不包含完整 ASR transcript。

因此模型训练时面对两类目标：

```text
短而固定的 WAIT target  → 很容易学
长而复杂的 ANSWER target → 很难学
```

优化过程自然偏向最容易降低 loss 的 WAIT 模式。

---

### 8.3 小数据集加剧过拟合

当前训练集只有 70 条样本，其中 30 条 streaming WAIT 使用完全相同的 target。小数据 + 固定 WAIT + 长答案目标共同导致 collapse。

---

## 9. 诊断结论

本次实验不是失败在推理脚本，也不是 `--init-adapter` 加载顺序问题。加载日志显示：

```text
merged pretrain adapter into base weights
loaded SFT adapter
<WAIT>/<ANSWER> ids present
```

因此模型状态是正确加载的。

问题在于：

```text
SFT V1 的训练目标设计导致模型学到了固定 WAIT 模式。
```

---

## 10. 下一步建议

### 10.1 先做 streaming-only SFT

去掉 `offline_answer`，只训练：

```text
streaming_wait
streaming_answer
```

这样任务更清晰：同一类问题在 `query_time` 应该 WAIT，在 `answer_time` 应该 ANSWER。

生成 streaming-only 数据：

```bash
python - <<'PY'
import json

inp = "azure_data/QA/results/8V649L5Q368_training_samples.jsonl"
out = "azure_data/QA/results/8V649L5Q368_streaming_train.jsonl"

n = 0
with open(inp, encoding="utf-8") as f, open(out, "w", encoding="utf-8") as g:
    for line in f:
        r = json.loads(line)
        if r.get("sample_type", "").startswith("streaming_"):
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1

print(f"wrote {n} streaming samples -> {out}")
PY
```

训练：

```bash
python QA/train/train.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --init-adapter azure_data/checkpoints/pretrain_qwen3vl_bf16 \
  --train-jsonl azure_data/QA/results/8V649L5Q368_streaming_train.jsonl \
  --default-video-path azure_data/videos/8V649L5Q368.mp4 \
  --output-dir azure_data/checkpoints/qwen3vl_2b_sft_streaming_e5 \
  --window-size 8 \
  --frame-size 336 \
  --num-train-epochs 5 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --learning-rate 2e-4 \
  --bf16 \
  --gradient-checkpointing \
  --early-stop-patience 3 \
  --early-stop-min-delta 0.001
```

推理：

```bash
python QA/eval/infer_qwen_lora.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --init-adapter azure_data/checkpoints/pretrain_qwen3vl_bf16 \
  --adapter-path azure_data/checkpoints/qwen3vl_2b_sft_streaming_e5 \
  --eval-jsonl azure_data/QA/results/8V649L5Q368_streaming_train.jsonl \
  --default-video-path azure_data/videos/8V649L5Q368.mp4 \
  --output azure_data/checkpoints/sft_streaming_e5_predictions.jsonl \
  --window-size 8 \
  --frame-size 336 \
  --limit 20 \
  --max-new-tokens 160 \
  --bf16
```

关键观察：

```text
pred label counts
```

只要不再是：

```text
{'WAIT': 20}
```

就说明已经缓解 collapse。

---

### 10.2 重新生成 wait_reason 版数据

正式版应该用新版 generator / merger，使 WAIT target 从固定句变为具体缺失证据：

```text
<WAIT> {specific missing evidence at query_time}
```

例如：

```text
<WAIT> The probe has not yet been placed on the chest, so lung sliding cannot be assessed.
```

这样模型不能靠背固定 WAIT 模板，需要学习不同问题下的具体证据缺失。

---

## 11. 当前产物

```text
SFT V1 adapter:
azure_data/checkpoints/qwen3vl_2b_sft_wait_answer

SFT V1 predictions:
azure_data/checkpoints/sft_predictions_limit20.jsonl

Qwen3-VL-2B-Instruct predictions:
azure_data/checkpoints/base_qwen_predictions_limit20.jsonl
```
