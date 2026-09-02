# 超声视频实时理解流程总结

## 1. 项目目标

我们的目标是构建一个面向超声视频的实时理解系统。模型在视频播放过程中持续接收超声画面和可能的旁白信息，并在用户提出问题时判断当前证据是否足够，并进行回答。

如果信息不足，模型应该输出：

```text
<WAIT> Specific reason
```

如果信息足够，模型应该输出：

```text
<ANSWER> Answer
```

当前主模型是：

```text
Qwen/Qwen3-VL-2B-Instruct + LoRA
```

---

## 2. 数据准备

目前爬取了 300 个视频，295 个成功下载，随机取 250 个用于训练，45 个用于评测：

```text
train_full295: 250 videos
eval_full295 : 45 videos
total        : 295 videos
```

视频总时长约：

```text
train: 39.49 h
eval : 11.89 h
total: 51.38 h
```

数据目录：

```text
/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding/cluster_data
```

---

## 3. ASR 转录

每个视频先通过 Whisper 生成 ASR transcript。

ASR job 已完成，当前 transcript 数量齐全：

```text
train: 250 / 250
eval : 45 / 45
```

这说明 full295 已经具备后续 pretrain 所需的文本监督。

---

## 4. ASR 质量过滤

ASR transcript 不直接全部使用，而是先通过规则过滤，去掉明显质量较差或不相关的转录。

过滤脚本：

```text
scripts/data/filter_asr_transcripts.py
```

当前过滤规则：

```text
language == en                  # 只保留英文视频
language_probability >= 0.8     # 是英文的概率 >= 80%，Whisper 转录 ASR 时给出
word_count >= 300               # 避免单词过少
num_segments >= 20              # 避免 ASR segments 过少
ultrasound keyword hits >= 2    # 至少有两个超声关键词
repeated_line_ratio <= 0.35     # ASR segment 重复比例
```

train 过滤结果：

```text
transcripts : 250
keep        : 196
drop        : 54
keep rate   : 78.4%
```

eval 过滤结果：

```text
transcripts : 45
keep        : 41
drop        : 4
keep rate   : 91.1%
drop reasons:
  num_segments_lt_20: 4
  word_count_lt_300: 4
```

输出文件包括：

```text
cluster_data/QA/train_full295/transcripts_asr_keep
cluster_data/splits/train_full295_asr_keep_videos.json
cluster_data/splits/train_full295_asr_drop_videos.json
cluster_data/QA/train_full295/asr_filter_audit.jsonl

cluster_data/QA/eval_full295/transcripts_asr_keep
cluster_data/splits/eval_full295_asr_keep_videos.json
cluster_data/splits/eval_full295_asr_drop_videos.json
cluster_data/QA/eval_full295/asr_filter_audit.jsonl
```

过滤后的 full295 ASR-keep 视频统计：

| split | videos | size | duration | mean duration |
|---|---:|---:|---:|---:|
| train | 196 | 13.019 GiB / 13.979 GB | 37.825 h / 37:49:31 | 11.58 min/video |
| eval | 41 | 1.040 GiB / 1.116 GB | 11.771 h / 11:46:17 | 17.23 min/video |
| total | 237 | 14.058 GiB / 15.095 GB | 49.597 h / 49:35:48 | 12.56 min/video |

相比原始 full295：

| metric | original full295 | ASR-keep | kept ratio |
|---|---:|---:|---:|
| videos | 295 | 237 | 80.3% |
| duration | 51.38 h | 49.60 h | 96.5% |
| size | 14.726 GiB | 14.058 GiB | 95.5% |

结论：ASR 过滤删除了 58 个视频，但只损失约 1.78 小时视频，说明被过滤掉的大多是短视频或 ASR 内容不足的视频。

统计文件：

```text
cluster_data/splits/full295_asr_keep_video_stats.json
```

---

## 5. Pretrain

### 目的

通过大量超声视频 + ASR narration，把 Qwen3-VL 从通用视觉语言模型，进一步适配到超声影像场景，让它学会超声视频里的视觉模式、医学术语、检查流程和时序语义。

不是单纯做一个 ASR 文本续写模型，也不是简单做 captioning，而是做 **ultrasound-domain visual-text alignment pretraining**。

### 数据构建

过滤后的 ASR transcript 会被转成 pretrain samples。

当前主线是：

```text
V4 mixedmask sentence-level pretrain
```

也就是以句子为单位构造样本，而不是固定 10 秒 chunk。

样本形式大致是：

```text
history visual-text context
+ current visual context
-> target sentence
```

构建参数：

```text
unit: sentence
sentence mode: punctuation
format: mixedmask
history units: 3
frames per sentence: 3
min words: 3
sentence max words: 80
```

掩码策略：

```text
mask current visual probability: 0.33  # 遮蔽当前视频，只用历史 context 预测下一句话
no mask probability            : 0.33  # 不遮蔽，用历史 context 和当前视频预测下一句话
text modality mask probability : 0.34  # 遮蔽历史 context 里的所有文本，强迫模型通过视频生成下一句话
```

---

## 6. 视频帧处理

视频帧采样已经从原来的直接拉伸成正方形，改成：

```text
等比例缩放 + 黑色 padding 填充
```

也就是先保持原始宽高比 resize，再用黑边补齐到模型输入尺寸。

当前 Pretrain 和 SFT 统一使用相同的图像输入尺寸：

```text
224 × 224 RGB
```

具体处理方式是：先把原始帧按比例缩放，使长边变成 224；然后在短边方向用黑色 padding 补齐，最终得到 `224 × 224` 的正方形图像。

例如原始帧为 `640 × 360` 时：

```text
640 × 360
-> 等比例缩放到 224 × 126
-> 上下黑色 padding 到 224 × 224
```

这样可以保留超声图像原始形态，避免由于强行拉伸导致解剖结构和探头视野变形。

涉及模块：

```text
pretrain/video_sampling.py
QA/train/video_sampling.py
```

---

## 7. 当前 Pretrain 主线

训练：

```text
pretrain_v4_mixedmask_qwen3vl_2b_full295_asr_keep
```

使用：

```text
Qwen/Qwen3-VL-2B-Instruct + LoRA
```

训练目标是进行超声领域知识注入，让模型把超声视频中的视觉模式、医学术语、检查流程和时序语义与 ASR narration 对齐：

```text
超声视频帧
+ ASR narration history
-> 当前或下一段 narration
```

这一步的核心不是让模型单纯学习 ASR 文本续写，而是提升模型对超声视频画面、教学讲解文本以及二者对应关系的领域理解能力。

---

## 8. 如何评估 Pretrain 效果

Pretrain 本身不是最终的 QA 任务，所以评估重点不是看模型能不能直接回答临床问题，而是看它是否真正完成了“超声领域知识注入”。

也就是说，我们主要比较：

```text
原始 Qwen3-VL
vs
经过 Pretrain 的 Qwen3-VL + LoRA
```

在相同的 eval 视频、相同的输入条件下，观察经过 Pretrain 的模型是否更能理解超声画面，并生成更合理、更专业、更符合视频证据的描述。

评估分成两层：

```text
1. 自动文本指标
2. 不同 mask 条件下的 base vs pretrain 生成对比，并用大模型 blind judge 盲评
```

### 8.1 自动文本指标

使用过滤后的 eval ASR-keep samples 进行生成式评估：

```text
cluster_data/pretrain/eval_full295_asr_keep_punct_sentence_mixedmask_samples.jsonl
```

每条样本输入包括：

```text
历史超声视频帧
+ 历史 ASR narration
+ 当前超声视频帧
```

模型生成：

```text
当前或下一段 narration
```

然后和 ASR target narration 做文本级对比，主要统计两个指标：

| 指标 | 含义 |
|---|---|
| Word overlap F1 | 衡量 prediction 和 target 的词级重合度 |
| ROUGE-L | 衡量 prediction 和 target 在句子结构 / 最长公共子序列上的相似度 |

这一步的作用是给出一个可重复、低成本的自动量化结果。

预期结果是：

```text
Pretrain 后模型的 Word overlap F1 高于原始模型
Pretrain 后模型的 ROUGE-L 高于原始模型
```

需要注意的是，这里的目标不是要求模型逐字复现 ASR。Word overlap F1 和 ROUGE-L 只能衡量 prediction 与 target 的文本相似度，不能完全代表医学正确性。因此还需要结合下面的 blind judge 做语义和医学合理性评估。

### 8.2 不同 mask 条件下的生成对比

在相同 eval samples 上，分别测试三种 mixedmask eval mode：

```text
no_mask
mask_current_visual
text_modality_mask
```

每种条件下都让两个模型生成：

```text
原始 Qwen3-VL 生成 Prediction Base
Pretrain 后模型生成 Prediction Pretrain
```

三种条件的意义是：

| eval mode | 输入条件 | 目的 |
|---|---|---|
| `no_mask` | 历史文本 + 历史视频帧 + 当前视频帧 | 测完整信息下的超声视觉-语言对齐能力 |
| `mask_current_visual` | 历史文本 + 历史视频帧，遮蔽当前视频帧 | 测检查流程和时序预测能力 |
| `text_modality_mask` | 历史视频帧 + 当前视频帧，遮蔽历史 narration 文本 | 测模型是否真正利用超声视觉信息 |

其中最关键的是：

```text
text_modality_mask
```

因为它能检查模型是不是只学会了 ASR 文本续写。如果遮蔽历史文本后，Pretrain 模型仍然比 base 生成得更贴近超声画面，说明它确实学到了超声视觉知识。

### 8.3 大模型 Blind Judge 盲评

自动指标只能衡量文本相似度，不能可靠判断医学合理性。因此还需要使用大模型做 blind A/B judge：

```text
pretrain/llm_judge_predictions.py
```

盲评流程：

```text
1. 对同一批 eval samples，分别用 base model 和 pretrained model 生成结果
2. 每条样本随机打乱两个输出，记为 Prediction A 和 Prediction B
3. Judge 模型只看到 A/B，不知道哪个来自 base，哪个来自 pretrain
4. Judge 根据 target narration、历史 context 和评估标准判断哪个更好
5. 脚本再把 A/B 映射回 base/pretrain，统计胜率
```

Judge 依据：

```text
1. 是否更接近 target narration 的语义
2. 是否更符合超声画面和上下文
3. 是否使用更准确的医学/超声术语
4. 是否更像真实超声教学视频旁白
5. 是否更少出现泛泛描述或 hallucination
```

最终统计：

| 指标 | 含义 |
|---|---|
| pretrain win rate | Pretrain 模型被判更好的比例 |
| base win rate | 原始模型被判更好的比例 |
| tie rate | 两者打平的比例 |
| both_bad rate | 两者都差的比例 |
| mean score delta | Pretrain 分数 - Base 分数 |

### 8.4 最终判断标准

如果结果满足：

```text
1. Pretrain 模型的 Word overlap F1 更高
2. Pretrain 模型的 ROUGE-L 更高
3. 在 no_mask / mask_current_visual / text_modality_mask 三种条件下，Pretrain 都比 base 更好
4. 尤其在 text_modality_mask 下，Pretrain 仍有明显优势
5. Blind Judge 中 Pretrain win rate 高于 base win rate
```

就可以说明：Pretrain 不只是让模型学会了 ASR 文本续写，而是完成了一定程度的超声领域知识注入。

---

## 9. Streaming QA / SFT

Pretrain 后，模型进入 QA/SFT 阶段。

在实时 QA 中，模型持续观看视频流，并在不同时间点面对问题。

输出格式是：

```text
<WAIT> reason
```

或者：

```text
<ANSWER> answer
```

含义：

```text
<WAIT>   当前证据不足，需要继续看视频
<ANSWER> 当前证据充分，可以回答问题
```

SFT 的目标是让模型学会：

```text
什么时候该等
什么时候该答
以及如何基于视频证据回答
```

---

## 10. 评估方式

评估分两类。

### 自动评估

统计：

```text
WAIT / ANSWER 判断准确率
过早回答
过晚回答
最终答案质量
```

相关脚本：

```text
QA/eval/analyze_predictions.py
```

### Blind LLM Judge

新增了 blind A/B judge：

```text
pretrain/llm_judge_predictions.py
```

LLM 只看到：

```text
Prediction A
Prediction B
```

不知道哪个来自 base，哪个来自 pretrain。

脚本内部再把 A/B 映射回：

```text
base
pretrain
```

推荐 judge model：

```text
google/gemini-2.5-flash
```

---

## 11. 未来方向：Recurrent Summary Token Pretrain

未来计划新增：

```text
Pretrain V5-RecurrentSummary
```

核心引入一个 special token：

```text
<SUMMARY>
```

它不是文本摘要，而是 hidden-state memory。

每个视频 chunk 更新一次状态：

```text
S_{t-1} + V_t + T_t + <SUMMARY> -> S_t
```

然后用 summary state 预测下一段 narration：

```text
S_t + V_{t+1} -> T_{t+1}
```

训练 loss：

```text
CE(T_{t+1})
```

这样模型不能直接读取完整历史，只能依赖 `S_t`，因此 `<SUMMARY>` 会被迫学习压缩 past context。

---

## 12. 未来方向：Summary + Decide Streaming QA

QA 阶段可以进一步引入：

```text
<SUMMARY> = query-agnostic video memory
<DECIDE>  = query-aware answerability decision token
```

流程：

```text
1. 视频流持续更新 S_t
2. 用户问题 Q 到来
3. 构造 S_t + current visual + Q + <DECIDE>
4. 读取 <ANSWER> 和 <WAIT> logits
5. 用 logit(<ANSWER>) - logit(<WAIT>) 判断能否回答
```

也就是：

```text
answerability_logit = logit(<ANSWER>) - logit(<WAIT>)
```

如果偏向 `<WAIT>`：

```text
输出 <WAIT> reason
```

如果偏向 `<ANSWER>`：

```text
输出 <ANSWER> answer
```

第一版不额外加 binary head，而是直接利用 LM token logits。

---

## 13. 当前状态

已经完成：

```text
full295 split
ASR transcript generation
train ASR filtering
eval ASR filtering
filtered ASR-keep duration/size statistics
pretrain / ASR 工具脚本
letterbox video resize
blind LLM judge
future summary-token design docs
```

已 push 的相关 commits：

```text
899a935 pretrain: add data utilities and blind llm judge
1f3b3a6 asr: reuse whisper model in batch and add full295 job
279ae74 docs: add recurrent summary streaming qa designs
```

当前还需要做：

```text
1. build train/eval pretrain samples
2. launch full295 V4 mixedmask pretrain
3. evaluate pretrain effect on streaming QA
4. later explore V5 recurrent summary token architecture
```

---

## 14. 总流程

```text
full295 videos
↓
Whisper ASR
↓
rule-based ASR filtering
↓
sentence-level mixedmask pretrain sample construction
↓
Qwen3-VL-2B + LoRA V4 pretrain
↓
streaming QA / SFT with <WAIT>/<ANSWER>
↓
automatic metrics + blind LLM judge
```

未来增强：

```text
<SUMMARY> recurrent memory pretrain
↓
<SUMMARY> + <DECIDE> streaming answerability decision
```

最终目标是让模型能够在长超声视频流中持续维护视频理解状态，并在问题出现时判断当前证据是否足够，从而决定继续等待还是立即回答。