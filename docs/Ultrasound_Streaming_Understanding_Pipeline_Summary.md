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

需要注意：Qwen3-VL 本身支持 dynamic resolution，并不要求输入必须固定为 `224 × 224`。当前工程实现中固定使用 `224 × 224` 是一个 baseline setting，主要原因是方便控制显存占用，避免后续使用更大分辨率或动态高分辨率视频帧训练时出现 OOM。同时，固定尺寸也让 Pretrain 和 SFT 的视频输入分布保持一致，便于排查训练问题。

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

后续如果显存允许，可以再实验更高固定分辨率或 Qwen3-VL 原生 dynamic resolution / `max_pixels` 设置，比较细节保留、训练速度、显存占用和 QA 效果之间的 trade-off。

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

## 9. Streaming QA / SFT：基于 Summary Bank 和 `<WAIT>/<ANSWER>`

Pretrain 后，完整路线进入三阶段训练中的后两步：Information Compression / Learn Summary Bank，然后进入 Streaming QA / SFT。

这一阶段的目标不是把完整历史视频帧和文本全部塞给模型，而是让模型在视频播放过程中持续维护一个 query-agnostic 的视频理解状态：

```text
B_t = [S_{t-(K-1)Δ}, ..., S_t]
```

这里 `B_t` 是时间 `t` 的 sliding temporal summary bank。`K` 是最多保留的 summary token 数量；例如 `K=20, Δ=10s` 时，`B_200=[S_10,S_20,...,S_200]`；`S_210` 加入后移出 `S_10`，得到 `B_210=[S_20,...,S_210]`。`B_t` 不针对某个具体问题，而是持续压缩到当前时刻为止的视频和 narration 历史，例如：

```text
已经看到的解剖结构
已经出现的超声征象
当前扫查部位
探头移动和检查流程
已经积累的视觉证据
```

### 9.1 动态更新 Summary Bank

视频流按 chunk 持续进入模型。每来一个新 chunk，模型更新一次 summary bank：

```text
B_t = [S_{t-(K-1)Δ}, ..., S_t]
S_t = hidden_state(<SUMMARY> | B_{t-Δ}, V_t, optional T_t)
B_t = append(B_{t-Δ}, S_t)
if len(B_t) > K: B_t = keep_last_K(B_t)
```

其中：

```text
S_{t-1} = 上一时刻的 summary bank
V_t     = 当前视频 chunk 的超声帧
T_t     = 当前 chunk 的 ASR narration，可选
S_t     = 更新后的 query-agnostic summary bank
```

这样模型不需要每次重新读取完整历史，而是通过滑动窗口 summary bank `B_t` 持续维护视频流的理解状态。

### 9.2 Query 到来后的 Answerability 判断

当用户在时间 `t` 提出问题 `Q` 时，模型使用当前 summary bank 和当前视觉证据来判断是否可以回答：

```text
S_t + current visual + Q
```

本版本不引入额外分类模块。answerability 由语言模型的首 token 直接决定：

```text
p_WAIT   = P(<WAIT>   | S_t, V_t, Q)
p_ANSWER = P(<ANSWER> | S_t, V_t, Q)
```

如果 `<WAIT>` 概率更高，说明当前证据不足；如果 `<ANSWER>` 概率更高，说明当前证据足够。

### 9.3 `<WAIT>/<ANSWER>` 的分工

`<WAIT>` 和 `<ANSWER>` 既是输出格式前缀，也是 answerability decision token。模型通过普通 next-token prediction 学会在当前 summary bank、视觉证据和 query 下应该等待还是回答。

| token/state | 作用 |
|---|---|
| `S_t=[s_t^1,...,s_t^K]` | 不依赖 query 的视频历史 summary bank |
| `<WAIT>` | 首 token 决策：当前证据不足，并生成等待原因 |
| `<ANSWER>` | 首 token 决策：当前证据充分，并生成答案 |

训练时只需要普通语言生成 loss：

```text
loss = CE(target tokens)
```

### 9.4 WAIT 后继续看视频并更新 Summary

如果模型输出 `<WAIT>`，系统不会结束，而是继续接收后续视频 chunk，并继续更新 summary bank：

```text
S_{t+1} = hidden_state(<SUMMARY> | B_t, V_{t+1}, optional T_{t+1})
B_{t+1} = append(B_t, S_{t+1})
if len(B_{t+1}) > K: B_{t+1} = keep_last_K(B_{t+1})
```

然后用同一个问题 `Q` 重新判断：

```text
p_WAIT   = P(<WAIT>   | S_{t+1}, V_{t+1}, Q)
p_ANSWER = P(<ANSWER> | S_{t+1}, V_{t+1}, Q)
```

如果仍然证据不足，则继续等待并继续更新：

```text
S_{t+2}, S_{t+3}, ...
```

直到某个时刻 summary state 和当前视觉证据足够支持回答：

```text
S_k + current visual + Q -> <ANSWER> answer
```

### 9.5 SFT 训练目标

SFT 的训练目标是让模型同时学会两件事：

```text
1. 如何根据新视频 chunk 动态更新 query-agnostic summary bank
2. 如何结合 summary bank 和 query 直接生成 <WAIT> 或 <ANSWER>
```

WAIT 样本：

```text
Input:
S_t + current visual + Question

Target:
<WAIT> 当前证据不足的具体原因
```

ANSWER 样本：

```text
Input:
S_t + current visual + Question

Target:
<ANSWER> 基于当前证据的答案
```

### 9.6 和 Pretrain 的关系

当前 V4 mixedmask pretrain 先学习超声视频和 narration 的对齐关系：

```text
历史视觉/文本 + 当前视觉 -> 当前或下一段 narration
```

后续 Information Compression / recurrent summary-bank pretrain 会进一步训练：

```text
B_t = [S_{t-(K-1)Δ}, ..., S_t]
S_t = hidden_state(<SUMMARY> | B_{t-Δ}, V_t, optional T_t)
B_t = append(B_{t-Δ}, S_t)
if len(B_t) > K: B_t = keep_last_K(B_t)
S_t -> reconstruct compressed information / predict future narration
```

也就是让 `S_t` 这个 summary bank 学会承载过去视频和 narration 信息。

在 QA/SFT 阶段，再引入 query：

```text
S_t + current visual + Q -> <WAIT>/<ANSWER> + text
```

这样模型先学会“看视频并维护状态”，再学会“根据问题判断证据是否充分”。

### 9.7 推理流程

实际推理时，`S_t` 不是自然语言 summary，而是 sliding temporal hidden-state summary bank。每个视频 chunk 到来后，模型执行一次 summary-bank update forward pass：

```text
B_t = [S_{t-(K-1)Δ}, ..., S_t]
S_{t+1} = hidden_state(<SUMMARY> | B_t, V_{t+1}, optional T_{t+1})
B_{t+1} = append(B_t, S_{t+1})
if len(B_{t+1}) > K: B_{t+1} = keep_last_K(B_{t+1})
```

当用户问题 `Q` 存在时，模型使用最新的 summary bank、当前视觉证据和 query 直接计算 `<WAIT>/<ANSWER>` 首 token 概率：

```text
p_WAIT   = P(<WAIT>   | S_{t+1}, V_{t+1}, Q)
p_ANSWER = P(<ANSWER> | S_{t+1}, V_{t+1}, Q)
```

完整在线流程如下：

```text
1. 初始化 summary bank B_0 = []
2. 视频 chunk V_{t+1} 持续到来
3. 使用 S_t、V_{t+1} 和 optional T_{t+1} forward 得到新的 summary bank S_{t+1}
4. 如果用户问题 Q 尚未出现，只持续更新 query-agnostic memory
5. 如果用户问题 Q 已出现，比较 P(<WAIT>) 和 P(<ANSWER>) 或直接生成首 token
6. 如果首 token 是 <WAIT>：
   - generate <WAIT> reason
   - continue streaming
   - 后续 chunk 到来后继续更新 summary 并重新判断
7. 如果首 token 是 <ANSWER>：
   - generate <ANSWER> answer
   - stop or return answer
```

核心思想是：

```text
S_t=[s_t^1,...,s_t^K] 负责持续维护 query-agnostic hidden summary bank
<WAIT>/<ANSWER> 负责首 token answerability decision 和可解析文本输出
```

因此，SFT 阶段的目标不是简单训练一个静态 QA 模型，而是训练一个能够在长超声视频流中持续积累证据、动态判断 answerability，并在证据充分时回答的 streaming QA 模型。

---

## 10. Summary-based Streaming QA 评估方式

评估分为三类。

### 10.1 Answerability 判断

统计模型在每个 query time 上是否正确判断：

```text
当前证据不足 -> <WAIT>
当前证据充分 -> <ANSWER>
```

核心指标：

```text
WAIT / ANSWER accuracy
WAIT precision / recall
ANSWER precision / recall
```

### 10.2 时机评估

Streaming QA 不仅要答对，还要在正确时间回答。

需要统计：

```text
过早回答：证据还不够时提前输出 <ANSWER>
过晚回答：证据已经足够后仍然输出 <WAIT>
首次正确回答时间
answer delay
```

理想模型应该：

```text
不提前 hallucinate
也不过度保守一直 WAIT
```

### 10.3 答案质量评估

当模型输出 `<ANSWER>` 后，再评估 answer 内容是否正确。

可以使用：

```text
exact match / option accuracy
semantic correctness
LLM judge
```

对于开放式回答，可以继续使用 blind LLM judge，比较 base SFT 和 summary-decide SFT 的回答质量。

---

## 11. 当前状态

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

当前 pretrain samples：

```text
train: 18,868 samples
eval :  5,692 samples
total: 24,560 samples
```

已 push 的相关 commits：

```text
899a935 pretrain: add data utilities and blind llm judge
1f3b3a6 asr: reuse whisper model in batch and add full295 job
279ae74 docs: add recurrent summary streaming qa designs
a79b604 pretrain: replace random modality mask with text modality mask
```

当前还需要做：

```text
1. launch full295 V4 mixedmask pretrain
2. evaluate pretrain effect with Word overlap F1, ROUGE-L, and blind LLM judge
3. implement V5 Information Compression / recurrent summary-bank pretrain
4. implement summary-bank streaming QA/SFT
```

---

## 12. 总流程

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
Information Compression / recurrent summary-bank pretrain
↓
Summary-based streaming QA/SFT
↓
每个 chunk:
B_t = [S_{t-(K-1)Δ}, ..., S_t]
S_{t+1} = hidden_state(<SUMMARY> | B_t, V_{t+1}, optional T_{t+1})
B_{t+1} = append(B_t, S_{t+1})
if len(B_{t+1}) > K: B_{t+1} = keep_last_K(B_{t+1})
↓
当 query Q 存在:
p_WAIT   = P(<WAIT>   | S_{t+1}, V_{t+1}, Q)
p_ANSWER = P(<ANSWER> | S_{t+1}, V_{t+1}, Q)
↓
WAIT 则生成 <WAIT> reason 并继续观看、更新 summary
↓
ANSWER 则生成 <ANSWER> answer 并返回
↓
automatic metrics + blind LLM judge
```

最终目标是让模型能够在长超声视频流中持续维护 query-agnostic 的视频理解状态，并在用户问题出现后结合 query 判断当前证据是否足够，从而决定继续等待还是立即回答。