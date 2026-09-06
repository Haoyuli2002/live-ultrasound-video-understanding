# Live Ultrasound Video Understanding：完整设计文档

## 1. 总目标

目标是构建一个面向实时超声视频的理解系统。模型持续接收超声视频流和可选 ASR narration；当用户提出问题时，模型判断当前证据是否足够，并直接输出：

```text
<WAIT> reason
```

或：

```text
<ANSWER> answer
```

当前主模型：

```text
Qwen/Qwen3-VL-2B-Instruct + LoRA
```

---

## 2. 三阶段训练路线

```text
Stage 1: Pretrain / Ultrasound Knowledge Injection
Stage 2: Information Compression / Learn Summary Bank
Stage 3: SFT / QA + Instruction Tuning
```

在三阶段训练之前，数据处理流程先做：

```text
Raw videos
↓
Whisper ASR
↓
ASR rule-based filtering
↓
VLM video-type classification
↓
stage-specific keep/drop
```

VLM video-type classification 位于 ASR 后面，原因是 ASR rule filtering 成本更低，可以先减少候选视频数量，再对 ASR-keep 视频运行较贵的 VLM 分类。

### ASR 后 VLM 视频类型分类

ASR filtering 只能判断文本质量，不能准确判断视频视觉内容类型。因此在 ASR-keep 视频上增加 VLM 分类，将每个视频归入以下 6 类：

```text
A. ultrasound_cine
   纯超声视频 / real-time ultrasound scan / ultrasound cine loop

B. ultrasound_teaching_with_scan
   超声教学视频，包含真实超声动态画面 + 讲解

C. ultrasound_ppt_teaching
   超声 PPT / slide-based lecture / 大量文字或幻灯片

D. mixed_screen_recording
   混合屏幕录制，包括网页、软件界面、PPT、少量超声图

E. non_ultrasound_or_irrelevant
   非超声或明显无关视频

F. uncertain
   模型不确定，需要人工复核或保守保留
```

VLM 分类脚本：

```text
scripts/data/classify_video_type_vlm.py
```

分类模型使用 **Qwen3-VL**（视觉-语言模型），默认用 MoE 版本 `Qwen/Qwen3-VL-30B-A3B-Instruct`（总参数约 30B，激活约 3B），适合 H100 推理。通过 vLLM 起一个 OpenAI 兼容服务：

```bash
vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
  --port 8000 \
  --limit-mm-per-prompt image=16
```

然后脚本默认指向该本地服务：

```bash
python scripts/data/classify_video_type_vlm.py \
  --video-map cluster_data/splits/train_full295_asr_keep_videos.json \
  --output cluster_data/splits/train_full295_asr_vlm_video_type.jsonl \
  --n-frames 12 --frame-size 224 \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --base-url http://localhost:8000/v1 \
  --api-key-env VLLM_API_KEY
```

说明：纯文本的 `Qwen3-30B-A3B` 看不到画面，因此视频类型分类必须用视觉版本 `Qwen3-VL-30B-A3B`。

输入通常是 ASR-keep video map：

```text
cluster_data/splits/train_full295_asr_keep_videos.json
cluster_data/splits/eval_full295_asr_keep_videos.json
```

输出 JSONL audit：

```text
cluster_data/splits/train_full295_asr_vlm_video_type.jsonl
cluster_data/splits/eval_full295_asr_vlm_video_type.jsonl
```

每条记录包含：

```json
{
  "video_id": "...",
  "video_path": "...",
  "label": "ultrasound_teaching_with_scan",
  "confidence": 0.86,
  "visual_evidence": "...",
  "keep_for_pretrain": true,
  "keep_for_compression": true,
  "keep_for_sft": true
}
```

默认 stage-specific keep 策略：

| label | Pretrain | Compression | SFT |
|---|---:|---:|---:|
| `ultrasound_cine` | keep | keep | keep |
| `ultrasound_teaching_with_scan` | keep | keep | keep |
| `ultrasound_ppt_teaching` | keep | drop | drop |
| `mixed_screen_recording` | keep | drop | drop |
| `non_ultrasound_or_irrelevant` | drop | drop | drop |
| `uncertain` | keep/audit | drop/audit | drop/audit |

根据 VLM audit 生成各阶段 keep/drop map 的脚本：

```text
scripts/data/filter_by_vlm_video_type.py
```

### Stage 1

第一阶段使用超声视频和 ASR narration 做领域知识注入，目标不是简单 ASR 续写，而是让模型学习超声视觉模式、解剖结构、病灶表现、扫查流程，并进行 video-text 对齐。

当前 V4 mixedmask 形式：

```text
history visual/text + current visual -> narration
```

mask 策略：

```text
mask_current_visual_prob = 0.33 （遮盖当前的视频，让模型通过历史信息预测下一个句子，训练时许能力。）
no_mask_prob             = 0.33 （无掩码，通过历史的 context 和当前的视频，进行生产，训练理解能力）
text_modality_mask_prob  = 0.34 （遮盖文本模态，让模型只通过视觉特征进行学习，适配下游任务）
```

### Stage 2

第二阶段训练 information compression 能力，也就是把 past context 压缩进 summary bank，并通过 summary bank 重建或预测信息。

### Stage 3

第三阶段做 streaming QA / instruction tuning。模型使用 summary bank、当前视觉证据和 query，生成 `<WAIT> 及其原因` 或 `<ANSWER> 以及回复`。

---

## 3. Summary Bank 定义

Summary bank 采用 **sliding temporal bank** 设计：一开始为空，每个 chunk 生成一个新的 summary token，然后 append 到 bank；如果超过最大容量 `K`，就移出最旧的 summary token。

```text
B_0 = []
```

每个 chunk 生成一个 summary token：

```text
S_t = 当前 chunk 新生成的 summary token
```

当前时刻的 summary bank 是最近最多 `K` 个 summary tokens：

```text
B_t = [S_{t-(K-1)Δ}, ..., S_t]
```

其中：

```text
B_t = 当前 sliding temporal summary bank
S_t = 当前 chunk 的 summary token
K   = bank 最大容量
Δ   = chunk 时间间隔，例如 10s
```

例如：

```text
K = 20
Δ = 10s

B_0 = []
B_10  = [S_10]
B_20  = [S_10, S_20]
...
B_200 = [S_10, S_20, ..., S_200]
B_210 = [S_20, S_30, ..., S_210]  # S_210 加入后，S_10 被移出
```

因此，`K=20` 表示最多保存最近 20 个 chunk summary tokens，不表示每个 `S_t` 内部有 20 个 summary slots。

---

## 4. Summary Bank 更新

每个 chunk 或者每 10s：

```text
C_t = V_t + optional T_t（但是纯超声视频一般没有声音，目前只是一个placeholder）
```

先从当前 chunk 和已有 bank 生成新的 summary token：

```text
S_t = hidden_state(<SUMMARY> | B_{t-Δ}, V_t, optional T_t)
```

然后把新的 `S_t` append 到 bank：

```text
B_t = append(B_{t-Δ}, S_t)
if len(B_t) > K:
    B_t = keep_last_K(B_t)
```

其中 `Δ` 是 chunk size，例如 `10s`。

---

## 5. Information Compression Objective

Stage 2 的核心目标：

```text
past context -> B_t -> reconstruction / prediction
```

可选监督：

```text
B_t -> reconstruct masked / historical narration
B_t + V_{t+Δ} -> predict T_{t+Δ}
```

关键约束：reconstruction / prediction 只通过 summary bank 进行生成，不能看到完整 raw history，否则模型会绕过 summary bank。

---

## 6. Streaming QA：直接用 `<WAIT>/<ANSWER>` 决策

第三阶段不引入额外分类模块。

QA 输入：

```text
B_t + current visual + Q
```

模型直接以 `<WAIT>` 或 `<ANSWER>` 开头生成：

```text
<WAIT> reason
```

或：

```text
<ANSWER> answer
```

answerability 由首 token 概率决定：

```text
p_WAIT   = P(<WAIT>   | B_t, V_t, Q)
p_ANSWER = P(<ANSWER> | B_t, V_t, Q)
```

训练 loss 是普通 causal LM loss：

```text
loss = CE(target tokens)
```

WAIT 样本：

```text
Input:  B_t + current visual + Question
Target: <WAIT> 当前证据不足的具体原因
```

ANSWER 样本：

```text
Input:  B_t + current visual + Question
Target: <ANSWER> 基于当前证据的答案
```

---

## 7. WAIT reason 是否写回 Summary Bank

不写回。

`B_t` 是 query-agnostic video memory，只压缩视频和 optional narration 历史：

```text
S_{t+1} = hidden_state(<SUMMARY> | B_t, V_{t+1}, optional T_{t+1})
B_{t+1} = append(B_t, S_{t+1})
if len(B_{t+1}) > K:
    B_{t+1} = keep_last_K(B_{t+1})
```

previous `<WAIT>` reason 属于 query-specific interaction history。它可以在下一次 QA prompt 中作为可选上下文输入，但不要写入 `B_{t+1}`，否则 summary bank 会被某个具体 query 污染。

---

## 8. 视频帧处理

当前工程 baseline 使用：

```text
224 × 224 RGB
```

处理方式：

```text
等比例缩放 + 黑色 padding
```

Qwen3-VL 本身支持 dynamic resolution，并不要求固定 `224 × 224`。当前固定尺寸主要是为了方便控制显存，避免后续使用更大分辨率或动态高分辨率视频帧训练时 OOM，同时保持 Pretrain 和 SFT 输入分布一致。

后续可以实验更高固定分辨率或 Qwen3-VL 原生 dynamic resolution / `max_pixels` 设置，比较细节保留、训练速度、显存占用和 QA 效果之间的 trade-off。

---

## 9. 在线推理流程

```text
1. 初始化 summary bank B_0 = []
2. 视频流每 Δ 秒到来一个 chunk，例如 Δ=10s
3. 为当前 chunk 生成新的 summary token:
   S_{t+1} = hidden_state(<SUMMARY> | B_t, V_{t+1}, optional T_{t+1})
4. 将新 token append 到 bank:
   B_{t+1} = append(B_t, S_{t+1})
   if len(B_{t+1}) > K: B_{t+1} = keep_last_K(B_{t+1})
5. 如果用户问题 Q 尚未出现，只持续维护 sliding temporal summary bank
6. 如果用户问题 Q 已出现，计算 P(<WAIT>) 和 P(<ANSWER>)，或直接生成首 token
7. 如果首 token 是 <WAIT>，生成 reason 并继续 streaming
8. 如果首 token 是 <ANSWER>，生成 answer 并返回
```

---

## 10. 当前数据状态

```text
full295: 250 train videos + 45 eval videos
ASR: train 250/250, eval 45/45
ASR filtering: train keep 196/250, eval keep 41/45
filtered total: 237 videos, 14.058 GiB, 49:35:48
V4 mixedmask samples: train 18,868, eval 5,692, total 24,560
```

---

## 11. 总流程

```text
full295 videos
↓
Whisper ASR
↓
rule-based ASR filtering
↓
Stage 1: Ultrasound Knowledge Injection with V4 mixedmask pretrain
↓
Stage 2: Information Compression / recurrent summary-bank pretrain
    B_0 = []
    S_t = hidden_state(<SUMMARY> | B_{t-Δ}, V_t, optional T_t)
    B_t = append(B_{t-Δ}, S_t)
    B_t = keep_last_K(B_t)
    B_t -> reconstruction / prediction
↓
Stage 3: Summary-bank streaming QA / SFT
    B_t + current visual + Q
    p_WAIT   = P(<WAIT>   | B_t, V_t, Q)
    p_ANSWER = P(<ANSWER> | B_t, V_t, Q)
↓
WAIT 则生成 <WAIT> reason 并继续观看、更新 summary
↓
ANSWER 则生成 <ANSWER> answer 并返回
↓
automatic metrics + blind LLM judge
```