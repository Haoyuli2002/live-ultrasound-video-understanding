# Pretrain Full V1

## 1. 目标

Pretrain Full V1 是当前 ultrasound video understanding pipeline 的第一个较大规模预训练实验。

目标是先把 `Qwen3-VL-2B-Instruct` 适配到超声教学/操作视频的 narration continuation 任务上，然后再进入后续 QA / SFT 阶段。

预训练任务可以概括为：

```text
当前 narration segment 之前的视频窗口
+ 前面的 ASR narration 文本上下文
→ 预测当前 ASR narration segment
```

也就是说，模型需要根据：

1. 当前语音片段开始之前的一段视频画面；
2. 前面已经发生过的 narration transcript；

去生成下一段 narration。

这一步还不是 QA 训练，而是一个 caption / narration continuation 风格的 pretraining stage。

---

## 2. 数据来源与划分

原始视频目录：

```text
UltrasoundCrawler_KeyCode_20260323_v2/output/20260726_153916_youtube
```

我们从该目录下的 `.mp4` 文件中按唯一 video ID 做划分。

当前 split 统计：

```text
unique videos: 74
selected videos: 70
train videos: 50
eval videos: 20
unused videos: 4
random seed: 42
```

生成的 split 文件：

```text
cluster_data_splits/selected_70_videos.tsv
cluster_data_splits/train_50_videos.tsv
cluster_data_splits/eval_20_videos.tsv
cluster_data_splits/train_videos.json
cluster_data_splits/eval_videos.json
```

这些文件已经复制到 LRZ：

```text
cluster_data/splits/
cluster_data/videos/train/
cluster_data/videos/eval/
```

训练和评测时，代码通过下面两个文件解析 video ID 到视频路径的映射：

```text
cluster_data/splits/train_videos.json
cluster_data/splits/eval_videos.json
```

---

## 3. ASR Transcript 生成

每个选中的视频先通过 ASR pipeline 转成 transcript。

train 视频命令模式：

```bash
python QA/prepare/asr.py \
  --video <train_video_path> \
  --output-dir cluster_data/QA/train \
  --model large-v3
```

eval 视频命令模式：

```bash
python QA/prepare/asr.py \
  --video <eval_video_path> \
  --output-dir cluster_data/QA/eval \
  --model large-v3
```

ASR backend 使用 `faster-whisper`，模型是 `large-v3`。

每个 transcript 由多个带时间戳的 ASR segment 组成。一个 segment 大致长这样：

```json
{
  "start": 0.0,
  "end": 6.4,
  "text": "..."
}
```

需要注意：

- pretrain sample 不是由我们自己按标点或者句子切出来的。
- sample 是基于 **ASR segment** 生成的。
- ASR segment 通常接近一句话或半句话，但边界由 Whisper/faster-whisper 根据语音和模型输出决定。
- 因此，一个 target 可能是完整句子，也可能是半句话、短语或者一句话的延续。

当前 train ASR 状态：

```text
train transcripts: 50
```

---

## 4. Pretrain Sample 生成逻辑

样本由下面命令生成：

```bash
python pretrain/build_samples.py \
  --transcripts cluster_data/QA/train/transcripts \
  --output cluster_data/pretrain/train_pretrain_samples.jsonl \
  --window-sec 8 \
  --min-words 3 \
  --context-max-chars 400
```

对于每一个 ASR segment，builder 会尝试生成一个 pretrain sample。

假设当前 ASR segment 是：

```text
segment_start = t_start
segment_end = t_end
segment_text = 当前 ASR segment 的文本
```

那么生成的训练样本是：

```text
video_window = [max(0, t_start - window_sec), t_start]
prev_context = 当前 segment 之前的 ASR 文本，上限 context_max_chars
target = 当前 ASR segment 的文本
```

当前配置：

```text
window_sec = 8
min_words = 3
context_max_chars = 400
```

因此，每个样本中模型看到的是：

```text
当前 segment 开始前最多 8 秒的视频帧
+
当前 segment 之前最多 400 字符的 transcript 上下文
```

模型要预测的是：

```text
当前 ASR segment 的文本
```

概念上就是：

```text
Video frames from [t_start - 8s, t_start]
+
Previous transcript text
→
Current transcript segment text
```

如果某个 ASR segment 少于 `--min-words 3`，则会被跳过。

---

## 5. Sample Schema

每一行 JSONL 是一个 pretrain sample。

核心字段：

```json
{
  "sample_type": "pretrain_caption",
  "video_id": "...",
  "video_window": [start_time, end_time],
  "prev_context": "...",
  "target": "..."
}
```

字段含义：

| 字段 | 含义 |
|---|---|
| `sample_type` | 样本类型。当前阶段为 `pretrain_caption`。 |
| `video_id` | 视频 ID，通常是 `.mp4` 文件名去掉后缀。 |
| `video_window` | 视觉输入使用的视频时间窗口，结束点是当前 ASR segment 的开始时间。 |
| `prev_context` | 当前 segment 之前的 ASR 文本上下文。 |
| `target` | 当前 ASR segment 文本，也就是模型要生成的目标。 |

注意：每条 sample 不一定直接存 `video_path`。训练时视频路径通过下面文件解析：

```text
cluster_data/splits/train_videos.json
```

---

## 6. 当前数据统计

train sample build 结果：

```text
videos: 50
total train samples: 7928
output: cluster_data/pretrain/train_pretrain_samples.jsonl
```

有 3 个视频生成了 0 个训练样本：

```text
ERGxpZ4qdYI: 0 samples
GhkNXh0m5Nk: 0 samples
s4suV5lDB5g: 0 samples
```

这不一定是错误，可能原因包括：

- ASR 文本为空或非常短；
- 视频里几乎没有有用语音；
- segment 少于 `--min-words 3`；
- transcript 内容被 sample-building 规则过滤掉。

整体来看，当前 train set 已经足够大：

```text
7928 train samples
```

---

## 7. 真实样本示例

下面是真实来自：

```text
cluster_data/pretrain/train_pretrain_samples.jsonl
```

的样本。

### Example 1

```yaml
sample: 1
video_id: -1i1i9sbjqE
video_path: null
video_window: [0.0, 1.88]
prev_context: ""
target: "Thanks for tuning in. In this module, I'm going to go over a little bit more in detail the cardiac"
```

解释：

- 这是该视频的第一个可用 ASR segment。
- 因为前面没有 narration，所以 `prev_context` 为空。
- 视觉输入是视频开头到 `1.88s`。
- target 是第一段 narration fragment。

---

### Example 2

```yaml
sample: 2
video_id: -1i1i9sbjqE
video_path: null
video_window: [0.0, 6.54]
prev_context: "Thanks for tuning in. In this module, I'm going to go over a little bit more in detail the cardiac"
target: "ultrasound conventions that are commonly out there. As per usual, a number of the ultrasound"
```

解释：

- 模型看到当前 ASR segment 开始前的视频窗口。
- 上一个 segment 文本被放进 `prev_context`。
- target 是 narration 的下一段延续。

---

### Example 3

```yaml
sample: 3
video_id: -1i1i9sbjqE
video_path: null
video_window: [3.18, 11.18]
prev_context: "Thanks for tuning in. In this module, I'm going to go over a little bit more in detail the cardiac ultrasound conventions that are commonly out there. As per usual, a number of the ultrasound"
target: "images and videos are courtesy of the Division of Emergency Ultrasound at Massachusetts General"
```

解释：

- 这里的视频窗口长度正好是 8 秒：`3.18s` 到 `11.18s`。
- 前面的 ASR 文本被拼接成更长的上下文。
- 模型要生成下一段 ASR segment。

---

### Example 4

```yaml
sample: 4
video_id: -1i1i9sbjqE
video_path: null
video_window: [7.9, 15.9]
prev_context: "Thanks for tuning in. In this module, I'm going to go over a little bit more in detail the cardiac ultrasound conventions that are commonly out there. As per usual, a number of the ultrasound images and videos are courtesy of the Division of Emergency Ultrasound at Massachusetts General"
target: "Hospital. In my previous tutorial with respect to parasternal long axis cardiac imaging,"
```

解释：

- target 接在前文 `Massachusetts General` 后面，继续形成 `Massachusetts General Hospital`。
- 这个例子说明 ASR segment 的边界不一定是完整句子边界。
- 因此当前 pretrain 是 ASR-segment 级别的 narration continuation，而不是严格 sentence-level 任务。

---

### Example 5

```yaml
sample: 5
video_id: -1i1i9sbjqE
video_path: null
video_window: [16.02, 24.02]
prev_context: "Thanks for tuning in. In this module, I'm going to go over a little bit more in detail the cardiac ultrasound conventions that are commonly out there. As per usual, a number of the ultrasound images and videos are courtesy of the Division of Emergency Ultrasound at Massachusetts General Hospital. In my previous tutorial with respect to parasternal long axis cardiac imaging,"
target: "I mentioned that you're going to be pointing the transducer marker either to the patient's"
```

解释：

- 这是另一个 narration continuation 样本。
- 模型同时使用视觉上下文和前文 transcript。
- target 是当前时刻下一段 ASR segment。

---

## 8. 训练命令

50-video train set 的训练命令如下：

```bash
python pretrain/train.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --train-jsonl cluster_data/pretrain/train_pretrain_samples.jsonl \
  --video-path-map cluster_data/splits/train_videos.json \
  --output-dir cluster_data/checkpoints/pretrain_qwen3vl_train50 \
  --window-size 4 \
  --frame-size 224 \
  --num-train-epochs 3 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --bf16 \
  --gradient-checkpointing \
  --early-stop-patience 3 \
  --early-stop-min-delta 0.001
```

训练配置总结：

```text
base model: Qwen3-VL-2B-Instruct
HF model id: Qwen/Qwen3-VL-2B-Instruct
train samples: 7928
per-device batch size: 1
gradient accumulation: 8
effective batch size: 8
epochs: 3
estimated optimizer steps: about 2973
LoRA trainable params: about 17.4M
vision encoder: frozen
```

当前训练方式是 LoRA adapter training，不是 full fine-tuning。

---

## 9. 当前训练问题：视频解码失败

第一次训练尝试中，模型和 dataset 都成功加载，但在进入第一个 optimizer step 前失败：

```text
0%| | 0/2973
RuntimeError: Failed to read any frame from cluster_data/videos/train/ga3t4LCZ6P0.mp4
```

进一步扫描发现，50 个 train 视频中有 28 个无法被 OpenCV 读取首帧：

```text
bad_count: 28
```

这些 train video ID 是：

```text
v2uBpsEKte8
jEeJ8otZC0c
_Q0cTG3ZlHk
TlckvYhqaFE
b70tsWPz2P0
1E4NSR6yjMw
qB-vLjiupak
s4suV5lDB5g
R15Q11bLl34
GhkNXh0m5Nk
AxuGkz78pBA
ERGxpZ4qdYI
vmmrjy1bhEc
BwSJCkTBN0c
m9s5AD_lBkY
ODF8ZLr_52s
Flc8xS2foOU
SWzpcc4ZR-U
HJ9HX0CrL-U
Cg8Mzbc4TBY
ltEmNQ6yzi4
X1E7OgOLzw0
ga3t4LCZ6P0
fsRrC53sWus
6V-aAFVfiJk
CW3TLabogko
4Yu-ftQzHZM
bRPD-51UG7Y
```

可能原因是这些视频使用 AV1 或其他当前 OpenCV/FFmpeg stack 无法通过 `cv2.VideoCapture` 解码的格式。

ASR 成功是因为 ASR 只需要抽音频；pretrain 失败是因为训练时需要读取视频帧。

计划修复方式是：把这些视频批量转码成 H264，并保持原文件名/路径不变。这样已有 transcript 和 JSONL sample 都不用重新生成。

推荐转码参数：

```bash
ffmpeg -y \
  -i input.mp4 \
  -c:v libx264 \
  -preset veryfast \
  -crf 23 \
  -pix_fmt yuv420p \
  -c:a aac \
  -b:a 128k \
  output.mp4
```

为了把这个流程固定到代码中，新增了脚本：

```text
scripts/transcode_videos_for_opencv.py
```

该脚本做的事情：

1. 读取 `video_path_map.json`，例如 `cluster_data/splits/train_videos.json`。
2. 对每个视频调用 `cv2.VideoCapture` 检查是否能读取首帧。
3. 找出 OpenCV 无法读取的视频。
4. 可选地调用 `ffmpeg` 将不可读视频转码为 H264 / yuv420p MP4。
5. 转码后再次用 OpenCV 验证。
6. 验证通过后，用 H264 版本替换原路径。
7. 可选择保留或删除原始 backup。

只扫描，不替换：

```bash
python scripts/transcode_videos_for_opencv.py \
  --video-path-map cluster_data/splits/train_videos.json \
  --quiet-opencv
```

扫描并替换不可读视频，成功后删除 backup：

```bash
python scripts/transcode_videos_for_opencv.py \
  --video-path-map cluster_data/splits/train_videos.json \
  --replace \
  --delete-backup \
  --quiet-opencv
```

如果想先保留原始视频 backup：

```bash
python scripts/transcode_videos_for_opencv.py \
  --video-path-map cluster_data/splits/train_videos.json \
  --replace \
  --keep-backup \
  --quiet-opencv
```

eval 视频也可以用同一个脚本检查和修复：

```bash
python scripts/transcode_videos_for_opencv.py \
  --video-path-map cluster_data/splits/eval_videos.json \
  --replace \
  --delete-backup \
  --quiet-opencv
```

转码后重新扫描，期望：

```text
[summary] final_bad_count: 0
```

或者手动 OpenCV 扫描输出：

```text
bad_count: 0
```

然后删除失败训练输出目录并重新训练：

```bash
rm -rf cluster_data/checkpoints/pretrain_qwen3vl_train50
```

---

## 10. 存储管理说明

LRZ home directory 有容量限制，不适合长期存放大数据和多个 checkpoint。

当前实际策略：

- LRZ 上只保留当前需要的视频、transcript、JSONL 和正在训练的 checkpoint。
- 旧 checkpoint 用 `rsync` 拉回本地保存。
- 不把 checkpoint、视频、生成的 transcript 或 `cluster_data` push 到 GitHub。

旧 SFT checkpoint 已经被拉回本地：

```text
local_checkpoints/qwen3vl_2b_sft_streaming_v2_waitreason/
```

最终 adapter 文件在本地：

```text
local_checkpoints/qwen3vl_2b_sft_streaming_v2_waitreason/adapter_model.safetensors
```

中间的 `checkpoint-*` 目录主要用于 resume training，不是推理必须文件。

---

## 11. 下一步

1. 使用 `scripts/transcode_videos_for_opencv.py` 将 OpenCV 无法读取的 train 视频批量转码成 H264。
2. 验证 50 个 train 视频都能被 OpenCV 读取，即 `final_bad_count: 0`。
3. 如有需要，对 eval 视频也运行同样的 OpenCV 可读性检查和 H264 转码。
4. 重新启动 Pretrain Full V1 训练。
5. 如果 eval ASR 还没完成，补完 eval ASR。
6. 构建 eval pretrain samples。
7. 在 eval20 上分别跑 base 和 LoRA inference。
8. 用 word-overlap F1 和 semantic cosine similarity 对比 base vs LoRA。
9. 将最终 LoRA adapter 保存到本地，并删除 LRZ 上不必要的 checkpoint。
