# Qwen3-VL QA SFT 训练说明

本目录是 `QA/` 数据的第一版训练逻辑。目标是训练 Qwen3-VL-2B 学会：

```text
System Prompt
+ 当前时间之前的最后 N 帧 visual tokens
+ Question
→ <WAIT> 或 <ANSWER>
```

当前默认：

```text
MODEL = Qwen/Qwen3-VL-2B-Instruct
WINDOW_SIZE = 8 frames
FRAME_SAMPLING = last_n_frames
```

---

## 文件结构

```text
QA/train/
├── video_sampling.py   # 从 video_window 里采样最后 N 帧
├── dataset.py          # 读取 training_samples.jsonl，解析视频路径与帧
├── collator.py         # 构造 Qwen-VL multimodal chat 输入并 mask label
├── train.py            # HuggingFace + PEFT LoRA 训练入口
└── README.md           # 本文件
```

---

## 依赖与环境

推荐先安装项目依赖：

```bash
pip install -r requirements.txt
```

当前训练脚本依赖 HuggingFace + PEFT：

```text
transformers >= 4.57
accelerate >= 1.1
peft >= 0.13
numpy < 2
```

如果想用 flash attention，需要另装匹配环境的 `flash-attn`，第一版可以不装。

---

## Azure Standard_NC8as_T4_v3 实测环境

当前已在 Azure ML Compute Instance 上用以下环境跑通 smoke train：

```text
VM: Standard_NC8as_T4_v3
GPU: NVIDIA Tesla T4 16GB
CPU: 8 cores
RAM: 56 GB
Python env: azureml_py38
Torch: 2.9.1+cu128
Transformers: 4.57.6
NumPy: 1.26.4
Precision: fp16
```

### 关键环境变量

AzureML 预装环境里可能有 TensorFlow / Flax，与 Transformers import 链路冲突。训练前建议设置：

```bash
export TRANSFORMERS_NO_TF=1
export USE_TF=0
export USE_FLAX=0
```

### NumPy / pyarrow / pandas / sklearn 修复

如果遇到：

```text
_ARRAY_API not found
numpy.core.multiarray failed to import
```

说明当前环境里 `numpy==2.x` 与部分 NumPy-1.x 编译的包不兼容。按下面修复：

```bash
pip install --force-reinstall "numpy==1.26.4"
pip install --force-reinstall "pyarrow>=14,<16" "pandas>=2.0,<2.3" "scikit-learn>=1.3,<1.6"
```

### accelerate 版本修复

如果遇到：

```text
Accelerator.unwrap_model() got an unexpected keyword argument 'keep_torch_compile'
```

说明 `transformers` 和 `accelerate` 版本不匹配。升级：

```bash
pip install -U "accelerate>=1.1.0"
```

### HuggingFace cache / 磁盘空间

Qwen3-VL-2B 权重约 4.26GB，下载和缓存至少需要 8-10GB 空间。如果遇到：

```text
Not enough free disk space to download the file
```

可以清理缓存：

```bash
rm -rf ~/.cache/pip
rm -rf ~/.cache/huggingface
```

并把 HuggingFace cache 放到项目目录或大磁盘：

```bash
cd ~/live-ultrasound-video-understanding
mkdir -p hf_cache

export HF_HOME=$PWD/hf_cache
export HF_HUB_CACHE=$PWD/hf_cache/hub
```

---

## HuggingFace vs vLLM

- **训练**：使用 HuggingFace `transformers` + `peft`。
- **vLLM**：主要用于推理 / serving，不用于这个 LoRA SFT 训练脚本。

---

## 输入数据

训练输入文件：

```text
QA/results/8V649L5Q368_training_samples.jsonl
```

每行一条样本：

```json
{
  "sample_type": "streaming_wait",
  "video_id": "8V649L5Q368",
  "video_window": [167.0, 197.0],
  "question": "...",
  "target": "<WAIT> Not enough information yet. More video is needed.",
  "qa_type": "next_observation"
}
```

训练脚本会：

1. 读取 `video_window`
2. 从这个时间范围内采样最后 `WINDOW_SIZE` 帧
3. 构造 multimodal chat prompt
4. 只对 assistant target 算 loss

---

## 推荐第一版命令

```bash
python QA/train/train.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --train-jsonl QA/results/8V649L5Q368_training_samples.jsonl \
  --default-video-path UltrasoundCrawler_KeyCode_20260323_v2/output/20260520_162816_youtube/media/case_reasoning/8V649L5Q368.mp4 \
  --output-dir QA/checkpoints/qwen3vl_2b_lora_wait_answer \
  --window-size 8 \
  --frame-size 448 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --bf16
```

如果显存紧张：

```bash
python QA/train/train.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --train-jsonl QA/results/8V649L5Q368_training_samples.jsonl \
  --default-video-path UltrasoundCrawler_KeyCode_20260323_v2/output/20260520_162816_youtube/media/case_reasoning/8V649L5Q368.mp4 \
  --output-dir QA/checkpoints/qwen3vl_2b_lora_wait_answer \
  --window-size 8 \
  --frame-size 336 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 1e-4 \
  --bf16 \
  --gradient-checkpointing
```

如果显卡不支持 bf16，改用：

```bash
--fp16
```

---

## 小样本 smoke train

先只用 4 条样本测试 dataloader / processor / loss 是否能跑通：

```bash
python QA/train/train.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --train-jsonl QA/results/8V649L5Q368_training_samples.jsonl \
  --default-video-path UltrasoundCrawler_KeyCode_20260323_v2/output/20260520_162816_youtube/media/case_reasoning/8V649L5Q368.mp4 \
  --output-dir QA/checkpoints/smoke_qwen3vl \
  --window-size 8 \
  --frame-size 336 \
  --limit 4 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --learning-rate 1e-4 \
  --bf16
```

---

## 输出

训练完成后输出 LoRA adapter：

```text
QA/checkpoints/qwen3vl_2b_lora_wait_answer/
```

包括：
- adapter 权重
- tokenizer / processor 文件
- trainer state

---

## 当前限制

1. 第一版推荐 batch size = 1。多样本 multimodal padding 在不同 Qwen-VL processor 版本之间差异较大。
2. Collator 使用 image blocks（8 张图）而不是 video block，这是为了兼容 Qwen2-VL / Qwen2.5-VL / Qwen3-VL。
3. 当前没有 eval loop；先保证 SFT 能跑通。
4. 当前没有 memory bank；只训练 recent-window WAIT/ANSWER。
5. 当前默认 freeze vision encoder，只对 LLM 侧 LoRA 做训练。