# Live Ultrasound Video Understanding — 完整 Pipeline

目标：从 YouTube / Bilibili 超声教学视频，构建数据并训练一个 answerability-aware 的实时超声视频理解模型（Qwen/Qwen3-VL-2B-Instruct）。

---

## 0. 总览

```text
1. 视频爬取         UltrasoundCrawler_KeyCode_20260323_v2/
2. 视频过滤         src/video_filter.py / scripts/video_filter_vlm.py
3. 数据准备         QA/prepare/run_prepare.py  (ASR转录获取字幕 + 视频切分clipping)
        │            -> transcripts/{id}.json  +  clips/{id}_clips.json
        │
        ├── Pretrain 分支（预训练，让模型学习超声视觉-语言的对应关系）
        │     4. build_samples   pretrain/build_samples.py  -> pretrain_samples.jsonl
        │     5. 预训练           pretrain/train.py          (Stage 1, LoRA)
        │
        └── QA / SFT 分支（学 WAIT/ANSWER 决策，判断什么时候可以回答问题，什么时候无法回答问题。）
              6. QA 生成         QA/run.py = offline → streaming → validator → merger
                                 -> {id}_training_samples.jsonl
              7. SFT             QA/train/train.py          (Stage 2, LoRA)

8. 推理 / 评测     QA/eval/ , pretrain/infer.py
```

关键点：

- Pretrain 分支和 QA/SFT 分支**共享前面的爬取 / 过滤 / 数据准备（ASR + clipping）**，之后分成两条独立数据流。
- 两条训练分支目前**完全解耦**，是否用 Stage 1 预训练的 adapter 作为 Stage 2 SFT的起点，是可选实验。

---

## 1. 视频爬取

目录：`UltrasoundCrawler_KeyCode_20260323_v2/`

- 从 YouTube / Bilibili 爬取超声教学视频。
- 有 Web UI（`webapp.py` / `run_ui.bat`）和 CLI（`cli.py`）。

**产物**：原始视频 `output/.../media/<category>/{video_id}.mp4`。

---

## 2. 视频过滤

- `src/video_filter.py`
- `scripts/video_filter_vlm.py`
- `scripts/batch_filter.py`

用 VLM 判断视频是否为"有用的超声教学内容"，过滤掉无关视频（纯讲座、广告、非超声等）。

**产物**：过滤后的视频清单。

---

## 3. 数据准备（ASR + Clipping，一键）

入口：`QA/prepare/run_prepare.py`（内部依次调用 `QA/prepare/asr.py` 和 `QA/prepare/clipping.py`）。

```bash
python QA/prepare/run_prepare.py \
    --video path/to/video.mp4 \
    --output-dir QA/results \
    --whisper-model base
# 可选: --language en / --skip-asr / --skip-clipping / --no-llm-clipping
```

**产物**：

```text
QA/results/transcripts/{video_id}.json
QA/results/clips/{video_id}_clips.json
```

### 3.1 ASR 转录

- ffmpeg 抽音频 → faster-whisper 转写 → 带时间戳 JSON。
- 模型策略：默认 `medium`；检测到 GPU 时自动升级 `large-v3`；GPU 用 `float16`，CPU 用 `int8`。医学术语需要较大模型，避免污染下游文本。
- 也可单独运行：`python QA/prepare/asr.py --video ... --output-dir QA/results`

transcript 格式：

```json
{
  "video_id": "TlckvYhqaFE",
  "duration_sec": 623.4,
  "segments": [{"start": 4.3, "end": 12.3, "text": "So the first thing we do is..."}, ...],
  "full_text": "..."
}
```

### 3.2 Clipping（视频切片，超声友好、离线、无 LLM）

- 核心实现：`scripts/video_clipping.py`（函数 `clip_video`）；wrapper：`QA/prepare/clipping.py`。
- 也可单独运行：`python QA/prepare/clipping.py --video ... --output-dir QA/results`
- 逻辑：
  1. **视觉变化检测**（固定时间网格，默认每 1.5s 一帧）。支持多种 `--visual-method`：
     - **`qwen_embed`（默认）**：用 **torchcodec** 抽**原始彩色帧**（不经 OpenCV、不转灰度），过 **`Qwen/Qwen3-VL-Embedding-2B`**（与 SFT 基座 Qwen3-VL-2B 同源）得图像 embedding → L2 归一化 → 相邻帧**余弦相似度**；`similarity < scene_threshold`（默认 `0.85`）且间隔 ≥ `min_scene_gap(3s)` 记为 scene change。**需要 GPU**；缺 GPU / sentence-transformers / torchcodec 时自动回退 `ssim`。
     - `ssim` / `framediff`：OpenCV 灰度网格 SSIM（默认阈值 `0.6`，无 scikit-image 回退 framediff）。
     - `histogram`：legacy 逐 segment 直方图。
  2. 句子边界：英文标点 `.?!` + 停顿 gap(0.8s) fallback + 全弱边界兜底。
  3. 对齐：视觉切点找最近句边界（`tolerance=5s`），找不到就放弃该切点（保句子完整）。
  4. 组装：`min_clip=30s`、`max_clip=240s`；短尾（< min_clip）合并到前一个 clip。
  5. 超长（> max_clip）按句边界细分。
- 阈值调参：加 `--save-trace` 输出每个采样点的相似度，便于确定 `--scene-threshold`。
- 相关参数：`--qwen-embed-model`（默认 `Qwen/Qwen3-VL-Embedding-2B`）、`--qwen-embed-device`（auto）、`--qwen-embed-batch`（16）。

clips 格式（含 method / params / coverage_pct / 每 clip 的 start/end/duration/text/cut_reason）：

```json
{
  "video_id": "...",
  "clips": [
    {"clip_idx": 1, "start": 87.45, "end": 250.17, "duration": 162.72,
     "text": "...", "cut_reason": "scene_change"}
  ]
}
```

---

## 4. Pretrain 分支：构造预训练样本

入口：`pretrain/build_samples.py`

- **任务**：整句 caption completion。对每个 ASR segment，看这句开始前最近 N 秒的帧，续写这句超声解说。
- `current_time = segment.start`，`video_window = [max(0, start - window_sec), start]`。
- 带 `prev_context`（前文解说，可截断，可 `--no-context` 关闭）。
- 过滤：空句、词数 < `--min-words`、含 `[music]/[applause]/[laughter]/[inaudible]/[noise]` 的 segment。

```bash
python pretrain/build_samples.py \
  --transcripts QA/results/transcripts \
  --output pretrain/data/pretrain_samples.jsonl \
  --window-sec 8 --min-words 3 --context-max-chars 400
```

**产物**：`pretrain/data/pretrain_samples.jsonl`（`sample_type=pretrain_caption`）：

```json
{
  "sample_type": "pretrain_caption",
  "video_id": "TlckvYhqaFE",
  "video_window": [4.3, 12.3],
  "prev_context": "前文解说（截断到 context-max-chars）",
  "target": "So the first thing we do is place the linear probe...",
  "meta": {"segment_idx": 5, "seg_start": 12.3, "seg_end": 15.8}
}
```

其中 `video_window = [max(0, seg_start - window_sec), seg_start]`。

---

## 5. 预训练（Stage 1）

入口：`pretrain/train.py`

- LoRA 预训练，**不加 special token，词表不变**（adapter 小、快、省显存）。
- System prompt：`You are an ultrasound teaching assistant...Continue the spoken narration...`
- User：`[N frames] + (可选) "Narration so far: {prev_context}\nContinue..."`；Assistant：`{target}`。
- **Loss**：只对 assistant `target` 计算，其余（system / 图像 token / user / prev_context）都 `-100`。label mask 通过在 `input_ids` 中定位 target token 子序列实现，兼容 Qwen-VL image placeholder 展开。
- Early Stopping（每 epoch 平均 loss，连续 N epoch 改善 < min_delta 停）+ TensorBoard。
- 多视频用 `--video-path-map`（JSON: `{video_id: mp4_path}`）；单视频用 `--default-video-path`。

```bash
python pretrain/train.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --train-jsonl pretrain/data/pretrain_samples.jsonl \
  --video-path-map pretrain/data/video_path_map.json \
  --output-dir /mnt/cache/qwenFT/pretrain_qwen3vl_bf16 \
  --window-size 4 --frame-size 224 \
  --num-train-epochs 3 \
  --per-device-train-batch-size 1 --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 --bf16 --gradient-checkpointing \
  --early-stop-patience 3 --early-stop-min-delta 0.001
```

**产物**：预训练 LoRA adapter。

推理检查：`pretrain/infer.py`（caption 续写）。

> 当前限制：batch size = 1；用 image blocks（N 张图）而非 video block（兼容多版本 Qwen-VL processor）；只做整句 completion（句中续写是后续 ablation）。

---

## 6. QA 生成（一体化 `QA/run.py`）

入口：`QA/run.py`，内部按顺序执行四步：**offline_generator → generator(streaming) → validator → merger**。

```bash
python QA/run.py \
  --video path/to/{video_id}.mp4 \
  --expand-wait-answer
# 默认 clips = results/clips/{id}_clips.json, transcript = results/transcripts/{id}.json
# 可用 --clips / --transcript / --out-dir 覆盖；--skip-offline/-generation/-validation/-merge 跳步
```

生成器默认模型 `google/gemini-2.5-flash`（通过 OpenRouter；需要 API key）。

### 6.1 Offline QA — `QA/offline_generator.py`

- 每个 clip 生成 **1 条** `clip_summary`（完整 clip 理解：扫查过程 + 关键视觉细节 + 相关医学知识）。
- clip 超长按 `--clip-max-sec`（默认 300s）截断上传。

**产物**：`QA/results/{id}_offline_qa.json`

### 6.2 Streaming QA — `QA/generator.py`（**当前语义，已更新**）

- **单锚点**：`TIME_RATIOS = [0.5]`（clip 中点），且 `MAX_QA_PER_ANCHOR = 1` → **每 clip 只产 1 条 streaming QA**。
- QA 类型：`next_action`（接下来该做什么：探头调整、加压、切模式、调体位）/ `next_observation`（接下来该看什么：胸膜线、A/B-lines、spine/curtain sign、回声/运动模式）。每条 QA 带 `query_time`（证据不足）与 `answer_time`（证据充分），`answer_time - query_time ≥ MIN_ANSWER_DELAY_SEC(5s)`。
- **`wait_reason`（新增，抗坍塌关键）**：每条 QA 必须给出一句"为何在 `query_time` 尚不可答"的**具体缺失证据**（哪一个结构还没出现 / 哪个探头动作还没发生 / 哪个 sign 还看不到），且必须与 `answer` 对应。
  - 质量控制 `is_generic_wait_reason()`：空 / 少于 5 词 / 命中通用套话黑名单（如 "not enough information"、"more video is needed"）→ 视为无效，该 QA 被 drop。

**产物**：`QA/results/{id}_streaming_qa.json`（含每条 QA 的 `question / answer / wait_reason / evidence / query_time / answer_time / type / clip_idx`）。

### 6.3 校验 — `QA/validator.py`

- 三条硬约束 + VLM 校验 QA 质量/格式：
  1. `question_no_leak`：问题不能泄露 future 信息。
  2. `not_answerable_at_query_time`：`query_time` 时证据不足。
  3. `answerable_at_answer_time`：`answer_time` 时证据充分。
- **本地 `wait_reason` 质量**（无额外 API 调用）：若 `wait_reason` 缺失 / 太短 / 通用，也强制 downgrade 成 `fail` 并附注说明（黑名单与 generator 一致）。
- `QA/run.py` 支持三种 validation mode：
  - `--validation-mode all`：全量 VLM validation，并用 validated streaming QA 进入 merger。适合小规模调试 / eval set。
  - `--validation-mode sample`：只抽样 validation，产出 audit 文件；训练数据 merge 使用 raw streaming QA。适合 train set 批量生成时控制成本。
  - `--validation-mode none`：完全跳过 VLM validation，直接 merge raw streaming QA。适合 generator 质量已稳定后的大规模 train generation。
- 抽样参数：
  - `--validation-sample-rate 0.1`
  - `--validation-max-qa 20`
  - `--validation-sample-seed 42`
- `--keep-failed` 只对 `validation-mode all` 的 validated 输出有意义；默认丢弃 failed QA。

**产物**：
- `all`：`QA/results/{id}_streaming_qa_validated.json`
- `sample`：`QA/results/{id}_streaming_qa_validation_sample.json` + `QA/results/{id}_streaming_qa_validation_audit.json`
- `none`：不生成 validation 产物

### 6.4 合并 / 展开 — `QA/merger.py`

- 合并 offline + streaming，产出 per-video 记录 `{id}.jsonl`。
  - `validation-mode all`：使用 validated streaming QA。
  - `validation-mode sample/none`：使用 raw streaming QA。
- `--expand-wait-answer` 展开成 WAIT/ANSWER 训练样本。
- **WAIT 目标多样化**：`_wait_target_for(qa)` 在 `wait_reason` 具体且非通用时用 `"<WAIT> {wait_reason}"`（多样化 WAIT 目标，缓解坍塌）；否则回退固定 `WAIT_TARGET = "<WAIT> Not enough information yet. More video is needed."`（兼容老数据）。

**最终产物**：`QA/results/{video_id}_training_samples.jsonl`，每行一条训练样本，三类：

```text
offline_answer     -> <ANSWER> clip_summary          (整段 clip 均匀采样)
streaming_wait     -> <WAIT> {wait_reason} / 固定兜底  (query_time 前末尾窗口)
streaming_answer   -> <ANSWER> {answer}                (answer_time 前末尾窗口)
```

样本 schema（简化）：

```json
{
  "sample_type": "streaming_wait",
  "video_id": "8V649L5Q368",
  "clip_idx": 1,
  "video_window": [167.0, 197.0],
  "question": "...",
  "target": "<WAIT> The pleural line is not yet centered between the rib shadows...",
  "qa_type": "next_observation",
  "meta": {"query_time": 197.0, "answer_time": 215.0, "wait_reason": "..."}
}
```

> 数据 schema 唯一权威：`QA/schema.md`。

---

## 7. SFT（Stage 2）

入口：`QA/train/train.py`

**训练目标**：answerability —— 证据不足输出 `<WAIT>`，证据充分输出 `<ANSWER> answer`。

统一输入形式：

```text
System Prompt
+ current_time 之前最后 N 帧 visual tokens
+ Question
→ <WAIT> 或 <ANSWER> answer
```

- 新增 special token `<WAIT>` / `<ANSWER>`，并让 `embed_tokens` / `lm_head` 通过 `modules_to_save` 可训练。
- 冻结 vision encoder；LoRA SFT；只对 assistant `target` 计算 loss（其余 `-100`）。
- Early Stopping + TensorBoard；T4 用 `--fp16`，Ampere/A100+ 用 `--bf16`。

**视觉输入构造**（`QA/train/video_sampling.py`；`video_window=[start,end]` 只定义时间范围，帧数由训练脚本决定，便于帧数消融）：

- streaming（`sample_last_n_frames`）：`current_time = video_window.end`，在 `[max(start, current_time-WINDOW_SIZE), current_time]` 内均匀取 `WINDOW_SIZE` 帧（"末尾 N 秒取 N 帧"，聚焦最近上下文；不足则复制末帧补齐）。
- offline（`sample_uniform_frames`）：整段 clip 均匀采样 `WINDOW_SIZE` 帧。

```bash
python QA/train/train.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --train-jsonl QA/results/{video_id}_training_samples.jsonl \
  --default-video-path path/to/{video_id}.mp4 \
  --output-dir /mnt/cache/qwenFT/qwen3vl_2b_lora_wait_answer \
  --window-size 8 --frame-size 336 \
  --num-train-epochs 100 \
  --per-device-train-batch-size 1 --gradient-accumulation-steps 4 \
  --learning-rate 2e-4 --bf16 --gradient-checkpointing \
  --early-stop-patience 3 --early-stop-min-delta 0.001
# smoke: 加 --limit 4 --num-train-epochs 1
# 显存紧张: --frame-size 336 --gradient-checkpointing --gradient-accumulation-steps 16
# 后续消融: WINDOW_SIZE ∈ {8,16,32}, FRAME_SIZE ∈ {336,448}
```

**产物**：SFT LoRA adapter。

---

## 8. 推理 / 评测

- `QA/eval/infer_qwen.py`：base model raw 推理（answerability baseline）。
- `QA/eval/infer_qwen_lora.py`：base + LoRA adapter 推理，`skip_special_tokens=False` 保留 `<WAIT>`/`<ANSWER>`，统计 answerability accuracy。
- `QA/eval/analyze_predictions.py`：分析预测结果。
- `QA/eval/test_openrouter_video_model.py`：OpenRouter 视频模型连通性/对照测试。
- `pretrain/infer.py`：预训练 adapter 的 caption 续写推理。
- `QA/test/check_collator_labels.py`：验证 collator 的 label mask 只监督 `<WAIT>`/`<ANSWER>` target。

SFT adapter 推理示例：

```bash
python QA/eval/infer_qwen_lora.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --adapter-path /mnt/cache/qwenFT/qwen3vl_2b_lora_wait_answer \
  --eval-jsonl QA/results/{video_id}_training_samples.jsonl \
  --default-video-path path/to/{video_id}.mp4 \
  --output /mnt/cache/qwenFT/predictions.jsonl \
  --window-size 8 --frame-size 336 --limit 20 --max-new-tokens 160 --bf16
```

---

## 9. 目录速查

```text
UltrasoundCrawler_KeyCode_20260323_v2/  1. 爬取
src/video_filter.py                     2. 过滤
scripts/video_filter_vlm.py             2. 过滤
QA/prepare/run_prepare.py               3. 数据准备（ASR + clipping 一键）
  QA/prepare/asr.py                     3. ASR（core: scripts/asr_pipeline.py）
  QA/prepare/clipping.py                3. Clipping（core: scripts/video_clipping.py, 默认 qwen_embed, no LLM）
pretrain/                               4-5. 预训练分支
  build_samples.py / dataset.py / collator.py / train.py / infer.py / video_sampling.py
QA/run.py                               6. QA 生成一体化入口
  offline_generator.py                  6.1 offline QA
  generator.py                          6.2 streaming QA（单锚点 + wait_reason）
  validator.py                          6.3 校验（含本地 wait_reason 质量门）
  merger.py                             6.4 合并/展开（WAIT 目标多样化）
  smoke_test.py                         6.5 本地逻辑自检
QA/train/                               7. SFT
  train.py / dataset.py / collator.py / video_sampling.py
QA/eval/                                8. 推理/评测
QA/schema.md                            数据 schema 唯一权威
```

---

## 10. 环境要点

- GPU 训练环境：Azure T4（`azureml_py38`，torch 2.9.1+cu128）。本项目 T4 环境 `is_bf16_supported=True`，可用 `--bf16`；否则用 `--fp16`。
- 大文件（模型下载 / checkpoint）放可写大盘，例如 `/mnt/cache`；根分区通常空间紧张。
- Transformers 在混合环境里可能误 import TensorFlow/Keras：训练脚本已在顶部 `os.environ.setdefault("TRANSFORMERS_NO_TF", "1")`，或运行前 export。
- QA 生成走 OpenRouter，需要 API key（`.env` 或 `--api-key`）。
- 依赖见 `requirements.txt`（含 `faster-whisper`、`scikit-image`(SSIM)、`tensorboard`）。

---

## 11. 两阶段训练关系

```text
Stage 1 (pretrain/)  : ASR caption completion   -> 学超声视觉-语言知识（不加 special token）
Stage 2 (QA/train/)  : WAIT/ANSWER answerability -> 学决策格式（加 <WAIT>/<ANSWER>）

两套代码解耦、可独立运行。
可选：Stage 2 从 Stage 1 的 adapter 继续，作为后续实验。
```
