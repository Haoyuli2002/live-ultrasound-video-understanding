# 数据流水线

> Live Ultrasound Video Understanding — 数据处理全流程

---

## 总览

1. **数据采集（Crawler）** — 从 YouTube/B站搜索并下载超声相关视频
2. **数据清洗（Filter）** — 分析视频帧，过滤非纯超声画面
3. **ASR 转录（WhisperX）** — 语音识别，获取逐词时间对齐文本
4. **QA 生成（GPT-4o）** — 结合视频帧+ASR，自动生成多类型问答对
5. **模型训练（Qwen2-VL）** — 在超声QA数据上微调流式视频理解模型
6. **评估（Benchmark）** — 在领域测试集和通用测试集上评估

---

## Step 1：数据采集

### 工具
`UltrasoundCrawler_KeyCode_20260323_v2/`

### 做什么
从 YouTube 和 B站 自动搜索并下载超声相关视频。

### 输入
搜索关键词（如 "ultrasound real-time scanning", "超声检查 全程"）

### 输出
```
output/YYYYMMDD_HHMMSS_youtube/
├── videos.jsonl          # 视频元数据（标题、描述、时长、章节等）
├── media/                # 下载的视频文件（按分类子目录）
│   ├── scan_tutorial/
│   ├── case_reasoning/
│   └── organ_system_lecture/
├── thumbnails/           # 缩略图
└── raw/                  # 原始API数据
```

### 运行方式
```bash
# Web UI
cd UltrasoundCrawler_KeyCode_20260323_v2
source ../.venv/bin/activate
python webapp.py  # 打开 http://127.0.0.1:5088

# 命令行
python cli.py --source youtube --max-results 100 --download-media \
  --keywords "ultrasound real-time scanning,POCUS live scan,abdominal ultrasound full exam"
```

### 关键参数
| 参数 | 说明 |
|------|------|
| `--max-results` | 最多保留视频数 |
| `--search-per-term` | 每个关键词检索数量 |
| `--download-media` | 是否下载视频文件 |
| `--keywords` | 自定义搜索关键词 |

---

## Step 2：数据清洗

### 工具
`video_filter.py`

### 做什么
- 过滤：
  - 对每个视频**每分钟采样 10 帧**，自动分析每帧的 7 个视觉维度，过滤含有人工标注（文字、箭头、PPT、人脸讲课）的视频，只保留纯超声扫查画面。 
- 去标注：
  - 如果视频大部分内容是带有人工标注的超声视频的话，应用CV或者Stable Diffusion Model来去标注。

### 输入
Step 1 下载的视频文件目录

### 输出
```
filter_report.json        # 每个视频的详细分析数据
sample_frames/            # 每个视频的截图（用于人工确认）
```

### 运行方式
```bash
source .venv/bin/activate
python video_filter.py --input-dir "UltrasoundCrawler_KeyCode_20260323_v2/output/最新目录" --save-frames
```

### 过滤维度
| 维度 | 纯超声特征 | 非超声特征 |
|------|-----------|-----------|
| 灰度比例 | > 85% | < 70%（有彩色PPT） |
| 彩色像素 | < 15% | > 30%（彩色动画） |
| 文字区域 | < 2%（仅机器参数） | > 5%（大量标注） |
| 人脸检测 | 0帧 | 多帧有人脸 |
| 亮色块面积 | < 10% | > 30%（PPT白底） |

### 分类结果
- **≥ 75分**：纯超声 ✅ → 进入下一步
- **55-74分**：轻微标注 ⚠️ → 可考虑保留
- **< 55分**：拒绝 ❌ → 丢弃

---

## Step 3：ASR 转录

### 工具
[WhisperX](https://github.com/m-bain/whisperX)（large-v3-turbo 模型）

### 做什么
对通过过滤的视频进行语音识别，获得**逐词时间对齐**的转录文本。

### 输入
Step 2 保留的视频文件（.mp4）

### 输出格式
```json
{
  "video": "path/to/video.mp4",
  "content": [
    [0.00, 0.45, "Now"],
    [0.45, 0.82, "we're"],
    [0.82, 1.20, "looking"],
    [1.20, 1.55, "at"],
    [1.55, 1.90, "the"],
    [1.90, 2.40, "right"],
    [2.40, 2.95, "kidney"],
    [2.95, 3.50, "in"],
    [3.50, 4.10, "longitudinal"],
    [4.10, 4.80, "view"]
  ],
  "title": "Renal Ultrasound Scan",
  "language": "en"
}
```

每个词是一个三元组：`[开始时间(秒), 结束时间(秒), 词]`

### 运行方式（参考 LiveCC）
```bash
# 需要 GPU 环境
pip install whisperx

# 单个视频
whisperx video.mp4 --model large-v3-turbo --language en --output_format json

# 批量处理（参考 livecc/data/production/distributed_whisperx.py）
torchrun --nproc_per_node=8 distributed_whisperx.py --input_dir videos/ --output_dir transcripts/
```

### 后处理：视频分段
参考 LiveCC 的 `sft_to_clips.py`，按以下规则切分：
- 片段长度：30-240秒
- 片段起始必须是句子开头（大写字母，前一词以句号结尾）
- 遇到超过3秒的静默则断开

---

## Step 4：QA 生成

### 工具
GPT-4o / Claude API（AI Agent 方式）

### 做什么
对每个视频片段，结合 ASR 文本和视频帧，自动生成多种类型的问答对。

### 输入
- 视频片段 + ASR 转录（Step 3 输出）
- 视频关键帧（抽取5-10帧发给 Vision API）

### 输出格式
```json
{
  "video": "clips/kidney_scan_001.mp4",
  "video_start": 0.0,
  "video_end": 30.5,
  "asr_text": "Now we're looking at the right kidney in longitudinal view...",
  "text_stream": [
    [0.0, 0.45, "Now"],
    [0.45, 0.82, "we're"],
    ...
  ],
  "qa_pairs": [
    {
      "timestamp": 2.0,
      "type": "scene_description",
      "question": "What anatomical structure is currently visible?",
      "answer": "The right kidney is visible in longitudinal section..."
    },
    {
      "timestamp": 2.0,
      "type": "sonographer_intent",
      "question": "What is the sonographer trying to assess?",
      "answer": "The sonographer is evaluating renal morphology..."
    },
    {
      "timestamp": 5.0,
      "type": "action_guidance",
      "question": "How can I better visualize the upper pole?",
      "answer": "Tilt the probe cranially and ask the patient to take a deep breath..."
    }
  ]
}
```

### QA 类型

| 类型 | 代号 | Prompt 模板 |
|------|------|------------|
| 基础知识 | `basic_knowledge` | "基于当前扫查部位，需要知道哪些基础知识？" |
| 场景描述 | `scene_description` | "描述当前画面中可见的解剖结构" |
| 操作指导 | `action_guidance` | "如何调整探头才能更好地显示目标结构？" |
| 操作者意图 | `sonographer_intent` | "操作者当前在做什么/想找什么？" |
| 细粒度属性 | `fine_grained` | "描述具体位置、大小、回声特征等" |

### Agent Prompt
```python
SYSTEM_PROMPT = """你是一位资深超声科主治医师和教学专家。
给定一段超声扫查视频的片段和对应的ASR转录，请从以下5个角度生成问答对：

1. SCENE（场景描述）：当前画面中可见什么解剖结构？客观描述。
2. INTENT（操作意图）：操作者在做什么/在寻找什么？
3. GUIDANCE（操作指导）：学习者应该如何操作才能获得更好的切面？
4. KNOWLEDGE（先验知识）：与当前扫查相关的医学知识是什么？
5. FINE-GRAINED（细粒度属性）：描述具体位置、大小、回声特征等。

每个问答对请标注对应的时间戳。输出JSON格式。"""
```

### 质量控制
- GPT-4o 先判断 ASR 是否在描述实时画面内容（参考 LiveCC 的 `make_prompt.py`）
- 过滤掉"闲聊"、"个人经历分享"、"多人对话"类内容
- 人工抽检 200-500 条确认质量

---

## Step 5：模型训练

### 基础模型
**Qwen2-VL-7B** + LiveCC 流式架构

### 训练数据格式（LiveCC 兼容）
```json
[
  {
    "role": "user",
    "content": [
      {"type": "video", "video": "clips/kidney_001.mp4", "video_start": 0.0, "video_end": 30.5},
      {"type": "text", "text": "请描述当前超声画面中可见的解剖结构。"}
    ]
  },
  {
    "role": "assistant",
    "content": [
      {"type": "text_stream", "text_stream": [[0.0, 2.5, "当前"], [2.5, 4.0, "可见"], [4.0, 6.0, "右肾"], ...]}
    ]
  }
]
```

### 训练配置
```bash
export VIDEO_MIN_PIXELS=78400
export FPS_MAX_FRAMES=480
export VIDEO_MAX_PIXELS=19267584

torchrun --nproc_per_node=8 train.py \
  --deepspeed zero2.json \
  --pretrained_model_name_or_path Qwen/Qwen2-VL-7B \
  --annotation_paths ultrasound_train.jsonl \
  --freeze_modules visual \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 64 \
  --bf16 True
```

### 训练策略
| 阶段 | 数据 | 目的 |
|------|------|------|
| Stage 1（可选） | 大规模超声视频+ASR | 领域适配 |
| Stage 2 | 超声QA数据集 | 学习多类型问答 |
| Stage 3 | 流式数据（text_stream） | 学习实时输出 |

---

## Step 6：评估

### 评估基准

| Benchmark | 衡量什么 | 指标 |
|-----------|---------|------|
| **UltrasoundQA（我们的）** | 领域特定实时理解 | LLM-Judge 胜率, 属性F1 |
| **VideoMME** | 通用视频理解 | 准确率 |
| **OVOBench** | 在线视频理解 | 准确率 |

### Baseline 对比

| 模型 | 特点 |
|------|------|
| Qwen2-VL-7B（zero-shot） | 无超声训练，通用能力 |
| GPT-4o Vision | 强大但非流式 |
| LiveCC-7B-Instruct | 流式但非医学 |
| **Ours** | 流式 + 医学 + 意图建模 |

### LLM-as-Judge 评估协议
```python
judge_prompt = """比较以下两个关于超声视频的回答。
参考答案：{reference}
回答A：{model_a_output}
回答B：{model_b_output}
哪个回答更准确、更有临床价值、更详细？请给出判断和理由。"""
```

### 评估维度

| 维度 | 评估方式 |
|------|---------|
| 解剖结构识别准确性 | 与ground truth对比 |
| 操作意图推断正确性 | LLM-Judge |
| 指导建议可行性 | 医学专家评审 |
| 细粒度属性准确性 | 属性级别的Precision/Recall/F1 |
| 实时性 | 输出延迟测量 |

---

## 数据流转

1. YouTube 视频 → **Step 1 Crawler** → 原始视频 + 元数据 (videos.jsonl)
2. 原始视频 → **Step 2 video_filter.py** → 纯超声视频 (score ≥ 55)
3. 纯超声视频 → **Step 3 WhisperX** → 视频 + 逐词 ASR: [[start, end, word], ...]
4. ASR 输出 → **Step 3.5 sft_to_clips** → 30-240 秒片段 + ASR
5. 片段 + ASR → **Step 4 GPT-4o Agent** → 片段 + ASR + QA 对 (5种类型)
6. QA 数据 → **Step 4.5 质量过滤 + 人工抽检** → 高质量训练数据 (JSONL)
7. 训练数据 → **Step 5 Qwen2-VL-7B 训练** → 超声实时理解模型
8. 模型 → **Step 6 Benchmark 评估** → 性能指标 + 对比报告

---

## 当前进度

| 步骤 | 状态 | 说明 |
|------|------|------|
| Step 1 数据采集 | ✅ 完成 | 已下载11个视频 |
| Step 2 数据清洗 | ✅ 完成 | 筛出5个可用视频（score≥55） |
| Step 3 ASR转录 | 🔲 待做 | 需要GPU环境运行WhisperX |
| Step 4 QA生成 | 🔲 待做 | 需要OpenAI/Claude API |
| Step 5 模型训练 | 🔲 待做 | 需要8×A100 |
| Step 6 评估 | 🔲 待做 | 需要先完成benchmark构建 |

---

## 文件组织结构（目标）

```
ultrasound_benchmark/
├── raw_videos/                    # Step 1 原始下载
├── filtered_videos/               # Step 2 过滤后保留
├── transcripts/                   # Step 3 ASR转录结果
│   └── whisperx_output.jsonl
├── clips/                         # Step 3.5 切分后片段
│   ├── kidney_001.mp4
│   └── ...
├── annotations/                   # Step 4 QA标注
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── checkpoints/                   # Step 5 模型权重
└── evaluation/                    # Step 6 评估结果
    ├── results.json
    └── judges/
```
