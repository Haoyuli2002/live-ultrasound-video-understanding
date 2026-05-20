# Agentic Benchmark Construction Pipeline

> 用 AI Agent 自动化构建超声视频理解 Benchmark

---

## 总览

输入：搜索关键词 + 质量要求
输出：可直接用于训练/评估的超声视频 QA 数据集

---

## Agent 节点

### 1. Crawler Agent

- 根据关键词自动搜索 YouTube/B站
- VLM 判断缩略图/标题是否与超声相关
- 下载相关视频 + 元数据

### 2. Filter Agent

- 每分钟采样 10 帧，逐帧分析视觉特征
- VLM 判断每帧是否为纯超声画面
- 自动裁剪 ROI / 按时间分段，只保留干净片段

### 3. Transcription Agent

- WhisperX 逐词 ASR 转录
- VLM 判断 ASR 是否在描述画面内容（过滤闲聊/无关内容）
- 按规则切分为 30-240 秒片段

### 4. QA Generation Agent

- 结合视频帧 + ASR，多角色 prompt 生成 5 种类型 QA：
  - Scene（场景描述）
  - Intent（操作意图）
  - Guidance（操作指导）
  - Knowledge（先验知识）
  - Fine-grained（细粒度属性）
- 每条 QA 带时间戳对齐

### 5. Quality Control Agent

- 自动检验 QA 准确性与一致性
- 剔除低质量/幻觉内容
- 生成质量统计报告

---

## 与现有工作的对比

| 维度 | OpenClaw | 本项目 |
|------|----------|--------|
| 领域 | 机器人操作 | 医学超声视频 |
| 数据来源 | 仿真环境生成 | YouTube 真实视频 |
| Agent 输入 | 任务描述 | 视频 + 音频 + 文本 |
| Agent 输出 | 机器人任务集 | 视频 QA 对 |
| 多模态 | 无 | 视觉 + 语音 + 文本 |
| 验证方式 | 仿真执行 | LLM-Judge + 人工抽检 |

---

## 可扩展性

- **新器官**：更换搜索关键词即可扩展到心脏/甲状腺/乳腺等
- **新语言**：支持中英文 ASR + 多语 prompt
- **新任务类型**：增加 QA 模板即可生成新类型标注