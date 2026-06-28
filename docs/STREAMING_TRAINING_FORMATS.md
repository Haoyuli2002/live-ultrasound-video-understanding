# 超声流式视频理解的两种训练格式

## 格式 A：Interleaved ASR Pretraining

### 目的

让模型先学习：

```text
超声视频帧 ↔ 同步讲解 ASR
```

也就是学习超声视频流和讲解文本之间的时间对齐关系。

这个阶段可以帮助模型理解超声视频的流式特征，但它**不直接训练 QA 能力**。

### 输入 / 输出

```text
输入：Context + 当前时间段 Frames
输出：当前时间段对应的 ASR Words
```

### 序列形式

```text
Context
Frames_0 → Words_0
Frames_1 → Words_1
Frames_2 → Words_2
...
```

### 示例

```text
Context:
这是一个肾脏超声教学视频。

User:
Time=0.0-3.0s
[ultrasound frames]

Assistant:
we are placing the probe in the flank ...

User:
Time=3.0-4.0s
[ultrasound frames]

Assistant:
now the kidney is coming into view ...
```

### 训练模型学到什么

格式 A 主要训练模型：

- 以流式方式处理超声视频；
- 将视觉变化和同步讲解联系起来；
- 学习超声相关语言和术语；
- 生成与视频时间对齐的短文本片段。

### 在项目中的作用

格式 A 适合用于：

```text
domain-adaptive streaming pretraining
```

也就是超声领域的流式预训练。

在项目早期，这一步是可选的。  
当我们有足够多的超声讲解视频和高质量 ASR 后，再做格式 A 会更有价值。

---

## 格式 B：Online QA SFT

### 目的

让模型学会：

```text
基于当前已经看到的视频，以及可选的 ASR，回答在线问题
```

也就是训练真正的流式超声理解能力。

模型只能使用 `query_time` 之前已经出现的信息，不能依赖未来视频或未来 ASR。  
其中 `Seen ASR` 是可选输入：当 ASR 不存在、质量不可靠，或者任务希望评估纯视觉理解能力时，模型应只基于 `Seen Frames` 和 `Question` 回答。

### 输入 / 输出

```text
输入：Context + Seen Frames + [Optional Seen ASR] + Question
输出：Answer
```

### 序列形式

```text
Context
Seen Frames up to query_time
[Optional] Seen ASR up to query_time
Question
→ Answer
```

#### Video Clip

#### Proactive QA
```text
Context
Seen Frames up to query_time
[Optional] Seen ASR up to query_time
Question （判断什么时候能够回答）
→ Answer
```

#### Reactive QA
```text
Context
Seen Frames up to query_time <从 Frames bank 里面提取和当前最相关的帧：Reactive QA>
    1. VLM-as-Encoder 
    2. Reactive QA: Q-Former - 视觉表征向量 
    3. Proactive：Q-Former - 视觉表征向量 - lightweight MLP for BCE -> Yes , No, No, Yes, Retrieve frames / visual tokens
    4. MLLM Decodoer -> 
[Optional] Seen ASR up to query_time
Question ---
→ Answer
```

### 示例

```text
Context:
你是一个超声助手。只能基于目前已经看到的视频和听到的讲解回答问题。

User:
Observed video: 0.0-18.0s
[ultrasound frames]

ASR so far:
"... now we are trying to obtain a long-axis kidney view ..."

Question:
What is the sonographer currently trying to do?

Assistant:
The sonographer appears to be trying to obtain a longitudinal kidney view and bring the full renal contour into the field of view.
```

### 两种输入模式

格式 B 支持两种主要输入模式。

#### Mode 1：Video-only Online QA

```text
Context + Seen Frames + Question → Answer
```

适合：

- 没有音频 / ASR 的任务；
- 静音超声视频；
- 只有机器屏幕录制的视频；
- 测试模型纯视觉超声理解能力；
- 部署时只有超声画面输入；
- 避免 ASR 信息泄露。

#### Mode 2：Video + ASR Online QA

```text
Context + Seen Frames + Seen ASR + Question → Answer
```

适合：

- 超声教学视频；
- 有医生或讲师讲解的视频；
- 需要结合操作者语言判断意图的任务；
- 需要利用 narration 辅助理解的任务。

### 训练模型学到什么

格式 B 主要训练模型回答在线超声问题，例如：

- 当前切面识别；
- 当前可见解剖结构识别；
- 图像质量评估；
- 操作者意图理解；
- 下一步动作预测；
- 下一步操作指导；
- 当前可见发现相关的医学知识解释。

### Online 约束

对于严格的 online QA，输入和答案都必须只基于 `query_time` 之前的信息。

也就是说：

```text
允许使用：
- clip_start 到 query_time 之间的视频帧
- clip_start 到 query_time 之间的 ASR（如果有）
- 用户问题

不允许使用：
- query_time 之后的未来视频帧
- query_time 之后的未来 ASR
- 只有在后续片段中才出现的发现或结论
```

### 在项目中的作用

格式 B 是本项目的核心格式，适合用于：

```text
online ultrasound QA / instruction tuning
```

如果训练资源有限，应该优先做格式 B，而不是格式 A。

---

## 两种格式对比

| 格式 | 输入 | 输出 | 主要目标 |
|---|---|---|---|
| 格式 A：Interleaved ASR Pretraining | Context + Frames | ASR Words | 学习流式视频-语言时间对齐 |
| 格式 B：Online QA SFT | Context + Seen Frames + [Optional Seen ASR] + Question | Answer | 学习在线超声问答能力 |

---

## 推荐训练顺序

长期理想路线是：

```text
格式 A：Interleaved ASR Pretraining
→ 格式 B：Online QA SFT
```

也就是：

1. 先让模型通过超声视频和 ASR 学会流式视觉-语言对齐；
2. 再用 online QA 数据训练它回答超声相关问题。

但对于本项目早期，更现实的路线是：

```text
先构建 Online QA 数据
→ 优先训练 / 评估格式 B
→ 当积累足够多超声讲解视频和 ASR 后，再加入格式 A
```

---

## 关键设计原则

```text
Pretraining:
Context + Frames → Words

QA SFT:
Context + Seen Frames + [Optional Seen ASR] + Question → Answer
```

其中：

- `Words` 可以作为预训练目标，也可以作为 QA 阶段的 seen ASR 上下文；
- `Seen ASR` 是可选输入，不应该成为模型回答 online QA 的硬性依赖；
- `Question` 通常应该是输入；
- `Answer` 才是 QA 训练时的监督输出。

---

## 简短结论

如果目标是训练流式超声视频理解模型，推荐使用两阶段思想：

```text
格式 A：让模型懂超声视频流和讲解之间的时间关系
格式 B：让模型学会基于当前已见内容回答在线问题
```

但在项目初期，最重要的是先做好格式 B：

```text
Context + Seen Frames + [Optional Seen ASR] + Question → Answer
```

因为它最直接对应我们的核心目标：

```text
online ultrasound video understanding
```


