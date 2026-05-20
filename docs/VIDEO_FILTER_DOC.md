# 超声视频纯度过滤器 — 技术文档

> 脚本：`video_filter.py`  
> 目的：自动过滤含有人工标注/非超声内容的视频  
> 依赖：`opencv-python`、`numpy`

---

## 1. 概述

分析下载的超声视频，为每个视频计算一个**纯度评分（0-100）**。高分视频包含干净的超声扫查画面；低分视频包含PPT幻灯片、人脸讲课、重度标注或非超声内容。

### 过滤目标

| 伪影类型 | 描述 | 检测方法 |
|----------|------|----------|
| 文字标注 | 箭头、测量线、文字标签叠加在超声画面上 | 形态学文字检测 |
| PPT/幻灯片 | 教学幻灯片、示意图、项目符号列表 | 大面积亮色块检测 |
| 人脸讲课 | 讲师出现在画面中 | Haar级联人脸检测 |
| 彩色覆盖层 | 非超声的彩色区域（非多普勒） | HSV饱和度分析 |
| 非医学内容 | 动画、Logo、标题画面 | 综合启发式判断 |

---

## 2. 处理流程

```
输入：视频文件（.mp4, .webm, .mkv, .avi, .mov）
    │
    ├── [1] 均匀抽取 N 帧（默认8帧/视频）
    │
    ├── [2] 逐帧分析（5个维度）
    │       ├── 灰度比例
    │       ├── 彩色像素比
    │       ├── 文字区域得分
    │       ├── 人脸检测
    │       └── 大面积亮色块
    │
    ├── [3] 帧级得分聚合为视频级得分
    │
    ├── [4] 应用扣分规则
    │
    └── [5] 分类
            ├── pure_ultrasound  纯超声（≥75分）
            ├── mildly_annotated 轻微标注（55-74分）
            ├── heavily_annotated 重度标注（35-54分）
            └── rejected 拒绝（<35分）
```

---

## 3. 检测维度详解

### 3.1 灰度比例（Grayscale Ratio）

**目的：** 真实 B-mode 超声图像以灰度为主。

**方法：**
```python
# 对每个像素，检查 max(R,G,B) - min(R,G,B) 是否小于 25
max_channel = frame.max(axis=2)
min_channel = frame.min(axis=2)
diff = max_channel - min_channel
grayscale_mask = diff < 25
ratio = grayscale_mask.sum() / 总像素数
```

**判读标准：**
- `> 85%` → 很可能是纯超声
- `70-85%` → 可接受（可能有少量多普勒）
- `< 70%` → 很可能包含非超声内容（PPT、动画）

---

### 3.2 彩色像素比（Color Pixel Ratio）

**目的：** 检测大量彩色内容（非超声来源）。

**方法：**
```python
# 转HSV色彩空间，找饱和度高且亮度高的像素
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
colored_mask = (hsv[:,:,1] > 50) & (hsv[:,:,2] > 50)  # 饱和度>50 且 明度>50
ratio = colored_mask.sum() / 总像素数
```

**判读标准：**
- `< 15%` → 正常（超声画面极少彩色）
- `15-30%` → 可能有多普勒或轻微彩色覆盖
- `> 30%` → 很可能是非超声内容

**已知局限：** 彩色多普勒超声本身就有彩色区域。含多普勒的视频会被误扣分。

---

### 3.3 文字区域得分（Text Region Score）

**目的：** 检测叠加在超声画面上的文字标注。

**方法：**
```python
# 1. Canny边缘检测
edges = cv2.Canny(gray, 100, 200)

# 2. 形态学膨胀，连接相邻文字字符
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))  # 宽而矮的核
dilated = cv2.dilate(edges, kernel, iterations=1)

# 3. 查找符合文字特征的轮廓（宽且矮的矩形）
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    宽高比 = w / h
    # 文字块通常宽且矮（宽高比 > 2）
    if 宽高比 > 2.0 and 面积 > 200 and h < 帧高度 * 0.1:
        文字面积 += 面积

得分 = 文字面积 / 总面积
```

**判读标准：**
- `< 0.02` → 文字极少（可接受，可能只是机器参数显示）
- `0.02 - 0.05` → 中等文字（有标注存在）
- `> 0.05` → 大量文字覆盖

**已知局限：** 超声机器屏幕边缘始终显示参数文字（深度、增益、TGC、患者信息），这是不可避免的，会贡献文字得分。

---

### 3.4 人脸检测（Face Detection）

**目的：** 检测画面中是否出现讲课者面部。

**方法：**
```python
# OpenCV Haar级联正脸检测器
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 缩小帧尺寸加速检测（最大边320px）
faces = face_cascade.detectMultiScale(small_gray,
    scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

有人脸 = len(faces) > 0
```

**判读标准：**
- 0帧检测到人脸 → 纯扫查视频
- 1-2帧有人脸 → 偶尔出现讲师
- 4+帧有人脸 → 以讲课为主

---

### 3.5 大面积亮色块（Bright Uniform Blocks）

**目的：** 检测PPT幻灯片、白色背景、标题画面。

**方法：**
```python
# 1. 亮度阈值（像素值 > 200）
_, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# 2. 查找大面积连通亮区（> 帧面积的5%）
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 总面积 * 0.05:
        大亮区面积 += area

比例 = 大亮区面积 / 总面积
```

**判读标准：**
- `< 10%` → 正常（超声图像背景大多较暗）
- `10-30%` → 有一些亮色内容（可能有信息面板）
- `> 30%` → 很可能是PPT幻灯片或白色背景

**原理：** 真实超声图像有深色背景（扇区/矩形区域），内有不同灰度的组织回声。大面积均匀亮区几乎不可能出现在超声图像本身中。

---

## 4. 评分算法

### 起始分：100分

### 扣分规则

| 触发条件 | 扣分公式 | 最大扣分 |
|----------|----------|----------|
| 彩色比 > 15% | `min(30, 彩色比 × 100)` | 30分 |
| 灰度比 < 70% | `(0.7 - 灰度比) × 60` | 42分 |
| 检测到人脸 | `(有人脸帧数 / 总帧数) × 40` | 40分 |
| 文字得分 > 0.02 | `min(25, 文字得分 × 500)` | 25分 |
| 亮色块 > 10% | `min(35, 亮色块比 × 100)` | 35分 |

### 最终得分

```
纯度得分 = max(0, min(100, 100 - 总扣分))
```

### 分类阈值

| 分数范围 | 类别 | 含义 |
|----------|------|------|
| ≥ 75 | `pure_ultrasound` | 干净的超声扫查画面 |
| 55 – 74 | `mildly_annotated` | 以超声为主，有少量标注/彩色 |
| 35 – 54 | `heavily_annotated` | 有大量非超声内容 |
| < 35 | `rejected` | 以非超声内容为主 |

---

## 5. 使用方法

### 基本用法

```bash
cd /Users/I761836/Documents/Semester3/Guided\ Research
source .venv/bin/activate

# 分析默认输出目录中的所有视频
python video_filter.py

# 分析指定目录
python video_filter.py --input-dir "UltrasoundCrawler_KeyCode_20260323_v2/output/20260502_152417_youtube"

# 保存样本帧用于人工检查
python video_filter.py --save-frames

# 更严格阈值（只保留75分以上）
python video_filter.py --min-score 75

# 每个视频抽更多帧提高准确度
python video_filter.py --frames 16
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input-dir` | `UltrasoundCrawler output/` | 扫描视频的目录 |
| `--frames` | 8 | 每个视频抽取的帧数 |
| `--output-report` | `filter_report.json` | JSON报告输出路径 |
| `--save-frames` | 关闭 | 保存样本帧用于人工审核 |
| `--min-score` | 55.0 | "保留"的最低分数阈值 |

### 输出文件

| 文件 | 内容 |
|------|------|
| `filter_report.json` | 每个视频和每帧的详细分析数据 |
| `sample_frames/` | 每个视频3帧截图（10%、50%、90%位置），文件名含分类和分数 |

---

## 6. 实际运行结果

### 对11个下载视频的分析（2026-05-02）：

```
✅ [ 75.0/100] HmPR3D-Eekk    — 42秒, 灰度97.5%, 病例推理类
✅ [ 75.0/100] d07t_9YLmXs    — 18分钟, 灰度91.6%, 扫查教程类
⚠️  [ 70.3/100] 9gMGOaPBE0w   — 5分钟, 有亮色块(11.7%)
⚠️  [ 65.9/100] 6zJ7k5m4kJE   — 1.8分钟, 多普勒彩色(50.6%)
⚠️  [ 63.0/100] I1Bdp2tMFsY   — 1.7分钟, 彩色覆盖(25.3%)
❌ [ 50.0/100] 3I3NzAECfrQ   — 彩色32.1%, 检测到文字
❌ [ 47.9/100] adJepggTLd4   — 彩色55.8%, 1/8帧有人脸
❌ [ 40.2/100] 0AcrQEtQn_8   — 亮色块87.3%（基本是幻灯片）
🚫 [ 26.0/100] WyscpAee3vw   — 亮色块61%, 彩色22%
🚫 [ 20.9/100] hjGaM7B2GRI   — 5/8帧有人脸, 彩色54.5%
🚫 [  0.0/100] mgra4WihmfA   — 彩色60.2%, 有人脸, 亮色块, 文字
```

---

## 7. 已知局限与改进方向

### 当前局限

| 局限 | 影响 | 可能的改进 |
|------|------|-----------|
| 多普勒超声被误罚 | 含彩色多普勒的有效超声视频得分偏低 | 检测多普勒特有的颜色模式（仅出现在扫描区内） |
| 机器参数文字被计入 | 所有超声机器都显示设置文字 | 只检测中心超声区域内的文字 |
| Haar级联误检 | 超声纹理偶尔触发人脸检测 | 使用更鲁棒的人脸检测器（如MediaPipe） |
| 固定帧采样 | 可能漏掉短暂的PPT插入片段 | 增加帧数或使用场景切换检测 |
| 无运动分析 | 无法区分"冻结帧"和实时扫查 | 添加光流分析 |

### 改进方向

1. **超声区域检测** — 先定位画面中的超声扇区/矩形区域，只在该区域内做分析
2. **多普勒感知** — 区分彩色多普勒信号和非医学彩色覆盖
3. **时序一致性** — 标记内容类型变化的视频（如扫查和PPT交替出现）
4. **OCR集成** — 用实际文字识别区分机器参数和人工标注
5. **VLM辅助验证** — 对边界案例（50-75分）使用GPT-4o或Qwen2-VL进行二次确认

---

## 8. 代码架构

```
video_filter.py
│
├── find_videos()           ← 递归查找所有 .mp4/.webm 文件
│
├── analyze_video()         ← 主分析函数（每个视频）
│   │
│   ├── 均匀采样 N 帧
│   │
│   ├── analyze_frame() × N
│   │   ├── compute_grayscale_ratio()      灰度比例
│   │   ├── compute_color_pixel_ratio()    彩色像素比
│   │   ├── compute_text_score()           文字区域检测
│   │   ├── detect_face()                  人脸检测
│   │   └── compute_bright_uniform_blocks() 大亮区检测
│   │
│   ├── 聚合帧级得分 → 视频级得分
│   │
│   └── _compute_verdict()  ← 应用扣分规则，分配类别
│
├── print_report()          ← 控制台报告
├── save_report()           ← JSON导出
└── save_frame_samples()    ← 保存截图用于人工审核
```

---

## 9. 环境依赖

```
opencv-python>=4.8.0    # 图像处理、人脸检测
numpy>=1.24.0           # 数组运算
```

安装：
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install opencv-python numpy
```

无需GPU。MacBook Pro上每个视频约5秒完成分析。
