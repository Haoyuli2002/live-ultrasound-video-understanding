# 超声讲解视频抓取工具（YouTube + B站）

## 功能
- 英文版按钮：抓取 YouTube 超声扫描讲解视频。
- 中文版按钮：抓取 B 站超声扫描讲解视频。
- 自动输出：
  - `videos.csv`
  - `videos.jsonl`
  - `summary.json`
  - `failures.json`
  - `filtered_out.jsonl`
  - `raw/seed_payloads.jsonl`
  - `raw/detail_payloads.jsonl`
- 可选下载：
  - 音视频文件（尽量高清）
  - 缩略图（可作为配套超声图片）

## 分类逻辑
程序会按标题、简介、分段章节关键词自动归类为：
1. `扫查教学型`
2. `病例讲解型`
3. `器官系统教学型`

未命中分类关键词时会标记为 `待人工分类`。

## 过滤逻辑
- 仅保留命中超声关键词的视频。
- 启发式过滤疑似“纯音频/无画面”的内容。
- 语言过滤：
  - YouTube 任务偏向英文内容
- B站任务偏向中文内容

## 2026-03-22 改进
- YouTube 抓取改为“动态扩种 + 处理中补种”，不再固定卡在单批种子上限。
- 对 `详情为空` 增加二次回补尝试（shorts/watch 备用 URL）。
- 新增断点状态文件：`output\\_resume_state_youtube.json`，异常中断后可自动续跑。
- 新增更细粒度统计：`detail_attempted/detail_success/detail_empty`、下载成功/失败计数。

## 2026-03-23 改进
- 检索关键词支持自定义（Web 表单 + CLI `--keywords`）。
- 下载视频按分类目录保存（`media/scan_tutorial`、`media/case_reasoning`、`media/organ_system_lecture`、`media/uncategorized`）。
- 视频下载超时支持自定义（Web 表单 + CLI `--download-timeout-sec`）。

## 启动方式

### 方式一：前端按钮（推荐）
```powershell
python -m pip install -r requirements.txt
python webapp.py
```
浏览器打开 `http://127.0.0.1:5088`，点击对应按钮运行。
点击后页面会显示 **实时日志面板**（后台任务状态、进度日志、完成/失败状态）。

### 方式二：命令行
```powershell
python cli.py --source youtube --max-results 80 --search-per-term 25 --download-media
python cli.py --source bilibili --max-results 80 --pages-per-term 4 --download-media
python cli.py --source youtube --keywords "lung ultrasound tutorial,ivc pocus scan" --download-timeout-sec 45
```

## 输出目录
每次任务单独输出到：
`output/YYYYMMDD_HHMMSS_youtube` 或 `output/YYYYMMDD_HHMMSS_bilibili`

## 说明
- 由于平台反爬与内容形态复杂，`“无超声图像”` 的判断是启发式，建议交付前人工抽检。
- 分类是自动初分，适合后续人工复核。
- 部分视频可能因地区、登录或风控无法下载，失败会记录在 `failures.json`。
