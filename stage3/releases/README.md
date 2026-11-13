# Stage3 数据发布

2026-11-13 将既有本地标签及来源记录导出到 Git；没有重新生成标签或重新逐题视觉验收。

| 目录 | 内容 | 范围 |
|---|---|---|
| `visual_asr_20261112/` | 当前 Visual+ASR 系列封装 | 223 视频，1,015 QA，1,505 决策样本 |
| `legacy_visual_only_stage3_batch/` | 历史 Visual-only QA，保留作比较 | 50 视频，205 QA；任务格式，未展开训练行 |
| `../examples/two_video_overfit/` | P3 + CzLE 调试样本 | 2 train 视频，12 QA，57 决策样本 |

当前封装 train 为185视频/832 QA/1,237行；eval为38视频/183 QA/268行，视频ID不重叠。
根目录 train/eval 是全量；`data/memory_candidates/` 是690 QA、1,055决策样本的候选子集。
便捷训练脚本仍默认使用候选子集，不能把该运行称为1,015 QA全量训练。
325条旧audio/joint标签QA完整保留，但视觉充分性没有在本次逐题复核。
当前模型只消费视觉memory、问题和query/current时间，不会因为标签来自ASR就自动接收ASR输入。
211视频附标注用ASR，12视频没有现成ASR；完整ASR不能注入因果模型输入。

发布包含原任务、修订记录、审核文本和来源哈希；不包含源视频、审核图片、权重及本机video map。
审核文件内的原始绝对路径/图片引用是来源记录，不表示其他电脑上文件存在。
`source_checksums.json` 是原包档案；`checksums.json` 只针对本次实际导出的文件。
`validation.json` 是本机结构/加载器检查，不是新医学审核、GPU测试或模型效果。

## 另一台机器使用

先 `git pull --ff-only origin main`，再按实际视频根目录生成一个新映射：

```bash
RELEASE="$PWD/stage3/releases/visual_asr_20261112"
python "$RELEASE/code/make_video_map.py" --video-root /path/to/original_videos \
  --output /path/to/new_stage3_video_map.json
python "$RELEASE/code/validate.py" --project-root "$PWD" \
  --video-path-map /path/to/new_stage3_video_map.json
```

映射要求每个ID唯一匹配；检查器验证路径和样本合同，不替代实际视频内容哈希核验。
`code/build.py` 是原封装构建逻辑，依赖原来源文件；已发布样本可直接使用，无需在新机器重建。

选择候选子集训练（需自行分配GPU，且用正确的Stage2及可选Stage1权重链）：

```bash
export PROJECT_ROOT="$PWD"
export VIDEO_PATH_MAP=/path/to/new_stage3_video_map.json
export STAGE2_ADAPTER=/path/to/stage2/final
export OUTPUT_DIR=/path/to/new_stage3_run
# STAGE1_ADAPTER 仅在该 Stage2 使用了它时设置。
bash "$RELEASE/run_training.sh" --epochs 1 --dtype bf16 --frame-size 224
```

全量包保留小数秒，只监督已审核端点，不制造中间WAIT或负例。
现有双视频 `stage3.overfit/metrics` 不可原样套到全量小数端点；使用包内 `code/score_endpoints.py` 评估自回归预测。
历史 Visual-only tasks 和两视频 overfit 与全量包可能有重叠，不能直接拼接、重复计数或混用split。
