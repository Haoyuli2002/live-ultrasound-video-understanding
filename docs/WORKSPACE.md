# 主仓库与数据目录

跨电脑接手请先阅读 [Stage2 交接文档（含 Stage3 数据与 SSH 视频核验）](STAGE2_HANDOFF_20261113.md)。

本机主仓库：`/Users/I761836/Desktop/Semester 3/live-ultrasound-video-understanding`。
后续修改、测试、提交和推送均从这里进行。
`Documents/Codex/2026-10-23/wo-y/work/stage2-overfit-ab` 是历史工作副本。

## 代码入口

- Stage1：`stage1/train.py`；评测 `scripts/slurm/submit_stage1_eval.sh`。
  Base / LoRA 共用训练目录的 `training_config.json` 输入设置，自回归保存预测。
- Stage2 当前全量：`scripts/slurm/submit_stage2_reviewed.sh` → `stage2/train_full.py`。
  `DATASET_VERSION=visual_only|visual_asr` 选择监督版本；模型输入仍是视频。
  `ordinary|qformer` 选择结构；详见 `stage2/FULL_TRAINING.md`。
- Stage2 保留入口：`stage2/train.py` 及 `run_stage2_train_ddp.sbatch` 是旧的合并 loss / 每分钟六次更新训练器，不能与全量独立监督结果混称。
- Stage3：`stage3/train.py`、`stage3/inference.py`；小样本训练与生成见 `stage3/OVERFIT.md`。
- Demo：`scripts/viz_demo.sh`，需要本地视频及 `viz/data/` 来源文件。

## 数据与版本控制

- `cluster_data/stage2/`：本地 Visual+ASR 审核快照与发布副本；不直接提交 Git。
- `stage2/releases/accepted_v4_20261105_990b9b48/`：已入 Git 的 Visual-only 发布。
- `stage2/releases/accepted_av_asr_v1/`：已入 Git 的 Visual+ASR 发布。
- `cluster_data/`、`viz/data/`、视频、权重、日志、转写/抓取缓存保留本地。
- Stage3 标签已独立发布到 `stage3/releases/`，包括当前 Visual+ASR 封装与历史 Visual-only QA；双视频调试标签在 `stage3/examples/two_video_overfit/`。原视频、mock 权重和审核图片仍保留本地，详见 `stage3/releases/README.md`。
- `test.py` 与 `notebooks/fence_reinforcement_practice.ipynb` 是个人练习，保留原位置并由本机 Git exclude 排除。
- `livecc` 是嵌套仓库；其中原有 teaser 视频删除未提交，保持原状，不替用户恢复或发布。

## 本次整合与备份（2026-11-13）

Desktop 从 `fb6af49` 快进到 `0d5ddc3` 后，恢复本地改动并解决四处冲突。
完整改动备份位于：
`/Users/I761836/Documents/Codex/2026-10-23/wo-y/outputs/desktop_repo_cleanup_20261113_123520/`。
同时保留名为 `Desktop integration backup before syncing main 2026-11-13` 的 Git stash。

Stage2 通用模型接口随本次整合变化。旧 smoke / full run 的代码快照不变；
新提交仍执行实现哈希门禁，若旧 smoke 报告的哈希不匹配，需要用当前实现重跑 smoke，不能绕过检查。
本次只整理代码，不提交或取消任何 Slurm 作业，也不重新做数据视觉验收。
