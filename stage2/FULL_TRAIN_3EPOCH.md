# Stage 2 全量训练 — 3 Epoch 运行说明

目标：在原有单 epoch 全量训练（job 5919253）基础上，跑一个 **3 epoch** 的版本，
用于观察多 epoch 下 short/block/global loss 是否逐步下降（单 epoch 看不出收敛）。

## 关键事实

- **不需要改训练代码**。核心提交脚本 `scripts/slurm/submit_stage2_full_training.sh`
  已经把 `EPOCHS` 作为环境变量（默认 1），并透传给
  `stage2.train_full --epochs`（见 `run_stage2_full_training.sbatch:17`）。
- 本次只新增一个薄 wrapper：`scripts/slurm/submit_stage2_full_training_3ep.sh`，
  它固定 `EPOCHS=3` 后调用核心脚本，其余参数全部沿用默认
  （lr=1e-4，save_steps=100，nproc=4，walltime=24h，frame_size=224）。

## 前置条件（提交前确认）

1. **不能有同名 job 在跑**。核心脚本会拒绝已有 `stage2-full-qformer` 的提交
   （`submit_stage2_full_training.sh:69-70`）。先让当前 1-epoch 的 `5919253`
   结束（预计还需约 4 小时），或先 `scancel 5919253`。
2. **代码已提交**。核心脚本要求 `stage2/` 与两个 slurm 脚本无未提交改动
   （第 28-30 行）。本 wrapper 已随本次提交入库。
3. **qformer 冒烟门禁**。qformer 架构要求 `QFORMER_SMOKE_REPORT` 指向一份
   `status=functional_smoke_passed` 的 `smoke_report.json`，且其
   `implementation_sha256` 与当前 `stage2/{model,qformer,short_encoding,train_overfit_ab}.py`
   一致（第 51-63 行）。若这些文件自上次冒烟后有改动，需要先重跑冒烟。

## 在 SSH 集群上的执行步骤

```bash
cd ~/haoyu/live-ultrasound-video-understanding
git pull                      # 拉到本次新增的 wrapper

# 若当前 1-epoch job 还在跑，先等它结束或取消：
# squeue -u $USER
# scancel 5919253             # 需要时才执行

# qformer 需要指向一份通过的冒烟报告（用你最近那次冒烟）：
export QFORMER_SMOKE_REPORT=/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding/cluster_data/experiments/stage2/qformer_smoke_P3_20261112T111302Z_aq2X2Z/smoke_report.json

# 1) 先 DRY RUN 预检（不创建 job，只跑所有门禁并打印 sbatch 命令）
DRY_RUN=1 bash scripts/slurm/submit_stage2_full_training_3ep.sh qformer

# 2) 预检通过后，真正提交
bash scripts/slurm/submit_stage2_full_training_3ep.sh qformer
```

提交成功会打印 `Job: <id>` 和 `Run: <run_dir>`。记下 run_dir。

## 时长预估

- 单 epoch ≈ 4465 step、约 12.9 小时（按 ~347 step/小时实测）。
- **3 epoch ≈ 13395 step、约 38~39 小时** → **会超过单次 24h walltime**。
- 已配置 `--resume` 且每 100 step 存 `checkpoint-latest`，所以撞墙后可续训：

```bash
# 续训：指向同一个 run 目录，用同一 code commit
RESUME_RUN=<上面的 run_dir> EPOCHS=3 \
  bash scripts/slurm/submit_stage2_full_training_3ep.sh qformer
```

> 提示：也可以用 `WALLTIME=48:00:00`（若分区允许）一次跑完，
> 在提交前 `export WALLTIME=48:00:00`。能否用取决于集群 QOS 上限。

## 跑完后做什么（质量验收，不是看 loss）

loss 绝对值不代表成功。跑完（或每个 epoch 存盘后）应对 checkpoint 跑
`stage2.reconstruct`，在留出视频上看 real vs zero vs other_window 的重建判别性，
方法同 `stage2/OVERFIT_RECON_5919286.md` 的逐条对照。
