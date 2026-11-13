# SSH 验证清单（Stage 1 / 2 / 3）

> 历史记录：下文保留 2026-10-20 的进展和命令，不代表当前数据、训练状态或默认入口。当前入口见 [主仓库说明](WORKSPACE.md)，Stage2 全量训练见 [FULL_TRAINING.md](../stage2/FULL_TRAINING.md)。请勿直接按旧作业编号取消或重提任务。

更新日期：2026-10-20。本文把需要在 **SSH 集群上实际核验**的事项整理成可勾选清单。原则一句话：**“代码已实现 ≠ SSH 上已验证”**。需要验证的本质有三类：①代码里有、但从未在真机跑过的链路；②数据与产物只有“代码能生成”、没人逐条核对过内容；③当前在跑作业的真实状态与吞吐瓶颈。

配套文档：三阶段进展见 `docs/STAGE_1_2_3_PROGRESS_zh.md`，命令见 `docs/HOW_TO_RUN_PIPELINE.md`，设计见 `README_zh.md`。

约定：
- `$REPO` = `/dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding`
- `$DATA` = `/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding`
- 日志：`$REPO/logs/stage1_pretrain_{jobid}.{out,err}`
- ⚠️ 标记为“代码已确认存在的坑”，文档虽列为待办，但会直接导致 SSH 验证失败或误判，必须优先处理。

每条勾选时请记录：**真实路径、作业 ID、关键日志片段**，作为“全量完成”的凭据。

---

## P0 — 当前阻塞主线：Stage 1 吞吐与产物

这是整条 pipeline 的关键路径，Stage 2/3 全都在等 Stage 1 的 `final/` adapter（2B 为关键路径）。

- [ ] **1. 旧作业是否还在跑、用的是不是旧代码（`num_workers=0`）**
  - `squeue -u $USER`
  - `scontrol show job 5861391 5861392 | grep -E 'JobState|SubmitTime|Command'`
  - 判据：若 `SubmitTime` 早于 worker 修复提交时间 → 用的是旧代码 → 必须 `scancel 5861391 5861392` 后重交，修复才会生效。

- [ ] **2. worker 修复是否真生效：GPU 利用率 / step 时间**
  - 重交后：`tail -f $REPO/logs/stage1_pretrain_<jobid>.err`
  - 另开：`srun --jobid=<id> --pty nvidia-smi dmon -s u`（或 `watch -n2 nvidia-smi`）
  - 判据：GPU `util` 不再长期停在 0%；日志 `s/it` 明显低于旧的 190–253s/step。

- [ ] **3. 4B 的 allocator OOM 风险**
  - `grep -iE 'memory allocation failed|OutOfMemory|CUDA out of memory' $REPO/logs/stage1_pretrain_5861392.err`
  - 判据：若只是 C++ caching-allocator 的 `[W...] memory allocation failed with OOM` 警告（**不是** Python `OutOfMemoryError`/`RuntimeError`），则 `_is_cuda_oom`（`stage1/train.py:73-77`）**不会**触发 120→48→36→24 的自动帧预算回退 → 4B 需用更低 `FRAME_BUDGET=32` 重交做显式保护。

- [ ] **4. Stage 1 产物：是否真出了 `final/` adapter**
  - `ls -la $DATA/.../*_2b_full/{checkpoint-*,final}/`
  - 判据：`adapter_model.safetensors`、`adapter_config.json`、`training_config.json` 存在且非空。

- [ ] **5. 正式作业用的输入 JSONL 是不是合并后的 193/186 视频**
  - `cat <output-dir>/training_config.json`（看 `train_jsonl`）
  - 对照 sbatch / `scontrol show job` 里的 `TRAIN_JSONL`
  - ⚠️ 判据：Slurm 默认 `TRAIN_JSONL` 仍指旧文件；193 视频主线必须显式指定合并 JSONL，否则跑的是旧数据。

---

## P1 — Stage 1 数据内容审核（结构核验替代不了内容审核）

> **已核实状态（2026-10-20）：当前 Stage 1 训练 = 186 视频，不是 193。**
> preflight 日志 `Stage 1 preflight: 186 videos (map covers 193), 53244 rows, 17748 targets`
> ——路径 map 列了 193，但样本只覆盖 186。训练 JSONL 为
> `$DATA/cluster_data/pretrain/train_full295_stage1_samples.jsonl`（425MB，10/1 生成）。
> **决策：先用 186 跑出 `final/` adapter 解锁 Stage 2 smoke，7 个 pilot 视频留到后面补成 193。**
>
> **7 个 pilot 视频（`OKtdZVXtZ0M OQJ2RhnuOQQ TlckvYhqaFE WhNxq98szE8 YYcTdkP7xhs d7IwDHAVcGQ vmmrjy1bhEc`）
> 当前 `clean_asr` 全部失败**（audit：`cluster_data/QA/train_full295/stage1_seven_clean_audit.jsonl`，
> 作业 5832433，7/7 `"status":"failed"`）。根因不是转录出错，而是清洗器护栏拒了 teacher
> (Qwen3.5-27B) 的输出——失败类型：`Correction is not grounded` / `Spurious lexical correction`
> / `Undeclared lexical change`（LLM 臆造/多余/未声明的改词），以及 `WhNxq98szE8` 的
> `LLM did not return segments JSON`（被 `max_tokens` 截断）。护栏行为正确（防幻觉进训练数据）。
>
> **补 193 已放弃（2026-10-20 决定）：主线以 186 视频为准，不再补齐。**
> `stage1.seven_video_pilot` 合并代码与 `stage1.verify_merged_193` 核验器保留但不再是主线路径。
> （若将来仍想补：改用 `stage1/restore_punctuation.py` 纯标点恢复不改词 / 对截断视频提高
> `--max-tokens` / 收紧 teacher prompt 禁止 lexical correction，再 `stage1.data` 生成样本后 merge。）

- [ ] **6. 训练 JSONL 是否就是 186 视频主线数据**
  - `jq -r '.video_id' "$DATA/cluster_data/pretrain/train_full295_stage1_samples.jsonl" | sort -u | wc -l` 应 = 186
  - 行数 ÷ 3（三条件）= target 句数，应与 preflight 的 `17748 targets` 一致。
  - （193 已放弃；`stage1.verify_merged_193` 核验器保留但非主线。）

- [ ] **7. train / eval 视频不重叠**
  - 分别取 train、eval split 的 `video_id`，`comm -12 <(sort train_ids) <(sort eval_ids)` 应为空。

- [ ] **8. 句子时间戳插值准确性**（人工抽查）
  - 随机抽几条 `[start,end)`，用 `ffmpeg -ss <start> -to <end>` 截该时段音频，听是否对得上目标讲解句。

---

## P1 — Stage 2 双卡 GPU smoke（文档明确“双卡 smoke 待跑”）

直接下一步，依赖 P0 的 Stage 1 `final/`。

- [ ] **10. `run_stage2_ddp_smoke.sbatch` 双卡能跑通**
  - `--nproc_per_node=2 --validation-fraction 0 --verify-ddp-gradients`，全新 `--output-dir`，≥2 视频。
  - 判据：日志同时出现 `[ddp-check] synchronized gradient` **和** `[smoke] PASS`。

- [ ] **11. ⚠️ 梯度同步验证覆盖面不足（代码已确认）**
  - `_verify_ddp_gradients`（`stage2/train.py:250`）只比较 **第一个** LoRA 参数，**没覆盖 `<SHORT_MEM>`/`<LONG_MEM>` 两行 embedding** 的梯度。
  - 需手动 all-gather 这两行 embedding 的 grad 做跨 rank 一致性验证，或增强该函数后再下结论。

- [ ] **12. Stage 1 adapter 完整加载链**
  - smoke 时 `--export ...,STAGE1_ADAPTER=<..._2b_full/final>`
  - 判据：base → Stage 1 merge → Stage 2 加载全链不报错。

- [ ] **13. gradient-checkpointing（`use_reentrant=False`）与 DDP 交互**
  - 判据：smoke 日志无 “parameters that didn't receive grad” / “mark_ready twice” 等 DDP 报错。

---

## P1 — Stage 2 数据构建（批量标注前必须修）

- [ ] **14. ⚠️ `VIDEO_IDS=""` 空字符串回退 bug（代码已确认）**
  - `scripts/slurm/run_stage2_annotate_build.sbatch` / `stage2/prepare_ddp_smoke.py` 使用 `VIDEO_IDS=${VIDEO_IDS:-default}` → 空串仍回退默认单视频。
  - `README_zh.md` 的 `VIDEO_IDS=""` 全量标注示例**照用会只跑一个视频**。批量前先 `bash -n` + 小规模实跑，确认真的遍历了 split 内全部视频。

- [ ] **15. 每完整分钟恰好 6 local + 1 block + 1 global，窗口左闭右开**
  - 对一个真实视频跑 `stage2.annotate` + `stage2.build_data`，用 `jq` 统计每个 block 的 label 数与窗口 `[T-60,T)` / `[0,T)`。

- [ ] **16. 尾部不足 60s 不虚构整分钟摘要**
  - 判据：最后一个不完整分钟只有完整的十秒 local，无 block / global。

---

## P2 — Stage 3（训练 CLI 尚未实现，先别提交全量）

代码已确认：`stage3/model.py` 只有双模型加载 + 单样本 BCE/NTP forward，**没有训练 CLI / 数据加载器 / 优化器 / checkpoint / 在线生成**。

- [ ] **17. 双模型 GPU smoke（待先实现 loader + loop）**
  - 用真实 Stage 1/2 adapter：核验显存、memory 侧 **无梯度**、decision 侧 LoRA/新 token/BCE head **有梯度**、WAIT/ANSWER 标签一致、损失有限。

- [ ] **18. QA 数据 `answer_time` 是否“视觉首次可答时刻”**（人工）
  - 现有 QA 工具受 ASR/讲解影响，`answer_time` 不是已验证的视觉首次充分时刻。需专家按视觉证据重审，SSH 上按决策时刻重放视频核对。

---

## 建议的最小动作顺序

1. **先做 P0 第 1–5 项即可解锁主线**：`squeue` 确认旧作业用旧代码 → `scancel 5861391 5861392` → 带 `DATALOADER_NUM_WORKERS=6` + 降 `FRAME_BUDGET`（2B=48 / 4B=32）、保持同一 output-dir 以便 `--resume` 续训重交 → 盯 GPU util 和 step 时间（第 2、3 项）。
2. 等 `..._2b_full/final/` 出来 → 进 Stage 2 双卡 smoke（第 10–13 项），注意第 11 项梯度验证覆盖面。
3. Stage 2 批量标注前先修第 14 项回退 bug。
4. Stage 3 待先实现 loader + 训练循环，再做第 17 项 GPU smoke。

每次状态升级，保留：数据清单、配置、代码版本（commit）、日志、checkpoint 路径、评测输出。
---

## 附：Stage 1 重交已验证命令（2026-10-20，worker 修复后）

背景：作业 5861391(2B)/5861392(4B) 用 04:31 提交的旧代码（`num_workers=0`），
`156–229 s/it`、ETA 32–35 天 ≫ 14h TimeLimit，且两者都在 allocator OOM 边缘，必须重交。
修复 commit `8d327a0` 已 push 并在集群 `git pull` 到位（`--dataloader-num-workers` 已生效）。

旧作业真实配置（`sacct` + `.out` 核实）：
- 2B：`MODEL_NAME=Qwen/Qwen3.5-2B`，`RUN_NAME=stage1_qwen35_comparison_2b_full`，JobName `q35-2b-train`
- 4B：`MODEL_NAME=Qwen/Qwen3.5-4B`，`RUN_NAME=stage1_qwen35_comparison_4b_full`，JobName `q35-4b-train`

```bash
scancel 5861391 5861392
sleep 5 && squeue -u $USER

# 2B: workers=6 + frame_budget 48
sbatch --job-name=q35-2b-train \
  --export=ALL,RUN_NAME=stage1_qwen35_comparison_2b_full,MODEL_NAME=Qwen/Qwen3.5-2B,DATALOADER_NUM_WORKERS=6,FRAME_BUDGET=48 \
  scripts/slurm/run_stage1_pretrain_merged.sbatch

# 4B: workers=6 + frame_budget 32（OOM 更严重，给更大余量）
sbatch --job-name=q35-4b-train \
  --export=ALL,RUN_NAME=stage1_qwen35_comparison_4b_full,MODEL_NAME=Qwen/Qwen3.5-4B,DATALOADER_NUM_WORKERS=6,FRAME_BUDGET=32 \
  scripts/slurm/run_stage1_pretrain_merged.sbatch
```

重交后 ~5–8 分钟验证（对应 P0 #2 / #3）：

```bash
NEW2B=<新2B作业号>; NEW4B=<新4B作业号>
grep -E 'Output:|frame_budget|preflight' logs/stage1_pretrain_${NEW2B}.out
grep -iE 's/it|it/s' logs/stage1_pretrain_${NEW2B}.err | tail -5     # 应远低于 156 s/it
grep -iE 's/it|it/s' logs/stage1_pretrain_${NEW4B}.err | tail -5
grep -iE 'memory allocation failed|OutOfMemory|CUDA out of memory' logs/stage1_pretrain_${NEW4B}.err | tail
srun --jobid=${NEW2B} --pty nvidia-smi dmon -s u                     # GPU util 不再长期 0%
```

注意：`--resume` 已内置于 sbatch；旧作业仅 26/19 步、`SAVE_STEPS=50` 未存过 checkpoint，
故保留同名 output-dir 续训此刻等于从 0，不会与 120 帧旧状态冲突。
在 login 节点（`base` 环境）跑单测会 `ModuleNotFoundError: No module named 'torch'`，
属环境问题而非代码问题；需 `conda activate ultrasound` 或在作业节点上验证。

