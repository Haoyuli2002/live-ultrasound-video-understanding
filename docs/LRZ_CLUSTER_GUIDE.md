# LRZ Cluster 常用操作总结

本文档记录本项目在 LRZ AI Systems / Slurm cluster 上的常用操作流程，尤其是 GPU interactive shell、`sbatch`、job 监控、环境激活、日志查看和本项目 pretrain/eval 相关命令。

---

## 1. 登录 LRZ

本地 Mac terminal：

```bash
ssh ai
```

如果没有配置 `ai` alias，则使用：

```bash
ssh ge75vid2@login.ai.lrz.de
```

登录后通常会进入 home 目录：

```bash
/dss/dsshome1/04/ge75vid2
```

可以用：

```bash
pwd
```

确认当前路径。

---

## 2. 进入项目 repo

本项目在 LRZ 上的路径是：

```text
/dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding
```

进入 repo：

```bash
cd /dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding
```

---

## 3. 激活 conda 环境

在新开的 login shell 或 compute shell 里，`conda` 可能不在 PATH 中。先加载 conda 初始化脚本：

```bash
source /dss/dsshome1/04/ge75vid2/miniconda3/etc/profile.d/conda.sh
conda activate ultrasound
```

确认 Python 环境：

```bash
which python
python -c "import torch; print(torch.__version__)"
```

如果在 GPU 节点上，可以确认 CUDA：

```bash
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("bf16:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None)
PY
```

---

## 4. 检查当前有哪些 job

查看当前用户所有 Slurm job：

```bash
squeue -u ge75vid2
```

更详细版本：

```bash
squeue -u ge75vid2 -o "%.18i %.20P %.20j %.8u %.2t %.10M %.10l %.20S %.40R"
```

字段含义：

```text
JOBID       job id
PARTITION   分区
NAME        job 名称
USER        用户
ST          状态
TIME        已运行时间
TIME_LIMIT  最大运行时间
START_TIME  预计开始时间
NODELIST(REASON) 节点名或排队原因
```

状态含义：

```text
PD = pending，排队中
R  = running，正在运行
CG = completing，快结束
```

取消某个 job：

```bash
scancel <job_id>
```

例如：

```bash
scancel 5720242
```

只统计当前用户 job 数量：

```bash
squeue -h -u ge75vid2 | wc -l
```

---

## 5. 申请 GPU interactive shell

### 5.1 推荐：MCML H100

通常本项目优先尝试 MCML H100：

```bash
srun \
  --partition=mcml-hgx-h100-94x4 \
  --qos=mcml \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=04:00:00 \
  --pty bash
```

如果需要更短任务，例如 ASR / eval debug：

```bash
srun \
  --partition=mcml-hgx-h100-94x4 \
  --qos=mcml \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=02:00:00 \
  --pty bash
```

### 5.2 LRZ H100

如果想用 LRZ H100：

```bash
srun \
  --partition=lrz-hgx-h100-94x4 \
  --qos=gpu \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=04:00:00 \
  --pty bash
```

### 5.3 LRZ A100

A100 80GB 对 Qwen3-VL-2B LoRA 也可以使用：

```bash
srun \
  --partition=lrz-hgx-a100-80x4 \
  --qos=gpu \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=04:00:00 \
  --pty bash
```

---

## 6. 进入 GPU 节点后要做什么

进入 GPU 节点后，会看到类似：

```text
(base) ge75vid2@mcml-hgx-h100-021:~$
```

先进入 repo：

```bash
cd /dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding
```

加载 conda 并激活环境：

```bash
source /dss/dsshome1/04/ge75vid2/miniconda3/etc/profile.d/conda.sh
conda activate ultrasound
```

确认 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

---

## 7. 检查 GPU shell 什么时候能启动

提交 `srun` 后会显示 job id，例如：

```text
srun: job 5720686 queued and waiting for resources
```

在另一个 login terminal 中检查：

```bash
squeue -j 5720686 -o "%.18i %.20P %.20j %.8u %.2t %.10M %.10l %.20S %.40R"
```

重点看：

```text
START_TIME
NODELIST(REASON)
```

如果 `START_TIME` 是具体时间：

```text
2026-07-29T10:30:00
```

则表示预计开始时间。

如果是：

```text
N/A
```

表示 Slurm 暂时无法给出可靠预计时间。

看更详细信息：

```bash
scontrol show job 5720686 | egrep "JobState|Reason|StartTime|EndTime|SubmitTime|EligibleTime|RunTime|TimeLimit|Partition|QOS|Gres|TRES|NodeList"
```

---

## 8. 检查当前 GPU shell 还剩多少时间

先找 job id：

```bash
squeue -u ge75vid2
```

然后查看详细信息：

```bash
scontrol show job <job_id> | egrep "JobState|StartTime|EndTime|RunTime|TimeLimit|Partition|NodeList"
```

例子：

```text
RunTime=01:15:10
TimeLimit=02:00:00
EndTime=2026-07-29T10:29:01
```

说明这个 job 已运行 1 小时 15 分钟，总时长 2 小时，还剩约 45 分钟。

也可以用 `squeue` 看简表：

```bash
squeue -j <job_id> -o "%.18i %.20P %.20j %.2t %.10M %.10l %.20S %.40R"
```

其中：

```text
TIME       已运行时间
TIME_LIMIT 最大运行时间
```

剩余时间约等于：

```text
TIME_LIMIT - TIME
```

---

## 9. 查看 GPU 使用情况

必须在 GPU compute node 上执行：

```bash
nvidia-smi
```

如果看到 Python 进程占用 GPU 显存，说明训练或推理正在使用 GPU。

查看当前用户进程：

```bash
ps -u $USER -o pid,ppid,cmd,%cpu,%mem --sort=-%cpu | head -30
```

---

## 10. 查看有哪些 GPU partition

查看所有 GPU 相关 partition：

```bash
sinfo -o "%P %a %l %D %t %G %N" | grep -Ei "gpu|h100|a100|v100|p100|lrz|mcml"
```

常用组合：

```text
mcml-hgx-h100-94x4 + qos=mcml
lrz-hgx-h100-94x4  + qos=gpu
lrz-hgx-a100-80x4  + qos=gpu
```

查看某个 partition：

```bash
sinfo -p mcml-hgx-h100-94x4 -o "%P %a %l %D %t %G %N"
```

```bash
sinfo -p lrz-hgx-h100-94x4 -o "%P %a %l %D %t %G %N"
```

---

## 11. `srun` vs `sbatch`

### 11.1 `srun --pty bash`

`interactive shell`，适合：

```text
debug
短任务
手动运行 ASR / inference
检查环境
```

特点：

- 你会进入 GPU shell。
- 需要手动输入命令。
- terminal 断开后任务有风险。
- 不适合非常长的训练。

示例：

```bash
srun \
  --partition=mcml-hgx-h100-94x4 \
  --qos=mcml \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=04:00:00 \
  --pty bash
```

### 11.2 `sbatch`

后台批处理任务，适合：

```text
长训练
长 eval
不想一直保持 SSH 连接的任务
```

特点：

- 提交脚本后自动排队和运行。
- 不依赖当前 terminal。
- 输出写到 log 文件。
- 更适合正式长任务。

提交：

```bash
sbatch scripts/slurm/pretrain_train50.slurm
```

查看：

```bash
squeue -u ge75vid2
```

取消：

```bash
scancel <job_id>
```

---

## 12. 查看 sbatch 日志

本项目 sbatch 日志一般写到：

```text
cluster_data/logs/
```

例如：

```text
cluster_data/logs/pretrain_train50_<job_id>.out
cluster_data/logs/pretrain_train50_<job_id>.err
```

实时看 stdout：

```bash
tail -f cluster_data/logs/pretrain_train50_<job_id>.out
```

实时看 error：

```bash
tail -f cluster_data/logs/pretrain_train50_<job_id>.err
```

如果 log 文件不存在，通常说明 job 还没开始运行。

查看完成状态：

```bash
sacct -j <job_id> --format=JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS
```

---

## 13. Git 操作：本地 push，cluster pull

推荐代码源头在本地 Mac。

### 13.1 本地 Mac push

先看状态：

```bash
git status --short
```

只 add 代码/文档，不 add 数据：

```bash
git add <code_or_doc_files>
git commit -m "<message>"
git push origin main
```

不要 push：

```text
cluster_data/
local_checkpoints/
QA/results_*/
*.mp4
*.wav
checkpoint-*
```

### 13.2 LRZ pull

```bash
cd /dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding
git pull origin main
```

如果 pull 被 untracked 文件挡住，例如：

```text
error: The following untracked working tree files would be overwritten by merge
```

可以把本地 cluster 文件备份到 `cluster_data/local_backups/`：

```bash
mkdir -p cluster_data/local_backups
mv <file> cluster_data/local_backups/<file>.$(date +%Y%m%d_%H%M%S)
git pull origin main
```

---

## 14. 本项目常用路径

Repo：

```text
/dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding
```

训练视频：

```text
cluster_data/videos/train/
```

评估视频：

```text
cluster_data/videos/eval/
```

split：

```text
cluster_data/splits/train_videos.json
cluster_data/splits/eval_videos.json
```

ASR transcripts：

```text
cluster_data/QA/train/transcripts/
cluster_data/QA/eval/transcripts/
```

pretrain samples：

```text
cluster_data/pretrain/train_pretrain_samples.jsonl
cluster_data/pretrain/eval_pretrain_samples.jsonl
cluster_data/pretrain/train_pretrain_sentence_samples.jsonl
```

checkpoints：

```text
cluster_data/checkpoints/pretrain_qwen3vl_train50/
```

eval outputs：

```text
cluster_data/eval/
```

本地 checkpoint 备份：

```text
local_checkpoints/
```

---

## 15. Pretrain 训练常用命令

训练 50-video segment-level pretrain：

```bash
python pretrain/train.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --train-jsonl cluster_data/pretrain/train_pretrain_samples.jsonl \
  --video-path-map cluster_data/splits/train_videos.json \
  --output-dir cluster_data/checkpoints/pretrain_qwen3vl_train50 \
  --window-size 4 \
  --frame-size 224 \
  --num-train-epochs 3 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --bf16 \
  --gradient-checkpointing \
  --early-stop-patience 3 \
  --early-stop-min-delta 0.001
```

断点续训：

```bash
python pretrain/train.py \
  ... \
  --resume-from-checkpoint auto
```

指定 checkpoint 续训：

```bash
python pretrain/train.py \
  ... \
  --resume-from-checkpoint cluster_data/checkpoints/pretrain_qwen3vl_train50/checkpoint-2800
```

---

## 16. Eval 常用命令

### 16.1 检查 eval transcripts

```bash
echo eval_transcripts $(find cluster_data/QA/eval/transcripts -name "*.json" 2>/dev/null | wc -l)
```

期望：

```text
eval_transcripts 20
```

### 16.2 构建 eval pretrain samples

```bash
python pretrain/build_samples.py \
  --transcripts cluster_data/QA/eval/transcripts \
  --output cluster_data/pretrain/eval_pretrain_samples.jsonl \
  --window-sec 8 \
  --min-words 3 \
  --context-max-chars 400 \
  --unit segment
```

### 16.3 检查 / 转码 eval 视频

只检查：

```bash
python scripts/transcode_videos_for_opencv.py \
  --video-path-map cluster_data/splits/eval_videos.json \
  --quiet-opencv
```

自动修复：

```bash
python scripts/transcode_videos_for_opencv.py \
  --video-path-map cluster_data/splits/eval_videos.json \
  --replace \
  --delete-backup \
  --quiet-opencv
```

期望：

```text
bad_count: 0
```

### 16.4 Base full inference，支持断点续跑

```bash
python pretrain/infer.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --no-adapter \
  --eval-jsonl cluster_data/pretrain/eval_pretrain_samples.jsonl \
  --video-path-map cluster_data/splits/eval_videos.json \
  --output cluster_data/eval/pretrain_eval20_base_full.jsonl \
  --window-size 4 \
  --frame-size 224 \
  --max-new-tokens 128 \
  --bf16 \
  --quiet \
  --resume
```

如果要覆盖旧结果：

```bash
--overwrite
```

### 16.5 LoRA full inference，支持断点续跑

```bash
python pretrain/infer.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --adapter-path cluster_data/checkpoints/pretrain_qwen3vl_train50 \
  --eval-jsonl cluster_data/pretrain/eval_pretrain_samples.jsonl \
  --video-path-map cluster_data/splits/eval_videos.json \
  --output cluster_data/eval/pretrain_eval20_lora_full.jsonl \
  --window-size 4 \
  --frame-size 224 \
  --max-new-tokens 128 \
  --bf16 \
  --quiet \
  --resume
```

### 16.6 对比 base vs LoRA

```bash
python pretrain/compare_predictions.py \
  --base cluster_data/eval/pretrain_eval20_base_full.jsonl \
  --lora cluster_data/eval/pretrain_eval20_lora_full.jsonl \
  --output cluster_data/eval/pretrain_eval20_compare_full.jsonl \
  --embed-device cpu
```

检查行数：

```bash
wc -l cluster_data/pretrain/eval_pretrain_samples.jsonl
wc -l cluster_data/eval/pretrain_eval20_base_full.jsonl
wc -l cluster_data/eval/pretrain_eval20_lora_full.jsonl
```

三者应一致，例如：

```text
2199
2199
2199
```

---

## 17. 保存 checkpoint 到本地

训练完成后，把 LoRA adapter 拉回本地 Mac：

```bash
/bin/mkdir -p local_checkpoints

/usr/bin/rsync -av --progress -e /usr/bin/ssh \
  --exclude 'checkpoint-*' \
  --exclude 'runs' \
  ai:/dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding/cluster_data/checkpoints/pretrain_qwen3vl_train50/ \
  local_checkpoints/pretrain_qwen3vl_train50/
```

本地检查：

```bash
/usr/bin/du -sh local_checkpoints/pretrain_qwen3vl_train50
/bin/ls -lh local_checkpoints/pretrain_qwen3vl_train50
```

---

## 18. 常见问题

### 18.1 `conda: command not found`

执行：

```bash
source /dss/dsshome1/04/ge75vid2/miniconda3/etc/profile.d/conda.sh
conda activate ultrasound
```

### 18.2 `cv2` 找不到

说明没激活 `ultrasound` 环境：

```bash
source /dss/dsshome1/04/ge75vid2/miniconda3/etc/profile.d/conda.sh
conda activate ultrasound
python -c "import cv2; print(cv2.__version__)"
```

### 18.3 视频 OpenCV 读不了

检查：

```bash
python scripts/transcode_videos_for_opencv.py \
  --video-path-map cluster_data/splits/eval_videos.json \
  --quiet-opencv
```

修复：

```bash
python scripts/transcode_videos_for_opencv.py \
  --video-path-map cluster_data/splits/eval_videos.json \
  --replace \
  --delete-backup \
  --quiet-opencv
```

### 18.4 interactive shell 没了

先检查中间产物：

```bash
wc -l cluster_data/eval/pretrain_eval20_base_full.jsonl 2>/dev/null || echo "no base output yet"
```

重新申请 GPU shell 后用 `--resume` 续跑。

### 18.5 output 已存在，infer 报错

默认保护输出，避免误覆盖。

续跑：

```bash
--resume
```

覆盖：

```bash
--overwrite
```

### 18.6 Slurm 显示 `Priority`

说明 job 配置没错，只是在等调度优先级。

查看预计开始：

```bash
squeue -j <job_id> -o "%.18i %.20P %.20j %.2t %.10M %.10l %.20S %.40R"
```

如果 `START_TIME=N/A`，说明 Slurm 暂时无法预测。

---

## 19. 建议工作方式

- 代码改动：本地 Mac 修改、commit、push。
- LRZ：只 pull 代码，跑实验。
- 大数据和 checkpoint：不要进 Git。
- 长任务：优先 `sbatch`。
- 短调试：使用 `srun --pty bash`。
- eval/inference：使用 `--resume`，避免 shell 中断后重跑。