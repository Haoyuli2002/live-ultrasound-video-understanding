# Stage2 交接：供另一台笔记本的 Codex 接手

更新日期：2026-11-13。本文根据本机代码、冻结 manifest 和用户在聊天中提供的 Slurm 回执整理；本次没有连接 SSH，不能把历史排队状态当作当前状态。

## 1. 接手目标与约束

我们在做流式超声视频的分层记忆学习，比较 ordinary memory 与秒级 Q-Former，使用 Local / Block / Global 摘要监督。数据有 Visual-only 和 Visual+ASR 两版。

- 主仓库是 Desktop 仓库，已将本地独立改动与远端合并并分批 push。
- 不覆盖旧数据、旧训练目录、原始生成文本和审核证据。
- 训练完成 / 功能 smoke 通过不等于生成质量合格。
- 对模型效果的评测必须看不输入参考答案的自回归结果；teacher-forced loss 只能作为诊断。
- 新作业提交前检查 squeue、sacct 和已有提交记录，防止重复。
- 不绕过 Q-Former smoke 实现哈希检查。

## 2. 仓库与本地文件

GitHub：https://github.com/Haoyuli2002/live-ultrasound-video-understanding
分支：main。本文核查的最新提交：2e507bb595072e215c491d348acc2a1fb47239f9。

原笔记本主仓库：
`/Users/I761836/Desktop/Semester 3/live-ultrasound-video-understanding`

历史工作副本（不再作为主仓库）：
`/Users/I761836/Documents/Codex/2026-10-23/wo-y/work/stage2-overfit-ab`

原始 Visual+ASR 生成/审核档案：
`/Users/I761836/Documents/Codex/2026-10-23/wo-y/outputs/stage2_av_asr_full_20261105/codex_native_w7ft9xf1`

本地主仓库下 `cluster_data/stage2/` 保存整理后的审核快照、video_map、videos.json 和发布副本。它不在 Git 中。已冻结的标签和 manifest 在下述 `stage2/releases/` 中，已入 Git。

视频主要在 `cluster_data/videos_from_ssh/{train_full295,eval_full295}/`；Git 不包含原视频、模型权重、虚拟环境、审核帧和完整原始审核档案。另一台笔记本需要单独传输这些文件或使用可访问的源路径。文档里的原电脑绝对路径不是另一台电脑上已存在的路径。

Desktop 整理前的改动已备份到：
`/Users/I761836/Documents/Codex/2026-10-23/wo-y/outputs/desktop_repo_cleanup_20261113_123520/`
并保留了 stash。主仓库目前仅剩 livecc 嵌套仓库内原有 teaser 视频删除，未提交。不要为“清理”擅自恢复或删除用户文件。

## 3. 两版 Stage2 数据

| 字段 | Visual-only | Visual+ASR |
|---|---|---|
| DATASET_VERSION | visual_only | visual_asr |
| Git 中发布目录 | stage2/releases/accepted_v4_20261105_990b9b48 | stage2/releases/accepted_av_asr_v1 |
| 视频 | 196 train + 41 eval | 196 train + 41 eval |
| 总监督目标 | 23,450 | 23,450 |
| JSONL 分组行数 | 2,328 train + 721 eval | 2,328 train + 721 eval |
| 模型训练输入 | 视频 | 视频 |
| 标签来源 | 视觉证据 | 视觉为主，窗口内 ASR 补充 |

Visual+ASR：17,860 个 train 目标、5,590 个 eval 目标。一行 JSONL 是分组，不是单个监督目标。完整分钟有 6 Local + 1 Block + 1 Global；尾段保留完整 10 秒的 Local，不为不足一分钟的尾段制造 Block/Global。

两版 manifest 均为 `stage2_accepted_release_v1`、training_ready=true、full_dataset_complete=true。原始审核快照的 training_ready=false 是预冻结状态，不能与冻结发布混淆。

Visual-only 保留用户明确接受的首轮核验豁免，不能声称每条都是重新原尺寸完整视频盲审。Visual+ASR 发布继承已有逐窗审核记录；本次冻结做的是哈希、覆盖、split 等结构核验，没有重做全量视觉内容审核。

Visual+ASR manifest 声明 label_modalities=[video,audio_via_window_asr,asr]，direct_audio_listening=false。即使用了音频转写的信息，不能声称每条都独立听过音频。两版 clinical_expert_review=false。

用户约定的审核采样：每 10 秒固定 +1/+5/+9 秒原尺寸帧；Block 汇合当前分钟六组、共 18 帧；Global 使用从开头到当前时刻的所有已抽帧。抽帧不能证明连续运动或每一帧均被检查。

生成/核验边界：Local 只取当前 10 秒；Block 取当前完整 60 秒并核对 Local；Global 只用当前时刻以前的证据。Visual+ASR 的文字增量也应时间对齐。发现问题直接修正并保留原文、原因、证据，旧版本不覆盖。

### 发布文件哈希

Visual-only train：860620f15bc9e618ad5a58454fe24fcdfa5c927fbca4c96b5de346c6051c32df
Visual-only eval：97a7e4c2c9238eb14fc8a77c54c862fe690208fa74c7998be001671e0ff13cfe
Visual+ASR train：0571ce7ad2039cb9dc93e8230a2a23bd3521f325014fb67b31b95bdfb9363e84
Visual+ASR eval：6d75a51e43db19af804fa308268fdfe0821762e78cb82f9404681790f9ffb368

## 4. 当前模型与训练入口

基座 Qwen/Qwen3.5-2B，Qwen hidden width D=2048。全量默认合并新版 Stage1 adapter，然后训练新 Stage2 参数。训练输入每秒一帧、224 像素 padding；原尺寸是标注/核验设置，不是训练输入大小。

Ordinary：每帧视觉 patches + 一个共享 SHORT_MEM token，经共享 Qwen backbone，取 SHORT_MEM hidden state 为一个 Si；10 个 Si 重建 Local；60 个 Si 重建 Block；旧 Long + 当前 60 Short + 60 个 LONG_MEM readout 更新 60 个 Long，重建 Global。

Q-Former 当前为 qformer_secondwise_v2：
- Short 和 Long 是两个独立参数的模块；width=512、2 层、8 heads。
- Short 每秒复用 1 个 learned query；self-attention 读取当前分钟已有的 512-d Short 历史以及当前 query；cross-attention 读取当前帧 visual patches。
- 新 512-d state 留在分钟内缓存，另投影为 2048-d Short 输出；不是把历史 2048-d 输出反复降维。每分钟/视频重置缓存。
- 每 10 秒输出 10 个 Short 供 Local；一分钟累积 60 个供 Block。
- Long 的 60 个 learned queries 在 512-d 内 self-attention；cross-attention 读取投影后的旧 Long 和本分钟 60 Short；输出再投影为 60×2048，供 Global 和下分钟递归。
- 因为 Short 读取分钟内历史，Local memory 并非仅含本 10 秒信息；评测需检查旧信息重复污染。

当前全量入口：
`scripts/slurm/submit_stage2_reviewed.sh ordinary|qformer [visual_only|visual_asr]`
→ `submit_stage2_full_training.sh` → `stage2.train_full`。

默认每组 4 张 H100 或 A100 80GB、1 epoch、24h、lr=1e-4、每100步保存；eval 不进入训练。Local/Block/Global 分别监督。历史 Long detach，视频按时间顺序保持在同一 rank。

保留的 `stage2.train` 与 `run_stage2_train_ddp.sbatch` 是旧的每分钟6次更新、分钟末合并三项 loss 的训练器，不能与当前全量方案混淆。

数据选择通过 DATASET_VERSION，写入提交/训练配置与 resume invariants。visual 是 visual_only 兼容别名。Visual+ASR 显式 override 优先，其次 `$DATA/cluster_data/stage2/releases/accepted_av_asr_v1`，不存在则用 Git 中发布；外部存在但无效时应报错，不静默改用另一个版本。

## 5. 已确认实验与未知状态

下面状态来自聊天中用户贴出的回执，不是2026-11-13实时查询。

| 作业 | 用途 | 最后确认 |
|---|---|---|
| 5902188 | 旧 Q-Former smoke | 已取消，被秒级版本替代 |
| 5903390 | 秒级 Q-Former，P3，3 epoch | COMPLETED，0:0，11分48秒；functional_smoke_passed |
| 5919286 | 秒级 Q-Former，P3，24 epoch overfit + 自回归 | 已提交，一张A100、4h；最后回执 PENDING，需查当前状态 |
| 5903250 | ordinary 全量，Visual-only | 已提交；没有本次可确认的完成证据 |
| 5919253 | Q-Former 全量，Visual-only | 已提交；最后回执 PENDING，需查当前状态 |

未见 Visual+ASR 全量训练的提交回执；不能宣称已经开始或完成。

3 epoch smoke 共48更新（16监督目标×3），48生成结果（16目标×real/zero/other_window）；无空输出、无截断；Short/Long 参数发生有限更新。报告 quality_approved=false，只证明功能链路。

3 epoch 训练视频重建（词汇指标）：
| 条件 | Short F1 / ROUGE-L | Block F1 / ROUGE-L | Global F1 / ROUGE-L |
|---|---|---|---|
| real | .524 / .473 | .531 / .265 | .298 / .190 |
| zero | .298 / .244 | .350 / .240 | .314 / .225 |
| other_window | .346 / .286 | .531 / .265 | .288 / .197 |

Short 对真实记忆有一定响应；Block 的汇总指标与错窗相同，Global 未显示稳定优势。生成中出现切面混淆、模板重复及“红圈表示植入设备”等无依据说法。不能据此认定 overfit 成功或有泛化能力；只有2个Block和2个Global，样本极少。

### 集群路径

SSH alias：ai；历史连接经常失败，用户可自行运行命令。
仓库：`~/haoyu/live-ultrasound-video-understanding`
DATA：`/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding`

Stage1 初始化：
`$DATA/cluster_data/experiments/stage1/stage1_v21_train191_2b_4gpu_f30_e1_20261101T071541Z_sCe9Fb/final`

3 epoch smoke：
`$DATA/cluster_data/experiments/stage2/qformer_smoke_P3_20261105T074123Z_AIjxFg`
其中已有用户读取的 `smoke_report.json` 和 `reconstruct_final/{report.json,review.txt}`。

24 epoch：
`$DATA/cluster_data/experiments/stage2/qformer_smoke_P3_20261112T111302Z_aq2X2Z`

ordinary 全量：
`$DATA/cluster_data/experiments/stage2/full_ordinary_20261105T052448Z_O9iKJr`

Q-Former 全量：
`$DATA/cluster_data/experiments/stage2/full_qformer_20261112T104925Z_9FcvVy`

注意：`submit_stage2_reconstruct.sh` 是旧 official-initialized overfit 的特定提交器，会拒绝带 init_adapter 的 run。不要不加检查地用于新版 Stage1 初始化 Q-Former。优先检查 smoke 自动生成结果和其代码快照里的 reconstruction 命令。

## 6. 本次代码整合的影响

2026-11-13 已 push：
- b12a057 / c46bb5c / 0d5ddc3：数据版本选择、冻结工具、Visual+ASR 发布。
- b5a41de：Stage1 评测继承训练输入配置。
- 8ee0795：保留并整合 legacy Stage2 DDP、模型接口。
- 5c2b9cc：Stage3 门控自回归推理与 overfit 工具。
- 7bf9bb7 / 2e507bb：Demo、主仓库与本地产物整理。

80项相关CPU测试通过，含两进程Gloo同步；13视频Demo及237视频总览构建通过。不是GPU全量训练验证。

8ee0795 修改了 stage2/model.py，因此旧 smoke 的实现哈希可能不再匹配当前代码。已提交作业的冻结代码不变；新 Q-Former 全量提交必须有当前同实现、同初始化的功能 smoke。不要删除哈希检查或改写旧报告。

## 7. 另一台电脑的接手步骤

已有仓库先看 git status，保护未提交改动，然后更新；没有仓库则 clone：

```bash
git clone https://github.com/Haoyuli2002/live-ultrasound-video-understanding.git
cd live-ultrasound-video-understanding
git status --short
git pull --ff-only origin main
git log -1 --oneline
python -m stage2.release --manifest stage2/releases/accepted_v4_20261105_990b9b48/manifest.json
python -m stage2.release --manifest stage2/releases/accepted_av_asr_v1/manifest.json
```

配置项目依赖后再运行 Python 检查；不要求另一台电脑复制原机器的绝对路径。先读 docs/WORKSPACE.md、stage2/FULL_TRAINING.md、stage2/QFORMER_SECONDWISE.md。

在 SSH 上只读检查现有作业：

```bash
cd ~/haoyu/live-ultrasound-video-understanding
squeue -u "$USER" -o '%.18i %.30j %.12T %.12M %.40R'
sacct -j 5903250,5903390,5919253,5919286 --format=JobID%22,JobName%28,State,Elapsed,ExitCode
export DATA=/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding
RUN="$DATA/cluster_data/experiments/stage2/qformer_smoke_P3_20261112T111302Z_aq2X2Z"
tail -n 80 logs/stage2_qformer_smoke_5919286.out
tail -n 80 logs/stage2_qformer_smoke_5919286.err
find "$RUN" -maxdepth 4 -type f \( -name smoke_report.json -o -name report.json -o -name review.txt -o -name predictions.jsonl -o -name completed.json \) -print
```

接着读取24 epoch训练/自回归结果，把每条 GT、real、zero、other_window 水平对比。不要只看聚合F1。确认两个全量 run 的完成状态、checkpoint、训练配置和实际数据哈希。

后续需要新训练时：先检查已有提交记录。reviewed helper会实际提交，且内部固定 DRY_RUN=0；预检必须使用 generic helper 的 DRY_RUN=1：

```bash
# 只示范预检；Q-Former 还必须设置有效的当前实现 QFORMER_SMOKE_REPORT。
DATASET_VERSION=visual_asr GPU_TYPE=auto NPROC_PER_NODE=4 DRY_RUN=1 \
  bash scripts/slurm/submit_stage2_full_training.sh ordinary
```

用户已授权历史任务的推进，但本文没有创建新作业。另一位 Codex 应先报告核查到的实际状态，再承接后续工作。不要把旧会话的“代理还在运行”当作当前事实。

## 8. 补充：SSH 原视频路径及接手核验（2026-11-13）

按已有记录，SSH 集群保存了原视频，训练不要求另一台笔记本也下载一份。本次没有实时连接SSH确认文件是否仍齐全；下面命令用于实际核验。只核验，不提交GPU任务。

原视频常用位置：

- `/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding/cluster_data/videos/train_full295/`
- `/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding/cluster_data/videos/eval_full295/`

JSONL不包含画面；Stage2训练/视频重建需要原视频，Stage3需由冻结Stage2读取视频产生memory。只检查标签或给已有预测计算文本指标则不需要视频。不要用重新编码或裁剪的视频冒充原始文件。

### Stage2：两版标签、237个视频的存在性和SHA256核验

在SSH仓库执行。全量视频哈希会读取所有文件，可能需要一定时间；不是新的模型推理或内容审核。

```bash
(
set -euo pipefail
cd ~/haoyu/live-ultrasound-video-understanding
git pull --ff-only origin main
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ultrasound
export DATA=/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding
export CHECK_DIR
CHECK_DIR=$(mktemp -d "$DATA/cluster_data/stage23_media_check_XXXXXX")

python -m stage2.release \
  --manifest stage2/releases/accepted_v4_20261105_990b9b48/manifest.json \
  --video-root "$DATA/cluster_data/videos" \
  --output-video-map "$CHECK_DIR/stage2_visual_only_video_map.json"

python -m stage2.release \
  --manifest stage2/releases/accepted_av_asr_v1/manifest.json \
  --video-root "$DATA/cluster_data/videos" \
  --output-video-map "$CHECK_DIR/stage2_visual_asr_video_map.json"

python - <<'PY'
import json, os
from pathlib import Path
root=Path(os.environ['CHECK_DIR'])
for name in ['stage2_visual_only_video_map.json','stage2_visual_asr_video_map.json']:
    mapping=json.loads((root/name).read_text())
    assert len(mapping)==237, (name,len(mapping))
    print(f'{name}: {len(mapping)} videos; release/source SHA256 checks passed')
PY
printf '核验输出目录：%s\n' "$CHECK_DIR"
)
```

该命令会检查发布JSONL哈希、split身份、跨split源哈希重叠，并逐视频核对源SHA256。缺文件或哈希不一致会失败，不能只改映射或manifest哈希跳过。若集群实际路径不同，先定位正确原件，再使用 `--input-video-map` 指定映射；保持原冻结manifest不变。

### Stage3：数据已push，223个视频映射及合同核验

交接文档生成后新增提交：`71241749ed7e8b374c03ce62eb1cc62258ceaffa`，已push到main。
原第2节的 `2e507bb` 是交接初稿核查版本；接手时应至少包含 `7124174`（或其后继提交）。

已上传：
- `stage3/releases/visual_asr_20261112/`：223视频、1,015 QA、1,505决策样本，含标注ASR及审核文本。
- `stage3/releases/legacy_visual_only_stage3_batch/`：历史50视频、205 QA，任务格式。
- `stage3/examples/two_video_overfit/`：2视频、12 QA、57决策样本的标签与manifest。
- 原视频、审核图片、模型/mock权重未上传。

以下在SSH执行，检查223个视频唯一可定位、所有数据视图和当前Stage3加载器合同；它不执行GPU训练，也不重新审核医学内容。

```bash
(
set -euo pipefail
cd ~/haoyu/live-ultrasound-video-understanding
git pull --ff-only origin main
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ultrasound
DATA=/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding
RELEASE="$PWD/stage3/releases/visual_asr_20261112"
CHECK_DIR=$(mktemp -d "$DATA/cluster_data/stage3_media_check_XXXXXX")
VIDEO_MAP="$CHECK_DIR/video_map.json"

python "$RELEASE/code/make_video_map.py" \
  --video-root "$DATA/cluster_data/videos" --output "$VIDEO_MAP"

# 直接调用检查函数，将新报告写在CHECK_DIR，避免改动Git里的历史validation.json。
python - "$RELEASE" "$PWD" "$VIDEO_MAP" "$CHECK_DIR/validation.json" <<'PY'
import json, runpy, sys
from pathlib import Path
release, project, mapping, output=map(Path,sys.argv[1:])
validate=runpy.run_path(str(release/'code/validate.py'))['validate']
report=validate(release,project,mapping)
output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
PY
printf 'Stage3映射：%s\n核验报告：%s\n' "$VIDEO_MAP" "$CHECK_DIR/validation.json"
)
```

成功时映射工具输出 `Wrote 223 video paths`，报告应为 status=passed、train_videos=185、eval_videos=38、train_eval_video_overlap=0。

区别：Stage3映射工具按文件名ID查找，重复或缺失即报错；包内检查器验证路径存在及样本合同，**不对每个源视频重新计算SHA256**。需要逐字节源确认时，使用上面的Stage2哈希核验覆盖相应源视频，并核对Stage3来源身份。路径存在不能替代内容相同。

Stage3主数据的325条旧audio/joint标签QA仍保留视觉充分性未逐题确认的限制。便捷run_training.sh默认选择690 QA的memory_candidates子集，不是1,015 QA全量；现有模型不直接接收ASR。
