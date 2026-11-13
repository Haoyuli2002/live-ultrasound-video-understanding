# Stage3 交接：进展、数据与后续工作

更新日期：2026-11-13。供另一台笔记本的 Codex 在 pull 后接手。此次整理文档，没有重新生成 QA 或重新观看全部视频。

## 先看这里

**223 个 High/Medium 视频已有 1,015 条英文 QA，展开为 1,505 条 WAIT/ANSWER 样本。数据已封装，尚无此全量版本的真实 GPU 训练与模型效果结果。**

| 划分 | 视频 | QA | Immediate | Proactive | 决策样本 |
|---|---:|---:|---:|---:|---:|
| train | 185 | 832 | 427 | 405 | 1,237 |
| eval | 38 | 183 | 98 | 85 | 268 |
| 合计 | 223 | 1,015 | 525 | 490 | 1,505 |

每条 Immediate 产生 1 条 ANSWER，每条 Proactive 产生 1 条 WAIT 和 1 条 ANSWER：

- train：427 + 405 × 2 = 1,237（832 ANSWER + 405 WAIT）。
- eval：98 + 85 × 2 = 268（183 ANSWER + 85 WAIT）。
- 全视频库 237 个：Stage3 High 130、Medium 93、Low 14；223 个 High/Medium 均已覆盖。

### pull 后可以直接找到

- [当前 Stage3 发布包](../stage3/releases/visual_asr_20261112/)：统一 QA、训练样本、标注 ASR、来源及审核文本。
- [发布目录使用说明](../stage3/releases/README.md)。
- [Stage3 代码](../stage3/README.md)。
- [Stage2 交接](STAGE2_HANDOFF_20261113.md)：Stage2 发布及实验状态以该文档为准。

Git 包与本地完整包的主要标签相同；Git 未包含原视频、审核图片、权重或本机 video map。源记录中的原电脑绝对路径只用于追溯，不能在新电脑直接使用。

### 接手时避免三个误解

1. **325 条旧 audio/joint 标签 QA 不等于不能视觉回答。** 它们完整保留，需要逐题进一步核对视觉充分性；不能按标签直接删除。690 条 memory_candidates 也是同一 visual+ASR 系列的候选视图，不是旧 visual-only 数据。
2. **默认训练脚本只用候选视图。** train 为 857 行，不能把该运行称为全量 1,237 行训练。
3. **“12 个视频没有 ASR”是旧归档查找遗漏。** 后来已经找到这 12 个视频的 Stage2 窗口转写。Git 冻结包仍归档 211 个，补充文件尚未合入；不能推断原 Stage3 生成时使用过后来找到的转写。

12 个视频 ID：`-UDinlF3ha4`、`7cQ_3GtvNxo`、`A_CVfJFPp-E`、`C60jGZsiPrs`、`GPHyZfm8T2s`、`J_kORTc8hxY`、`KGSdQB80ois`、`Q9ZcHkmIBzY`、`VFPcSCetCxg`、`hazy6gol4rE`、`o_Fmsk3QCkc`、`xBDOuUxBswo`。

原电脑的定位审计：`/Users/I761836/Documents/Codex/2026-11-04/mi/outputs/stage3_status_draft_20261112/asr_lookup_correction.json`。12 份转写在已导出的 `stage3_handoff_20261113.zip/asr_supplement_unintegrated/` 中，也可按下文 Stage2 AV 工作目录中的 `videos/{train,eval}/{id}/audio_asr_windows.jsonl` 找到。窗口可能未覆盖视频尾部，不应声称全片转写完整。

旧发布说明/manifest 的“ASR missing”字段是冻结时的归档状态；此处记录勘误，不覆盖冻结数据。

## 一、当前主版本与来源

原部署目录：/Users/I761836/Desktop/Semester 3/live-ultrasound-video-understanding/cluster_data/stage3

当前版本stage3_av_training_20261112：223视频，1015英文QA，1505决策样本。

类型：current_observation 308；history_recall 217；temporal_comparison 245；event_notification 245。

前两类Immediate合计525，后两类Proactive合计490，接近1:1，并非逐视频硬凑相同数量。

train/eval按视频分开，185/38，无视频ID交叉；没有验证跨不同ID的内容重复。

五个来源（互斥视频集合）：

stage3_av50_20261105：50视频200QA。

stage3_av_next50_20261106：50视频200QA。

stage3_av_visualfirst_next50_20261110：30视频100QA。

stage3_av_visualfirst_regen14_reviewed_20261111：14视频99QA。

stage3_remaining79/export_accepted79_20261111：79视频416QA。

4条补审修订已叠加到最终版本，旧源文件保留。旧stage3_batch visual-only不混入此统计。

## 二、生成和审核约定

所有QA用英文；使用视频视觉为主要依据，ASR辅助理解术语、步骤和上下文。

ASR可以提示观察方向，但不能据讲解推定画面必然显示某现象。

不调用外部模型API代替助手生成。每视频按信息量生成，不规定固定题数。

第一遍：理解全片主题、结构和过程，形成内容概览、时间线、候选可观察事件与价值判断；不是先硬套题目。

第二遍：围绕当前观察、历史回忆、时序比较、条件触发设计QA，记录query_time、answer_time、证据及来源。

第三遍：带问题查原尺寸全幅画面，检查问题自然、历史指代唯一、答案有依据；核对query/answer前后和历史证据。

Proactive的query前缀不能完整回答，answer前缀才足以回答。若query已经能答，改题、改时间或删除后重新检查。

持续条件核对每次观察；重现条件核对初次存在、退出和再次出现。防止提前、漏报、重复通知。

发现问题直接修正再查，没问题保存；记录真正看过的帧与范围；单独导出验收版，不覆盖旧数据。

检查仅凭问题、字幕、颜色、剪辑模板猜答案的风险；标注审核不等于完成模型消融实验。

同一助手先看全片再审题必须称自审；不能声称独立盲审。

## 三、已完成审核的实际含义

当前所有记录已有accepted类源审核状态，partial_review为0；并不等于所有样本都经过同等强度的独立复核。

本次交接重新核对文件/计数/加载合同，没有重新看全部223个视频。

最近79视频中：前21视频62QA为主线自审；后58视频354QA由A生成、B复核，历史记录2179张原尺寸帧。

这是关键帧和时间边界审查，不是连续完整播放所有视频，不是独立盲审，也不能等同真实音频试听。

最终4题补审查看38张原尺寸画面：

  pClYyRZlYE8_01：删去不清楚的针体判断，保留两管腔相对位置，query=answer=330。

  pClYyRZlYE8_02：answer 346→340.5，query335；改问管腔内独立亮回声，不断言精确针尖成功入腔。

  OJB8ZNtVAvA_01：query=answer=745，保留胸前探头位置描述。

  OJB8ZNtVAvA_02：answer802→801.125，query790；逐秒阴性及边界帧核查频谱出现。

这些4条的原记录、修订和日志已随Git发布。38张复核帧在本地完整包和ZIP中；Git发布不含审核图片。其余历史审核材料也未全量复制。

## 四、训练接口与数据结构

train_tasks.jsonl/eval_tasks.jsonl：832/183条统一QA任务。

train.jsonl/eval.jsonl：1237/268条展开决策样本；data/all中有相同副本。

annotations/source_tasks.jsonl：1015条保留来源、证据和审核字段的原QA记录（已含4条修订）。

样本schema_version=stage3_train_sample_v1：这是格式版本名，不是训练轮次；当前不需要另找v2。

样本字段：sample_id、task_id、video_id、question、video_window=[0,current_time]、target、meta。

target为<WAIT>或<ANSWER>加参考回答。

meta保留query_time/answer_time/current_time/qa_type/y/split用于标签和校验。

训练loader输出不含gold answer_time；不能把未来答案、完整ASR或审核说明放入模型输入。

Immediate在query_time生成ANSWER；Proactive在query_time生成WAIT、answer_time生成ANSWER。

保留小数秒。不补造中间WAIT、任务完成后ANSWER或未触发负例。

当前模型读取Stage2因果视觉memory、问题、query/current时间，不直接读取ASR/音频。

stage3/train.py：冻结Stage2 memory模型，训练决策/回答模型LoRA，BCE与文本生成损失。

因果memory按时间推进；小数时刻采用临时探测，不污染后续整秒递推。

原尺寸用于标注复核；当前训练默认frame-size224。原尺寸可辨认不代表224输入也可辨认，仍需实验。

data/memory_candidates：690QA/1055样本（train560QA857行，eval130QA198行）。

data/audio_or_joint_tagged：325QA/450样本（train272QA380行，eval53QA70行）。

两个视图按QA分组，视频可跨视图重复；不能相加视频数。它们都来自同一visual+ASR系列。

325条旧audio/joint标签只提示需要进一步逐题核查视觉充分性，不能据标签断言不能训练，也不能直接删除。

run_training.sh默认候选视图。数据全量存在，不等于默认脚本会训练全部1015QA。

## 五、已验证与未完成

已验证：文件哈希、样本数、任务端点一致性、train/eval视频分离；AST提取真实load_examples、decision_question和answerability_label进行合同执行。

未完成：此全量版本的真实Stage2 checkpoint加载、GPU训练、模型重建和模型质量评测。

历史mock或两个视频overfit流程不能当作此全量版本已训练的证据。

code/score_endpoints.py可评端点预测，支持小数秒，拒绝缺失/重复/未知样本。

现有stage3.overfit/metrics针对两个视频及整数密集网格，不能原样用于全量小数端点。

端点分类/回答得分不代表连续流式首次触发准确率；密集评测需另设计。

本地合同测试没有导入完整torch模型，没有临床独立验收。

## 六、建议接手顺序（待办，不代表已执行）

1. 传输校验；定位新电脑的完整项目和原视频，重建video map，按真实项目运行加载合同。

2. 关联后来找到的12份Stage2窗口ASR（ZIP另附；Git发布暂未合入），并核对时间覆盖；生成新版本，保留冻结版。不要声称原QA已使用后来找出的ASR。

3. 对325条旧标签QA逐题检查视觉充分性，必要时改英文问答和时间，记录证据；不得只改标签或直接排除。

4. 确认真实Stage2 adapter及其配套Stage1配置，先跑小规模真实overfit→重建→端点评测，再推进全量。

5. 按实际证据选择训练全量/修订后全量，并明确数据选择；必要时实现真正因果的ASR输入，不能注入全片转写。

## 七、原电脑资源定位

项目：/Users/I761836/Desktop/Semester 3/live-ultrasound-video-understanding

视频：项目/cluster_data/videos_from_ssh/train_full295（196个）、eval_full295（41个）。文件夹名不等于实际视频数。

Stage2已发布视觉数据：/Users/I761836/Documents/Codex/2026-10-23/wo-y/outputs/stage2_training_preparation_20261104/release_20261105/

Stage2 AV工作目录：/Users/I761836/Documents/Codex/2026-10-23/wo-y/outputs/stage2_av_asr_full_20261105/codex_native_w7ft9xf1/

后者是原始工作档案，历史manifest的training_release_approved=false不能用于判断后来冻结发布的状态。当前Stage2 Visual+ASR冻结版已在stage2/releases/accepted_av_asr_v1；最新Stage2情况见STAGE2_HANDOFF_20261113.md。

历史Stage3产物：/Users/I761836/Documents/Codex/2026-11-04/mi/outputs/

Notion：https://www.notion.so/401bd7b59648803a8193ccc86213dc18

最近SSH排查没有成功登录，尚未到密码验证阶段；换机需重测VPN/网络。交接包不携带认证资料。

ZIP内examples/first_pass_label_example.json是为解释流程事后整理的示例，不能当真实第一遍日志；它不是本Git目录下的文件。

## 新电脑接手与校验

在完整项目根目录操作。先 pull，再定位原视频；无需重跑原始 build.py。

```bash
git pull --ff-only origin main
export PROJECT_ROOT="$PWD"
export RELEASE="$PROJECT_ROOT/stage3/releases/visual_asr_20261112"
# 将 /path/to/original_videos 改为实际视频根目录。
export CHECK_DIR="$(mktemp -d)"
python3 "$RELEASE/code/make_video_map.py" \
  --video-root /path/to/original_videos \
  --output "$CHECK_DIR/video_map.json"

# 只调用校验函数；报告写到新目录，不修改Git内冻结validation.json。
python3 - <<'CHECK'
import json, os, runpy
from pathlib import Path
release = Path(os.environ['RELEASE'])
project = Path(os.environ['PROJECT_ROOT'])
out = Path(os.environ['CHECK_DIR'])
validate = runpy.run_path(str(release / 'code/validate.py'))['validate']
report = validate(release, project, out / 'video_map.json')
(out / 'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False, indent=2))
CHECK
```

预期：223 个视频映射，train 185 / eval 38，无视频 ID 交叉，all 视图加载 1,505 条样本。

映射工具按文件名 ID 定位；缺失或重复会报错。合同校验检查路径存在与数据结构，**不逐个重新计算原视频哈希**，也不执行 GPU 模型。需要源视频身份确认时，结合 Stage2 交接文档中的源哈希校验。

训练前需要真实 Stage2 adapter、对应 Stage1 配置和 GPU 环境。数据完整不等于这些条件已满足；先做小规模真实 overfit、自由生成重建与端点评测。保存到新目录，保留旧数据和实验。

## 给接手 Codex 的指令

> 阅读本文件与 Stage2 交接，先核对当前仓库和 Stage3 发布 manifest，再定位新电脑上的原视频和真实 checkpoint。完成映射和加载合同检查，简要汇报已确认状态、缺失资源与下一步。不要直接启动全量训练。所有 QA 保持英文；视频为主、ASR 为辅；Proactive 在 query 时不能完整回答，在 answer 时才可回答。325 条旧 audio/joint 标签不能直接排除；发现证据或时间问题应实际修改并复查，输出新版本。不要把 mock、结构检查或端点评分当成全量模型质量验收。
