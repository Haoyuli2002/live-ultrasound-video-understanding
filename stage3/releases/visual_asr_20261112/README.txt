Git发布说明（2026-11-13）：本目录为原封装的文本导出。原视频、38张审核图像和本机绝对路径映射未随包上传；annotations中的图像引用需回原档案查看。source_checksums.json / source_validation.json保留原包记录；checksums.json仅校验实际上传文件。本文下方的原包描述不表示所有原包媒体已入Git。

Stage3 visual+ASR 系列数据 · 2026-11-12 完整封装

223个High/Medium视频，1015条英文QA。
训练185视频832QA，展开1237条决策样本。
评测38视频183QA，展开268条决策样本。
Immediate QA525条；Proactive QA490条。共1505个决策样本。
本包包含所选visual+ASR系列版本；未加入旧stage3_batch visual-only版本。
原视频不复制或重编码，通过video_path_map.local.json引用原始文件。

主入口（全量数据，没有因旧模态标签删除QA）
train_tasks.jsonl / eval_tasks.jsonl：统一QA。
train.jsonl / eval.jsonl：匹配stage3.train.load_examples的WAIT/ANSWER决策样本。
data/all/：与主入口内容一致。
annotations/source_tasks.jsonl：带原证据、审核、来源字段的1015条QA（含本次修正）。
annotations/review_overrides.jsonl：本次4条复核结果。
annotations/reviews/review4_20261112/：修改前数据、逐题审核日志、实际查看的38张原尺寸画面。
annotations/asr/：211个视频的现有完整ASR归档，仅用于标注审核。
media_manifest.json：223个视频的原路径、原尺寸、时长、ASR与音频可用性。
dataset_manifest.json：各视图计数、审核范围、接口约束。
source_manifest.json / checksums.json：来源哈希及包内文件哈希。

4条补复核
全部已核验并保存，partial_review现为0。
心脏频谱通知：query790，answer802改为801.125秒。
亮回声通知：query335，answer346改为340.5秒；改为画面能支持的管腔内独立亮回声，不断言精确针尖入腔。
当前观察题删除未明确可见的针体判断，保留两管腔的相对位置。
胸部探头位置题保持原问题和答案。
复核为同一助手带题查看原尺寸全幅关键帧及边界时刻；并非盲审或连续全片回放。

旧模态标签仅作为附加训练视图，不表示生成时是否使用ASR
所有QA均来自visual+ASR系列；有12个视频无现成ASR，已如实列明。
data/memory_candidates/：按旧证据与审核标签筛出的690条候选QA，1055个样本（train857/eval198）。
data/audio_or_joint_tagged/：325条带旧audio/joint标签的QA，完整保留。
这325条尚未重新逐题核验视觉充分性；不能仅因生成时用了ASR就断言它们不适合视觉训练。
data/partial_review/：空集合，保留目录以兼容统计工具。
不同视图中视频会重复，不能把各视图视频数相加。主入口始终是全部1015条。

当前训练接口
现有stage3模型消费Stage2视觉memory、问题、query/current时间，不直接消费ASR或音频。
训练样本顶层字段：sample_id/task_id/video_id/question/video_window/target/meta。
video_window严格为[0,current_time]；目标为裸<WAIT>或<ANSWER>加参考答案。
meta包含gold answer_time及标签，仅用于真实load_examples的结构校验，不进入模型问题前缀。
Immediate QA只产生query时的ANSWER；Proactive QA产生query时的WAIT和answer时的ANSWER。
只使用已复核端点，保留小数秒；不补造中间WAIT或未触发负例，不增加完成后样本。
原尺寸复核不能替代224输入可读性检查，本次没有改变训练frame-size或模型架构。

当前模型的便捷启动脚本run_training.sh沿用memory_candidates视图。
这是显式的候选训练配置；全量数据仍在根目录，325条旧audio/joint标签并未删除。
若训练全量，请先处理实际视觉证据不足的题，或实现真实因果ASR输入；本包不自动注入完整ASR。

训练机使用
export PROJECT_ROOT='/path/to/live-ultrasound-video-understanding'
export STAGE2_ADAPTER='/path/to/exact_stage2/final'
export OUTPUT_DIR='/path/to/new_stage3_run'
# 仅当该Stage2确实使用对应Stage1时设置STAGE1_ADAPTER。
# 如果训练机的视频路径不同，先生成新映射：
python3 code/make_video_map.py --video-root '/cluster/original_videos' --output '/cluster/stage3_video_paths.json'
export VIDEO_PATH_MAP='/cluster/stage3_video_paths.json'
bash run_training.sh --epochs 1 --dtype bf16 --frame-size 224

校验
python3 code/validate.py --project-root '/path/to/project' --video-path-map '/path/to/video_paths.json'
本机未安装torch：校验直接执行真实train.py/model.py中AST提取的加载/标签/问题格式函数体；并非完整GPU模块运行。
所有视图样本已通过该合同校验，验证集与训练集视频ID无交叉。外部视频路径与原生分辨率已检查。
未加载模型、未进行GPU训练；未重新逐题审核其余1011条QA或检查跨ID内容重复。

端点评测
python3 code/score_endpoints.py --samples data/all/eval.jsonl --predictions /path/predictions.jsonl --output /path/new_report.json
预测格式：{"sample_id":"...","status":"WAIT|ANSWER","prediction":"ANSWER时的文本"}。
评分支持小数时间，拒绝未知、重复、缺失样本；只评已审核端点，不代表密集流式首次触发测试。
项目现有stage3.overfit/metrics限定两视频及整秒网格，不能原样用于本包。

来源原始文件及2026-11-11封装均保留；本目录是补复核后的独立版本。
