# 训练结果 · 2026-11-27

| 作业 | 数据与目标 | 实际完成情况 |
|---|---|---|
| 5963544 · Event Retrieval v3 | QA v3，10 视频、97 题、246 个监督时点。同集 overfit，纯 NTP。 | 1×A100 80 GB，BS8、GA2，有效 BS16。20/20 轮，320/320 次更新，完成逐秒回放，用时 1:10:50。 |
| 5958344 · StreamReady 全量无 TR | 171 个训练视频、16 个 holdout 视频。答案 NTP、同题 WAIT–ANSWER 配对 CL、等权 readiness BCE，TR 权重 0。 | 2×H100，每卡 BS96、GA1。计划 2 轮、132 步；完成 step 0 holdout 后，合成全 ANSWER 压力测试 OOM，退出时训练为 0/132 步，用时 1:48:05。 |

## 5963544 · 新架构同集回放

历史 Top-5 event 经过 Long Q-Former；近期 30 帧中，前 28 帧经过 Short Q-Former，最近两帧空间 token 直通。训练两个 Q-Former 的视觉路径、投影与完整 Qwen3.5-2B 语言模型。第 1、5、10、15、20 轮 NTP 分别为 2.4933、0.0780、0.0350、0.0521、0.0583。[配置](../cluster_data/experiments/event_retrieval/v3_overfit_bs8_ga2_kmrfbeyn/plan.json) · [训练日志](../cluster_data/experiments/event_retrieval/v3_overfit_bs8_ga2_kmrfbeyn/logs/slurm-5963544.log) · [第 20 轮 checkpoint](../cluster_data/experiments/event_retrieval/v3_overfit_bs8_ga2_kmrfbeyn/fit/latest.pt)

| 首次作答 | 即时 81 题 | 延迟 15 题 | 全部可答 96 题 |
|---|---:|---:|---:|
| 准时 | 81 | 13 | 94 |
| 准时且精确匹配 | 76 | 11 | 87 |
| 准时且语义正确（含同义改写） | 80 | 12 | 92 |

共回放 792 个时点，零提前、零晚答。另 1 道无答案题保持 WAIT 至片尾。4 道未完全通过的可答题中，CW3TLabogko/Q004、Q006 一直等待；az6h_u_8nhA/Q010、lREUPXCpK8Y/Q006 准时作答但空间关系描述不完整。相同视频、相同轮数的 BS8/GA1 基线为 90/96 准时且语义正确。[回放汇总](../cluster_data/experiments/event_retrieval/v3_overfit_bs8_ga2_kmrfbeyn/replay/timing_metrics.json) · [97 题生成、参考答案与人工复核记录](../results/event_retrieval_overfit_research/5963544.jsonl)

## 5958344 · StreamReady 的初始化基线

唯一可用的 holdout 评测来自优化器第 0 步，动作阈值 0.5。全量目标损失 5.4938，其中答案 CE 4.1075、CL 0.6931、BCE 0.6931、TR 0。464/464 个 ANSWER 时点触发，0/530 个 WAIT 时点保持等待；准时且答案自动匹配 0/464，强制生成自动匹配 0/464。此次运行没有训练后的 holdout 指标，无法据此判断去掉 TR 的效果。[初始评测](../cluster_data/experiments/streamready/full_train171_holdout16_bs96_block8_2gpu2ep_20261125T1403Z/holdout_metrics.jsonl) · [压力测试](../cluster_data/experiments/streamready/full_train171_holdout16_bs96_block8_2gpu2ep_20261125T1403Z/synthetic_stress.jsonl) · [终止记录](../cluster_data/experiments/streamready/full_train171_holdout16_bs96_block8_2gpu2ep_20261125T1403Z/status.json)
