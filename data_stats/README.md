# 数据量统计

记录日期：2026-10-01。

以下数据来自本次提供的集群统计结果，未在本地重新计算。原始统计文件路径为 `cluster_data/splits/full295_qwen35_video_stats.json`。

| 维度 | 项目 | 数量／统计 | 说明 |
|---|---|---:|---|
| 总体规模 | 视频总数 | 237 | Train + Eval |
| 总体规模 | 总时长 | 49.597 小时 | 原始视频时长 |
| 总体规模 | 总大小 | 15.095 GB | |
| 总体规模 | 平均时长 | 12.56 分钟／视频 | |
| 总体规模 | 最长视频 | 95.20 分钟 | |
| 总体规模 | 错误记录 | 6 | 需检查错误原因并重试 |
| 数据划分 | Train | 196 个；37.825 小时；13.979 GB | 平均 11.58 分钟／视频 |
| 数据划分 | Eval | 41 个；11.771 小时；1.116 GB | 平均 17.23 分钟／视频 |
| 视频类型 | 混合超声教学（mixed_ultrasound_teaching） | 130 | Stage 2/3 使用前通常需要切片 |
| 视频类型 | 实操超声教学（hands_on_ultrasound_teaching） | 58 | Stage 2/3 直接保留的主要来源 |
| 视频类型 | 超声幻灯片讲座（ultrasound_ppt_lecture） | 39 | 主要用于 Stage 1 |
| 视频类型 | 不确定（uncertain） | 6 | 待审核 |
| 视频类型 | 非超声／无关（non_ultrasound_or_irrelevant） | 3 | 排除 |
| 视频类型 | 纯超声扫描（pure_ultrasound_scan） | 1 | Stage 2/3 直接保留 |
| 阶段筛选 | keep_for_pretrain | 234 | 当前统计的保留标记数 |
| 阶段筛选 | keep_for_compression | 59 | Train + Eval 合计，不能全部作为训练集 |
| 阶段筛选 | keep_for_sft | 59 | Train + Eval 合计，不能全部作为训练集 |
| 阶段筛选 | needs_clipping | 125 | 切片后可重新评估 Stage 2/3 适用性 |
| 语言 | 英语（en） | 225 | 占绝大多数 |
| 语言 | 未知（unknown） | 5 | |
| 语言 | 德语（de） | 1 | |
| 语言 | 缺失（missing） | 6 | 是否对应错误记录需核对 |
| 解剖部位 | 血管（vascular） | 79 | 多标签统计 |
| 解剖部位 | 肺（lung） | 60 | 多标签统计 |
| 解剖部位 | 神经／区域麻醉（nerve_regional_anesthesia） | 49 | 多标签统计 |
| 解剖部位 | 腹部（abdomen） | 40 | 多标签统计 |
| 解剖部位 | 肌骨（msk） | 36 | 多标签统计 |
| 解剖部位 | 甲状腺／颈部（thyroid_neck） | 35 | 多标签统计 |
| 解剖部位 | 心脏（cardiac） | 26 | 多标签统计 |
| 解剖部位 | 盆腔／妇产科（pelvis_obgyn） | 21 | 多标签统计 |
| 解剖部位 | 肝胆（hepatobiliary） | 20 | 多标签统计 |
| 解剖部位 | 软组织（soft_tissue） | 19 | 多标签统计 |
| 解剖部位 | 肾脏（renal） | 17 | 多标签统计 |
| 临床场景 | 操作引导（procedure_guidance） | 117 | 多标签统计 |
| 临床场景 | 穿刺针引导（needle_guidance） | 66 | 多标签统计 |
| 临床场景 | 血管通路（vascular_access） | 54 | 多标签统计 |
| 临床场景 | 通用扫描教学（general_scanning_tutorial） | 50 | 多标签统计 |
| 临床场景 | 神经阻滞（nerve_block） | 48 | 多标签统计 |
| 临床场景 | 胸腔积液评估（pleural_effusion_assessment） | 43 | 多标签统计 |
| 临床场景 | 气胸评估（pneumothorax_assessment） | 43 | 多标签统计 |
| 临床场景 | 肺水肿／B 线（pulmonary_edema_b_lines） | 34 | 多标签统计 |
| 临床场景 | 心功能评估（cardiac_function） | 23 | 多标签统计 |

## 统计口径

- 49.597 小时是全部 237 个视频的总时长，并非 234 个预训练保留视频的精确时长，也不是 Stage 1 训练集时长。
- Train 和 Eval 时长独立四舍五入，因此显示值相加与总计相差 0.001 小时。
- 59 个 compression / sft 保留视频是 Train + Eval 合计；后续生成标签与训练时应保持原有划分。
- `keep_for_pretrain: 234` 是否包含 6 个 `uncertain`，需要逐条核对原始标记。仓库合并后的策略会排除 `uncertain`，应区分原始教师标记与最终筛选结果。
- 6 个错误记录、6 个 `uncertain` 和 6 个语言缺失记录是否属于同一组视频，尚未核对。
- 解剖部位与临床场景为 Top 项目、多标签统计；同一视频可计入多项，不能相加作为视频总数。
- `needs_clipping` 是独立标记，不能直接等同于全部混合教学视频；切片后仍需审核是否适用于 Stage 2/3。
