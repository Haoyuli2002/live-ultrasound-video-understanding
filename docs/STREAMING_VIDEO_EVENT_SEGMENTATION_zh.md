# 流式视频如何切分事件、检索证据并决定何时回答

在超声流式 QA 中，系统每秒面对两个不同的问题：**画面是否进入了新的片段？截至这一秒，证据是否足以回答用户的问题？** 第一个问题可以在用户提问前处理，结果供多个问题复用。第二个问题取决于具体提问：即使切面不变，新出现的针尖也可能让模型从 WAIT 转为 ANSWER。反过来，切面发生变化时，跨时比较所需的第二份证据可能仍未出现。

## 1. 七篇论文中的 event 分别指什么

| 论文 | 历史片段如何形成 | 切完后做什么，以及与超声 QA 的关系 |
|---|---|---|
| ReKV · ICLR 2025 | 每帧一个候选，或把连续 $b$ 帧组成固定块。 | 先保存视频的 LLM KV cache，提问时检索相关帧或块，再载入对应 KV 生成答案。适合作为固定块检索基线。[论文 §3](https://proceedings.iclr.cc/paper_files/paper/2025/file/67a9b444cbcd647572c88194619f72d5-Paper-Conference.pdf) |
| Dispider · CVPR 2025 | 1 fps 采样，依据归一化 SigLIP 帧特征的余弦相似度找画面变化，用排除窗口控制片段间距。 | 感知模块形成不等长 clip，决策模块结合历史判断 wait/respond，触发后由反应模块异步生成。切段依据与回答时机可以分别建模。[论文 §3.1–3.2](https://openaccess.thecvf.com/content/CVPR2025/papers/Qian_Dispider_Enabling_Video_LLMs_with_Active_Real-Time_Interaction_via_Disentangled_CVPR_2025_paper.pdf) |
| StreamKV · AAAI 2026 | 相邻采样帧的视觉特征余弦相似度较低时产生候选边界，排除窗口抑制过密边界。段过长时合并段内最相似的相邻帧对。 | 为每段生成摘要向量，压缩帧级 KV，按问题检索。作者在多个压缩率下比较了语义切分和均匀切分，前者在论文的消融实验中表现更好。它提供了沿 ReKV 技术路线验证切分价值的对照。[AAAI 论文](https://ojs.aaai.org/index.php/AAAI/article/download/37305/41267) |
| StreamRAG · CVPR 2026 | 在滑窗内找相邻帧 ViT [CLS] 相似度的局部低谷，按相对谷深确认边界。 | 把相邻边界间的视频封装成 event，生成描述并增量入库，再根据问题的时间需求和相关性调整检索。与我们的历史事件索引最接近。[论文 §3](https://openaccess.thecvf.com/content/CVPR2026/papers/Xie_StreamRAG_Enhancing_Real-Time_Video_Understanding_with_Retrieval_Augmentation_CVPR_2026_paper.pdf) |
| OASIS · CVPR 2026 | 固定、不重叠的窗形成叶节点。根节点数超过预算时，按摘要相似度与树深合并时间相邻的节点。 | 每个节点有起止时间、关键帧、摘要、摘要向量和层级，并保留父子关系。适合参考其证据引用设计：粗粒度命中后仍能定位到可回放的叶片段。[论文式 1–3](https://openaccess.thecvf.com/content/CVPR2026/papers/Liang_OASIS_On-Demand_Hierarchical_Event_Memory_for_Streaming_Video_Reasoning_CVPR_2026_paper.pdf) |
| Online Generic Event Boundary Detection · ICCV 2025 | ESTimator 根据预测与实际观测之间的差异，在没有未来帧的条件下检测通用事件边界。 | 输出边界，可单独评估检测器。值得检验这种信号能否区分持续扫查时的正常运动与真正的切面转换。[ICCV 论文](https://openaccess.thecvf.com/content/ICCV2025/html/Jung_Online_Generic_Event_Boundary_Detection_ICCV_2025_paper.html) |
| StreamForest · NeurIPS 2025 | 把流入帧自适应组织成多棵事件树，根据时间距离、内容相似度及合并频率控制长期记忆。 | 适用于视频变长、历史记忆受限的场景。可与 OASIS 一起研究树结构如何保留早期证据。[NeurIPS 论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6dd91fec726dbed8915a1fbadd91d1d2-Abstract-Conference.html) |

## 2. ReKV 如何给问题和视频块打分？怎么监督？

**内置检索。** 在 LLM 的第 $\ell$ 层，把同一视觉 token 各注意力头的 Key 拼接起来，再对一帧的所有视觉 token 求平均，得到帧向量。对块 $B$ 内各帧向量再平均，得到 $\bar k_B^\ell$。提问时，在同一层对问题各 token 的 Query 求平均，得到 $\bar q^\ell$。每层独立计算下式并选 Top-K，因此不同层可以取回不同的视频块。[ReKV §3，式 2](https://proceedings.iclr.cc/paper_files/paper/2025/file/67a9b444cbcd647572c88194619f72d5-Paper-Conference.pdf)

$$
s^\ell(B,q)=\frac{\bar k_B^\ell\cdot\bar q^\ell}{\lVert\bar k_B^\ell\rVert\,\lVert\bar q^\ell\rVert}.
$$

平均向量负责排序。回答时取回命中块各视觉 token 的完整 Key 和 Value，供生成答案时做注意力计算。外部检索基线则用预训练 CLIP 类模型分别编码帧和问题，计算带温度的余弦相似度。**两种检索均可直接使用预训练参数；ReKV 没有额外用 QA 证据区间训练相关性打分器，也没有通过答案 NTP 更新检索器。** QAEgo4D 包含相关片段标注，属于评测数据的信息。相关分数表达“哪里值得看”，单凭它无法判断跨时比较题的两份证据是否都已出现。[ReKV §3–4.1](https://proceedings.iclr.cc/paper_files/paper/2025/file/67a9b444cbcd647572c88194619f72d5-Paper-Conference.pdf)

## 3. Dispider 具体在什么条件下切一刀？

**论文能确定的规则。** 每秒抽一帧，用预训练 SigLIP 得到向量并做 L2 归一化。设归一化后的帧特征为 $z_i,z_j$，二者的余弦相似度就是 $z_i^\top z_j$。显著视觉变化提供候选场景边界；在已选边界附近设置排除窗口，防止重复切出极短的段。正文 §3.2 没有给出足以逐帧复现的完整判据：没有明确 $j$ 必须是上一帧，也没有给出对 event 中心向量的比较规则、数值阈值、排除窗口秒数或候选冲突的取舍顺序。因此我们工程上的 0.02 不能当作 Dispider 的论文参数。[Dispider §3.2](https://openaccess.thecvf.com/content/CVPR2025/papers/Qian_Dispider_Enabling_Video_LLMs_with_Active_Real-Time_Interaction_via_Disentangled_CVPR_2025_paper.pdf)

**官方代码的默认行为。** 已核对的流式入口默认每 16 秒形成一个决策窗口，每窗均匀采 16 帧。这里的 16 秒是决策间隔，不能据此反推论文场景检测器的排除窗口长度。报告复现实验时应注明采用了哪个入口和哪种切分。[官方流式入口](https://github.com/Mark12Ding/Dispider/blob/master/stream_inference.py#L42-L58) · [仓库内的 Dispider 实现核对](DISPIDER_LOGIC_zh.md)

## 4. 对超声模型的可验证实验

StreamRAG 给出了一条可实现的对照规则。先计算相邻帧相似度 $c_i=\cos(z_{i-1},z_i)$。设 $c_L,c_R$ 为当前已观测窗口内候选位置左右的相似度峰值，谷深为 $d_i=(c_L+c_R-2c_i)/2$。当 $d_i>\mu_d+\tau\sigma_d$ 时确认边界，其中均值与标准差取自该窗口的谷深。由于需要观察右峰，边界可能发生于第 32 秒，到第 35 秒才确认。此例中的 32 秒用于回放定位，35 秒才是可检索时间；两者相差 3 秒。[StreamRAG §3.2](https://openaccess.thecvf.com/content/CVPR2026/papers/Xie_StreamRAG_Enhancing_Real-Time_Video_Understanding_with_Retrieval_Augmentation_CVPR_2026_paper.pdf)

| 对照项 | 建议的实验口径 |
|---|---|
| 唯一主要变量 | 在同一组 10 个视频比较固定时长块、当前 event 切分、StreamRAG 式局部低谷切分。固定视觉编码器、1 fps、检索器、Top-5、每段最多 24 帧和最近帧分支。记录实际送入模型的历史帧数。 |
| 先验收证据检索 | 对需要历史证据的问题，在标注答案秒检索当时已确认的片段。以“Top-5 实际采样帧至少命中一次”的历史证据窗口数除以标注历史窗口总数；另报所有必需历史窗口均命中的题数。事件区间与标注有交集，但实际采样漏过关键帧，应计为未命中。 |
| 再验收在线时机 | 记录边界发生秒、确认秒及两者之差。近期分支保持每秒更新，逐题比较首次回答是否提前、准时、晚答或至片尾仍等待，并统计准时且语义正确率。 |
| 已有证据与限制 | BS8/GA1 同集回放的 3 道延迟漏答题，标注证据已进入模型，其中两题强制生成能精确匹配。换切分是否能改善其他题的证据召回值得测，但现有记录不足以把这 3 题归因于漏检索。[实验记录](../research.MD) · [漏答题诊断](../results/event_retrieval_overfit_research/5963495_forced.jsonl) |
