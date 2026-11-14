# Stage 2 Q-Former Overfit 重建结果 vs GT（job 5919286）

本文件逐条列举 Stage 2 **Q-Former 秒级架构（`qformer_secondwise_v2`）** 过拟合冒烟测试
的自由重建输出与教师标注（GT）的对照，用于"检查重建质量"的人工对照记录。

## Run 元信息

| 字段 | 值 |
|---|---|
| Slurm Job | `5919286`（`stage2-qformer-smoke`，COMPLETED，ExitCode 0:0，用时 00:46:17） |
| Run 目录 | `cluster_data/experiments/stage2/qformer_smoke_P3_20261112T111302Z_aq2X2Z` |
| 入口 | `stage2.train_overfit_ab --mode independent --memory-architecture qformer` |
| 架构 | `qformer_secondwise_v2`（width=512, layers=2, heads=8, short_queries=1, long_queries=60） |
| 视频 | P3（单视频），2 分钟 → 12 个 short + 2 个 block + 2 个 global |
| 训练 | 24 epoch，384 次 optimizer 更新（16 目标 × 24） |
| 重建 | 自由贪心解码（`stage2.reconstruct`），`--max-new-tokens 96`，**reference 不进入解码 prefix** |
| 状态 | `functional_smoke_passed` / `quality_approved: false`（仅功能冒烟，非质量验收） |

> 说明：`real` = 该窗口真实记忆；`zero` = 全零记忆（下界对照）；`other_window` =
> 故意用同类型另一窗口的记忆（错窗诊断对照）。这些是**纯词汇层面**的重建，
> 不衡量医学正确性或视频忠实度，且是**训练视频自身重建**，非泛化。

## 词汇指标汇总（word_overlap_f1）

| kind | window | real_f1 | zero_f1 | otw_f1 | real>otw? | 标注 |
|---|---|---|---|---|---|---|
| short | [0, 10] | 0.111 | 0.062 | 0.216 | ❌ | 窗口错位 |
| short | [10, 20] | 0.300 | 0.286 | 0.182 | ✅ | 窗口错位 |
| short | [20, 30] | 0.188 | 0.118 | 0.263 | ❌ | 窗口错位 |
| short | [30, 40] | 0.500 | 0.333 | 0.474 | ✅ | 窗口错位 |
| short | [40, 50] | 0.529 | 0.375 | 0.294 | ✅ | 窗口错位 |
| short | [50, 60] | **1.000** | 0.312 | 0.278 | ✅ | 完美重建 |
| short | [60, 70] | **1.000** | 0.235 | 0.256 | ✅ | 完美重建 |
| short | [70, 80] | **1.000** | 0.343 | 0.485 | ✅ | 完美重建 |
| short | [80, 90] | **1.000** | 0.286 | 0.375 | ✅ | 完美重建 |
| short | [90, 100] | **1.000** | 0.529 | 0.389 | ✅ | 完美重建 |
| short | [100, 110] | **1.000** | 0.312 | 0.353 | ✅ | 完美重建 |
| short | [110, 120] | 0.312 | 1.000 | 0.235 | ✅ | 窗口错位（zero 反而完美） |
| block | [0, 60] | **1.000** | 0.218 | 0.378 | ✅ | 完美重建 |
| block | [60, 120] | **1.000** | 0.245 | 0.378 | ✅ | 完美重建 |
| global | [0, 60] | **1.000** | 0.377 | 0.556 | ✅ | 完美重建 |
| global | [0, 120] | **1.000** | 0.369 | 0.556 | ✅ | 完美重建 |

- 带 flag 的可疑条目共 6 条，全部集中在 short 的前半段 + [110,120]。
- 后半段 short（≥50s）与所有 block/global 均完美重建（real=1.0，且明显优于 zero/other_window）。


## 逐条全文对照

### SHORT

#### short [0, 10]
- **GT**：A bedside cardiac scanning demonstration is followed by a sector ultrasound image and a patient-position inset.
- **real**：RV, LV, RA and LA labels identify chambers in the apical four-chamber image; valve labels are subsequently added.  → 实为 [60,70] 的 GT
- **zero**：The labeled apical five-chamber view continues, with atrial, ventricular and central outflow spaces visible.
- **other_window**：The labeled apical four-chamber view is followed by an image captioned apical four chamber with coronary sinus, marked CS.

#### short [10, 20]
- **GT**：The sector image is labeled parasternal short axis, with a rounded ventricular cavity and an inset showing chest probe positioning.
- **real**：The labeled apical four-chamber view is followed by an image captioned apical four chamber with coronary sinus, marked CS.  → 实为 [70,80] 的 GT
- **zero**：The labeled apical five-chamber view continues, with atrial, ventricular and central outflow spaces visible.
- **other_window**：The coronary-sinus teaching view returns to a standard apical four-chamber image.

#### short [20, 30]
- **GT**：A red ring highlights a circular structure in the parasternal short-axis image, paired with an anatomical heart illustration.
- **real**：The coronary-sinus teaching view returns to a standard apical four-chamber image.  → 实为 [80,90] 的 GT
- **zero**：The labeled apical five-chamber view continues, with atrial, ventricular and central outflow spaces visible.
- **other_window**：The view caption changes to apical five chamber, with an additional central outflow region visible in the sector image.

#### short [30, 40]
- **GT**：The labeled view changes from parasternal short axis to apical four chamber, with a chest-position inset alongside the sector image.
- **real**：The view caption changes to apical five chamber, with an additional central outflow region visible in the sector image.  → 实为 [90,100] 的 GT
- **zero**：The labeled apical five-chamber view continues, with atrial, ventricular and central outflow spaces visible.
- **other_window**：The apical five-chamber image is paired with the same inset demonstrating the chest probe contact position.

#### short [40, 50]
- **GT**：An apical four-chamber image shows paired atrial and ventricular cavities, alongside the chest probe-position inset.
- **real**：The apical five-chamber image is paired with the same inset demonstrating the chest probe contact position.  → 实为 [100,110] 的 GT
- **zero**：The labeled apical five-chamber view continues, with atrial, ventricular and central outflow spaces visible.
- **other_window**：The apical four-chamber image remains visible, with a caption suggesting finding the point of maximal impulse.

#### short [50, 60] ✅
- **GT**：The apical four-chamber image remains visible, with a caption suggesting finding the point of maximal impulse.
- **real**：The apical four-chamber image remains visible, with a caption suggesting finding the point of maximal impulse.  → 完全命中
- **zero**：The labeled apical five-chamber view continues, with atrial, ventricular and central outflow spaces visible.

#### short [60, 70] ✅
- **GT**：RV, LV, RA and LA labels identify chambers in the apical four-chamber image; valve labels are subsequently added.
- **real**：RV, LV, RA and LA labels identify chambers in the apical four-chamber image; valve labels are subsequently added.  → 完全命中
- **zero**：The labeled apical five-chamber view continues, with atrial, ventricular and central outflow spaces visible.
- **other_window**：The labeled apical four-chamber view is followed by an image captioned apical four chamber with coronary sinus, marked CS.

#### short [70, 80] ✅
- **GT**：The labeled apical four-chamber view is followed by an image captioned apical four chamber with coronary sinus, marked CS.
- **real**：The labeled apical four-chamber view is followed by an image captioned apical four chamber with coronary sinus, marked CS.  → 完全命中
- **zero**：The labeled apical five-chamber view continues, with atrial, ventricular and central outflow spaces visible.
- **other_window**：The coronary-sinus teaching view returns to a standard apical four-chamber image.

#### short [80, 90] ✅
- **GT**：The coronary-sinus teaching view returns to a standard apical four-chamber image.
- **real**：The coronary-sinus teaching view returns to a standard apical four-chamber image.  → 完全命中
- **zero**：The labeled apical five-chamber view continues, with atrial, ventricular and central outflow spaces visible.
- **other_window**：The view caption changes to apical five chamber, with an additional central outflow region visible in the sector image.

#### short [90, 100] ✅
- **GT**：The view caption changes to apical five chamber, with an additional central outflow region visible in the sector image.
- **real**：The view caption changes to apical five chamber, with an additional central outflow region visible in the sector image.  → 完全命中
- **zero**：The labeled apical five-chamber view continues, with atrial, ventricular and central outflow spaces visible.
- **other_window**：The apical five-chamber image is paired with the same inset demonstrating the chest probe contact position.

#### short [100, 110] ✅
- **GT**：The apical five-chamber image is paired with the same inset demonstrating the chest probe contact position.
- **real**：The apical five-chamber image is paired with the same inset demonstrating the chest probe contact position.  → 完全命中
- **zero**：The labeled apical five-chamber view continues, with atrial, ventricular and central outflow spaces visible.
- **other_window**：The apical four-chamber image remains visible, with a caption suggesting finding the point of maximal impulse.

#### short [110, 120]
- **GT**：The labeled apical five-chamber view continues, with atrial, ventricular and central outflow spaces visible.
- **real**：The apical four-chamber image remains visible, with a caption suggesting finding the point of maximal impulse.  → 实为 [50,60] 的 GT
- **zero**：The labeled apical five-chamber view continues, with atrial, ventricular and central outflow spaces visible.  → 恰好命中 GT
- **other_window**：RV, LV, RA and LA labels identify chambers in the apical four-chamber image; valve labels are subsequently added.


### BLOCK

#### block [0, 60] ✅
- **GT**：The cardiac tutorial progresses from parasternal short-axis images with a red anatomical teaching ring to an apical four-chamber view. A persistent inset demonstrates chest probe placement, and the later caption refers to finding the point of maximal impulse.
- **real**：The cardiac tutorial progresses from parasternal short-axis images with a red anatomical teaching ring to an apical four-chamber view. A persistent inset demonstrates chest probe placement, and the later caption refers to finding the point of maximal impulse.  → 完全命中
- **zero**：The labeled apical five-chamber view continues, with atrial, ventricular and central outflow spaces visible.
- **other_window**：The second minute labels the four cardiac chambers and the mitral and tricuspid valves, briefly demonstrates the coronary sinus, then presents an apical five-chamber view. A patient-position inset accompanies the ultrasound images.

#### block [60, 120] ✅
- **GT**：The second minute labels the four cardiac chambers and the mitral and tricuspid valves, briefly demonstrates the coronary sinus, then presents an apical five-chamber view. A patient-position inset accompanies the ultrasound images.
- **real**：The second minute labels the four cardiac chambers and the mitral and tricuspid valves, briefly demonstrates the coronary sinus, then presents an apical five-chamber view. A patient-position inset accompanies the ultrasound images.  → 完全命中
- **zero**：The labeled apical five-chamber view continues, with atrial, ventricular and central outflow spaces visible.
- **other_window**：The cardiac tutorial progresses from parasternal short-axis images with a red anatomical teaching ring to an apical four-chamber view. A persistent inset demonstrates chest probe placement, and the later caption refers to finding the point of maximal impulse.

### GLOBAL

#### global [0, 60] ✅
- **GT**：The tutorial relates bedside chest probe positioning to parasternal short-axis and apical four-chamber ultrasound images, using a heart illustration, a red highlight and an external patient-position inset.
- **real**：The tutorial relates bedside chest probe positioning to parasternal short-axis and apical four-chamber ultrasound images, using a heart illustration, a red highlight and an external patient-position inset.  → 完全命中
- **zero**：The labeled apical five-chamber view is followed by an image captioned apical four chamber. A chest-position inset accompanies the ultrasound frames.
- **other_window**：The tutorial links chest probe positioning with parasternal short-axis, apical four-chamber and apical five-chamber images. It adds chamber and valve labels, briefly identifies the coronary sinus, and uses an external patient-position inset to relate anatomy to the scan.

#### global [0, 120] ✅
- **GT**：The tutorial links chest probe positioning with parasternal short-axis, apical four-chamber and apical five-chamber images. It adds chamber and valve labels, briefly identifies the coronary sinus, and uses an external patient-position inset to relate anatomy to the scan.
- **real**：The tutorial links chest probe positioning with parasternal short-axis, apical four-chamber and apical five-chamber images. It adds chamber and valve labels, briefly identifies the coronary sinus, and uses an external patient-position inset to relate anatomy to the scan.  → 完全命中
- **zero**：The labeled apical five-chamber view is followed by an image captioned apical four chamber. A chest-position inset accompanies the ultrasound frames.
- **other_window**：The tutorial relates bedside chest probe positioning to parasternal short-axis and apical four-chamber ultrasound images, using a heart illustration, a red highlight and an external patient-position inset.

## 观察到的"窗口错位"模式

前半段 short 的 `real` 输出系统性地"平移了约一分钟"，错位到后面窗口的 GT：

| 窗口 | real 实际吐出的是哪个窗口的 GT |
|---|---|
| short [0, 10] | → [60, 70] |
| short [10, 20] | → [70, 80] |
| short [20, 30] | → [80, 90] |
| short [30, 40] | → [90, 100] |
| short [40, 50] | → [100, 110] |
| short [50, 60] ~ [100, 110] | → 各自命中（正确） |
| short [110, 120] | → [50, 60] |

## 结论

1. **功能链路通过**：后半段 short（≥50s）与全部 block/global 完美重建（real=1.0），
   且 `real` 明显优于 `zero` / `other_window` → 记忆向量确实被使用、具判别性。
2. **前半段 short 存在系统性窗口错位**：`real` 吐出的是本视频真实出现过的*相邻窗口* GT
   （近似"平移一分钟"），而非乱码或凭空捏造。
3. **这不是实现 bug**：`stage2/test_secondwise_qformer.py` 全部 9 个用例通过，其中
   `test_training_replay_equals_online_prefix`（训练==评测编码一致，rtol=0）等已从数值上
   排除"编码错位 / 配对错位 / 因果泄漏 / 跨分钟串扰"。错位是 overfit 下
   **short 记忆向量不可分 + 重建 prompt 不含窗口身份** 导致的近邻检索式混淆，属设计局限。
4. **未见明显幻觉**：`real` 用词均为本视频真实术语，未出现"红圈=植入设备"一类无依据编造。
5. **样本量极小**：P3 单视频、block/global 各 2 条，是训练视频自身重建、纯词汇指标，
   `quality_approved: false`。**不能据此断言收敛或泛化**。

### 可改进方向（非 bug 修复）
- 给 short 重建注入"窗口身份"：prompt 带 `window j of 6 / seconds a-b`，可选加 segment embedding。
- 先把 `reconstruction_loss`（`stage2/model.py`）与 `greedy_reconstruct`（`stage2/reconstruct.py`）
  手写重复的 prompt 抽成共享常量，消除"改一处忘改另一处"的脆弱点，并为上一条打基础。
- 增加 short 的时序判别（相邻窗口对比损失 / 早窗 loss 加权 / 位置编码增强）。

- **other_window**：RV, LV, RA and LA labels identify chambers in the apical four-chamber image; valve labels are subsequently added.
