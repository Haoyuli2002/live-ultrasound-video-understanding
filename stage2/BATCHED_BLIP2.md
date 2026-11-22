# Stage2 BLIP2 批量重建训练

入口是 `python -m stage2.blip2_overfit --run-dir RUN --arm ARM`，ARM 为
`lavis_bert_query_1_60` 或 `qformer`。这是 Stage2 visual/ASR 摘要重建实验，
不是 StreamReady 的 readiness/QA 训练入口。

## 批量与并行

`experiment.json` 中 `gpus` 必须等于实际 WORLD_SIZE；单卡可直接运行，多卡用
`torchrun --nproc-per-node=N -m stage2.blip2_overfit ...`，当前入口按单节点设计。
`per_rank_targets` 是每卡每次前向的目标数，GA=1，全局 batch 上限为
`per_rank_targets * gpus`。每个 epoch 全覆盖，不重复填充样本；最后一批可能更小。
目标总数必须能被卡数整除。56 个目标支持 1/2/4 卡。

- LAVIS Short 将同批内独立帧合并计算，`short_frame_batch` 默认 128。
- Secondwise Short 保留分钟内的时间递归，只对不同分钟并行。
- 每次前向按视频和分钟共享 Short 表示及历史 Long 更新；历史 Long 保持 detach。
- 不同目标的 memory、prompt、answer 右侧补齐后，共享一次 decoder 前向。
- CE 先在每个目标的答案 token（含 EOS）内取均值，再在目标间取均值，
  与原有逐目标损失的平均方式相同；memory/prompt 不受 CE 监督。
- 训练态共享表示也共享该表示的 dropout 采样；关闭 dropout 后损失及参数梯度已与
  独立计算对齐。扩大 batch 会改变每 epoch 更新次数，不能按相同 epoch 宣称训练量相同。

只有冻结的视觉特征可跨 optimizer step 缓存，当前保存在 CPU。
可训练的 Short/Long 表示仅在本次前向中共享：更新参数后重新计算，防止用旧表示训练。

## 显存与 attention

`gradient_checkpointing` 默认 false。LAVIS 普通绝对位置 attention 使用 PyTorch SDPA；
需要返回 attention、head mask 或相对位置时回退原实现。
`fast_linear_attention: true` 尝试启用 Qwen 的 FLA 加速；实际结果记录在
`training_config.json` 的 `acceleration` 字段，应以运行记录为准。
仅输出可能预测目标的位置的 logits，减少未监督前缀上的词表张量。

## 零更新性能验证

添加 `--profile-only`，使用单卡和新的输出目录。该模式不执行 optimizer.step，
不保存训练 checkpoint，也不进行生成评估；显式分配 Adam 状态以估计训练显存。
先比较真实 BF16 模型的串行/批量损失，然后比较串行 BS1 和 `batch_candidates`。
每项一次预热，`profile_repeats` 次前向/反向计时；较长 Global 目标优先用于压力测试。

查看每个 arm 的 `batch_parity.json`、`batch_profiles.json`、`status.json`。
只有 peak reserved < 总显存 80% 且剩余显存 > 15% 的候选才标记 safe；
推荐值取安全候选中吞吐最高者。推荐是单卡压力测试结果，不代表多卡扩展效率或收敛结果。
公平 A/B 比较应使用两组都安全的配置，并明确记录全局 batch 和更新次数。
GPU 利用率应按 `timed_start_unix/timed_end_unix` 对齐采样，避免混入模型加载和编译。

## 验证与当前实验

CPU 回归：

```bash
python -m unittest stage2.test_batched_reconstruction stage2.test_blip2_launcher stage2.test_secondwise_qformer -v
```

17 项检查通过，覆盖损失/梯度一致性、因果性、历史 detach、缓存失效、SDPA 和样本覆盖。
另外双进程 gloo DDP 检查通过：跨 rank 的 Short/Global 混合任务梯度等于独立目标的全局均值。
这不替代真实多 GPU/NCCL 的性能测试。

单卡零更新测试 job 5945806 已提交，提交时 PENDING；完整配置、冻结源码、数据和日志见
`programs.md` 的 `20261121T162716Z` 条目。GPU 性能结果尚待实测，正式 A/B 训练尚未重提。
