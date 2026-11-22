# Full WAIT–ANSWER pairing

Set `timing_pair_mode: "all_pairs"` in `experiment.json` with
`sampler: "question_groups"` and a contrastive readiness objective. Existing
configurations default to `"sampled"`; evaluation always uses all pairs.

For each `(video_id, task_id)`, every gold ANSWER endpoint is paired with every
gold WAIT endpoint. The loss is `mean(softplus(s_wait - s_answer))` within each
question, followed by a mean across questions with both labels. Questions with
more WAIT endpoints therefore do not receive greater total contrastive weight.
Pairs never cross videos or questions. Identical timestamps are deduplicated;
conflicting labels at the same timestamp are rejected.

This uses the existing endpoint logits, without additional video/Q-Former
forwards. The current ten-video set has 35 paired questions and 100 pairs per
epoch, versus 35 sampled pairs previously. All 175 endpoints remain available:
71 ANSWER and 104 WAIT. Single-label questions have no contrastive pair; BCE,
when enabled, still supervises every endpoint. This does not add new timestamps
or turn sparse annotations into dense frame-level supervision.

NTP still supervises every gold ANSWER regardless of predicted readiness.
Temporal regularization, loss coefficients, readiness gradient isolation,
preprocessing, memory and attention implementations remain unchanged.
`timing_pairs` reports available pairs, `timing_used_pairs` reports actual pairs,
and `timing_pair_mode` distinguishes training policies. `timing_sampled_pairs`
retains its old meaning and is zero for deterministic all-pairs training.

## Four-arm selection and matched rerun

`python -B -m endtoend.select_all_pairs --run-dir <frozen-run>` consumes the
four runs recorded in `selection_plan.json`. It requires completed epoch-100
results, identical initial weight fingerprints, fresh Adam state, 300 optimizer
updates, identical data and planned batch order. Its predeclared selection rule
scans thresholds, maximizing first-trigger answers with both correct timing and
text, then minimizing WAIT false triggers, then maximizing readiness accuracy.
Remaining ties prefer thresholds closer to 0.5, then the declared arm order.
Fixed 0.5 and 0.35 results are also retained for all six requested metrics.

If a contrastive arm wins, the runner starts from the same original pretrained
initialization and fresh optimizer, changing only pairing mode and descriptive
metadata. Training-time evaluation keeps threshold 0.5 for comparison; the
source-selected threshold is evaluated separately. It verifies 100 used pairs
in each of 100 epochs, identical batch order and initial weights after training.
All checkpoints and artifacts live on scratch. The frozen launcher retains the
GPU utilization and memory monitor.

If pure BCE wins, its checkpoint already uses every WAIT/ANSWER label. The
runner records this outcome and reuses that checkpoint without adding CL or
performing an identical training run. The selected threshold is in
`selection.json`; the original checkpoint is preserved.

Selection and all reported metrics use the same ten training videos, with no
holdout. They are overfit diagnostics, not generalization estimates or dense
streaming ARS. The artifacts are `four_arm_comparison.json`, `selection.json`
and, when an all-pairs run occurs, `all_pairs_result.json`.
