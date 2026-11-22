# Ten-video fresh-initialization comparison, 2026-11-22

These are the exact four candidate configurations and the predeclared selection
plan. Absolute paths refer to the LRZ server. Raw QA, media, weights and optimizer
state remain on scratch and are not bundled. Each run uses the same original
pretrained initialization, seed, 100 epochs/300 updates, BS64 and GA1.

`completed_three_arm_results.json` records the already completed three-arm suite
(job 5946883). The fourth ANSWER1/WAIT1 hybrid is job 5947329; it was pending when
this snapshot was written. Dependent selection/all-pairs job 5947342 waits for
its success. No fourth-arm or all-pairs result is claimed here. Current status and
new results must be read from the scratch run named in `programs.md`.

The launchers require a prepared scratch RUN_DIR containing frozen `code/`,
`source_hashes.json`, `selection_plan.json`, `planned_batch_order.json`,
`samples.jsonl`, `tasks.jsonl`, `video_map.json`, and `logs/`. The selection plan
references the four completed runs. Their original hashes, data, optimizer-step
counts and batch order are verified before selection. See ../../ALL_PAIRS.md.
