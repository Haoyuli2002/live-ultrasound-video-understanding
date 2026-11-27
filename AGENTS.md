# Repository collaboration rules

## Required project log updates

The user requires `programs.md` at the repository root to be updated whenever a model training job is submitted or a code commit is created. This applies to full training, overfit/smoke training, resubmissions, and resumed training jobs.

- Before submitting training, record the intended experiment, configuration, and validation performed in `programs.md`. After submission, promptly record the returned job ID, run directory, and actual submission status. Record submission failures honestly; do not label queued jobs as running or completed.
- Record the model/architecture, initialization checkpoint or adapter, data version and split (including holdout), preprocessing, epochs, GPU count, learning rate, relevant changes, and paths to frozen configuration/code and logs. Link to detailed configuration rather than guessing missing values.
- Before each code commit, update `programs.md` with the change, its purpose, validation results or checks not run, and relevant experiment references. Include that update in the same commit. Do not create a follow-up commit solely to record the commit's own hash; use a descriptive change identifier when the hash is not yet available.
- Preserve previous entries and distinguish planned actions, submitted jobs, observed results, and unresolved issues. Never invent results or treat an older configuration's overfit result as validation of a new configuration.
- This logging requirement does not authorize submitting training or committing code on its own. Follow the user's requested scope.

## Checkpoint storage

- User requires all future training checkpoints, optimizer states, adapters and temporary checkpoint writes to live on scratch, not home. This applies to every stage, smoke/overfit runs, continuations and resubmissions.
- Use `/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding/cluster_data/experiments/<stage-or-model>/<unique-run>/` for new training runs. The repository `cluster_data` link may be used after resolving and verifying its scratch target.
- Resolve output paths and symlinks before submission. Keep the full run on scratch; home `results/` may contain small reports or links to scratch runs. Freeze this path in the submitted configuration and record it in `programs.md`.
- Existing checkpoints remain intact unless migration or deletion is separately requested.

## Independent model architectures

Keep the event-based streaming QA architecture under `event_retrieval/` and its new training runs under the scratch experiment namespace `event_retrieval/`. Keep StreamReady reproduction code and runs in their existing namespaces. Attribute checkpoints, evaluation results and demo outputs to their respective architecture. Shared source videos and splits can be used for comparisons.

## Live training progress

For future new or resumed training jobs, expose planned total optimizer updates, successfully completed updates, epoch progress, throughput and estimated remaining time in flushed logs and a machine-readable progress file. Count optimizer updates, not gradient-accumulation microbatches or the sum of per-rank steps. Restore the completed update offset on resume. Distinguish training, holdout and checkpoint phases; label ETA scope and show unknown until sufficient timing observations exist. Verify that the submitted frozen entry point includes this reporting. Do not restart or modify an already running frozen job solely to add progress logging.
