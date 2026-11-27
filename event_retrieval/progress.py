"""Optimizer-update progress for terminals, Slurm logs and JSON readers (CPU only)."""
from collections import deque
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time


def duration(seconds):
    if seconds is None:
        return 'unknown'
    seconds = max(0, round(seconds))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


class TrainingProgress:
    def __init__(self, run, *, examples, batch_size, world_size, accumulation, epochs,
                 completed_epochs=0, holdout=False, save_every=1, log_every=10,
                 log_seconds=30, enabled=True, clock=time.monotonic, emit=print):
        if min(examples, batch_size, world_size, accumulation, epochs, save_every, log_every, log_seconds) <= 0:
            raise ValueError('Progress counts/intervals must be positive')
        if not 0 <= completed_epochs <= epochs:
            raise ValueError('Completed epochs outside requested budget')
        self.enabled, self.clock, self.emit = enabled, clock, emit
        self.run = Path(run)
        self.examples, self.epochs = examples, epochs
        self.steps_per_epoch = math.ceil(examples / (batch_size * world_size * accumulation))
        self.total = self.steps_per_epoch * epochs
        self.step = self.initial_step = completed_epochs * self.steps_per_epoch
        self.epoch = completed_epochs
        self.processed = 0
        self.holdout = holdout
        self.save_epochs = {e for e in range(completed_epochs + 1, epochs + 1) if e % save_every == 0 or e == epochs}
        self.holdout_done = set(range(1, completed_epochs + 1))
        self.saved = set()
        self.phase_times = {'holdout': [], 'checkpoint': []}
        self.intervals = deque(maxlen=50)
        self.phase_name = 'initializing'
        self.started = self.phase_started = self.last_update = clock()
        self.last_logged = self.started
        self.last_logged_step = self.step
        self.log_every, self.log_seconds = log_every, log_seconds
        self.error = None
        self.holdout_processed = self.holdout_total = None

    def __enter__(self):
        if self.enabled:
            self.run.mkdir(parents=True, exist_ok=True)
            self.write(force=True)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc is not None and self.enabled:
            self.error = repr(exc)
            try:
                self.phase('failed')
            except Exception as reporting_error:
                self.emit(f'[progress] Failed to record failure: {reporting_error}', flush=True)
        return False

    def _close_phase(self, now):
        if self.phase_name in self.phase_times:
            self.phase_times[self.phase_name].append(max(0, now - self.phase_started))
            if self.phase_name == 'holdout':
                self.holdout_done.add(self.epoch)
            else:
                self.saved.add(self.epoch)

    def phase(self, name):
        if not self.enabled:
            return
        now = self.clock()
        if name != 'failed':
            self._close_phase(now)
        self.phase_name, self.phase_started = name, now
        if name == 'train':
            self.last_update = now
        self.write(force=True)

    def start_epoch(self, epoch):
        if not self.enabled:
            return
        if self.step != (epoch - 1) * self.steps_per_epoch:
            raise ValueError('Observed optimizer steps differ from epoch budget')
        self.epoch, self.processed = epoch, 0
        self.holdout_processed = self.holdout_total = None
        self.phase('train')

    def update(self, *, processed, loss=None):
        """Called only after an optimizer update returns successfully, once on rank zero."""
        if not self.enabled:
            return
        if self.phase_name != 'train' or self.step >= self.epoch * self.steps_per_epoch:
            raise ValueError('Unexpected optimizer update')
        now = self.clock()
        self.intervals.append(max(0, now - self.last_update))
        self.last_update = now
        self.step += 1
        self.processed = processed
        self.write(force=self.step == self.epoch * self.steps_per_epoch, loss=loss)

    def holdout_batch(self, processed, total):
        if self.enabled:
            self.holdout_processed, self.holdout_total = processed, total
            self.write(force=processed == total)

    def finish(self):
        if self.enabled:
            if self.step != self.total:
                raise ValueError('Training finished before all planned optimizer updates')
            self.phase('completed')

    def snapshot(self):
        now = self.clock()
        remaining = self.total - self.step
        per_step = sum(self.intervals) / len(self.intervals) if self.intervals else None
        train_eta = remaining * per_step if per_step is not None else (0. if remaining == 0 else None)
        overhead = 0.
        unknown = []
        for name, count in [('holdout', self.epochs - len(self.holdout_done) if self.holdout else 0),
                            ('checkpoint', len(self.save_epochs - self.saved))]:
            if not count:
                continue
            history = self.phase_times[name]
            if not history:
                unknown.append(name)
            else:
                estimate = sum(history) / len(history)
                current = min(estimate, now - self.phase_started) if self.phase_name == name else 0.
                overhead += max(0, count * estimate - current)
        job_eta = train_eta + overhead if train_eta is not None and not unknown else None
        if self.phase_name == 'completed':
            train_eta = job_eta = 0.
        if self.phase_name in {'initializing', 'indexing', 'distributed_setup', 'failed'}:
            job_eta = None
        return dict(timestamp=datetime.now(timezone.utc).isoformat(), phase=self.phase_name,
                    epoch=self.epoch, epochs=self.epochs, global_step=self.step, total_steps=self.total,
                    epoch_step=max(0, self.step - (self.epoch - 1) * self.steps_per_epoch) if self.epoch else 0,
                    steps_per_epoch=self.steps_per_epoch, session_start_step=self.initial_step,
                    session_completed_steps=self.step-self.initial_step, percent=100*self.step/self.total,
                    train_endpoints_processed=self.processed, train_endpoints_per_epoch=self.examples,
                    holdout_endpoints_processed=self.holdout_processed, holdout_endpoints_total=self.holdout_total,
                    elapsed_seconds=now-self.started, phase_elapsed_seconds=now-self.phase_started,
                    recent_seconds_per_update=per_step, recent_updates_per_second=1/per_step if per_step else None,
                    eta_training_updates_seconds=train_eta, eta_job_seconds=job_eta,
                    eta_unmeasured_phases=unknown, eta_window_updates=len(self.intervals),
                    eta_scope='Remaining training updates include retrieval/data preparation; job ETA adds measured holdout/checkpoint costs. Excludes separate generation evaluation and queue time.',
                    error=self.error)

    def write(self, *, force=False, loss=None):
        if not self.enabled:
            return
        row = self.snapshot()
        if loss is not None:
            row['last_microbatch_loss'] = loss
        path = self.run / 'progress.json'
        tmp = path.with_suffix('.json.partial')
        tmp.write_text(json.dumps(row, allow_nan=False) + '\n')
        tmp.replace(path)
        now = self.clock()
        if not (force or self.step-self.last_logged_step >= self.log_every or now-self.last_logged >= self.log_seconds):
            return
        with (self.run / 'progress.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        speed = 'unknown' if row['recent_seconds_per_update'] is None else f"{row['recent_seconds_per_update']:.2f}s/update"
        detail = '' if self.holdout_processed is None or self.phase_name != 'holdout' else f' | holdout {self.holdout_processed}/{self.holdout_total}'
        self.emit(f"[{self.phase_name}] epoch {self.epoch}/{self.epochs} | optimizer steps {self.step}/{self.total} ({row['percent']:.1f}%) | epoch steps {row['epoch_step']}/{self.steps_per_epoch} | {speed} | ETA updates {duration(row['eta_training_updates_seconds'])} | ETA job {duration(row['eta_job_seconds'])}{detail}", flush=True)
        self.last_logged, self.last_logged_step = now, self.step
