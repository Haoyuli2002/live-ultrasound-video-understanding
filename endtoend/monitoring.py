"""Independent holdout loading and evaluation state isolation."""
from contextlib import contextmanager
from pathlib import Path
import random
import numpy as np
import torch
from .data import load_examples


def validate_split_separation(train,holdout):
    if not train or not holdout:raise ValueError('Train and holdout must both be nonempty')
    if any(r['split']!='train' for r in train) or any(r['split']!='holdout' for r in holdout):
        raise ValueError('Unexpected train/holdout split labels')
    for field in ['video_id','task_id','sample_id']:
        if {r[field] for r in train}&{r[field] for r in holdout}:
            raise ValueError(f'Train/holdout overlap: {field}')
    if {str(Path(r['video_path']).resolve()) for r in train}&{str(Path(r['video_path']).resolve()) for r in holdout}:
        raise ValueError('Train/holdout media path overlap')


def load_holdout(run,cfg,train):
    if not cfg.get('holdout'):return []
    directory=Path(run)/cfg['holdout']['directory']
    rows=load_examples(directory/'samples.jsonl',directory/'tasks.jsonl',directory/'video_map.json')
    validate_split_separation(train,rows)
    return rows


@contextmanager
def evaluation_state(model):
    training=model.training;py_state=random.getstate();np_state=np.random.get_state()
    devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices),torch.no_grad():
            model.eval()
            yield
    finally:
        model.train(training);random.setstate(py_state);np.random.set_state(np_state)


def evaluation_due(cfg, step, epoch, epoch_end=False, last_step=None):
    """A step interval replaces epoch scheduling; final evaluation is mandatory."""
    if step == last_step:return False
    interval = int(cfg.get('eval_every_steps', 0))
    if interval < 0:raise ValueError('eval_every_steps must be nonnegative')
    if interval:
        return (epoch == cfg['epochs']) if epoch_end else step > 0 and step % interval == 0
    return epoch_end and (epoch in cfg.get('eval_epochs', []) or epoch == cfg['epochs'])


def token_mean_answer_ce(records):
    total = sum(r.get('answer_token_count', 0) for r in records)
    return sum(r.get('answer_token_nll', 0.) for r in records) / total if total else 0.
