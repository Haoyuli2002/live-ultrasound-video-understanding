"""Unit checks plus a real two-rank CPU Trainer smoke test (no downloads).

Run integration: ACCELERATE_USE_CPU=true OMP_NUM_THREADS=1 torchrun --standalone
  --nproc-per-node=2 -m stage1.test_ddp --worker-output /tmp/stage1-ddp-test
Add --resume-smoke to test restoring checkpoint-2 and advancing to step 3.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import PretrainedConfig, TrainerCallback
from stage1 import train


class DDPTests(unittest.TestCase):
    def test_bind_rank_before_nccl(self):
        events = []
        with patch.dict(os.environ, WORLD_SIZE='2', RANK='1', LOCAL_RANK='1'), \
             patch.object(torch.cuda, 'set_device', side_effect=lambda i: events.append(('device', i))), \
             patch.object(torch.multiprocessing, 'set_start_method') as start, \
             patch.object(torch.distributed, 'init_process_group', side_effect=lambda *a, **k: events.append(('init', a[0]))):
            self.assertEqual(train.setup_distributed(), (1, 2))
        self.assertEqual(events, [('device', 1), ('init', 'nccl')])
        start.assert_called_once_with('spawn', force=True)

    def test_resume_rejects_changed_world_size_or_accumulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'training_config.json').write_text(json.dumps({
                'world_size': 2, 'gradient_accumulation_steps': 4}))
            train.validate_resume_topology(root, 'checkpoint-25', 2, 4)
            for world, accum in [(1, 8), (4, 2), (2, 8)]:
                with self.assertRaisesRegex(ValueError, 'Resume requires unchanged'):
                    train.validate_resume_topology(root, 'checkpoint-25', world, accum)
            train.validate_resume_topology(root, None, 4, 2)

    def test_non_primary_rank_does_not_write_shared_metadata_or_retry_oom(self):
        from stage1.data import build_rows
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = build_rows({'video_id': 'v', 'segments': [
                {'start': 0, 'end': 5, 'text': 'This is the liver.'},
                {'start': 5, 'end': 10, 'text': 'This is the kidney.'}]})
            (root/'data.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (root/'map.json').write_text('{"v": "unused.mp4"}')
            argv = ['train', '--train-jsonl', str(root/'data.jsonl'),
                    '--video-path-map', str(root/'map.json'), '--output-dir', str(root/'out'),
                    '--dataloader-num-workers', '0']
            with patch.object(sys, 'argv', argv), \
                 patch.object(torch.cuda, 'is_available', return_value=True), \
                 patch.object(train, 'setup_distributed', return_value=(1, 2)), \
                 patch.object(torch.distributed, 'barrier'), \
                 patch.object(train, 'preflight') as probe, \
                 patch.object(train, 'build_and_train', side_effect=torch.cuda.OutOfMemoryError('oom')) as attempt:
                with self.assertRaisesRegex(RuntimeError, 'DDP rank 1'):
                    train.main()
                attempt.assert_called_once()
                probe.assert_not_called()
            self.assertFalse((root/'out'/'training_config.json').exists())
            self.assertFalse((root/'out'/'video_split.json').exists())
            self.assertFalse((root/'out'/'validation_samples.jsonl').exists())


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = PretrainedConfig()
        self.linear = torch.nn.Linear(2, 1)
        # Mimic a conditional model's parameters unused by this input modality.
        self.unused = torch.nn.Parameter(torch.ones(1))
        self.seen = []
        self.checkpoint_kwargs = None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.checkpoint_kwargs = gradient_checkpointing_kwargs

    def forward(self, input_ids, labels):
        if self.training:
            self.seen.extend(int(x) for x in input_ids[:, 0].tolist())
        if self.training and self.checkpoint_kwargs is not None:
            logits = torch.utils.checkpoint.checkpoint(self.linear, input_ids, **self.checkpoint_kwargs)
        else:
            logits = self.linear(input_ids)
        return {'loss': ((logits - labels) ** 2).mean(), 'logits': logits}


class TinyDataset:
    def __init__(self, rows, *args):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        value = self.rows[i]
        return {'input_ids': torch.tensor([float(value), 1.]),
                'labels': torch.tensor([float(value) / 2])}


class TinyCollator:
    def __init__(self, *args):
        pass
    def __call__(self, features):
        return {k: torch.stack([f[k] for f in features]) for k in features[0]}


class TinyProcessor:
    def save_pretrained(self, path):
        with (Path(path)/'processor_writers.txt').open('a') as f:
            f.write(os.environ['RANK']+'\n')


def integration_worker(output, resume):
    output.mkdir(parents=True, exist_ok=True)
    args = train.build_arg_parser().parse_args([
        '--train-jsonl', 'unused', '--video-path-map', 'unused', '--output-dir', str(output),
        '--max-steps', '3', '--dtype', 'fp32',
        '--gradient-accumulation-steps', '2', '--dataloader-num-workers', '0',
        '--save-steps', '1', '--logging-steps', '1'])
    torch.manual_seed(7)
    model = TinyModel()
    initial = model.linear.weight.detach().clone()
    if resume:
        from safetensors.torch import load_file
        resumed_weight = load_file(str(output/'checkpoint-2'/'model.safetensors'))['linear.weight']
    class StopAfterTwo(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 2:
                control.should_training_stop = True
            return control
    original_trainer = train.Trainer
    def trainer_factory(*a, **kw):
        return original_trainer(*a, callbacks=[] if resume else [StopAfterTwo()], **kw)
    original_arguments = train.make_training_arguments
    def cpu_arguments(*a, **kw):
        result = original_arguments(*a, **kw)
        # Accelerate on this macOS version returns cpu:0; torch.load accepts cpu.
        # Normalize only the CPU test harness; production CUDA mapping is unchanged.
        result.distributed_state.device = torch.device("cpu")
        result.__dict__["_setup_devices"] = torch.device("cpu")
        return result
    with patch.object(train, 'Trainer', side_effect=trainer_factory), \
         patch.object(train, 'make_training_arguments', side_effect=cpu_arguments), \
         patch.object(train, 'load_model', return_value=(model, TinyProcessor())), \
         patch.object(train, 'Examples', TinyDataset), patch.object(train, 'Collator', TinyCollator):
        model, _, trainer = train.build_and_train(
            args, frame_budget=2, train_rows=list(range(8)), val_rows=list(range(4)), paths={},
            output=output, dtype=torch.float32,
            resume_checkpoint=str(output/'checkpoint-2') if resume else None)
    assert trainer.state.global_step == (3 if resume else 2)
    assert model.checkpoint_kwargs == {'use_reentrant': False}
    assert not torch.equal(initial, model.linear.weight)
    if resume:
        assert not torch.equal(resumed_weight, model.linear.weight), "No optimizer update after resume"
    assert any('eval_loss' in item for item in trainer.state.log_history)
    vector = torch.cat([p.detach().flatten() for p in model.parameters()])
    gathered = [torch.empty_like(vector) for _ in range(2)]
    torch.distributed.all_gather(gathered, vector)
    assert torch.equal(gathered[0], gathered[1]), 'DDP parameters diverged'
    seen = [None, None]
    torch.distributed.all_gather_object(seen, model.seen)
    if trainer.is_world_process_zero():
        if not resume:
            assert set(seen[0]).isdisjoint(seen[1]), seen
            assert sorted(seen[0] + seen[1]) == list(range(8)), seen
        writers = (output/'final'/'processor_writers.txt').read_text().splitlines()
        assert writers == (['0', '0'] if resume else ['0']), writers
        assert (output/f'checkpoint-{trainer.state.global_step}'/'trainer_state.json').is_file()
        print(f'DDP integration passed: step={trainer.state.global_step}, shards={seen}, weights synchronized', flush=True)
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    if '--worker-output' in sys.argv:
        path = Path(sys.argv[sys.argv.index('--worker-output') + 1])
        integration_worker(path, '--resume-smoke' in sys.argv)
    else:
        unittest.main()
