"""CPU tests of decoder leakage, chronology and independent memory controls."""
import unittest
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

import torch

from .model import MemoryModel
from .reconstruct import (collect_memories, controlled_memory, greedy_reconstruct,
                          summarize, text_metrics)


class Tokenizer:
    eos_token_id = 9

    def __call__(self, text, **kwargs):
        self.texts = getattr(self, 'texts', [])+[text]
        return SimpleNamespace(input_ids=torch.tensor([[1, 2]]))

    def decode(self, ids, skip_special_tokens=False):
        return ' '.join(str(i) for i in ids if not (skip_special_tokens and i == 9))


class Decoder:
    device = torch.device('cpu')
    core_model = SimpleNamespace(training=False)

    def __init__(self, script=(3, 4, 9)):
        self.tokenizer = Tokenizer()
        self.script = script
        self.calls = []
        # The frozen overfit architecture predates the DDP core_model wrapper.
        self.model = self._forward

    def embedding(self, ids):
        return ids.to(torch.float32).unsqueeze(-1)

    def _forward(self, **kwargs):
        self.calls.append(kwargs)
        logits = torch.zeros(1, kwargs['inputs_embeds'].shape[1], 10)
        logits[:, -1, self.script[(len(self.calls)-1) % len(self.script)]] = 1
        return SimpleNamespace(logits=logits, loss=torch.tensor(0.0))


class Encoder:
    def __init__(self):
        self.frames = []
        self.updates = []

    def encode_frame(self, frame):
        self.frames.append(frame)
        return torch.tensor([[float(frame)]])

    def update_long(self, previous, current):
        self.updates.append((None if previous is None else previous.clone(), current.clone()))
        return current if previous is None else previous + current


def blocks():
    return [{'block_window': [a, a+60],
             'local_sub_summaries': [{'local_summary_target': f'reference-{a+j}'}
                                     for j in range(6)],
             'block_summary_target': f'block-{a}',
             'global_summary_target': f'global-{a}'} for a in (0, 60)]


class ReconstructionTests(unittest.TestCase):
    def test_decode_only_memory_prompt_and_own_generated_tokens(self):
        memory = Decoder()
        tokens = torch.tensor([[[20.], [21.]]])
        out = greedy_reconstruct(memory, tokens, 'short', 10)
        self.assertEqual(out['generated_token_ids'], [3, 4, 9])
        self.assertEqual(out['stop_reason'], 'eos')
        self.assertEqual(memory.tokenizer.texts,
                         ['Summarize the visible ultrasound evidence (short):\n'])
        self.assertEqual(memory.calls[0]['inputs_embeds'].flatten().tolist(), [20, 21, 1, 2])
        self.assertEqual(memory.calls[-1]['inputs_embeds'].flatten().tolist(), [20, 21, 1, 2, 3, 4])
        self.assertTrue(all(c['use_cache'] is False and 'labels' not in c for c in memory.calls))

    def test_frozen_pre_ddp_model_interface(self):
        memory = Decoder()
        class Backbone:
            training = False
            def __call__(self, **kwargs):
                return memory._forward(**kwargs)
        legacy = SimpleNamespace(model=Backbone(), device=memory.device,
                                 tokenizer=memory.tokenizer, embedding=memory.embedding)
        result = greedy_reconstruct(legacy, torch.ones(1,10,1), 'short', 5)
        self.assertEqual(result['generated_token_ids'], [3,4,9])

    def test_decode_prefix_exactly_matches_training_prefix(self):
        memory = Decoder()
        tokens = torch.tensor([[[20.], [21.]]])
        MemoryModel.reconstruction_loss(memory, tokens, 'reference supplied only to loss', 'global')
        training = memory.calls[-1]
        prefix = training['inputs_embeds'][:, :4, :].clone()
        memory.calls.clear()
        greedy_reconstruct(memory, tokens, 'global', 1)
        self.assertTrue(torch.equal(prefix, memory.calls[0]['inputs_embeds']))
        self.assertEqual(memory.tokenizer.texts[-1], memory.tokenizer.texts[0])

    def test_max_tokens_and_training_mode_rejected(self):
        memory = Decoder((3,))
        out = greedy_reconstruct(memory, torch.ones(1, 10, 1), 'short', 2)
        self.assertEqual(out['stop_reason'], 'max_new_tokens')
        self.assertEqual(out['generated_tokens'], 2)
        memory.core_model = SimpleNamespace(training=True)
        with self.assertRaisesRegex(ValueError, 'eval'):
            greedy_reconstruct(memory, torch.ones(1, 10, 1), 'short')

    def test_causal_encoding_and_recurrent_long_without_targets(self):
        memory = Encoder()
        reader = SimpleNamespace(frame_ending_at=lambda t: t)
        rows = collect_memories(memory, blocks(), reader)
        self.assertEqual(memory.frames, list(range(120)))
        self.assertEqual(len(rows), 16)
        self.assertEqual([r['memory'].shape[1] for r in rows], [10]*6+[60, 60]+[10]*6+[60, 60])
        self.assertIsNone(memory.updates[0][0])
        self.assertTrue(torch.equal(memory.updates[1][0], memory.updates[0][1]))
        self.assertEqual(rows[-1]['window'], [0, 120])
        self.assertEqual(rows[-1]['memory'][0, 0, 0], 60)
        self.assertEqual(rows[0]['memory'][0, -1, 0], 9)

    def test_controls_do_not_mutate_or_recompute_real_state(self):
        memory = Encoder()
        rows = collect_memories(memory, blocks(), SimpleNamespace(frame_ending_at=lambda t:t))
        original = [r['memory'].clone() for r in rows]
        zero, _ = controlled_memory(rows, 0, 'zero')
        self.assertEqual(zero.count_nonzero(), 0)
        swapped, meta = controlled_memory(rows, 0, 'other_window')
        self.assertEqual(meta['memory_window'], [10, 20])
        self.assertTrue(torch.equal(swapped, rows[1]['memory']))
        self.assertTrue(meta['diagnostic_only_noncausal_control'])
        long, meta = controlled_memory(rows, 7, 'other_window')
        self.assertEqual(meta['memory_window'], [0, 120])
        self.assertEqual(len(memory.updates), 2)
        self.assertTrue(all(torch.equal(o, r['memory']) for o,r in zip(original,rows)))

    def test_missing_donor_skipped_and_lexical_metrics(self):
        rows = [{'kind':'global','window':[0,60],'memory':torch.ones(1,60,1)}]
        self.assertIn('skipped', controlled_memory(rows, 0, 'other_window')[1])
        metrics = text_metrics('A vessel, visible.', 'a vessel visible')
        self.assertTrue(metrics['normalized_exact_match'])
        self.assertEqual(metrics['rouge_l_f1'], 1)
        self.assertEqual(text_metrics('', 'vessel')['word_overlap_f1'], 0)
        self.assertEqual(summarize([{'condition':'real','kind':'short','stop_reason':'eos',**metrics}])['real/short']['samples'], 1)


class SubmissionTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        train = root/'training run'
        (train/'code/stage2').mkdir(parents=True)
        (train/'code/stage2/model.py').write_text('# frozen architecture, not current repo\n')
        (train/'training/final').mkdir(parents=True)
        (train/'training/final/adapter_config.json').write_text('{}')
        (train/'training/training_config.json').write_text(json.dumps({'init_adapter':None}))
        (train/'blocks.jsonl').write_text('unchanged labels\n')
        video = root/'v.mp4'
        video.write_bytes(b'presence fixture only')
        (train/'video_map.json').write_text(json.dumps({'v':str(video)}))
        conda=root/'conda.sh'
        conda.write_text('conda() { :; }\n')
        bindir=root/'bin'
        bindir.mkdir()
        sbatch=bindir/'sbatch'
        sbatch.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\nprintf "123456;fixture\\n"\n')
        sbatch.chmod(0o755)
        repo=Path(__file__).resolve().parents[1]
        env=dict(os.environ,REPO=str(repo),TRAIN_RUN=str(train),CONDA_SH=str(conda),
                 GPU_TYPE='a100',WALLTIME='01:00:00',DRY_RUN='0',
                 PATH=str(bindir)+os.pathsep+os.environ['PATH'],
                 CAPTURE_ARGS=str(root/'args.txt'))
        return root,train,repo,env

    def test_submission_freezes_training_architecture_and_requests_one_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root,train,repo,env=self.fixture(directory)
            completed=subprocess.run(['bash',str(repo/'scripts/slurm/submit_stage2_reconstruct.sh')],
                                     env=env,capture_output=True,text=True)
            self.assertEqual(completed.returncode,0,completed.stderr)
            self.assertIn('Job: 123456',completed.stdout)
            runs=list(train.glob('reconstruct_final_*'))
            self.assertEqual(len(runs),1)
            run=runs[0]
            self.assertEqual((run/'code/stage2/model.py').read_text(),
                             (train/'code/stage2/model.py').read_text())
            self.assertEqual((run/'code/stage2/reconstruct.py').read_bytes(),
                             (repo/'stage2/reconstruct.py').read_bytes())
            args=(root/'args.txt').read_text()
            for expected in ['--gres=gpu:1','--mem=64G','--time=01:00:00',
                             '--partition=lrz-dgx-a100-80x8,lrz-hgx-a100-80x4']:
                self.assertIn(expected,args)
            plan=json.loads((run/'submission.json').read_text())
            self.assertFalse(plan['target_used_in_decoding'])
            self.assertFalse(plan['training'])
            self.assertEqual((train/'blocks.jsonl').read_text(),'unchanged labels\n')
            self.assertEqual((run/'job_id.txt').read_text().strip(),'123456')

    def test_failed_preflight_stops_before_copy_or_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root,train,repo,env=self.fixture(directory)
            (train/'training/final/adapter_config.json').unlink()
            completed=subprocess.run(['bash',str(repo/'scripts/slurm/submit_stage2_reconstruct.sh')],
                                     env=env,capture_output=True,text=True)
            self.assertNotEqual(completed.returncode,0)
            self.assertIn('Missing frozen experiment input',completed.stderr)
            self.assertFalse((root/'args.txt').exists())
            self.assertFalse(list(train.glob('reconstruct_final_*')))


if __name__ == '__main__':
    unittest.main()
