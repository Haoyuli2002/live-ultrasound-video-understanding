import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import torch
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers
from lavis.models.blip2_models.blip2 import Blip2Base
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
from lavis.models.blip2_models.streamready import StreamReady
from .data import causal_indices, load_examples, prompt, sample_times
from .model import forward_batch, predict_batch
from .gpu_monitor import select_gpu
from .optimization import task_batches, backward_objectives

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('qwen_test_helpers', ROOT / 'StreamReady/tests/test_qwen35_backbone.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def tiny_model():
    raw = Tokenizer(models.WordLevel({'[PAD]': 0, '[UNK]': 1, '[EOS]': 2, 'what': 3, 'yes': 4}, unk_token='[UNK]'))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token='[PAD]', unk_token='[UNK]', eos_token='[EOS]')
    backbone = helpers.tiny_backbone()
    backbone.tokenizer = tok

    def qformer(queries, width, **kwargs):
        cfg = BertConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=2,
                         num_attention_heads=4, intermediate_size=64,
                         hidden_dropout_prob=0., attention_probs_dropout_prob=0.)
        cfg.encoder_width = width
        cfg.add_cross_attention = True
        cfg.cross_attention_freq = 2
        cfg.query_length = queries
        cfg.gradient_checkpointing = True
        return BertLMHeadModel(cfg), torch.nn.Parameter(torch.randn(1, queries, 32) * .02)

    with patch('lavis.models.blip2_models.streamready.build_backbone', return_value=backbone), \
         patch.object(Blip2Base, 'init_tokenizer', return_value=tok), \
         patch.object(Blip2Base, 'init_Qformer', side_effect=qformer):
        return StreamReady(backbone_model='tiny', backbone_precision='fp32', num_query_token=3,
                           qformer_encoder_width=24, frame_capacity=2, num_centroids=3, num_prototypes=2,
                           topk_prototypes=2, topk_centroids=3, use_contextual_memory=False,
                           gradient_checkpointing=True, readiness_threshold=.5)


def example(ready=1):
    return dict(sample_id='x', task_id='x', video_id='x', question='what',
                query_time=1., current_time=2., qa_type='current_observation', ready=ready,
                answer='yes' if ready else '')


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_pts_and_no_future(self):
        self.assertEqual(sample_times(2.3), [0., 1., 2., 2.3])
        self.assertEqual(causal_indices([0., .99, 1.01, 2.01], [0., 1., 2.]), [0, 1, 2])
        with self.assertRaises(ValueError):
            causal_indices([.1, .5], [0.])

    def test_gpu_identity_uses_driver_uuid(self):
        uuid = 'GPU-12345678-1234-1234-1234-123456789abc'
        rows = [('0', 'GPU-ffffffff-ffff-ffff-ffff-ffffffffffff'), ('2', uuid)]
        self.assertEqual(select_gpu(rows, "UUID('12345678-1234-1234-1234-123456789abc')", {}), uuid)
        self.assertEqual(select_gpu(rows, 'unknown', {'SLURM_JOB_GPUS': '2'}), uuid)
        with self.assertRaises(RuntimeError):
            select_gpu(rows, 'unknown', {})

    def test_video_bridge_and_real_cpu_preprocessing(self):
        from .train import prepare_features, video_array
        from decord import bridge
        tensor = torch.zeros(2, 32, 32, 3, dtype=torch.uint8)
        self.assertEqual(video_array(tensor).shape, (2, 32, 32, 3))
        self.assertEqual(video_array(SimpleNamespace(asnumpy=lambda: tensor.numpy())).shape, (2, 32, 32, 3))
        video = ROOT / 'cluster_data/videos/train_full295/P3FoSFIrfQA.mp4'
        if not video.exists():
            self.skipTest('Real video is only available in the shared server workspace')
        def encode(pixels):
            self.assertEqual(tuple(pixels.shape[:2]), (1, 3))
            self.assertEqual(tuple(pixels.shape[-2:]), (256, 256))
            self.assertTrue(torch.isfinite(pixels).all())
            return torch.zeros(1, pixels.shape[2], 1, 32)
        model = SimpleNamespace(backbone=SimpleNamespace(encode_frames=encode))
        rows = [dict(sample_id='a', video_id='P3FoSFIrfQA', video_path=str(video), current_time=2.3),
                dict(sample_id='b', video_id='P3FoSFIrfQA', video_path=str(video), current_time=3.)]
        bridge.set_bridge('torch')
        with tempfile.TemporaryDirectory() as tmp, patch.object(torch.Tensor, 'cuda', lambda x: x):
            cache = prepare_features(model, rows, Path(tmp), 256, 2)
            self.assertEqual(cache['a'].shape, (1, 4, 1, 32))
            self.assertEqual(cache['b'].shape, (1, 4, 1, 32))
            audit = json.loads((Path(tmp) / 'frame_audit.json').read_text())[0]
            self.assertTrue(all(p <= t for p, t in zip(audit['actual_pts'], audit['requested_times'])))

    def test_no_gold_in_prompt(self):
        e = example(); e['answer_time'] = 9876; e['answer'] = 'SECRET'; e['evidence'] = 'SECRET'
        self.assertNotIn('SECRET', prompt(e)); self.assertNotIn('9876', prompt(e))

    def test_task_sampler_preserves_endpoints_without_duplicates(self):
        rows = [dict(example(y), sample_id=f'{task}-{i}', task_id=task, current_time=float(i))
                for task, labels in [('a', [0, 0, 0, 1]), ('b', [1]), ('c', [0, 1])]
                for i, y in enumerate(labels)]
        batches = task_batches(rows, 4, 7)
        self.assertEqual(sorted(r['sample_id'] for b in batches for r in b), sorted(r['sample_id'] for r in rows))
        for task in ('a', 'b', 'c'):
            self.assertEqual(sum(any(r['task_id'] == task for r in b) for b in batches), 1)
        self.assertEqual(batches, task_batches(rows, 4, 7))
        with self.assertRaises(ValueError):
            task_batches(rows, 2, 7)

    def test_corrected_joint_backward_and_ntp_when_predicted_wait(self):
        model = tiny_model().train()
        with torch.no_grad():
            model.core.readiness_head.net[-1].bias.fill_(-4.)
        rows = [example(0), example(1)]
        feats = [torch.randn(1, 3, 2, 32), torch.randn(1, 4, 2, 32)]
        with patch.object(model, '_reasoning_loss', wraps=model._reasoning_loss) as answer:
            out = forward_batch(model, rows, feats)
            self.assertTrue(bool((out['logits'] < 0).all()))
            answer.assert_called_once()
            self.assertTrue(out['answer_loss'].requires_grad)
            balance = backward_objectives(model, out, 'readiness_isolated')
        self.assertTrue(torch.isfinite(model.core.output_proj.weight.grad).all())
        self.assertGreater(float(model.core.output_proj.weight.grad.norm()), 0)
        self.assertGreater(float(model.core.readiness_head.net[-1].weight.grad.norm()), 0)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        self.assertEqual(balance['strategy'], 'readiness_isolated')

    def test_paper_timing_gradient_isolation_in_one_training_step(self):
        model = tiny_model().train()
        rows = [example(0), example(1)]
        features = [torch.randn(1, 3, 2, 32), torch.randn(1, 4, 2, 32)]
        out = forward_batch(model, rows, features)
        params = [model.core.long_query_tokens, model.core.vision_proj.weight, model.core.output_proj.weight]
        expected = torch.autograd.grad(out['answer_loss'], params, retain_graph=True, allow_unused=True)
        backward_objectives(model, out, 'readiness_isolated')
        for p, reference in zip(params, expected):
            if reference is not None:
                torch.testing.assert_close(p.grad, reference)
        self.assertGreater(float(model.core.readiness_head.net[-1].weight.grad.norm()), 0)
        model.zero_grad(set_to_none=True)
        out = forward_batch(model, [example(0)], features[:1])
        backward_objectives(model, out, 'readiness_isolated')
        self.assertTrue(all(p.grad is None for p in model.reasoning_parameters()))
        self.assertTrue(any(p.grad is not None for p in model.readiness_parameters()))

    def test_endpoint_matches_sequential_with_memory_eviction(self):
        model = tiny_model().eval()
        feats = torch.randn(1, 6, 2, 32)
        with torch.no_grad():
            full = model.core(feats, ['what'])
            endpoint = model.core(feats, ['what'], endpoint_only=True)
        torch.testing.assert_close(full['prefix_embeds'], endpoint['prefix_embeds'])
        torch.testing.assert_close(full['readiness'][:, -1], endpoint['readiness'][:, -1])

    def test_first_answer_token_retained(self):
        model = tiny_model()
        ids, lengths = model.concat_text_input_output(torch.tensor([[3, 0]]), torch.tensor([[1, 0]]),
                                                      torch.tensor([[4, 2]]), torch.ones(1, 2, dtype=torch.long))
        self.assertEqual(ids['input_ids'].tolist(), [[3, 4, 2, 0]])
        self.assertEqual(int(lengths[0]), 1)

    def test_variable_length_batch_gradients_and_wait(self):
        model = tiny_model().train()
        features = [torch.randn(1, 5, 2, 32), torch.randn(1, 3, 2, 32)]
        rows = [example(), example(0)]
        # First fusion output starts at zero. A real update must open the short path.
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            out = forward_batch(model, rows, features)
            self.assertTrue(torch.isfinite(out['loss']))
            out['loss'].backward()
            opt.step()
        self.assertGreater(float(model.core.vision_proj.weight.grad.norm()), 0)
        self.assertGreater(float(model.core.short_query_tokens.grad.norm()), 0)
        self.assertGreater(float(model.core.output_proj.weight.grad.norm()), 0)
        self.assertGreater(float(model.core.rdy_token.grad.norm()), 0)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        with patch.object(model, '_reasoning_loss', side_effect=AssertionError('WAIT must not train answer')):
            out = forward_batch(model, [example(0)], features[:1])
            self.assertEqual(float(out['answer_loss']), 0.)

    def test_wait_inference_does_not_force_answer(self):
        model = tiny_model().eval(); model.readiness_threshold = 1.
        with patch.object(model, '_decode', side_effect=AssertionError('No forced answer')):
            result, _ = predict_batch(model, [example(0)], [torch.randn(1, 3, 2, 32)])
        self.assertFalse(result[0]['ready']); self.assertEqual(result[0]['answer'], '')

    def test_loader_rejects_premature_and_accepts_untriggered(self):
        task = dict(task_id='x', video_id='v', question='what', query_time=1, answer_time=3,
                    reference_answer='yes', qa_type='event_notification', split='train', review_horizon=3)
        row = dict(sample_id='x@2', task_id='x', video_id='v', question='what', video_window=[0, 2], target='<WAIT>')
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / 'map.json').write_text(json.dumps({'v': '/unused.mp4'}))
            def load():
                (p / 'tasks.jsonl').write_text(json.dumps(task) + '\n')
                (p / 'samples.jsonl').write_text(json.dumps(row) + '\n')
                return load_examples(p / 'samples.jsonl', p / 'tasks.jsonl', p / 'map.json')
            self.assertEqual(load()[0]['ready'], 0)
            row['target'] = '<ANSWER> yes'
            with self.assertRaises(ValueError):
                load()
            row['target'] = '<WAIT>'; task.update(answer_time=None, reference_answer=None, untriggered=True)
            self.assertEqual(load()[0]['ready'], 0)


if __name__ == '__main__':
    unittest.main()
