"""Real-attention CPU checks: causal recurrence, gradients and legacy loading."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from .qformer import (QFormerMemoryModel, QueryResampler, SecondwiseShortEncoder,
                      SECONDWISE_ARCHITECTURE, LEGACY_ARCHITECTURE)
from .train_overfit_ab import encode_interval
from .train_full import Objective
from .reconstruct import collect_memories
from .evaluate import evaluate_video


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1))
        self.model = nn.Module()
        self.model.visual = nn.Linear(24, 24)
        self.decoder = nn.Linear(24, 1)
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=24))

    def get_base_model(self):
        return self

    def get_image_features(self, pixel_values, **kwargs):
        return SimpleNamespace(pooler_output=tuple(frame+self.p for frame in pixel_values))


class TinyImageProcessor:
    def __call__(self, images, **kwargs):
        return dict(pixel_values=torch.stack(images), image_grid_thw=torch.ones(len(images),3,dtype=torch.long))


def init_base(memory, *args, **kwargs):
    memory.model = TinyBackbone()
    memory.processor = SimpleNamespace(image_processor=TinyImageProcessor())


def tiny_memory(checkpoint=None):
    with patch('stage2.qformer.MemoryModel.__init__', init_base):
        memory = QFormerMemoryModel('tiny', adapter_path=checkpoint, query_width=16,
                                    query_layers=2, query_heads=4)
    memory.reconstruction_loss = lambda tokens,target,kind: (memory.model.decoder(tokens)-.3).square().mean()
    return memory


def frames(count):
    generator = torch.Generator().manual_seed(77)
    return [torch.randn(3,24,generator=generator) for _ in range(count)]


def blocks():
    return [dict(block_window=[a,a+60],
                 local_sub_summaries=[dict(local_summary_target='target') for _ in range(6)],
                 block_summary_target='block',global_summary_target='global') for a in (0,60)]


class SecondwiseTests(unittest.TestCase):
    def test_one_query_internal_cache_width_and_immutable_history(self):
        encoder = SecondwiseShortEncoder(24,32,16,2,4)
        history = None
        for i in range(20):
            old = None if history is None else history.clone()
            token, history = encoder(torch.randn(1,3,24),history)
            self.assertEqual(token.shape,(1,1,32))
            self.assertEqual(history.shape,(1,i+1,16))
            if old is not None:torch.testing.assert_close(history[:,:-1],old,rtol=0,atol=0)
        self.assertEqual(encoder.queries.shape,(1,16))

    def test_output_projection_is_not_on_the_history_feedback_path(self):
        encoder = SecondwiseShortEncoder(24,32,16,2,4).eval()
        images = frames(3)
        def run():
            history=None;outputs=[]
            for image in images:
                token,history=encoder(image.unsqueeze(0),history);outputs.append(token)
            return torch.cat(outputs,1),history
        a,h1=run()
        with torch.no_grad():encoder.output_projection[1].weight.zero_();encoder.output_projection[1].bias.zero_()
        b,h2=run()
        self.assertFalse(torch.allclose(a,b))
        torch.testing.assert_close(h1,h2,rtol=0,atol=0)

    def test_history_and_current_frame_receive_gradients_and_history_affects_output(self):
        encoder=SecondwiseShortEncoder(24,32,16,2,4)
        early=torch.randn(1,3,24,requires_grad=True);current=torch.randn(1,3,24,requires_grad=True)
        _,history=encoder(early)
        token,updated=encoder(current,history)
        changed,_=encoder(current,history+torch.randn_like(history))
        self.assertFalse(torch.allclose(token,changed))
        token.square().mean().backward()
        self.assertGreater(float(early.grad.abs().sum()),0)
        self.assertGreater(float(current.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in encoder.parameters()))
        self.assertIsNotNone(encoder.queries.grad)

    def test_batch_chunking_matches_per_second_and_future_does_not_change_prefix(self):
        memory=tiny_memory().eval();images=frames(20)
        joint=torch.cat([memory.encode_short(images[:10]),memory.encode_short(images[10:])],1)
        memory.reset_short_memory()
        seconds=torch.stack([memory.encode_frame(image) for image in images],1)
        torch.testing.assert_close(joint,seconds,rtol=1e-6,atol=1e-6)
        memory.reset_short_memory()
        changed=images[:5]+[x*100 for x in images[5:10]]
        perturbed=memory.encode_short(changed)
        torch.testing.assert_close(joint[:,:5],perturbed[:,:5],rtol=0,atol=0)
        self.assertFalse(torch.allclose(joint[:,5:10],perturbed[:,5:10]))
        self.assertEqual(memory._short_history.shape,(1,10,16))
        self.assertIsNone(memory.model.p.grad)

    def test_reset_required_after_60_and_explicit_reset_starts_a_new_minute(self):
        memory=tiny_memory().eval();images=frames(10)
        for _ in range(6):memory.encode_short(images)
        self.assertEqual(memory._short_history.shape[1],60)
        with self.assertRaisesRegex(ValueError,'minute'):memory.encode_frame(images[0])
        memory.reset_short_memory()
        self.assertEqual(memory.encode_frame(images[0]).shape,(1,24))
        self.assertEqual(memory._short_history.shape,(1,1,16))

    def test_training_replay_equals_online_prefix_and_releases_cache(self):
        memory=tiny_memory().eval();images=frames(120);seen=[]
        reader=SimpleNamespace(frame_ending_at=lambda t:seen.append(t) or images[t])
        memory.encode_short(images[:10]);online=memory.encode_short(images[10:20]).detach()
        replay=encode_interval(memory,reader,10,20)
        torch.testing.assert_close(online,replay,rtol=0,atol=0)
        self.assertEqual(seen,list(range(20)))
        self.assertIsNone(memory._short_history)
        replay.square().mean().backward()
        self.assertGreater(float(memory.compressors['short'].queries.grad.abs().sum()),0)
        seen.clear();encode_interval(memory,reader,90,100)
        self.assertEqual(seen,list(range(60,100)))
        self.assertIsNone(memory._short_history)
        with self.assertRaisesRegex(ValueError,'one minute'):encode_interval(memory,reader,50,70)
        with self.assertRaisesRegex(ValueError,'nonempty'):encode_interval(memory,reader,10,10)

    def test_independent_optimizer_steps_do_not_reuse_stale_short_graphs(self):
        memory=tiny_memory();images=frames(60)
        reader=SimpleNamespace(frame_ending_at=lambda t:images[t])
        objective=Objective(memory);optimizer=torch.optim.AdamW(objective.parameters(),lr=.001)
        for kind,start,end in [('short',0,10),('short',10,20),('block',0,60),('global',0,60)]:
            optimizer.zero_grad(set_to_none=True)
            loss,_=objective(dict(kind=kind,start=start,end=end,target='t'),reader,None)
            loss.backward();optimizer.step()
            self.assertIsNone(memory._short_history)
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(float(memory.compressors['short'].queries.grad.abs().sum()),0)
        self.assertGreater(float(memory.compressors['long'].queries.grad.abs().sum()),0)

    def test_reconstruction_and_loss_eval_reset_at_minutes_and_video_boundaries(self):
        memory=tiny_memory().eval();images=frames(60)*2
        reader=SimpleNamespace(frame_ending_at=lambda t:images[t])
        records=collect_memories(memory,blocks(),reader)
        self.assertEqual(len(records),16)
        torch.testing.assert_close(records[0]['memory'],records[8]['memory'],rtol=0,atol=0)
        self.assertIsNone(memory._short_history)
        again=collect_memories(memory,blocks(),reader)
        for a,b in zip(records,again):torch.testing.assert_close(a['memory'],b['memory'],rtol=0,atol=0)
        rows=evaluate_video(memory,blocks(),reader)
        self.assertEqual(len(rows),16)
        with torch.no_grad():
            for row,record in zip(rows,records):
                self.assertAlmostEqual(row['loss'],float(memory.reconstruction_loss(record['memory'],'t',record['kind'])),places=6)
        self.assertIsNone(memory._short_history)

    def test_config_and_checkpoint_roundtrip_and_legacy_weights_stay_legacy(self):
        memory=tiny_memory().eval();images=frames(2)
        expected=memory.encode_short(images).detach()
        self.assertEqual(memory.qformer_config['architecture'],SECONDWISE_ARCHITECTURE)
        self.assertEqual(memory.qformer_config['short_queries'],1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch('stage2.qformer.MemoryModel.save',lambda self,path:Path(path).mkdir(exist_ok=True)):
                memory.save(str(root))
            loaded=tiny_memory(str(root)).eval()
            self.assertTrue(loaded.requires_short_history)
            self.assertIsNone(loaded._short_history)
            torch.testing.assert_close(loaded.encode_short(images),expected,rtol=0,atol=0)
            legacy=nn.ModuleDict({'short':QueryResampler(24,24,10,16,2,4,10),
                                  'long':QueryResampler(24,24,60,16,2,4,120)})
            cfg=dict(memory.qformer_config,architecture=LEGACY_ARCHITECTURE,short_queries=10)
            (root/'qformer_config.json').write_text(json.dumps(cfg));torch.save(legacy.state_dict(),root/'qformer.pt')
            old=tiny_memory(str(root)).eval()
            self.assertFalse(old.requires_short_history)
            self.assertEqual(old.encode_short(images).shape,(1,2,24))
            with self.assertRaisesRegex(RuntimeError,'Legacy'):old.encode_frame(images[0])
            cfg['architecture']='unknown';(root/'qformer_config.json').write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError,'Unsupported'):tiny_memory(str(root))


if __name__=='__main__':unittest.main()
