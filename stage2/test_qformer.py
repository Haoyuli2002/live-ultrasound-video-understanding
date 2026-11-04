import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from .qformer import QueryResampler, QFormerMemoryModel
from .train_full import Objective,schedules,tasks_for_video
from .train_overfit_ab import encode_interval


class ResamplerTests(unittest.TestCase):
    def test_different_feature_query_output_dimensions_and_full_gradient(self):
        model=QueryResampler(24,32,10,width=16,layers=2,heads=4,max_positions=10)
        features=torch.randn(1,30,24,requires_grad=True)
        positions=torch.arange(10).repeat_interleave(3).unsqueeze(0)
        output=model(features,positions,torch.zeros_like(positions))
        self.assertEqual(output.shape,(1,10,32))
        output.square().mean().backward()
        self.assertTrue(torch.all(features.grad.abs().sum(-1)>0))
        self.assertIsNotNone(model.queries.grad)
        self.assertFalse(torch.equal(model.queries[0],model.queries[1]))
        self.assertTrue(torch.isfinite(output).all())

    def test_observed_short_prefix_only_and_no_frame_readout_fallback(self):
        model=object.__new__(QFormerMemoryModel)
        # Exercise the real visual-feature API adapter without downloading weights.
        class Backbone(nn.Module):
            def __init__(self):super().__init__();self.p=nn.Parameter(torch.ones(1));self.calls=[]
            def get_base_model(self):return self
            def get_image_features(self,**kw):
                self.calls.append(kw)
                return SimpleNamespace(pooler_output=tuple(torch.ones(3,24)*i for i in range(len(kw['image_grid_thw']))))
        model.model=Backbone()
        model.qformer_config={'input_dim':24}
        model.compressors=nn.ModuleDict({'short':QueryResampler(24,32,10,16,1,4,10)})
        class Images:
            def __call__(self,images,**kwargs):
                self.images=images
                return {'pixel_values':torch.ones(len(images)*3,4),
                        'image_grid_thw':torch.ones(len(images),3,dtype=torch.long)}
        images=Images();model.processor=SimpleNamespace(image_processor=images)
        result=model.encode_short([1,2,3])
        self.assertEqual(result.shape,(1,3,32))
        self.assertEqual(images.images,[1,2,3])
        self.assertEqual(len(model.model.calls),1)
        result.square().mean().backward()
        self.assertIsNone(model.model.p.grad)
        self.assertIsNotNone(model.compressors['short'].queries.grad)
        with self.assertRaisesRegex(RuntimeError,'causal encode_short'):model.encode_frame(1)

    def test_long_only_backpropagates_current_window_and_compressor_roundtrip(self):
        model=object.__new__(QFormerMemoryModel)
        model.compressors=nn.ModuleDict({'long':QueryResampler(32,32,60,16,1,4,120)})
        previous=torch.randn(1,60,32,requires_grad=True)
        current=torch.randn(1,60,32,requires_grad=True)
        output=model.update_long(previous,current)
        self.assertEqual(output.shape,(1,60,32))
        output.square().mean().backward()
        self.assertIsNone(previous.grad)
        self.assertIsNotNone(current.grad)
        buffer=io.BytesIO();torch.save(model.compressors.state_dict(),buffer);buffer.seek(0)
        loaded=nn.ModuleDict({'long':QueryResampler(32,32,60,16,1,4,120)})
        loaded.load_state_dict(torch.load(buffer,weights_only=True))
        positions=torch.arange(120).unsqueeze(0);kinds=torch.ones_like(positions);kinds[:,:60]=0
        self.assertTrue(torch.equal(output,loaded['long'](torch.cat([previous.detach(),current],1),positions,kinds)))


def block():
    return [{'block_window':[0,60],
             'local_sub_summaries':[{'local_summary_target':'target'} for _ in range(6)],
             'block_summary_target':'block','global_summary_target':'global'}]


class ScheduleTests(unittest.TestCase):
    def test_no_video_split_or_padding_duplication_and_exact_8_targets(self):
        videos={str(i):block() for i in range(5)}
        batches=schedules(videos,4,42,0)
        self.assertEqual(sum(map(len,batches)),40)
        self.assertEqual(len(tasks_for_video('v',block())),8)
        owners={}
        for rank,batch in enumerate(batches):
            for task in batch:
                old=owners.setdefault(task['video_id'],rank);self.assertEqual(old,rank)
        self.assertEqual(batches,schedules(videos,4,42,0))

    def test_interval_encoder_groups_only_observed_10s_and_partial_prefix(self):
        calls=[]
        def encode(images):calls.append(images);return torch.ones(1,len(images),4)
        memory=SimpleNamespace(encode_short=encode)
        result=encode_interval(memory,SimpleNamespace(frame_ending_at=lambda t:t),60,85)
        self.assertEqual(calls,[list(range(60,70)),list(range(70,80)),list(range(80,85))])
        self.assertEqual(result.shape,(1,25,4))

    def test_padding_has_zero_gradient_and_never_reads_a_frame(self):
        memory=SimpleNamespace(model=nn.Linear(1,1))
        objective=Objective(memory)
        loss,state=objective(None)
        loss.backward()
        self.assertIsNone(state)
        self.assertEqual(float(loss.detach()),0)
        self.assertEqual(float(memory.model.weight.grad.abs().sum()),0)


if __name__=='__main__':unittest.main()
