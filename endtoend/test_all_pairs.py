import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch.nn import functional as F
from .timing_loss import contrastive_timing_loss, sample_timing_pairs
from .test_timing_loss import rows
from .test_endtoend import tiny_model
from .model import forward_batch
from .optimization import backward_objectives


class AllPairTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(81)

    def test_every_pair_gradient_and_question_balancing(self):
        # Different video IDs may reuse a task ID; never create cross-video pairs.
        e = rows([0,0,1,1], 'same', 'a') + rows([0,1], 'same', 'b')
        s = torch.tensor([-.7,.8,1.2,-.1,2.,3.], requires_grad=True)
        result = contrastive_timing_loss(s,e,lambda_reg=0,sample_pairs=False)
        group_a = torch.stack([-F.logsigmoid(s[p]-s[n]) for p in [2,3] for n in [0,1]]).mean()
        group_b = -F.logsigmoid(s[5]-s[4])
        expected = (group_a + group_b)/2
        torch.testing.assert_close(result['loss'],expected)
        actual_grad = torch.autograd.grad(result['loss'],s,retain_graph=True)[0]
        torch.testing.assert_close(actual_grad,torch.autograd.grad(expected,s)[0])
        self.assertTrue((actual_grad[[0,1,4]]>0).all())
        self.assertTrue((actual_grad[[2,3,5]]<0).all())
        self.assertEqual(result['used_pairs'],5)
        self.assertEqual(result['sampled_pairs'],0)

    def test_mode_backward_compatibility_and_eval(self):
        model = SimpleNamespace(training=True)
        self.assertTrue(sample_timing_pairs(model))
        model.timing_pair_mode = 'all_pairs'
        self.assertFalse(sample_timing_pairs(model))
        model.training = False; model.timing_pair_mode = 'sampled'
        self.assertFalse(sample_timing_pairs(model))
        model.timing_pair_mode = 'typo'
        with self.assertRaises(ValueError):sample_timing_pairs(model)

    def test_training_forward_uses_all_pairs_and_keeps_gradient_isolation(self):
        model=tiny_model().train();model.parallel_endpoints=True
        model.readiness_objective='contrastive_bce_gold';model.timing_pair_mode='all_pairs'
        model.timing_lambda_reg=.1;model.readiness_bce_positive_weight=1.
        with torch.no_grad():model.core.readiness_head.net[-1].weight.normal_(std=.1)
        e=rows([0,0,1])
        for i,r in enumerate(e):r['current_time']=float(i+1)
        x=torch.randn(1,4,2,32);features=[x[:,:2],x[:,:3],x]
        with patch('lavis.models.streamready.readiness.readiness_loss',side_effect=AssertionError('Sampler must not run')):
            out=forward_batch(model,e,features)
        self.assertEqual(out['timing']['used_pairs'],2)
        torch.testing.assert_close(out['loss'],out['answer_loss']+out['timing']['loss']+out['readiness_bce'])
        params=[model.core.long_query_tokens,model.core.vision_proj.weight,model.core.output_proj.weight]
        expected=torch.autograd.grad(out['answer_loss'],params,retain_graph=True)
        backward_objectives(model,out,'readiness_isolated')
        for parameter,gradient in zip(params,expected):torch.testing.assert_close(parameter.grad,gradient)
        self.assertGreater(float(model.core.rdy_token.grad.norm()),0.)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))


if __name__=='__main__':unittest.main()
