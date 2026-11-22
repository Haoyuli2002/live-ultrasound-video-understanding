import math
import unittest
import torch
from torch.nn import functional as F
from .timing_loss import readiness_loss_terms
from .test_timing_loss import rows
from .test_endtoend import tiny_model
from .model import forward_batch
from .optimization import backward_objectives


class WeightedBCETests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(81)

    def test_positive_loss_and_gradient_are_doubled(self):
        logits=torch.zeros(2,requires_grad=True)
        terms=readiness_loss_terms(logits,rows([0,1]),'contrastive_bce_gold',lambda_reg=0,sample_pairs=False)
        torch.testing.assert_close(terms['weighted_bce'],torch.tensor(1.5*math.log(2)))
        gradient=torch.autograd.grad(terms['weighted_bce'],logits)[0]
        torch.testing.assert_close(gradient,torch.tensor([.25,-.5]))
        torch.testing.assert_close(terms['loss'],terms['timing']['loss']+terms['weighted_bce'])

    def test_single_label_tasks_receive_supervision(self):
        for labels in [[1,1],[0,0]]:
            logits=torch.zeros(2,requires_grad=True)
            examples=rows(labels[0:1],'a')+rows(labels[1:],'b')
            terms=readiness_loss_terms(logits,examples,'contrastive_bce_gold',sample_pairs=False)
            self.assertEqual(terms['timing']['paired_tasks'],0)
            self.assertGreater(float(terms['loss']),0)
            grad=torch.autograd.grad(terms['loss'],logits)[0]
            self.assertTrue((grad<0).all() if labels[0] else (grad>0).all())

    def test_old_objectives_unchanged(self):
        logits=torch.tensor([-.3,.7],requires_grad=True);e=rows([0,1])
        for mode in ['bce','contrastive_gold']:
            result=readiness_loss_terms(logits,e,mode,sample_pairs=False)
            expected=result['bce'] if mode=='bce' else result['timing']['loss']
            torch.testing.assert_close(result['loss'],expected)
        hybrid=readiness_loss_terms(logits,e,'contrastive_bce_gold',bce_positive_weight=1,bce_weight=.4,sample_pairs=False)
        torch.testing.assert_close(hybrid['loss'],hybrid['timing']['loss']+.4*hybrid['bce'])
        with self.assertRaises(ValueError):readiness_loss_terms(logits,e,bce_positive_weight=float('nan'))

    def test_hybrid_gradient_isolation_and_ce_weight(self):
        model=tiny_model().eval();model.parallel_endpoints=True
        model.readiness_objective='contrastive_bce_gold';model.timing_lambda_reg=.1
        model.readiness_bce_positive_weight=2.;model.readiness_bce_weight=1.
        with torch.no_grad():model.core.readiness_head.net[-1].weight.normal_(std=.1)
        e=rows([0,1]);e[0]['current_time']=1.;e[1]['current_time']=3.
        x=torch.randn(1,4,2,32);features=[x[:,:2],x]
        out=forward_batch(model,e,features)
        torch.testing.assert_close(out['loss'],out['answer_loss']+out['timing']['loss']+out['readiness_weighted_bce'])
        params=[model.core.long_query_tokens,model.core.vision_proj.weight,model.core.output_proj.weight]
        expected=torch.autograd.grad(out['answer_loss'],params,retain_graph=True)
        backward_objectives(model,out,'readiness_isolated')
        for parameter,gradient in zip(params,expected):torch.testing.assert_close(parameter.grad,gradient)
        model.zero_grad(set_to_none=True)
        out=forward_batch(model,e,features);out['answer_loss']=out['answer_loss'].detach()*0
        backward_objectives(model,out,'readiness_isolated')
        self.assertTrue(all(p.grad is None for p in model.reasoning_parameters()))
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        self.assertGreater(float(model.core.rdy_token.grad.norm()),0)
        self.assertGreater(float(model.core.readiness_head.net[-1].weight.grad.norm()),0)


if __name__=='__main__':unittest.main()
