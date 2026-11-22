import unittest
import torch
from torch.nn import functional as F
from .timing_loss import contrastive_timing_loss
from .test_endtoend import tiny_model,example
from .model import forward_batch
from .optimization import backward_objectives,task_batches
from lavis.models.streamready.readiness import readiness_loss


def rows(labels,task='x',video='v'):
    return [dict(example(y),task_id=task,video_id=video,sample_id=f'{task}-{i}',current_time=float(i)) for i,y in enumerate(labels)]


class TimingTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(37)

    def test_matches_official_sampled_value_and_gradient(self):
        s=torch.tensor([[-2.,-.2,.4,1.3],[.1,-1.,.2,2.]],requires_grad=True)
        labels=torch.tensor([[False,False,False,True],[False,False,False,True]])
        examples=rows([0,0,0,1],'a')+rows([0,0,0,1],'b')
        torch.manual_seed(11);expected=readiness_loss(s,labels,~labels,lambda_reg=.1)
        ref=torch.autograd.grad(expected,s,retain_graph=True)[0]
        torch.manual_seed(11);actual=contrastive_timing_loss(s.flatten(),examples)
        torch.testing.assert_close(actual['loss'],expected)
        torch.testing.assert_close(torch.autograd.grad(actual['loss'],s)[0],ref)
        self.assertEqual(actual['sampled_pairs'],2)

    def test_eval_expectation_and_only_same_task_pairs(self):
        s=torch.tensor([-2.,0.,1.,8.,9.],requires_grad=True)
        e=rows([0,0,1],'a')+rows([1,1],'b')
        result=contrastive_timing_loss(s,e,sample_pairs=False)
        expected=-F.logsigmoid(torch.tensor([3.,1.])).mean()
        torch.testing.assert_close(result['contrastive'],expected)
        self.assertEqual(result['pairs'],2);self.assertEqual(result['paired_tasks'],1)
        self.assertEqual(result['unpaired_tasks'],1)
        g=torch.autograd.grad(result['contrastive'],s)[0]
        self.assertLess(float(g[2]),0);self.assertGreater(float(g[0]),0)
        self.assertTrue((g[3:]==0).all())

    def test_temporal_order_duplicates_and_no_pairs(self):
        e=rows([0,0,0]);s=torch.tensor([0.,1.,2.],requires_grad=True)
        result=contrastive_timing_loss(s,e,sample_pairs=False)
        expected=(s.sigmoid()[1:]-s.sigmoid()[:-1]).abs().mean()
        torch.testing.assert_close(result['temporal'],expected)
        self.assertEqual(result['paired_tasks'],0)
        order=[2,0,1]
        other=contrastive_timing_loss(s[order],[e[i] for i in order],sample_pairs=False)
        torch.testing.assert_close(other['loss'],result['loss'])
        dup=contrastive_timing_loss(s[:1].expand(4),[e[0]]*4)
        self.assertEqual(float(dup['loss']),0.)
        dup['loss'].backward();self.assertTrue(torch.isfinite(s.grad).all())

    def test_contrastive_forward_and_isolated_gradients(self):
        model=tiny_model().eval();model.parallel_endpoints=True
        model.readiness_objective='contrastive_gold';model.timing_lambda_reg=.1
        with torch.no_grad():model.core.readiness_head.net[-1].weight.normal_(std=.1)
        e=rows([0,1]);e[0]['current_time']=1.;e[1]['current_time']=3.
        x=torch.randn(1,4,2,32);features=[x[:,:2],x]
        out=forward_batch(model,e,features)
        torch.testing.assert_close(out['loss'],out['answer_loss']+out['timing']['contrastive']+.1*out['timing']['temporal'])
        params=[model.core.long_query_tokens,model.core.vision_proj.weight,model.core.output_proj.weight]
        ref=torch.autograd.grad(out['answer_loss'],params,retain_graph=True)
        backward_objectives(model,out,'readiness_isolated')
        for p,g in zip(params,ref):torch.testing.assert_close(p.grad,g)
        self.assertGreater(float(model.core.readiness_head.net[-1].weight.grad.norm()),0.)
        model.zero_grad(set_to_none=True)
        out=forward_batch(model,e,features);out['answer_loss']=out['answer_loss'].detach()*0
        backward_objectives(model,out,'readiness_isolated')
        self.assertTrue(all(p.grad is None for p in model.reasoning_parameters()))
        self.assertGreater(float(model.core.rdy_token.grad.norm()),0.)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))


if __name__=='__main__':unittest.main()
