import unittest
import torch
from .ablate_long_memory import intervene, replace_slots
from .parallel import forward_endpoints
from .test_endtoend import tiny_model, example


class AblationTests(unittest.TestCase):
    def test_slot_replacement_preserves_masked_entries(self):
        memory=torch.arange(12.).reshape(1,4,3)
        mask=torch.tensor([[1,0,1,0]],dtype=torch.bool)
        donor=torch.full((3,3),-7.)
        actual=replace_slots(memory,mask,donor)
        torch.testing.assert_close(actual[0,~mask[0]],memory[0,~mask[0]])
        self.assertTrue((actual[0,mask[0]]==-7).all())
        self.assertFalse(torch.equal(memory,actual))

    @torch.no_grad()
    def test_only_long_history_changes_and_patch_restores(self):
        torch.manual_seed(41);torch.set_num_threads(2)
        model=tiny_model().eval();core=model.core
        row=dict(example(),current_time=4.)
        features=[torch.randn(1,5,2,32)]
        captured={}
        with intervene(core,'real',True,captured):
            base=forward_endpoints(core,[row],features)[0]
        original=core._run_branch
        for condition in ('zero_long','other_video_long'):
            trace={}
            with intervene(core,condition,True,trace,captured['short'],captured['long_slots']+3):
                changed=forward_endpoints(core,[row],features)[0]
            self.assertEqual(core._run_branch,original)
            torch.testing.assert_close(trace['long_slots'],captured['long_slots'])
            self.assertFalse(torch.allclose(changed['prefix_embeds'],base['prefix_embeds']))
        # Before history exists, interventions must be exact no-ops.
        early=dict(row,current_time=1.)
        baseline=forward_endpoints(core,[early],[features[0][:,:2]])[0]
        for condition in ('zero_long','other_video_long'):
            with intervene(core,condition,False,{}):
                got=forward_endpoints(core,[early],[features[0][:,:2]])[0]
            torch.testing.assert_close(got['prefix_embeds'],baseline['prefix_embeds'],atol=0,rtol=0)


if __name__=='__main__':unittest.main()
