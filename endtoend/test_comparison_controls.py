import copy
import json
from pathlib import Path
import tempfile
import unittest
import torch
from .comparison_controls import preallocate_adamw_state, verify_initial_state, state_fingerprint


class ComparisonTests(unittest.TestCase):
    def test_zero_state_matches_lazy_adam_without_extra_update(self):
        for amsgrad in [False, True]:
            a=torch.nn.Linear(3,2); b=copy.deepcopy(a)
            before=state_fingerprint(a)
            oa=torch.optim.AdamW(a.parameters(),lr=.001,amsgrad=amsgrad)
            ob=torch.optim.AdamW(b.parameters(),lr=.001,amsgrad=amsgrad)
            preallocate_adamw_state(oa)
            self.assertEqual(state_fingerprint(a),before)
            self.assertTrue(all(s['step'].item()==0 for s in oa.state.values()))
            for step in range(3):
                x=torch.arange(6,dtype=torch.float32).reshape(2,3)
                for model,opt in [(a,oa),(b,ob)]:
                    model(x).square().mean().backward()
                    if step==1:model.bias.grad=None
                    opt.step();opt.zero_grad(set_to_none=True)
                for pa,pb in zip(a.parameters(),b.parameters()):torch.testing.assert_close(pa,pb,rtol=0,atol=0)
            for pa,pb in zip(a.parameters(),b.parameters()):
                for k in oa.state[pa]:torch.testing.assert_close(oa.state[pa][k],ob.state[pb][k],rtol=0,atol=0)

    def test_initialization_rejects_different_weights_or_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=torch.nn.Linear(3,2);b=copy.deepcopy(a)
            oa=torch.optim.AdamW(a.parameters());ob=torch.optim.AdamW(b.parameters())
            shared=root/'common.json'
            verify_initial_state(a,oa,root/'a.json',shared)
            verify_initial_state(b,ob,root/'b.json',shared)
            with torch.no_grad():b.bias.add_(1)
            with self.assertRaises(ValueError):verify_initial_state(b,ob,root/'bad.json',shared)
            preallocate_adamw_state(oa)
            with self.assertRaises(ValueError):verify_initial_state(a,oa,root/'badopt.json',shared)


if __name__=='__main__':unittest.main()
