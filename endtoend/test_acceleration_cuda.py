"""Run on the compute node before enabling the FLA adapters in an experiment."""
import unittest
import torch
from torch.nn import functional as F
from .acceleration import causal_conv, causal_update


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class AccelerationTests(unittest.TestCase):
    def test_convolution_forward_backward_and_decode_state(self):
        torch.manual_seed(7)
        x = torch.randn(2, 32, 17, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(32, 4, device='cuda', dtype=torch.bfloat16)
        b = torch.randn(32, device='cuda', dtype=torch.bfloat16)
        reference = F.silu(F.conv1d(x, w[:, None], b, padding=3, groups=32)[:, :, :17])
        accelerated = causal_conv(x, w, b, 'silu')
        torch.testing.assert_close(accelerated, reference, atol=.06, rtol=.025)
        g1 = torch.autograd.grad(reference.float().square().mean(), x)[0]
        g2 = torch.autograd.grad(accelerated.float().square().mean(), x)[0]
        torch.testing.assert_close(g2, g1, atol=.002, rtol=.05)
        state = x.detach()[:, :, -4:].contiguous().clone()
        expected_state = state.roll(-1, -1)
        token = torch.randn(2, 32, 1, device='cuda', dtype=torch.bfloat16)
        expected_state[:, :, -1:] = token
        y = causal_update(token, state, w, b, 'silu')
        ref = F.silu((expected_state.float() * w.float()[None]).sum(-1) + b.float()).to(x.dtype)[..., None]
        torch.testing.assert_close(state, expected_state)
        torch.testing.assert_close(y, ref, atol=.04, rtol=.025)

    def test_linear_attention_forward_backward(self):
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
        torch.manual_seed(8)
        tensors = [torch.randn(2, 65, 2, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
        q, k, v = tensors
        g = -torch.rand(2, 65, 2, device='cuda')
        beta = torch.rand(2, 65, 2, device='cuda', dtype=torch.bfloat16)
        args = dict(query=q, key=k, value=v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
        ref, _ = torch_chunk_gated_delta_rule(**args)
        fast, _ = chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
        torch.testing.assert_close(fast, ref, atol=.02, rtol=.05)
        expected = torch.autograd.grad(ref.float().square().mean(), tensors)
        actual = torch.autograd.grad(fast.float().square().mean(), tensors)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=.06)


if __name__ == '__main__':
    unittest.main()
