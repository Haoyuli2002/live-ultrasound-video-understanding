"""Explicit FLA/Triton adapters for Qwen3.5; no silent fallback when requested."""
import torch


def causal_conv(x, weight, bias=None, activation=None, seq_idx=None):
    if seq_idx is not None:
        raise ValueError('Packed sequence IDs are not supported by this adapter')
    from fla.modules.causal_conv1d import causal_conv1d
    y, _ = causal_conv1d(x.transpose(1, 2).contiguous(), weight=weight, bias=bias,
                        activation=activation, backend='triton')
    return y.transpose(1, 2)


def causal_update(x, state, weight, bias=None, activation=None):
    from fla.modules.causal_conv1d import causal_conv1d_update
    if x.ndim != 3 or x.shape[-1] != 1:
        raise ValueError('Cached decode expects [B,D,1]')
    y, _ = causal_conv1d_update(x.transpose(1, 2).contiguous(), state,
                               weight=weight, bias=bias, activation=activation)
    return y.transpose(1, 2)


def enable_fast_attention(model):
    if not torch.cuda.is_available():
        raise RuntimeError('FLA acceleration requires a CUDA compute node')
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    names = []
    for name, module in model.backbone.named_modules():
        if hasattr(module, 'chunk_gated_delta_rule') and hasattr(module, 'conv1d'):
            module.chunk_gated_delta_rule = chunk_gated_delta_rule
            module.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule
            module.causal_conv1d_fn = causal_conv
            module.causal_conv1d_update = causal_update
            names.append(name)
    if not names:
        raise RuntimeError('No Qwen3.5 GatedDeltaNet modules were accelerated')
    return dict(linear_attention='fla.ops.gated_delta_rule', causal_convolution='fla Triton',
                accelerated_layers=len(names), modules=names)
