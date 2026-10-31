"""Portable CUDA model settings shared by all three ultrasound stages."""
from __future__ import annotations

import importlib.util
import json
import os

import torch


def model_kwargs(dtype: torch.dtype, attention: str | None = None) -> dict:
    # SDPA lets PyTorch dispatch a kernel supported by the allocated GPU. It
    # avoids requiring FlashAttention 3, which is not an Ampere/A100 backend.
    attention = attention or os.environ.get('ATTN_IMPLEMENTATION', 'sdpa')
    if attention == 'auto':
        attention = 'sdpa'
    if attention not in {'sdpa', 'eager', 'flash_attention_2'}:
        raise ValueError('ATTN_IMPLEMENTATION must be auto, sdpa, eager or flash_attention_2')
    if attention == 'flash_attention_2':
        if not torch.cuda.is_available() or importlib.util.find_spec('flash_attn') is None:
            raise RuntimeError('flash_attention_2 needs CUDA and flash-attn; use sdpa for the portable default')
        if torch.cuda.get_device_capability()[0] < 8 or dtype == torch.float32:
            raise ValueError('flash_attention_2 requires Ampere or newer and bf16/fp16 weights')
    if torch.cuda.is_available() and dtype == torch.bfloat16:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('Allocated GPU does not support BF16; request an A100/H100 or change dtype')
    return {'dtype': dtype, 'attn_implementation': attention}


def cuda_info() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError('This training entry requires an allocated CUDA GPU')
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    return {'gpu': properties.name, 'capability': list(torch.cuda.get_device_capability(device)),
            'memory_gib': round(properties.total_memory/1024**3, 2),
            'cuda_device': device, 'visible_devices': torch.cuda.device_count(),
            'bf16_supported': torch.cuda.is_bf16_supported(), 'torch':torch.__version__,
            'torch_cuda':torch.version.cuda,
            'attention':os.environ.get('ATTN_IMPLEMENTATION','sdpa')}


if __name__ == '__main__':
    print(json.dumps(cuda_info(), indent=2))
