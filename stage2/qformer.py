"""Small query resamplers over Qwen visual patches; not pretrained BLIP-2 weights.

All frames supplied to encode_short must precede the current decision endpoint.
Queries can interact bidirectionally inside that observed window. They cannot
read future windows, text targets or previously generated summaries.
"""
from __future__ import annotations

import json
from pathlib import Path
import torch
from torch import nn

from .model import MemoryModel


class QueryLayer(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm_self = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm_cross = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4*width), nn.GELU(), nn.Linear(4*width, width))

    def forward(self, queries, features):
        q = self.norm_self(queries)
        queries = queries + self.self_attention(q, q, q, need_weights=False)[0]
        queries = queries + self.cross_attention(self.norm_cross(queries), features, features,
                                                  need_weights=False)[0]
        return queries + self.ff(self.norm_ff(queries))


class QueryResampler(nn.Module):
    def __init__(self, input_dim, output_dim, queries, width=512, layers=2, heads=8,
                 max_positions=120):
        super().__init__()
        if min(input_dim, output_dim, queries, width, layers, heads, max_positions) < 1 or width % heads:
            raise ValueError("Positive dimensions and width divisible by heads are required")
        self.queries = nn.Parameter(torch.randn(queries, width)*0.02)
        self.input_projection = nn.Linear(input_dim, width)
        self.position = nn.Embedding(max_positions, width)
        self.source_type = nn.Embedding(2, width)
        self.layers = nn.ModuleList([QueryLayer(width, heads) for _ in range(layers)])
        self.output_projection = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, output_dim))

    def forward(self, features, positions, source_types, count=None):
        count = len(self.queries) if count is None else count
        if not 1 <= count <= len(self.queries) or features.ndim != 3 or features.shape[1] == 0:
            raise ValueError("Invalid resampler input or query count")
        features = self.input_projection(features)
        features = features + self.position(positions) + self.source_type(source_types)
        queries = self.queries[:count].unsqueeze(0).expand(features.shape[0], -1, -1)
        for layer in self.layers:
            queries = layer(queries, features)
        return self.output_projection(queries)


class QFormerMemoryModel(MemoryModel):
    def __init__(self, *args, query_width=512, query_layers=2, query_heads=8, **kwargs):
        checkpoint = kwargs.get("adapter_path")
        config_file = Path(checkpoint)/"qformer_config.json" if checkpoint else None
        if config_file and config_file.exists():
            config = json.loads(config_file.read_text())
            query_width, query_layers, query_heads = (config[k] for k in ("width", "layers", "heads"))
        super().__init__(*args, **kwargs)
        base = self.model.get_base_model()
        if not hasattr(base, "get_image_features"):
            raise TypeError("Qwen backbone needs get_image_features; no single-vector fallback")
        hidden = base.config.text_config.hidden_size
        self.qformer_config = dict(width=query_width, layers=query_layers, heads=query_heads,
                                   input_dim=hidden, output_dim=hidden, short_queries=10,
                                   long_queries=60, architecture="qformer_style_v1")
        self.compressors = nn.ModuleDict({
            "short": QueryResampler(hidden, hidden, 10, query_width, query_layers, query_heads, 10),
            "long": QueryResampler(hidden, hidden, 60, query_width, query_layers, query_heads, 120)})
        # This feature path bypasses the language transformer. Vision is frozen;
        # decoder LoRA and both resamplers remain trainable in the paired recipe.
        visual = getattr(base.model, "visual", None)
        if visual is None:
            raise TypeError("Qwen backbone has no model.visual module")
        visual.requires_grad_(False)
        self.visual = visual
        if checkpoint:
            if not config_file.is_file() or not (Path(checkpoint)/"qformer.pt").is_file():
                raise FileNotFoundError("Q-Former checkpoint is missing compressor weights/config")
            self.compressors.load_state_dict(torch.load(Path(checkpoint)/"qformer.pt", map_location="cpu", weights_only=True))

    def to(self, device, dtype=None):
        self.model.to(device)
        dtype = dtype or next(self.model.parameters()).dtype
        self.compressors.to(device=device, dtype=dtype)
        return self

    def train(self, mode=True):
        self.model.train(mode)
        self.visual.eval()
        self.compressors.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def encode_short(self, images):
        if not 1 <= len(images) <= 10:
            raise ValueError("Short encoding needs 1..10 observed frames")
        pixels = self.processor.image_processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k,v in pixels.items()}
        with torch.no_grad():
            output = self.model.get_base_model().get_image_features(
                pixel_values=inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"],
                return_dict=True)
        frames = output.pooler_output
        if not isinstance(frames, (tuple, list)) or len(frames) != len(images):
            raise TypeError("Expected one merged visual-patch tensor per input frame")
        if any(f.ndim != 2 or f.shape[-1] != self.qformer_config["input_dim"] for f in frames):
            raise ValueError("Unexpected Qwen visual-feature shape")
        dtype = next(self.compressors.parameters()).dtype
        features = torch.cat(frames, dim=0).unsqueeze(0).to(dtype)
        positions = torch.cat([torch.full((len(f),),i,device=self.device,dtype=torch.long)
                               for i,f in enumerate(frames)]).unsqueeze(0)
        return self.compressors["short"](features, positions, torch.zeros_like(positions), len(images))

    def encode_frame(self, image):
        raise RuntimeError("Q-Former needs causal encode_short windows, not independent encode_frame")

    def update_long(self, previous, short):
        if short.shape[1] != 60:
            raise ValueError("Long update requires the current minute's 60 Short tokens")
        features = short if previous is None else torch.cat([previous.detach(),short],dim=1)
        positions = torch.arange(features.shape[1],device=features.device).unsqueeze(0)
        kinds = torch.ones_like(positions)
        if previous is not None:
            kinds[:, :previous.shape[1]] = 0
        return self.compressors["long"](features, positions, kinds)

    def save(self, path):
        super().save(path)
        torch.save(self.compressors.state_dict(), Path(path)/"qformer.pt")
        (Path(path)/"qformer_config.json").write_text(json.dumps(self.qformer_config,indent=2))


def load_memory_model(model_name, *, architecture=None, adapter_path=None, **kwargs):
    detected = bool(adapter_path and (Path(adapter_path)/"qformer_config.json").is_file())
    if architecture not in (None, "ordinary", "qformer"):
        raise ValueError("Unknown memory architecture")
    if adapter_path and architecture is not None and (architecture == "qformer") != detected:
        raise ValueError("Checkpoint architecture differs from requested memory architecture")
    cls = QFormerMemoryModel if detected or architecture == "qformer" else MemoryModel
    return cls(model_name, adapter_path=adapter_path, **kwargs)


def short_window(memory, images):
    encoder = getattr(memory, "encode_short", None)
    return encoder(images) if encoder else torch.stack([memory.encode_frame(i) for i in images],dim=1)
