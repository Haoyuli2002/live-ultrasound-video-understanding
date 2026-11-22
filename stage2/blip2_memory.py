"""LAVIS image-query path with Stage2's Short/Long interface."""
import json
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open
from transformers import BertConfig

from .lavis_query_layers import BertLayer
from .model import MemoryModel
from .qformer import QFormerMemoryModel

ARCHITECTURE = "lavis_bert_query_1_60"
REVISION = "05cb670f47296522e24e7727e366b4a997d28582"


class ImageQueries(nn.Module):
    def __init__(self, encoder_width, count, bert_dir=None):
        super().__init__()
        config = BertConfig() if bert_dir is None else BertConfig.from_json_file(
            str(Path(bert_dir) / "config.json"))
        config.encoder_width = encoder_width
        config.add_cross_attention = True
        config.cross_attention_freq = 2
        self.input_norm = nn.LayerNorm(encoder_width)  # LAVIS ln_vision
        self.query_tokens = nn.Parameter(torch.empty(1, count, config.hidden_size))
        nn.init.normal_(self.query_tokens, std=config.initializer_range)
        self.embedding_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layers = nn.ModuleList()
        for index in range(config.num_hidden_layers):
            layer = BertLayer(config, index)
            # In an image-only query call LAVIS uses the query FFN exclusively.
            del layer.intermediate, layer.output
            self.layers.append(layer)
        self.output_projection = nn.Linear(config.hidden_size, encoder_width)
        if bert_dir is not None:
            self.initialize_bert(Path(bert_dir), config.initializer_range)

    @torch.no_grad()
    def initialize_bert(self, directory, std):
        with safe_open(str(directory / "model.safetensors"), framework="pt") as weights:
            for name, param in self.named_parameters():
                if name.startswith(("query_tokens", "input_norm.", "output_projection.")):
                    continue
                if ".crossattention." in name:
                    if name.endswith(".bias"):
                        param.zero_()
                    elif ".LayerNorm." in name:
                        param.fill_(1)
                    else:
                        param.normal_(std=std)
                    continue
                key = name.replace("embedding_norm.", "bert.embeddings.LayerNorm.")
                key = key.replace("layers.", "bert.encoder.layer.").replace("_query.", ".")
                key = key.replace("LayerNorm.weight", "LayerNorm.gamma").replace("LayerNorm.bias", "LayerNorm.beta")
                param.copy_(weights.get_tensor(key))

    def forward(self, features):
        keys = self.input_norm(features)
        q = self.embedding_dropout(self.embedding_norm(self.query_tokens.expand(len(keys), -1, -1)))
        for layer in self.layers:
            q = layer(q, encoder_hidden_states=keys, query_length=q.size(1))[0]
        return self.output_projection(q)


class Blip2Memory(QFormerMemoryModel):
    def __init__(self, model_name, *, bert_dir=None, adapter_path=None, **kwargs):
        if adapter_path is None and bert_dir is None:
            raise ValueError("Fresh LAVIS training requires BERT-base weights")
        MemoryModel.__init__(self, model_name, adapter_path=adapter_path, **kwargs)
        base = self.model.get_base_model()
        hidden = base.config.text_config.hidden_size
        self.requires_short_history = False
        self.visual = base.model.visual
        self.visual.requires_grad_(False)
        self.compressors = nn.ModuleDict({
            name: ImageQueries(hidden, count, bert_dir if adapter_path is None else None)
            for name, count in (("short", 1), ("long", 60))
        })
        self.qformer_config = dict(architecture=ARCHITECTURE, short_queries=1, long_queries=60,
                                   width=768, layers=12, heads=12, input_dim=hidden, output_dim=hidden,
                                   cross_attention_freq=2, lavis_revision=REVISION)
        if adapter_path:
            path = Path(adapter_path)
            if json.loads((path/"qformer_config.json").read_text()) != self.qformer_config:
                raise ValueError("Incompatible LAVIS memory checkpoint")
            self.compressors.load_state_dict(torch.load(path/"qformer.pt", map_location="cpu", weights_only=True))

    def encode_visual_short(self, frames):
        # Batch independent images, then restore one chronological output per second.
        dtype = next(self.compressors.parameters()).dtype
        return self.compressors["short"](torch.stack(tuple(frames)).to(dtype)).transpose(0, 1)

    def encode_frame(self, image):
        return self.encode_short([image])[:, 0, :]

    def update_long(self, previous, short):
        if short.shape[1] != 60:
            raise ValueError("Long update requires a complete minute")
        keys = short if previous is None else torch.cat((previous.detach(), short), dim=1)
        return self.compressors["long"](keys)
