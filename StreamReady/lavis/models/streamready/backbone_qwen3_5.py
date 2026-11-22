"""Frozen Qwen3.5 (dense or MoE) host for StreamReady.

The input is the repository's CLIP-normalised [B,C,T,H,W] video tensor.
Undo only that normalisation, then let the checkpoint's image processor pack
and normalise each frame independently. Continuous StreamReady prefixes use
the official conditional-generation model, including its hybrid attention cache.
"""
from typing import Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageTextToText, AutoTokenizer


QWEN35_MODEL_TYPES = {"qwen3_5", "qwen3_5_moe"}
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class Qwen35Backbone(nn.Module):
    def __init__(self, model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa"):
        super().__init__()
        self.model_path = str(model_path)
        config = AutoConfig.from_pretrained(model_path)
        if config.model_type not in QWEN35_MODEL_TYPES:
            raise ValueError(f"Expected a multimodal Qwen3.5 checkpoint, got {config.model_type!r}")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, config=config, dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        self.config = self.model.config
        self.image_processor = AutoImageProcessor.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.requires_grad_(False)
        self.model.eval()

    @property
    def vision_width(self):
        return int(self.config.vision_config.out_hidden_size)

    @property
    def llm_width(self):
        return int(self.config.text_config.hidden_size)

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    def tokens_per_frame(self, image_size):
        factor = self.config.vision_config.patch_size * self.config.vision_config.spatial_merge_size
        if image_size % factor:
            raise ValueError(f"image_size must be divisible by patch_size*merge_size={factor}")
        return (image_size // factor) ** 2

    @torch.no_grad()
    def encode_frames(self, pixels):
        if pixels.ndim != 5 or pixels.shape[1] != 3:
            raise ValueError("Expected CLIP-normalised RGB pixels [B,3,T,H,W]")
        b, _, t, h, w = pixels.shape
        if not b or not t:
            raise ValueError("Video batch and time dimensions must be nonempty")
        self.tokens_per_frame(h)
        self.tokens_per_frame(w)
        frames = pixels.permute(0, 2, 1, 3, 4).reshape(b * t, 3, h, w).float()
        mean = frames.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std = frames.new_tensor(CLIP_STD).view(1, 3, 1, 1)
        frames = (frames * std + mean).clamp(0, 1)
        # Treat frames as independent images: no temporal patch can see a future
        # frame. Crop/resize already happened in the LAVIS video processor.
        processed = self.image_processor(
            images=list(frames.unbind(0)), do_resize=False, do_rescale=False,
            do_normalize=True, return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        processed = {k: v.to(device) for k, v in processed.items() if torch.is_tensor(v)}
        out = self.model.get_image_features(
            pixel_values=processed["pixel_values"],
            image_grid_thw=processed["image_grid_thw"], return_dict=True,
        )
        features = out.pooler_output
        if not isinstance(features, (tuple, list)) or len(features) != b * t:
            raise RuntimeError("Expected one merged visual-token tensor per frame")
        factor = self.config.vision_config.patch_size * self.config.vision_config.spatial_merge_size
        n = (h // factor) * (w // factor)
        if any(f.shape != (n, self.vision_width) for f in features):
            raise RuntimeError("Unexpected Qwen3.5 image feature shape")
        return torch.stack(features).reshape(b, t, n, self.vision_width)

    def embed_tokens(self, input_ids):
        return self.model.get_input_embeddings()(input_ids)

    def lm_forward(self, inputs_embeds, attention_mask, labels: Optional[torch.Tensor] = None):
        # Frozen weights must still propagate CE gradients into the prefix.
        # Use text-style positions for continuous tokens and reset any previous
        # multimodal/generation position state before an independent call.
        self.model.model.rope_deltas = None
        positions = attention_mask.long().cumsum(-1) - 1
        positions.masked_fill_(attention_mask == 0, 0)
        out = self.model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            position_ids=positions, labels=labels, use_cache=False, return_dict=True,
        )
        return out.logits.float(), out.loss

    @torch.no_grad()
    def lm_generate(self, inputs_embeds, attention_mask, **kwargs):
        self.model.model.rope_deltas = None
        kwargs.setdefault("max_new_tokens", 32)
        kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        kwargs.setdefault("do_sample", False)
        # With inputs_embeds alone, HF returns generated tokens only. Its native
        # generation handles dense/MoE hybrid caches, beam search and EOS.
        return self.model.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            use_cache=True, **kwargs,
        )

    def drop_language_model(self):
        """Release decoder weights for the offline visual-feature extractor."""
        del self.model.model.language_model
        del self.model.lm_head
