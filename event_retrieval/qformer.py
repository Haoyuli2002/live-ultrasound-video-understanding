"""Query-only InstructBLIP branches for recent frames and retrieved events."""
from copy import deepcopy

import torch
from torch import nn
from lavis.models.blip2_models.Qformer import BertConfig, BertModel


def event_frame_indices(frame_count, *, device=None):
    """Index chronological 1 FPS frames; callers retain the timestamps at these indices."""
    if frame_count < 0:
        raise ValueError("Negative event frame count")
    if frame_count <= 24:
        return torch.arange(frame_count, device=device)
    return torch.linspace(0, frame_count - 1, 24, dtype=torch.float64, device=device).round().long()


class VisualQueryBranch(nn.Module):
    def __init__(self, config, num_queries, llm_width):
        super().__init__()
        config = deepcopy(config)
        config.query_length = num_queries
        self.encoder = BertModel(config, add_pooling_layer=False)
        # In query-only mode BERT uses the query-specific FFNs and no text embeddings.
        self.encoder.embeddings.word_embeddings.requires_grad_(False)
        self.encoder.embeddings.position_embeddings.requires_grad_(False)
        for layer in self.encoder.encoder.layer:
            layer.intermediate.requires_grad_(False)
            layer.output.requires_grad_(False)
        self.queries = nn.Parameter(torch.empty(1, num_queries, config.hidden_size))
        nn.init.normal_(self.queries, std=config.initializer_range)
        self.output_projection = nn.Linear(config.hidden_size, llm_width)

    def forward(self, visual_tokens, token_mask):
        encoded = self.encoder(
            query_embeds=self.queries.expand(visual_tokens.shape[0], -1, -1),
            encoder_hidden_states=visual_tokens,
            encoder_attention_mask=token_mask.long(),
            return_dict=True,
        ).last_hidden_state
        return self.output_projection(encoded)


class EventQFormers(nn.Module):
    """Build Qwen-compatible visual prefixes from already observed frame features."""

    def __init__(self, config, vision_width, llm_width, *, raw_recent_frames=0):
        super().__init__()
        self.short = VisualQueryBranch(config, 32, llm_width)
        self.long = VisualQueryBranch(config, 128, llm_width)
        self.visual_projection = nn.Linear(vision_width, config.encoder_width)
        # Individual frame features have no temporal or event-slot coordinate.
        self.frame_positions = nn.Parameter(torch.zeros(24, config.encoder_width))
        self.event_positions = nn.Parameter(torch.zeros(5, config.encoder_width))

        self.set_raw_recent_frames(raw_recent_frames)

    def set_raw_recent_frames(self, count):
        if count not in (0, 2, 4):
            raise ValueError("Expected 0, 2 or 4 direct recent frames")
        if count and self.visual_projection.in_features != self.short.output_projection.out_features:
            raise ValueError("Direct Qwen visual tokens must already have language embedding width")
        self.recent_window = 30 if count == 2 else 24
        position_count = max(24, self.recent_window - count)
        if self.frame_positions.size(0) != position_count:
            positions = self.frame_positions.new_zeros(position_count, self.frame_positions.size(1))
            common = min(position_count, self.frame_positions.size(0))
            positions[:common].copy_(self.frame_positions.detach()[:common])
            self.frame_positions = nn.Parameter(positions)
        self.raw_recent_frames = count

    @classmethod
    def from_instructblip(cls, checkpoint, *, vision_width=2048, llm_width=2048, noise_seed=0, raw_recent_frames=0):
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)["model"]
        weights = {k.removeprefix("Qformer.bert."): v for k, v in state.items()
                   if k.startswith("Qformer.bert.")}
        queries = state["query_tokens"]
        if queries.shape != (1, 32, 768):
            raise ValueError(f"Expected InstructBLIP query_tokens [1,32,768], got {queries.shape}")
        # Match the checkpoint's BERT-base tower, including its extra vocab entry.
        config = BertConfig(vocab_size=weights["embeddings.word_embeddings.weight"].shape[0],
                            encoder_width=1408, add_cross_attention=True, cross_attention_freq=2)
        model = cls(config, vision_width, llm_width, raw_recent_frames=raw_recent_frames)
        for branch in (model.short, model.long):
            branch.encoder.load_state_dict(weights, strict=True)
        with torch.no_grad():
            model.short.queries.copy_(queries)
            model.long.queries[:, :32].copy_(queries)
            copies = queries.float().repeat(1, 3, 1)
            noise = torch.randn(copies.shape, generator=torch.Generator().manual_seed(noise_seed))
            model.long.queries[:, 32:].copy_(copies + noise * (0.01 * queries.float().std()))
        return model

    def _encode(self, branch, frames, frame_mask, *, history):
        if frames.ndim != 5:
            raise ValueError("Expected [batch, event, frame, patch, vision_width]")
        b, events, length, patches, width = frames.shape
        if events > 5 or length > (24 if history else self.frame_positions.size(0)) or patches == 0 or width != self.visual_projection.in_features:
            raise ValueError(f"Invalid visual input shape {tuple(frames.shape)}")
        if frame_mask is None:
            frame_mask = torch.ones((b, events, length), dtype=torch.bool, device=frames.device)
        if frame_mask.shape != (b, events, length) or frame_mask.dtype != torch.bool:
            raise ValueError("Expected a boolean mask for each input frame")
        has_frames = frame_mask.flatten(1).any(1)
        mask = has_frames[:, None].expand(b, branch.queries.shape[1])
        if not history and not self.raw_recent_frames and not has_frames.all():
            raise ValueError("Each recent window must have at least one observed frame")
        if events == 0 or length == 0:
            return branch.queries.new_zeros(b, mask.shape[1], branch.output_projection.out_features), mask

        # Clear padded values before projection (including NaNs in padded storage).
        frames = frames.masked_fill(~frame_mask[..., None, None], 0)
        memory = self.visual_projection(frames.to(self.visual_projection.weight.dtype))
        memory = memory + self.frame_positions[:length, None, :].to(memory.dtype)
        if history:
            memory = memory + self.event_positions[:events, None, None, :].to(memory.dtype)
        tokens = memory.flatten(1, 3)
        token_mask = frame_mask.unsqueeze(-1).expand(b, events, length, patches).reshape(b, -1)
        # Provide a dummy attention key for an empty row; mask all its outputs downstream.
        safe_mask = token_mask.clone()
        safe_mask[:, 0] |= ~has_frames
        prefix = branch(tokens, safe_mask).masked_fill(~mask.unsqueeze(-1), 0)
        return prefix, mask

    def encode_short(self, recent, frame_mask=None):
        """Encode the recent window, optionally appending the newest unpooled frames."""
        if not self.raw_recent_frames:
            return self._encode(self.short, recent.unsqueeze(1),
                                None if frame_mask is None else frame_mask.unsqueeze(1), history=False)
        if recent.ndim != 4 or not 1 <= recent.shape[1] <= self.recent_window:
            raise ValueError("Expected observed frames within the configured recent window in [batch, frame, patch, width]")
        batch, length, patches, _ = recent.shape
        if frame_mask is None:
            frame_mask = torch.ones((batch, length), device=recent.device, dtype=torch.bool)
        if frame_mask.shape != (batch, length) or frame_mask.dtype != torch.bool or not frame_mask.any(1).all():
            raise ValueError("Each recent window needs a boolean mask with at least one observed frame")
        count = min(self.raw_recent_frames, length)
        # Split per example, even when windows of different lengths share a batch.
        remaining = frame_mask.flip([1]).cumsum(1).flip([1])
        older = frame_mask & (remaining > count)
        older_slots = length - count
        short, short_mask = self._encode(self.short, recent[:, None, :older_slots],
                                         older[:, None, :older_slots], history=False)

        positions = torch.arange(length, device=recent.device).expand(batch, -1)
        selected = positions.masked_fill(~(frame_mask & ~older), length).sort(dim=1).values[:, :count]
        valid = selected < length
        raw = recent[torch.arange(batch, device=recent.device)[:, None], selected.clamp_max(length - 1)]
        raw = raw.masked_fill(~valid[..., None, None], 0).flatten(1, 2)
        raw_mask = valid.repeat_interleave(patches, dim=1)
        return torch.cat((short, raw), dim=1), torch.cat((short_mask, raw_mask), dim=1)

    def encode_long(self, events, frame_mask=None):
        """events [B,E,T,P,V] in retrieval order; returns 128 tokens and their mask."""
        return self._encode(self.long, events, frame_mask, history=True)

    def forward(self, recent, events, *, recent_mask=None, event_mask=None):
        long, long_mask = self.encode_long(events, event_mask)
        short, short_mask = self.encode_short(recent, recent_mask)
        return torch.cat((long, short), dim=1), torch.cat((long_mask, short_mask), dim=1)
