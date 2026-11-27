"""Question-conditioned event lookup and inference-only long-prefix reuse."""
from dataclasses import dataclass
import math

import torch

from .qformer import event_frame_indices


@dataclass
class LongEvidence:
    prefix: torch.Tensor
    mask: torch.Tensor
    sources: list


class QwenRelevance:
    """Frozen VLM cross-encoder. Scores log P(Yes) - log P(No) on a representative image."""

    def __init__(self, backbone):
        self.backbone = backbone
        choices = [backbone.tokenizer.encode(word, add_special_tokens=False) for word in ("Yes", "No")]
        if any(len(ids) != 1 for ids in choices):
            raise ValueError("Expected single-token Yes/No labels for Qwen")
        self.yes, self.no = (ids[0] for ids in choices)

    @torch.inference_mode()
    def __call__(self, question, images, events):
        model = self.backbone.model
        if model.training:
            raise ValueError("Retrieval requires a frozen model in eval mode")
        tok = self.backbone.tokenizer
        image_id = model.config.image_token_id
        n, patches, width = images.shape
        side = math.isqrt(patches)
        if side * side != patches or width != self.backbone.vision_width or n != len(events):
            raise ValueError("Expected one square-grid Qwen image per event")
        device = model.get_input_embeddings().weight.device
        prompts = []
        for event in events:
            prompt = (f"Video interval [{event.start:g}, {event.end:g}) seconds. "
                      f"This frame is at {event.representative_s:g} seconds. User question: {question}\n"
                      "Is this interval visually relevant to the question? Reply Yes or No.")
            messages = [dict(role="user", content=[dict(type="image"), dict(type="text", text=prompt)])]
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok.encode(text, add_special_tokens=False)
            if ids.count(image_id) != 1:
                raise ValueError("Expected a single image placeholder in retrieval prompt")
            loc = ids.index(image_id)
            prompts.append(ids[:loc] + [image_id] * patches + ids[loc + 1:])
        # Right-align to read the next-token logit at the last position for every row.
        length = max(map(len, prompts))
        ids = torch.full((n, length), tok.pad_token_id, dtype=torch.long, device=device)
        attention = torch.zeros_like(ids)
        for i, tokens in enumerate(prompts):
            ids[i, -len(tokens):] = ids.new_tensor(tokens)
            attention[i, -len(tokens):] = 1
        image_mask = ids == image_id
        grid_side = side * model.config.vision_config.spatial_merge_size
        grid = ids.new_tensor([1, grid_side, grid_side]).expand(n, -1)
        positions, _ = model.model.get_rope_index(ids, image_mask.int(), image_grid_thw=grid,
                                                  attention_mask=attention)
        embeds = self.backbone.embed_tokens(ids)
        embeds[image_mask] = images.reshape(n * patches, width).to(embeds)
        logits = model(inputs_embeds=embeds, position_ids=positions, attention_mask=attention,
                       use_cache=False, logits_to_keep=1, return_dict=True).logits[:, -1].float()
        return (logits[:, self.yes] - logits[:, self.no]).cpu()


class QuestionSession:
    """One video + one pending question with fixed inference weights."""
    def __init__(self, index, question, scorer, qformers, *, retrieval_batch_size=4):
        if retrieval_batch_size < 1:
            raise ValueError("Retrieval batch size must be positive")
        self.retrieval_batch_size = retrieval_batch_size
        if not question.strip():
            raise ValueError("Expected a nonempty question")
        self.index, self.question = index, question
        self.scorer, self.qformers = scorer, qformers
        self.scores = {}
        self.selected = self.cached = None

    @torch.inference_mode()
    def at(self, second):
        if self.qformers.training:
            raise ValueError("Use eval mode for cached long prefixes")
        ranked = self.rank(second)
        key = tuple(e.number for e in ranked)
        if key == self.selected:
            return self.cached

        visual, valid, sources = self.materialize(ranked)
        device = self.qformers.visual_projection.weight.device
        prefix, mask = self.qformers.encode_long(visual.to(device), valid.to(device))
        self.cached = LongEvidence(prefix, mask, sources)
        self.selected = key
        return self.cached

    @torch.no_grad()
    def rank(self, second):
        events = self.index.events_at(second, exclude_recent_frames=self.qformers.recent_window)
        missing = [e for e in events if e.number not in self.scores]
        for i in range(0, len(missing), self.retrieval_batch_size):
            batch = missing[i:i + self.retrieval_batch_size]
            images = torch.stack([self.index.frame(e.representative_s)[1] for e in batch])
            scores = self.scorer(self.question, images, batch)
            if scores.shape != (len(batch),) or not torch.isfinite(scores).all():
                raise ValueError("Expected one finite relevance score per event")
            self.scores.update((e.number, float(s)) for e, s in zip(batch, scores))
        return sorted(events, key=lambda e: (-self.scores[e.number], e.number))[:5]

    def materialize(self, ranked):
        patches, width = self.index.shape
        # Load each selected frame once. Keep source timestamps adjacent to its slot.
        slots, sources = [], []
        for event in ranked:
            times = (event_frame_indices(math.ceil(event.end) - event.start) + event.start).tolist()
            pts, frames = zip(*(self.index.frame(t) for t in times))
            slots.append(torch.stack(frames))
            sources.append(dict(slot=len(sources), video_id=self.index.video_id, event_id=event.event_id,
                                start_s=event.start, end_s=event.end, available_at_s=event.available_at,
                                representative_s=event.representative_s, score=self.scores[event.number],
                                sample_seconds=times, source_pts_s=list(pts)))
        length = max((len(s) for s in slots), default=0)
        visual = torch.zeros(1, len(slots), length, patches, width, dtype=self.index.dtype)
        valid = torch.zeros(1, len(slots), length, dtype=torch.bool)
        for i, slot in enumerate(slots):
            visual[0, i, :len(slot)] = slot
            valid[0, i, :len(slot)] = True
        return visual, valid, sources
