"""Causal visual-prefix conditioning, action decoding and language alignment for Qwen."""
from dataclasses import dataclass
import math

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import LogitsProcessor, LogitsProcessorList

from .actions import install_actions


SYSTEM = (
    "You are watching a video one second at a time. Use only the observed visual evidence. "
    "The visual prefix contains retrieved historical events followed by the most recent frames. "
    "When there is enough evidence, reply <ANSWER> followed by a concise answer. "
    "Otherwise reply <WAIT> followed by what is visible and what evidence is still missing. "
    "Do not invent observations."
)
# Replaced by continuous embeddings, never passed to the tokenizer as a visual token.
VISUAL_SLOT = "{visual_prefix_here}"


@dataclass(frozen=True)
class DecisionContext:
    question: str
    query_time: float
    current_time: float
    recent_start: int
    recent_end: int
    events: tuple = ()

    def prompt_parts(self, tokenizer):
        if (not self.question.strip() or not math.isfinite(self.current_time)
                or not 0 <= self.query_time <= self.current_time
                or not 0 <= self.recent_start <= self.recent_end <= self.current_time):
            raise ValueError("Invalid causal question or recent window")
        history = []
        for e in self.events:
            if not 0 <= e['start_s'] < e['end_s'] <= e['available_at_s'] <= self.current_time:
                raise ValueError("Event is unavailable at decision time")
            history.append(f"{e['event_id']} [{e['start_s']:g}, {e['end_s']:g}) s")
        text = (f"{VISUAL_SLOT}\nHistory in prefix order: {', '.join(history) or 'empty'}. "
                f"Recent frames: {self.recent_start}–{self.recent_end} s. "
                f"Question asked at {self.query_time:g} s. Current time: {self.current_time:g} s.\n"
                f"Question: {self.question}")
        rendered = tokenizer.apply_chat_template(
            [dict(role="system", content=SYSTEM), dict(role="user", content=text)],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        before, after = rendered.split(VISUAL_SLOT, 1)
        return before, after


class ActionFormat(LogitsProcessor):
    """Constrain the initial tag only; Qwen's logits choose the action and its text."""
    def __init__(self, tokenizer, first_action=None):
        self.first_action = first_action
        self.tokenizer = tokenizer
        self.tags = {name: tokenizer.encode(f"<{name}>", add_special_tokens=False)
                     for name in ("WAIT", "ANSWER")}
        self.special = tokenizer.all_special_ids

    def __call__(self, input_ids, scores):
        # Native inputs_embeds generation returns only newly generated token IDs.
        for i, sequence in enumerate(input_ids.tolist()):
            for tag in self.tags.values():
                if sequence[:len(tag)] == tag:
                    if not self.tokenizer.decode(sequence[len(tag):], skip_special_tokens=True).strip():
                        scores[i, self.special] = -torch.inf
                    break
            else:
                allowed = [tag[len(sequence)] for name, tag in self.tags.items()
                           if (self.first_action is None or name == self.first_action)
                           and len(sequence) < len(tag) and tag[:len(sequence)] == sequence]
                if not allowed:
                    raise ValueError("Generated action prefix does not match the action grammar")
                keep = scores[i, allowed]
                scores[i].fill_(-torch.inf)
                scores[i, allowed] = keep
        return scores

    def parse(self, sequence):
        for name, tag in self.tags.items():
            if sequence[:len(tag)] == tag:
                body = self.tokenizer.decode(sequence[len(tag):], skip_special_tokens=True).strip()
                if body:
                    return name, body
        raise ValueError("Decoder did not produce an action with nonempty text")


class AnswerDecoder:
    """Shared action vocabulary for generation and teacher-forced NTP."""
    def __init__(self, backbone, *, legacy_tags=False, separate_action_loss=False, token_mean_ntp=False):
        if separate_action_loss and token_mean_ntp:
            raise ValueError("Choose a single training objective")
        self.token_mean_ntp = token_mean_ntp
        self.separate_action_loss = separate_action_loss
        self.backbone = backbone
        if not legacy_tags:
            install_actions(backbone)
        self.format = ActionFormat(backbone.tokenizer)

    def _ids(self, text):
        ids = self.backbone.tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor(ids, dtype=torch.long, device=self.backbone.model.get_input_embeddings().weight.device)

    def _prompt(self, visual, mask, context):
        if (visual.ndim != 2 or visual.shape[-1] != self.backbone.llm_width
                or mask.shape != visual.shape[:1] or mask.dtype != torch.bool):
            raise ValueError("Expected visual prefix [tokens, LLM width] and boolean token mask")
        before, after = context.prompt_parts(self.backbone.tokenizer)
        head = self.backbone.embed_tokens(self._ids(before))
        tail = self.backbone.embed_tokens(self._ids(after))
        # Physically omit unavailable slots so recurrent layers never consume dummy positions.
        return torch.cat([head, visual[mask].to(head), tail])

    @torch.inference_mode()
    def generate(self, visual, mask, context, max_new_tokens=64, *, force_answer=False):
        return self.generate_batch([(visual, mask, context)], max_new_tokens, force_answer=force_answer)[0]

    @torch.inference_mode()
    def generate_batch(self, items, max_new_tokens=64, *, force_answer=False):
        if max_new_tokens <= max(map(len, self.format.tags.values())):
            raise ValueError("Token budget must leave room for the action and its text")
        prompts = [self._prompt(visual, mask, ctx) for visual, mask, ctx in items]
        if not prompts:
            return []
        length = max(map(len, prompts))
        embeddings = prompts[0].new_zeros(len(prompts), length, prompts[0].shape[-1])
        attention = torch.zeros(embeddings.shape[:2], dtype=torch.long, device=embeddings.device)
        for i, prompt in enumerate(prompts):
            embeddings[i, -len(prompt):] = prompt
            attention[i, -len(prompt):] = 1
        ids = self.backbone.lm_generate(
            embeddings, attention, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
            logits_processor=LogitsProcessorList([ActionFormat(self.backbone.tokenizer, "ANSWER")
                                                  if force_answer else self.format]),
        )
        result = []
        eos = self.backbone.tokenizer.eos_token_id
        for sequence in ids.tolist():
            ended = eos in sequence
            if ended:
                sequence = sequence[:sequence.index(eos)]
            action, body = self.format.parse(sequence)
            result.append(dict(action=action, finish_reason="eos" if ended else "length",
                               reason=body if action == "WAIT" else None,
                               answer=body if action == "ANSWER" else None, text=f"<{action}> {body}"))
        return result

    def alignment_loss(self, visual, mask, contexts, targets, *, average_endpoints=False, endpoint_weights=None, loss_sums=None):
        """Token-mean CE on gold assistant responses; gradients reach both visible branches."""
        if visual.ndim != 3 or mask.shape != visual.shape[:2] or not (
                len(visual) == len(contexts) == len(targets) and len(targets) > 0):
            raise ValueError("Expected aligned batches of visual prefixes, contexts and targets")
        if endpoint_weights is not None and (not average_endpoints or len(endpoint_weights) != len(contexts)):
            raise ValueError("Expected one weight per endpoint for endpoint-averaged loss")
        inputs, labels = [], []
        for features, valid, context, target in zip(visual, mask, contexts, targets):
            prefix = self._prompt(features, valid, context)
            parts = target.split(maxsplit=1)
            tag = parts[0] if parts else ""
            body = parts[1].strip() if len(parts) == 2 else ""
            if tag not in ("<WAIT>", "<ANSWER>") or (tag == "<ANSWER>" and not body.strip()):
                raise ValueError("Expected <WAIT> [reason] or <ANSWER> answer")
            # Encode tag and payload separately to match constrained generation exactly.
            action = self.format.tags[tag[1:-1]]
            if self.separate_action_loss and len(action) != 1:
                raise ValueError("Separate action loss requires one special token per action")
            target_ids = list(action)
            if body.strip():
                target_ids += self.backbone.tokenizer.encode(" " + body.strip(), add_special_tokens=False)
                target_ids.append(self.backbone.tokenizer.eos_token_id)
            # A bare WAIT supervises only the action; no invented reason or immediate EOS target.
            target_ids = torch.tensor(target_ids, dtype=torch.long, device=prefix.device)
            inputs.append(torch.cat([prefix, self.backbone.embed_tokens(target_ids)]))
            labels.append(torch.cat([target_ids.new_full((len(prefix),), -100), target_ids]))
        embeddings = pad_sequence(inputs, batch_first=True)
        lengths = torch.tensor([len(row) for row in inputs], device=embeddings.device)
        attention = torch.arange(embeddings.shape[1], device=embeddings.device)[None] < lengths[:, None]
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        if self.token_mean_ntp or self.separate_action_loss or (average_endpoints and len(contexts) > 1):
            # Only supervised next tokens need vocabulary logits. Keep the original
            # per-endpoint mean so one ANSWER does not outweigh several bare WAITs.
            host = self.backbone.model.model
            host.rope_deltas = None
            positions = (attention.long().cumsum(-1) - 1).masked_fill(~attention, 0)
            hidden = host(inputs_embeds=embeddings, attention_mask=attention.long(), position_ids=positions,
                          use_cache=False, return_dict=True).last_hidden_state
            valid = labels[:, 1:] != -100
            logits = self.backbone.model.get_output_embeddings()(hidden[:, :-1][valid])
            losses = torch.nn.functional.cross_entropy(logits.float(), labels[:, 1:][valid], reduction="none")
            if self.token_mean_ntp:
                if endpoint_weights is not None and any(w != 1 for w in endpoint_weights):
                    raise ValueError("Plain NTP expects equal weight for every supervised token")
                if loss_sums is not None:
                    loss_sums[0].add_(losses.detach().sum())
                    loss_sums[1].add_(losses.numel())
                return losses.mean()
            if self.separate_action_loss:
                counts = valid.sum(1)
                starts = counts.cumsum(0) - counts
                action_ce = losses[starts]
                if endpoint_weights is not None:
                    action_ce = action_ce * action_ce.new_tensor(endpoint_weights)
                # A bare WAIT has no body target. Give it zero body loss.
                body_counts = counts - 1
                body_positions = torch.ones_like(losses, dtype=torch.bool)
                body_positions[starts] = False
                body_weights = body_counts.clamp_min(1).float().reciprocal()
                body_sum = (losses[body_positions] * body_weights.repeat_interleave(body_counts)).sum()
                action_sum = action_ce.sum()
                if loss_sums is not None:
                    loss_sums.add_(torch.stack((action_sum.detach(), body_sum.detach())))
                return (action_sum + body_sum) / len(contexts)
            weights = valid.sum(1).float().reciprocal()
            if endpoint_weights is not None:
                weights = weights * weights.new_tensor(endpoint_weights)
            weights = weights[:, None].expand_as(valid)
            return (losses * weights[valid]).sum() / len(contexts)
        _, loss = self.backbone.lm_forward(embeddings, attention.long(), labels=labels)
        return loss if endpoint_weights is None else loss * endpoint_weights[0]


class StreamingAnswerSession:
    """One question. WAIT advances; ANSWER ends the session; repeated ticks are idempotent."""
    def __init__(self, retrieval, decoder, query_time):
        if not math.isfinite(query_time) or query_time < 0:
            raise ValueError("Invalid question time")
        self.retrieval, self.decoder, self.query_time = retrieval, decoder, query_time
        self.last = None

    @torch.inference_mode()
    def step(self, second, *, max_new_tokens=64):
        if not math.isfinite(second) or second < self.query_time:
            raise ValueError("Decision precedes the question or is not finite")
        if self.last is not None:
            if self.last['at'] == second:
                return self.last
            if second < self.last['at'] or self.last['action'] == 'ANSWER':
                raise ValueError("Question is complete or decision times are out of order")
        index, model = self.retrieval.index, self.retrieval.qformers
        history = self.retrieval.at(second)
        end = min(math.floor(second), index.detector.next_second - 1)
        start = max(0, end + 1 - model.recent_window)
        frames = torch.stack([index.frame(t)[1] for t in range(start, end + 1)])
        short, short_mask = model.encode_short(frames.unsqueeze(0).to(model.visual_projection.weight.device))
        visual = torch.cat([history.prefix, short], dim=1)
        mask = torch.cat([history.mask, short_mask], dim=1)
        context = DecisionContext(self.retrieval.question, self.query_time, second, start, end, tuple(history.sources))
        decision = self.decoder.generate(visual[0], mask[0], context, max_new_tokens)
        self.last = dict(video_id=index.video_id, at=second, question=context.question, **decision,
                         recent_window_s=[start, end], retrieved_events=history.sources,
                         long_tokens=int(history.mask.sum()), short_tokens=int(short_mask.sum()))
        return self.last
