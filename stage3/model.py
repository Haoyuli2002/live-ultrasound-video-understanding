"""Stage 3 model primitives: frozen Stage 2 memory + trainable QA decision model.

Both models reconstruct the exact Stage 1 -> Stage 2 checkpoint chain. The
memory model remains frozen. The decision model merges the Stage 2 adapter into
an independent backbone, then adds a new LoRA, three Stage 3 tokens and a BCE
head. Data scheduling, the full training CLI and multi-question KV branching
remain separate concerns; see ``stage3/README.md``.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

DEFAULT_MODEL = "Qwen/Qwen3.5-2B"
DECISION = "<DECISION>"
WAIT = "<WAIT>"
ANSWER = "<ANSWER>"
SPECIAL_TOKENS = [DECISION, WAIT, ANSWER]
SYSTEM = (
    "You are an ultrasound video assistant. Use only the supplied causal visual "
    "memory. Decide whether the visible evidence is sufficient for the question. "
    "If it is insufficient, wait and state which evidence is still missing. If it "
    "is sufficient, answer only from visible evidence."
)


def answerability_label(target_text: str) -> float:
    """Return the BCE label encoded by a WAIT/ANSWER training target."""
    target = target_text.strip()
    if target.startswith(WAIT):
        return 0.0
    if target.startswith(ANSWER):
        return 1.0
    raise ValueError(f"Stage 3 target must start with {WAIT} or {ANSWER}")


def load_frozen_memory_model(
    model_name: str = DEFAULT_MODEL,
    *,
    stage1_adapter: str | None = None,
    stage2_adapter: str,
    dtype: torch.dtype = torch.bfloat16,
):
    """Load and freeze the causal Stage 2 memory model.

    ``stage1_adapter`` is first merged into the Qwen3.5 backbone. The Stage 2
    adapter is then loaded after restoring the two Stage 2 memory-token rows.
    Callers still wrap video-memory generation in ``torch.no_grad()``.
    """
    from stage2.model import MemoryModel

    memory = MemoryModel(
        model_name,
        base_adapter=stage1_adapter,
        adapter_path=stage2_adapter,
        dtype=dtype,
    )
    memory.model.eval()
    for parameter in memory.model.parameters():
        parameter.requires_grad_(False)
    return memory


class DecisionModel(nn.Module):
    """Read continuous long/short memory and learn WAIT versus ANSWER.

    Input order is exactly::

        [fixed system/template] [Long memory] [Short memory]
        [Question] [<DECISION>] [assistant template] [target text]

    Only target text receives NTP labels. The hidden state at ``<DECISION>``
    drives the registered BCE head. Because this class is an ``nn.Module``, one
    ``decision.to(device)`` call moves the language model and BCE head together.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        stage2_adapter: str,
        stage1_adapter: str | None = None,
        decision_adapter: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        rank: int = 16,
    ):
        super().__init__()
        from peft import LoraConfig, PeftModel, get_peft_model

        # Recreate the exact Stage 2 base (Stage 1 merged, memory-token rows
        # restored, Stage 2 adapter loaded), then bake Stage 2 into this
        # independent decision backbone.
        stage2 = load_frozen_memory_model(
            model_name,
            stage1_adapter=stage1_adapter,
            stage2_adapter=stage2_adapter,
            dtype=dtype,
        )
        base = stage2.model.merge_and_unload()
        self.processor = stage2.processor
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        to_add = [token for token in SPECIAL_TOKENS if token not in self.tokenizer.get_vocab()]
        if to_add:
            self.tokenizer.add_special_tokens({"additional_special_tokens": to_add})
            base.resize_token_embeddings(len(self.tokenizer))
        self.decision_id = self.tokenizer.convert_tokens_to_ids(DECISION)
        trainable_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in SPECIAL_TOKENS]
        try:
            config = LoraConfig(
                r=rank,
                lora_alpha=2 * rank,
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                trainable_token_indices=trainable_ids,
                # Qwen output embeddings are not assumed to be tied. Saving the
                # LM head ensures the new WAIT/ANSWER rows can be learned.
                modules_to_save=["lm_head"],
            )
        except TypeError as exc:
            raise RuntimeError(
                "Stage 3 requires PEFT with trainable_token_indices support"
            ) from exc
        if decision_adapter:
            self.model = PeftModel.from_pretrained(
                base, decision_adapter, is_trainable=False
            )
        else:
            self.model = get_peft_model(base, config)
        text_config = getattr(base.config, "text_config", base.config)
        hidden_size = getattr(base.config, "hidden_size", None) or text_config.hidden_size
        # Keep the small classifier in fp32 for stable BCE even when the VLM
        # backbone runs in bf16/fp16. ``nn.Module.to(device)`` still moves it.
        self.decision_head = nn.Linear(hidden_size, 1, dtype=torch.float32)
        if decision_adapter:
            head_path = Path(decision_adapter) / "decision_head.pt"
            if not head_path.is_file():
                raise FileNotFoundError(f"Missing saved BCE head: {head_path}")
            self.decision_head.load_state_dict(
                torch.load(head_path, map_location="cpu", weights_only=True)
            )
        self.model_name = model_name
        self.stage1_adapter = stage1_adapter
        self.stage2_adapter = stage2_adapter
        self.decision_adapter = decision_adapter

    @property
    def device(self):
        return next(self.model.parameters()).device

    def embedding(self, ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(ids.to(self.device))

    def _ids(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.device)

    def _prefix(
        self, memory: torch.Tensor, question: str
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Build the documented causal prefix and return its decision position."""
        if memory.ndim != 3 or memory.shape[0] != 1 or memory.shape[1] == 0:
            raise ValueError("memory must have shape [1, M, hidden] with M > 0")
        if memory.requires_grad:
            raise ValueError("memory must be detached before the decision model consumes it")
        question = question.strip()
        if not question:
            raise ValueError("Stage 3 question is empty")
        system_ids = self._ids(f"System: {SYSTEM}\nVisual memory:\n")
        question_ids = self._ids(f"\nQuestion: {question}\nDecision: {DECISION}")
        system_embeds = self.embedding(system_ids)
        question_embeds = self.embedding(question_ids)
        memory = memory.to(self.device, dtype=system_embeds.dtype)
        embeds = torch.cat([system_embeds, memory, question_embeds], dim=1)
        decision_position = embeds.shape[1] - 1
        if question_ids[0, -1].item() != self.decision_id:
            raise RuntimeError("<DECISION> must be the final token of the decision prefix")
        attention = torch.ones(embeds.shape[:2], dtype=torch.long, device=self.device)
        return embeds, attention, decision_position

    def decision_logits(self, memory: torch.Tensor, question: str) -> torch.Tensor:
        """Compute answerability logits without teacher text."""
        embeds, attention, decision_position = self._prefix(memory, question)
        output = self.model(
            inputs_embeds=embeds,
            attention_mask=attention,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = output.hidden_states[-1][:, decision_position, :]
        return self.decision_head(hidden.to(self.decision_head.weight.dtype)).squeeze(-1)

    def forward_answerability(
        self,
        memory: torch.Tensor,
        question: str,
        target_text: str,
        y: float | None = None,
    ):
        """Return ``(BCE loss, NTP loss, logit)`` for one decision point."""
        encoded_y = answerability_label(target_text)
        if y is None:
            y = encoded_y
        if float(y) not in {0.0, 1.0} or float(y) != encoded_y:
            raise ValueError("BCE label disagrees with the <WAIT>/<ANSWER> target")

        prefix_embeds, prefix_attention, decision_position = self._prefix(memory, question)
        assistant_ids = self._ids("\nAssistant: ")
        target_ids = self._ids(target_text.strip())
        if target_ids.numel() == 0:
            raise ValueError("Empty WAIT/ANSWER text target")
        eos = torch.tensor([[self.tokenizer.eos_token_id]], device=self.device)
        target_ids = torch.cat([target_ids, eos], dim=1)
        tail_ids = torch.cat([assistant_ids, target_ids], dim=1)
        embeds = torch.cat([prefix_embeds, self.embedding(tail_ids)], dim=1)
        attention = torch.cat(
            [
                prefix_attention,
                torch.ones(tail_ids.shape, dtype=torch.long, device=self.device),
            ],
            dim=1,
        )
        labels = torch.full(embeds.shape[:2], -100, dtype=torch.long, device=self.device)
        target_start = prefix_embeds.shape[1] + assistant_ids.shape[1]
        labels[:, target_start:] = target_ids

        output = self.model(
            inputs_embeds=embeds,
            attention_mask=attention,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = output.hidden_states[-1][:, decision_position, :]
        logit = self.decision_head(hidden.to(self.decision_head.weight.dtype)).squeeze(-1)
        y_tensor = torch.tensor([float(y)], device=self.device, dtype=logit.dtype)
        decision_loss = nn.functional.binary_cross_entropy_with_logits(logit, y_tensor)
        return decision_loss, output.loss, logit

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(destination, save_embedding_layers=True)
        self.processor.save_pretrained(destination)
        torch.save(self.decision_head.state_dict(), destination / "decision_head.pt")
        (destination / "stage3_config.json").write_text(
            json.dumps(
                {
                    "model_name": self.model_name,
                    "stage1_adapter": self.stage1_adapter,
                    "stage2_adapter": self.stage2_adapter,
                    "decision_adapter": self.decision_adapter,
                    "special_tokens": SPECIAL_TOKENS,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


# Still open: QA/time-aligned data loading, the live two-model training loop,
# thresholded text generation, and multi-question shared-prefix KV branching.
