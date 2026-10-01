"""One Qwen3-VL backbone with LoRA for visual encoding and memory reconstruction."""
from __future__ import annotations

import torch

SHORT = "<SHORT_MEM>"
LONG = "<LONG_MEM>"


class MemoryModel:
    def __init__(self, model_name: str, *, base_adapter: str | None = None,
                 adapter_path: str | None = None,
                 dtype: torch.dtype = torch.bfloat16, rank: int = 16):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from peft import LoraConfig, PeftModel, get_peft_model

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = self.processor.tokenizer
        base = AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True)
        if base_adapter:
            base = PeftModel.from_pretrained(base, base_adapter).merge_and_unload()
        tokenizer.add_special_tokens({"additional_special_tokens": [SHORT, LONG]})
        base.resize_token_embeddings(len(tokenizer))
        short_id = tokenizer.convert_tokens_to_ids(SHORT)
        long_id = tokenizer.convert_tokens_to_ids(LONG)
        if adapter_path:
            self.model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
        else:
            try:
                config = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
                                    task_type="CAUSAL_LM",
                                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                    "gate_proj", "up_proj", "down_proj"],
                                    trainable_token_indices=[short_id, long_id])
            except TypeError as exc:
                raise RuntimeError("Stage 2 requires PEFT with trainable_token_indices support") from exc
            self.model = get_peft_model(base, config)
        self.short_id = short_id
        self.long_id = long_id
        self.tokenizer = tokenizer

    @property
    def device(self):
        return next(self.model.parameters()).device

    def embedding(self, ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(ids.to(self.device))

    def encode_frame(self, image) -> torch.Tensor:
        """One visual frame followed by a causal short-memory readout token."""
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": SHORT},
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=False)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v
                  for k, v in inputs.items()}
        positions = (inputs["input_ids"][0] == self.short_id).nonzero(as_tuple=True)[0]
        if positions.numel() != 1:
            raise RuntimeError("SHORT_MEM must appear exactly once after image tokens")
        output = self.model(**inputs, output_hidden_states=True, return_dict=True)
        return output.hidden_states[-1][:, positions.item(), :]

    def update_long(self, previous: torch.Tensor | None,
                    short: torch.Tensor) -> torch.Tensor:
        """Replace long memory using [old long, current 60 short, 60 shared queries]."""
        if short.shape[1] != 60:
            raise ValueError("A long update requires exactly 60 short tokens")
        query = self.embedding(torch.full((1, 60), self.long_id,
                                          device=self.device, dtype=torch.long))
        parts = [short, query] if previous is None else [previous.detach(), short, query]
        prefix = torch.cat(parts, dim=1)
        output = self.model(inputs_embeds=prefix,
                            attention_mask=torch.ones(prefix.shape[:2], device=self.device),
                            output_hidden_states=True, return_dict=True)
        return output.hidden_states[-1][:, -60:, :]

    def reconstruction_loss(self, memory: torch.Tensor, target: str,
                            kind: str) -> torch.Tensor:
        """Teacher-forced CE on target tokens; memory/prompt are never labels."""
        if kind not in {"short", "block", "global"} or not target.strip():
            raise ValueError("Expected a nonempty short/block/global target")
        prompt = f"Summarize the visible ultrasound evidence ({kind}):\n"
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False,
                                    return_tensors="pt").input_ids.to(self.device)
        target_ids = self.tokenizer(target, add_special_tokens=False,
                                    return_tensors="pt").input_ids.to(self.device)
        if target_ids.numel() == 0:
            raise ValueError("Target tokenization is empty")
        eos = torch.tensor([[self.tokenizer.eos_token_id]], device=self.device)
        target_ids = torch.cat([target_ids, eos], dim=1)
        ids = torch.cat([prompt_ids, target_ids], dim=1)
        embeds = torch.cat([memory, self.embedding(ids)], dim=1)
        labels = torch.full(embeds.shape[:2], -100, dtype=torch.long,
                            device=self.device)
        labels[:, memory.shape[1] + prompt_ids.shape[1]:] = target_ids
        output = self.model(inputs_embeds=embeds, labels=labels,
                            attention_mask=torch.ones_like(labels), return_dict=True)
        return output.loss

    def save(self, path: str):
        self.model.save_pretrained(path, save_embedding_layers=True)
        self.processor.save_pretrained(path)
