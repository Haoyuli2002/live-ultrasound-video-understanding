"""Atomic action tokens in the Qwen input embedding and generation vocabulary."""
import torch

ACTIONS = ("<WAIT>", "<ANSWER>")


@torch.no_grad()
def install_actions(backbone):
    tok, model = backbone.tokenizer, backbone.model
    old = {s: tok.encode(s, add_special_tokens=False) for s in ACTIONS}
    existing = set(tok.get_vocab()).intersection(ACTIONS)
    tok.add_special_tokens({"extra_special_tokens": list(ACTIONS)}, replace_extra_special_tokens=False)
    ids = {s: tok.convert_tokens_to_ids(s) for s in ACTIONS}
    if len(set(ids.values())) != len(ACTIONS) or any(
        tok.encode(s, add_special_tokens=False) != [i] or i not in tok.all_special_ids for s, i in ids.items()
    ):
        raise ValueError("Expected two distinct atomic special tokens")
    # Qwen snapshots may have more model rows than tokenizer entries.
    size = max(ids.values()) + 1
    if size > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(size, mean_resizing=False)
    tables = [model.get_input_embeddings().weight]
    out = model.get_output_embeddings().weight
    if out.data_ptr() != tables[0].data_ptr():
        tables.append(out)
    for s, i in ids.items():
        if s not in existing:
            for table in tables:
                table[i].copy_(table[old[s]].float().mean(0).to(table))
    return ids
