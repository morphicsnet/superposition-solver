from typing import List, Tuple
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model_and_tokenizer(model_name: str):
    """
    Load HF causal LM model and tokenizer. Model set to eval with grads disabled.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # GPT-2 family typically has no pad token; use eos as pad for batching
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def capture_layer_activations(model, tokenizer, texts: List[str], layer_index: int) -> np.ndarray:
    """
    Capture mean-pooled per-sequence activations from a given transformer block.
    We capture MLP input at the specified layer (pre-MLP forward hook).

    Args:
        model: HF Causal LM (e.g., GPT-2).
        tokenizer: Matching tokenizer.
        texts: List of strings.
        layer_index: Index of transformer block to hook.

    Returns:
        acts: np.ndarray of shape [N, D], mean over tokens (mask-aware).
    """
    # Validate layer index
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        raise ValueError("Model does not expose transformer.h blocks (expected GPT-2 style).")
    n_layers = len(model.transformer.h)
    if layer_index < 0 or layer_index >= n_layers:
        raise ValueError(f"layer_index {layer_index} out of bounds [0, {n_layers-1}]")

    device = torch.device("cpu")
    model.to(device)

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # Container for captured tensors and masks (per batch)
    captured: List[torch.Tensor] = []
    batch_masks: List[torch.Tensor] = []

    # Hook the MLP input at the chosen layer
    target_mlp = model.transformer.h[layer_index].mlp

    def pre_hook(module, inputs):
        # inputs is a tuple; inputs[0] has shape [B, T, D]
        hidden = inputs[0].detach().to("cpu")
        captured.append(hidden)

    handle = target_mlp.register_forward_pre_hook(pre_hook)

    try:
        batch_size = 16
        N = input_ids.size(0)
        with torch.no_grad():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                ids = input_ids[start:end]
                mask = attention_mask[start:end]
                batch_masks.append(mask.detach().to("cpu"))
                _ = model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        handle.remove()

    # Mean-pool across tokens using attention mask
    pooled_list: List[torch.Tensor] = []
    for hid, mask in zip(captured, batch_masks):
        # hid: [B, T, D]; mask: [B, T]
        if hid.shape[:2] != mask.shape:
            raise RuntimeError(f"Shape mismatch between hidden {hid.shape} and mask {mask.shape}")
        mask_f = mask.unsqueeze(-1).float()  # [B, T, 1]
        denom = mask_f.sum(dim=1).clamp(min=1.0)  # [B, 1]
        summed = (hid * mask_f).sum(dim=1)  # [B, D]
        pooled = summed / denom  # [B, D]
        pooled_list.append(pooled)

    acts = torch.cat(pooled_list, dim=0).numpy() if pooled_list else np.zeros((0, model.config.n_embd), dtype=np.float32)
    return acts