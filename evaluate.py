"""Greedy exact-match evaluation and JSON-safe metric helpers."""
from __future__ import annotations

import time
import torch
from rewards import exact_match_rewards


@torch.no_grad()
def evaluate(model, tokenizer, dataset, device: str, batch_size: int = 16) -> dict:
    was_training = model.training
    model.eval()
    correct, outputs = [], []
    started = time.perf_counter()
    for start in range(0, len(dataset), batch_size):
        batch = dataset[start:start + batch_size]
        enc = tokenizer([x.prompt for x in batch], padding=True, return_tensors="pt").to(device)
        width = enc.input_ids.shape[1]
        ids = model.generate(**enc, max_new_tokens=1, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        completion = tokenizer.batch_decode(ids[:, width:], skip_special_tokens=True)
        correct.extend(exact_match_rewards(completion, [x.targets for x in batch], device).tolist())
        outputs.extend(completion)
    if was_training:
        model.train()
    return {"exact_match": sum(correct) / len(correct), "n_examples": len(correct),
            "eval_seconds": time.perf_counter() - started, "examples": outputs[:5]}
