"""Task rewards for generated completions."""
from __future__ import annotations

import string
import torch


def first_word(text: str) -> str:
    return text.strip().split(maxsplit=1)[0].strip(string.punctuation).lower() if text.strip() else ""


def exact_match_rewards(completions: list[str], targets: list[tuple[str, ...]], device: str) -> torch.Tensor:
    values = [float(first_word(c) in {t.lower() for t in ts}) for c, ts in zip(completions, targets)]
    return torch.tensor(values, dtype=torch.float32, device=device)
