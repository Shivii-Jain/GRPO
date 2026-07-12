"""Rollout collection and critic-free group-relative policy-gradient objectives."""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F

from rewards import exact_match_rewards


@dataclass
class Rollout:
    logp: torch.Tensor
    ref_logp: torch.Tensor
    rewards: torch.Tensor
    completions: list[list[str]]


def _completion_logps(model, full_ids, full_attention, prompt_width: int) -> torch.Tensor:
    """Sum only generated-token log probabilities.

    With left padding, the generated first token is label position prompt_width - 1,
    not ``non_pad_prompt_length - 1``.  This position-based mask is padding-safe.
    """
    logits = model(input_ids=full_ids[:, :-1], attention_mask=full_attention[:, :-1]).logits
    chosen = F.log_softmax(logits, dim=-1).gather(-1, full_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    generated_mask = full_attention[:, 1:].bool()
    generated_mask[:, : prompt_width - 1] = False
    return (chosen * generated_mask).sum(dim=-1)


def collect_rollouts(policy, reference, tokenizer, batch, group_size: int, device: str, max_new_tokens: int = 1) -> Rollout:
    prompts = [x.prompt for x in batch]
    target_lists = [x.targets for x in batch]
    encoded = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    prompt_width = encoded.input_ids.shape[1]
    logps, ref_logps, rewards, groups = [], [], [], []
    for _ in range(group_size):
        with torch.no_grad():
            generated = policy.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=True,
                                        temperature=1.0, pad_token_id=tokenizer.pad_token_id)
        # ``pad_token_id`` is commonly the EOS token for decoder-only models.
        # Do not infer generated-token validity from token IDs: an EOS action must
        # still receive a log-probability. Generation here always has fixed length.
        generated_width = generated.shape[1] - prompt_width
        attention = torch.cat(
            [encoded.attention_mask, torch.ones((generated.shape[0], generated_width), dtype=encoded.attention_mask.dtype, device=device)],
            dim=1,
        )
        completions = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
        logps.append(_completion_logps(policy, generated, attention, prompt_width))
        with torch.no_grad():
            ref_logps.append(_completion_logps(reference, generated, attention, prompt_width))
        rewards.append(exact_match_rewards(completions, target_lists, device))
        groups.append(completions)
    return Rollout(torch.stack(logps), torch.stack(ref_logps), torch.stack(rewards), groups)


def policy_loss(rollout: Rollout, beta_kl: float, normalize_advantages: bool, eps: float = 1e-6):
    rewards = rollout.rewards
    advantages = rewards - rewards.mean(dim=0, keepdim=True)
    if normalize_advantages:
        advantages = advantages / (rewards.std(dim=0, unbiased=False, keepdim=True) + eps)
    # Monte-Carlo sampled KL estimate: E_{a~pi}[log pi(a|s)-log ref(a|s)].
    kl = (rollout.logp - rollout.ref_logp).mean()
    loss = -(advantages.detach() * rollout.logp).mean() + beta_kl * kl
    return loss, advantages.detach(), kl.detach()


def reinforce_loss(rollout: Rollout, beta_kl: float):
    advantages = rollout.rewards
    kl = (rollout.logp - rollout.ref_logp).mean()
    return -(advantages.detach() * rollout.logp).mean() + beta_kl * kl, advantages.detach(), kl.detach()
