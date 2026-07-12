# Design note: group-relative optimization, correctly scoped

## Why group-relative advantages?

For a given prompt, a binary reward is often sparse. Sampling several candidate continuations provides a local comparison set: candidates above the group's mean receive positive advantage and candidates below it receive negative advantage. This removes the need for a learned value network in this prototype.

## What makes it different from PPO?

PPO normally uses a value/critic baseline and a clipped ratio between current and rollout-policy probabilities. This project uses neither a critic nor a clipped surrogate. It is therefore a group-normalized REINFORCE-style method with a reference-policy penalty, described as **GRPO-style** rather than as an empirical PPO replacement.

## Token accounting invariant

The loss must only score tokens generated after the prompt. Since batches use left padding, a prompt's non-padding length is not the correct absolute sequence index. `grpo._completion_logps` masks every label position before the padded batch prompt width, then sums only generated labels. This invariant is covered by code comments and is central to interpreting the loss.

## Experimental claims we will and will not make

We report exact-match reward over held-out templates, mean ± standard deviation across fixed seeds, training duration, and CUDA peak allocated memory. We do not claim general LLM alignment, PPO superiority, compute savings, or statistical significance unless additional experiments directly establish them.
