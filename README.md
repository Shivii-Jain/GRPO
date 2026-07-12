# Group-Normalized Policy Optimization for LLM Fine-Tuning

A compact, reproducible PyTorch study of critic-free, group-relative policy gradients for a controlled GPT-Neo-125M analogy-completion task. It is intentionally small enough to inspect end-to-end.

## What this repository implements

- **GRPO-style policy gradient:** sample `G` completions per prompt; use each completion's reward relative to its group mean; optionally normalize by the within-group standard deviation.
- **Frozen-reference regularization:** a sampled KL estimate penalizes divergence from the initial GPT-Neo policy.
- **Padding-safe generated-token log probabilities:** the objective masks prompt positions and scores only generated tokens. This is important because left-padded batches cannot use each item's non-padding prompt length as an absolute token index.
- **Fairer evaluation:** four complete analogy templates are held out by seed. No repeated instance of a held-out relation enters training.
- **Baselines:** pretrained GPT-Neo (reported before training), supervised fine-tuning (SFT), and REINFORCE without relative group advantages. PPO is deliberately **not** claimed or included.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_experiments.py --config configs/base.yaml
```

This executes GRPO-style, REINFORCE, and SFT runs for three fixed seeds. It writes a checkpoint, per-step `training.csv`, per-run `metrics.json`, and aggregate `artifacts/summary.json`.

To run a single experiment:

```bash
python train.py --algorithm grpo --seed 7 --config configs/base.yaml
```

> GPT-Neo-125M weights are downloaded from Hugging Face on the first run. The runner automatically selects CUDA, Apple Silicon's MPS GPU backend, or CPU, in that order. Metrics record peak allocated VRAM when CUDA is available.

## Objective

For prompt \(x_i\), sampled responses \(y_{i,g}\), and binary reward \(r_{i,g}\), this project uses

\[
A_{i,g} = \frac{r_{i,g} - \operatorname{mean}_g(r_{i,g})}{\operatorname{std}_g(r_{i,g}) + \epsilon}
\]

and minimizes

\[
\mathcal{L} = -\mathbb{E}[A_{i,g}\log\pi_\theta(y_{i,g}|x_i)]
 + \beta\,\mathbb{E}[\log\pi_\theta(y_{i,g}|x_i)-\log\pi_{ref}(y_{i,g}|x_i)].
\]

The reward is 1 when the first generated word exactly matches an accepted target and 0 otherwise. `max_new_tokens: 1` keeps the task and token-level accounting deliberately controlled.

## Reporting protocol

Use `artifacts/summary.json` to report mean ± population standard deviation across the configured seeds. Never report a single run as a final comparison. Include the template-disjoint exact-match score, runtime, peak VRAM, seed list, config, and baseline results.

Run the complete, seed-controlled ablation grid with:

```bash
python run_ablations.py
```

`configs/ablations.yaml` varies group size, KL coefficient, and advantage normalization while retaining the same seeds and template-split procedure. Its aggregate table is written to `artifacts/ablations/summary.json`.

## Limitations

This is a controlled educational benchmark, not a general LLM-alignment result. The 20 manually curated analogy templates, binary exact-match reward, one-token completion, and small model limit external validity. A template-disjoint split is stricter than repeated-template evaluation, but it is still too small to establish broad reasoning generalization.

The implementation is **GRPO-style**, not a reproduction of every production GRPO detail: it has no clipped importance-ratio surrogate, multi-turn rollout, reward model, or distributed training. It also does not implement PPO; accordingly, this repository makes no PPO performance or compute claim.

