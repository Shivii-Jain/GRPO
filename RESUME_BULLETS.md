# Resume bullets — truthful now

Use these until the full seed sweep has completed and `artifacts/summary.json` contains results.

- Built a reproducible PyTorch implementation of a **GRPO-style**, critic-free policy-gradient method for GPT-Neo-125M; sampled groups of four one-token completions and optimized normalized within-group exact-match rewards.
- Implemented padding-safe generated-token log-probability accounting and a frozen-reference KL regularizer; added SFT and plain REINFORCE baselines for controlled comparison.
- Designed a template-disjoint evaluation protocol for a synthetic analogy-completion benchmark, with fixed-seed experiments that save checkpoints, per-step reward/KL curves, exact-match metrics, runtime, and CUDA memory usage.

## Do not claim yet

Do not state a 0.15 → 0.80 improvement, PPO outperformance, or reduced compute. The new protocol must first run the three-seed baseline and ablation suites; then replace these bullets with the measured mean ± standard deviation results.

## After experiments complete

Only if the reports support it, add a result sentence in this format:

> Across three fixed seeds, GRPO-style training achieved **X ± Y** template-disjoint exact-match reward versus **A ± B** for REINFORCE and **C ± D** for SFT.
