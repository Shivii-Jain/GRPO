"""Train GRPO-style, REINFORCE, or SFT baselines from one YAML config."""
from __future__ import annotations

import argparse, csv, json, random, time
from pathlib import Path
import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import repeat_templates, split_templates
from evaluate import evaluate
from grpo import collect_rollouts, policy_loss, reinforce_loss


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f: return yaml.safe_load(f)


def select_device() -> str:
    """Prefer CUDA, then the Apple Silicon MPS backend, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_sft(policy, tokenizer, data, cfg, device):
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg["learning_rate"])
    history = []
    for step in range(1, cfg["steps"] + 1):
        batch = random.sample(data, cfg["batch_size"])
        prompts = [x.prompt for x in batch]
        targets = [random.choice(x.targets) for x in batch]
        enc = tokenizer([p + " " + t for p, t in zip(prompts, targets)], padding=True, return_tensors="pt").to(device)
        labels = enc.input_ids.clone()
        target_lengths = [len(tokenizer(" " + target, add_special_tokens=False).input_ids) for target in targets]
        for row, target_length in enumerate(target_lengths):
            labels[row, : labels.shape[1] - target_length] = -100
        loss = policy(**enc, labels=labels).loss
        loss.backward(); optimizer.step(); optimizer.zero_grad()
        history.append({"step": step, "loss": loss.item(), "reward": None, "kl": None})
    return history


def train_rl(policy, reference, tokenizer, data, cfg, device, algorithm):
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg["learning_rate"])
    history = []
    for step in range(1, cfg["steps"] + 1):
        batch = random.sample(data, cfg["batch_size"])
        rollout = collect_rollouts(policy, reference, tokenizer, batch, cfg["group_size"], device, cfg["max_new_tokens"])
        if algorithm == "grpo":
            loss, _, kl = policy_loss(rollout, cfg["beta_kl"], cfg["normalize_advantages"])
        else:
            loss, _, kl = reinforce_loss(rollout, cfg["beta_kl"])
        loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg["max_grad_norm"])
        optimizer.step(); optimizer.zero_grad()
        history.append({"step": step, "loss": loss.item(), "reward": rollout.rewards.mean().item(), "kl": kl.item()})
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--algorithm", choices=["grpo", "reinforce", "sft"], default="grpo")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    seed_everything(seed)
    device = select_device()
    if device == "cuda": torch.cuda.reset_peak_memory_stats()
    out = Path(args.output_dir) / f"{args.algorithm}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    tokenizer.padding_side = "left"; tokenizer.pad_token = tokenizer.eos_token
    policy = AutoModelForCausalLM.from_pretrained(cfg["model_name"]).to(device)
    reference = None
    if args.algorithm != "sft":
        reference = AutoModelForCausalLM.from_pretrained(cfg["model_name"]).to(device).eval()
        for p in reference.parameters(): p.requires_grad_(False)
    train_templates, test_templates = split_templates(seed, cfg["held_out_templates"])
    train_data = repeat_templates(train_templates, cfg["train_repeats"], seed)
    test_data = repeat_templates(test_templates, cfg["eval_repeats"], seed + 1)
    baseline = evaluate(policy, tokenizer, test_data, device, cfg["eval_batch_size"])
    started = time.perf_counter()
    history = train_sft(policy, tokenizer, train_data, cfg, device) if args.algorithm == "sft" else train_rl(policy, reference, tokenizer, train_data, cfg, device, args.algorithm)
    runtime = time.perf_counter() - started
    final = evaluate(policy, tokenizer, test_data, device, cfg["eval_batch_size"])
    report = {"algorithm": args.algorithm, "seed": seed, "config": cfg, "device": device,
              "baseline": baseline, "final": final, "train_seconds": runtime,
              "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device == "cuda" else None,
              "train_templates": [x.prompt for x in train_templates], "held_out_templates": [x.prompt for x in test_templates]}
    with open(out / "metrics.json", "w") as f: json.dump(report, f, indent=2)
    with open(out / "training.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss", "reward", "kl"]); writer.writeheader(); writer.writerows(history)
    policy.save_pretrained(out / "checkpoint"); tokenizer.save_pretrained(out / "checkpoint")
    print(json.dumps({"output": str(out), "baseline_exact_match": baseline["exact_match"], "final_exact_match": final["exact_match"], "train_seconds": runtime}, indent=2))

if __name__ == "__main__": main()
