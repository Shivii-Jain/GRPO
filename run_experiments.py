"""Reproducibly run seed sweeps and configured GRPO ablations."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import yaml
import numpy as np

def main():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/base.yaml"); p.add_argument("--output-dir", default="artifacts")
    args = p.parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)
    root = Path(args.output_dir); root.mkdir(exist_ok=True)
    jobs = [(algorithm, seed) for algorithm in cfg["baselines"] for seed in cfg["seeds"]]
    for algorithm, seed in jobs:
        subprocess.run([sys.executable, "train.py", "--config", args.config, "--algorithm", algorithm, "--seed", str(seed), "--output-dir", str(root)], check=True)
    summary = {}
    for algorithm in cfg["baselines"]:
        metrics = [json.loads((root / f"{algorithm}_seed{s}" / "metrics.json").read_text()) for s in cfg["seeds"]]
        scores = [m["final"]["exact_match"] for m in metrics]
        summary[algorithm] = {"mean_exact_match": float(np.mean(scores)), "std_exact_match": float(np.std(scores)), "n_seeds": len(scores)}
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__": main()
