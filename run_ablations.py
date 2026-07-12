"""Run the configured GRPO ablation grid with identical seed/split protocol."""
from __future__ import annotations
import argparse, itertools, json, subprocess, sys
from pathlib import Path
import numpy as np
import yaml

def main():
    p = argparse.ArgumentParser(); p.add_argument("--base-config", default="configs/base.yaml"); p.add_argument("--ablation-config", default="configs/ablations.yaml"); p.add_argument("--output-dir", default="artifacts/ablations")
    args = p.parse_args()
    base = yaml.safe_load(Path(args.base_config).read_text())
    grid = yaml.safe_load(Path(args.ablation_config).read_text())
    root = Path(args.output_dir); root.mkdir(parents=True, exist_ok=True)
    results = []
    for values in itertools.product(*grid.values()):
        setting = dict(zip(grid.keys(), values))
        name = "_".join(f"{key}-{str(value).lower()}" for key, value in setting.items())
        cfg = {**base, **setting}
        cfg_path = root / f"{name}.yaml"; cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        scores = []
        for seed in base["seeds"]:
            subprocess.run([sys.executable, "train.py", "--config", str(cfg_path), "--algorithm", "grpo", "--seed", str(seed), "--output-dir", str(root / name)], check=True)
            metrics = json.loads((root / name / f"grpo_seed{seed}" / "metrics.json").read_text())
            scores.append(metrics["final"]["exact_match"])
        results.append({**setting, "mean_exact_match": float(np.mean(scores)), "std_exact_match": float(np.std(scores)), "n_seeds": len(scores)})
    (root / "summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

if __name__ == "__main__": main()
