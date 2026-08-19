#!/usr/bin/env python3
"""
UTA-Jensen Benchmark: Evaluate pure Jensen gap corrected UTA (UTAJensen)
where proxy_logit = z_bar + variance/2 + log(N), but numerator value is v_mean.

Methods evaluated:
  - UTA_Jensen@10pct
  - UTA_Jensen@5pct
  - UTA_Jensen@3pct

Usage:
  python run_uta_jensen_bench.py <device> <density_label>
  e.g.
  python run_uta_jensen_bench.py cuda:0 10pct
"""

import os
import sys
import json
import torch
from pathlib import Path

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

from sparse_attention_hub.adapters import ModelAdapterHF
from sparse_attention_hub.sparse_attention.uta_attention import (
    UTAAttentionConfig,
    UTAJensenConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    SinkMaskerConfig,
    LocalMaskerConfig,
    OracleTopKConfig,
)
from benchmark.ruler32k import Ruler32K

SUBSETS = ["qa_1", "qa_2", "vt", "fwe", "niah_multikey_2", "niah_multikey_3", "niah_multivalue"]
OUTPUT_DIR = Path("results_jensen")

DENSITY_MAP = {
    "10pct": 0.098,
    "5pct": 0.048,
    "3pct": 0.028,
}

def safe_name(name):
    return name.replace("@", "_at_").replace("-", "_")

def run_method(method_name, config, device):
    method_dir = OUTPUT_DIR / safe_name(method_name)
    adapter = ModelAdapterHF(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        sparse_attention_config=config,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=device,
    )

    for subset in SUBSETS:
        subset_dir = method_dir / subset
        if (subset_dir / "metrics.json").exists():
            print(f"  [SKIP] {method_name}/{subset} — already done")
            continue

        subset_dir.mkdir(exist_ok=True, parents=True)
        print(f"  [{device}] Running {method_name}/{subset}...")
        benchmark = Ruler32K(subsets_to_run=[subset])
        benchmark.run_benchmark(
            adapter,
            str(subset_dir),
            request_kwargs={"max_context_length": 32000},
            generation_kwargs={"max_new_tokens": 120},
        )

    # Collect results
    print(f"\n{'='*60}")
    print(f"Results for {method_name}:")
    scores = {}
    for subset in SUBSETS:
        metrics_file = method_dir / subset / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                data = json.load(f)
                score = data.get("task_scores", {}).get(subset, {}).get("string_match", 0.0)
                scores[subset] = round(score, 2)
                print(f"  {subset}: {score}")
    if len(scores) == len(SUBSETS):
        avg = round(sum(scores.values()) / len(scores), 2)
        print(f"  Average: {avg}")
    print(f"{'='*60}\n")
    del adapter
    torch.cuda.empty_cache()


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    density_label = sys.argv[2] if len(sys.argv) > 2 else "10pct"

    if density_label not in DENSITY_MAP:
        print(f"Unknown density label: {density_label}. Must be one of {list(DENSITY_MAP.keys())}")
        sys.exit(1)

    topk_size = DENSITY_MAP[density_label]
    OUTPUT_DIR.mkdir(exist_ok=True)

    method_name = f"UTA_Jensen@{density_label}"
    config = UTAJensenConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.001),
        LocalMaskerConfig(window_size=0.001),
        OracleTopKConfig(heavy_size=topk_size),
    ])

    print(f"\n=== Running {method_name} on {device} ===\n")
    run_method(method_name, config, device)


if __name__ == "__main__":
    main()
