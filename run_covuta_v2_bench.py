#!/usr/bin/env python3
"""CovUTA v2 Quick Benchmark: Test the fixed implementation at 10% density.

Usage:
  python run_covuta_v2_bench.py <device> <method>
  
  device: cuda:0 or cuda:1
  method: covuta or indep
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
    CovUTAConfig,
    CovUTAIndependentConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    SinkMaskerConfig,
    LocalMaskerConfig,
    OracleTopKConfig,
)
from benchmark.ruler32k import Ruler32K

SUBSETS = ["qa_1", "qa_2", "vt", "fwe", "niah_multikey_2", "niah_multikey_3", "niah_multivalue"]
OUTPUT_DIR = Path("results_covuta_v2")

def run_method(method_name, config, device):
    method_dir = OUTPUT_DIR / method_name
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


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    method = sys.argv[2] if len(sys.argv) > 2 else "covuta"

    OUTPUT_DIR.mkdir(exist_ok=True)

    if method == "covuta":
        config = CovUTAConfig(masker_configs=[
            SinkMaskerConfig(sink_size=0.001),
            LocalMaskerConfig(window_size=0.001),
            OracleTopKConfig(heavy_size=0.098),
        ])
        run_method("CovUTA_v2_at_10pct", config, device)
    elif method == "indep":
        config = CovUTAIndependentConfig(masker_configs=[
            SinkMaskerConfig(sink_size=0.001),
            LocalMaskerConfig(window_size=0.001),
            OracleTopKConfig(heavy_size=0.098),
        ])
        run_method("CovUTA_Indep_v2_at_10pct", config, device)
    else:
        print(f"Unknown method: {method}")
        sys.exit(1)


if __name__ == "__main__":
    main()
