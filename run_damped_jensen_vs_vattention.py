#!/usr/bin/env python3
"""
Full RULER32K Benchmark Comparison:
UTAJensen_Damped (alpha=0.25) vs vAttention at 10% density across all 7 subsets (200 samples each).

Usage:
  python run_damped_jensen_vs_vattention.py <device> <method_type>
  e.g.
  python run_damped_jensen_vs_vattention.py cuda:0 damped_jensen
  python run_damped_jensen_vs_vattention.py cuda:1 vattention
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
from sparse_attention_hub.sparse_attention.uta_attention import UTAJensenConfig
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    SinkMaskerConfig,
    LocalMaskerConfig,
    OracleTopKConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
    AdaptiveSamplingMaskerConfig,
)
from benchmark.ruler32k import Ruler32K

SUBSETS = ["qa_1", "qa_2", "vt", "fwe", "niah_multikey_2", "niah_multikey_3", "niah_multivalue"]
OUTPUT_DIR = Path("results_damped_jensen_full")


def get_config(method_type):
    if method_type == "damped_jensen":
        # UTAJensen with alpha=0.25 (10% density)
        return "UTAJensen_Damped_10pct", UTAJensenConfig(
            alpha=0.25,
            masker_configs=[
                SinkMaskerConfig(sink_size=0.001),
                LocalMaskerConfig(window_size=0.001),
                OracleTopKConfig(heavy_size=0.098),
            ]
        )
    elif method_type == "vattention":
        # vAttention (10% density: 4.8% top-k + 5% adaptive sampling)
        return "vAttention_10pct", ResearchAttentionConfig(
            masker_configs=[
                SinkMaskerConfig(sink_size=0.001),
                LocalMaskerConfig(window_size=0.001),
                OracleTopKConfig(heavy_size=0.048),
                AdaptiveSamplingMaskerConfig(
                    base_rate_sampling=0.05,
                    epsilon=0.25,
                    delta=0.25,
                    init_offset=0.001,
                    local_offset=0.001,
                ),
            ]
        )
    else:
        raise ValueError(f"Unknown method_type: {method_type}")


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
            print(f"  [SKIP] {method_name}/{subset} — already completed")
            continue

        subset_dir.mkdir(exist_ok=True, parents=True)
        print(f"\n  [{device}] Starting benchmark for {method_name} on subset: {subset} (all 200 samples)...")

        benchmark = Ruler32K(subsets_to_run=[subset])
        benchmark.run_benchmark(
            adapter,
            str(subset_dir),
            request_kwargs={"max_context_length": 32000},
            generation_kwargs={"max_new_tokens": 120},
        )

        metrics_file = subset_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                data = json.load(f)
                score = data.get("task_scores", {}).get(subset, {}).get("string_match", 0.0)
                print(f"  ✓ Finished {method_name}/{subset} — string_match: {score:.2f}")

    # Print summary of all subsets
    print(f"\n{'='*60}")
    print(f"Summary for {method_name}:")
    scores = {}
    for subset in SUBSETS:
        metrics_file = method_dir / subset / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                data = json.load(f)
                score = data.get("task_scores", {}).get(subset, {}).get("string_match", 0.0)
                scores[subset] = round(score, 2)
                print(f"  {subset:18s}: {score:.2f}")
    if len(scores) == len(SUBSETS):
        avg = round(sum(scores.values()) / len(scores), 2)
        print(f"  {'AVERAGE':18s}: {avg:.2f}")
    print(f"{'='*60}\n")

    del adapter
    torch.cuda.empty_cache()


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    method_type = sys.argv[2] if len(sys.argv) > 2 else "damped_jensen"

    method_name, config = get_config(method_type)
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"\n=== Launching {method_name} benchmark on {device} ===\n")
    run_method(method_name, config, device)


if __name__ == "__main__":
    main()
