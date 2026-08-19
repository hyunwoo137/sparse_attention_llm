#!/usr/bin/env python3
"""
Task-Contrastive Diagnostic Analysis:
Compare UTA (alpha=0.0) vs UTAJensen-Damped (alpha=0.25) vs UTAJensen-Full (alpha=1.0)
across degraded QA tasks (qa_1, qa_2) vs improved aggregation tasks (fwe, niah_multikey_2).

Usage:
  python run_jensen_diagnostic.py <device> <task_group>
  task_group: "qa" (qa_1, qa_2) or "agg" (fwe, niah_multikey_2)
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

OUTPUT_DIR = Path("results_jensen_diagnostic")

TASK_GROUPS = {
    "qa": ["qa_1", "qa_2"],
    "agg": ["fwe", "niah_multikey_2"],
}

METHODS = {
    "UTA_Base": UTAAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.001),
        LocalMaskerConfig(window_size=0.001),
        OracleTopKConfig(heavy_size=0.098),
    ]),
    "UTAJensen_Damped_0.25": UTAJensenConfig(alpha=0.25, masker_configs=[
        SinkMaskerConfig(sink_size=0.001),
        LocalMaskerConfig(window_size=0.001),
        OracleTopKConfig(heavy_size=0.098),
    ]),
    "UTAJensen_Full_1.0": UTAJensenConfig(alpha=1.0, masker_configs=[
        SinkMaskerConfig(sink_size=0.001),
        LocalMaskerConfig(window_size=0.001),
        OracleTopKConfig(heavy_size=0.098),
    ]),
}


def run_subset_diagnostic(method_name, config, subset, device):
    subset_dir = OUTPUT_DIR / method_name / subset
    if (subset_dir / "metrics.json").exists():
        print(f"  [SKIP] {method_name}/{subset} — already done")
        return

    subset_dir.mkdir(exist_ok=True, parents=True)
    print(f"  [{device}] Running {method_name} on {subset} (50 samples)...")

    adapter = ModelAdapterHF(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        sparse_attention_config=config,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=device,
    )

    benchmark = Ruler32K(subsets_to_run=[subset])
    benchmark.run_benchmark(
        adapter,
        str(subset_dir),
        request_kwargs={"max_context_length": 32000, "max_requests": 50},
        generation_kwargs={"max_new_tokens": 120},
    )

    metrics_file = subset_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            data = json.load(f)
            score = data.get("task_scores", {}).get(subset, {}).get("string_match", 0.0)
            print(f"  [{method_name} / {subset}] string_match = {score:.2f}")

    del adapter
    torch.cuda.empty_cache()


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    group_name = sys.argv[2] if len(sys.argv) > 2 else "qa"

    if group_name not in TASK_GROUPS:
        print(f"Unknown group: {group_name}. Must be 'qa' or 'agg'")
        sys.exit(1)

    subsets = TASK_GROUPS[group_name]
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"\n=== Starting Diagnostic Run for Group [{group_name}: {subsets}] on {device} ===\n")

    for subset in subsets:
        for method_name, config in METHODS.items():
            run_subset_diagnostic(method_name, config, subset, device)

    print(f"\n=== Completed Diagnostic Run for Group [{group_name}] ===\n")


if __name__ == "__main__":
    main()
