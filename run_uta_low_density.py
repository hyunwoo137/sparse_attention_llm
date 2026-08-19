#!/usr/bin/env python3
"""
Compare oracle-top-k vs UTA at lower density levels (5%, 3%).

At lower sparsity, oracle-top-k loses more information from tail tokens,
so UTA's mean-pooled proxy should provide a larger benefit.
"""

import os
import sys
import json
import torch
import pandas as pd
from pathlib import Path

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

from sparse_attention_hub.adapters import ModelAdapterHF
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.uta_attention import UTAAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    SinkMaskerConfig,
    LocalMaskerConfig,
    OracleTopKConfig,
)
from benchmark.ruler32k import Ruler32K


RULER32K_HARD_SUBSETS = [
    "qa_1", "qa_2", "vt", "fwe",
    "niah_multikey_2", "niah_multikey_3", "niah_multivalue",
]

# Density levels to test: 5% and 3%
# Each level: (name, top-k heavy_size, total_approx_density)
DENSITY_LEVELS = [
    ("5pct", 0.048, "~5%"),
    ("3pct", 0.028, "~3%"),
]


def build_configs_for_density(heavy_size):
    """Build oracle-top-k and UTA configs for a given density level."""
    configs = {}

    # oracle-top-k (no UTA): just sparse mask, no tail recovery
    configs["oracle-top-k"] = ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=0.001),
            LocalMaskerConfig(window_size=0.001),
            OracleTopKConfig(heavy_size=heavy_size),
        ]
    )

    # UTA: same mask + tail aggregation
    configs["UTA"] = UTAAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=0.001),
            LocalMaskerConfig(window_size=0.001),
            OracleTopKConfig(heavy_size=heavy_size),
        ]
    )

    return configs


def safe_method_dir(name):
    return name.replace("(", "_").replace(")", "")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare oracle-top-k vs UTA at low density")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--max-requests-per-subset", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="./results_uta_low_density")
    args = parser.parse_args()

    base_out_dir = Path(args.output_dir)

    print("=" * 80)
    print("oracle-top-k vs UTA at Low Density")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Density levels: {[(d[0], d[2]) for d in DENSITY_LEVELS]}")
    print("=" * 80)

    for density_name, heavy_size, density_label in DENSITY_LEVELS:
        configs = build_configs_for_density(heavy_size)

        print(f"\n{'=' * 60}")
        print(f"Density: {density_label} (heavy_size={heavy_size})")
        print(f"{'=' * 60}")

        for method_name, sparse_config in configs.items():
            full_method_name = f"{method_name}@{density_name}"
            print(f"\n{'─' * 40}")
            print(f"Method: {full_method_name}")
            print(f"{'─' * 40}")

            adapter = ModelAdapterHF(
                model_name=args.model,
                sparse_attention_config=sparse_config,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device=args.device,
            )

            for subset in RULER32K_HARD_SUBSETS:
                method_dir = base_out_dir / density_name / safe_method_dir(method_name)
                subset_dir = method_dir / subset

                if (subset_dir / "metrics.json").exists():
                    print(f"  [SKIP] {subset} already exists")
                    continue

                subset_dir.mkdir(exist_ok=True, parents=True)
                print(f"  Running {subset}...")

                benchmark = Ruler32K(subsets_to_run=[subset])
                benchmark.run_benchmark(
                    adapter,
                    str(subset_dir),
                    request_kwargs={
                        "max_requests": args.max_requests_per_subset,
                        "max_context_length": 32000,
                    },
                    generation_kwargs={"max_new_tokens": 120},
                )
                print(f"  ✓ {subset} done")

            del adapter
            torch.cuda.empty_cache()

    # ── Build comparison tables ──
    print("\n" + "=" * 80)
    print("RESULTS: oracle-top-k vs UTA at Low Density")
    print("=" * 80)

    all_rows = []
    for density_name, heavy_size, density_label in DENSITY_LEVELS:
        for method_name in ["oracle-top-k", "UTA"]:
            method_dir = base_out_dir / density_name / safe_method_dir(method_name)
            row = {"Method": f"{method_name}@{density_label}"}
            scores = []

            for subset in RULER32K_HARD_SUBSETS:
                metrics_file = method_dir / subset / "metrics.json"
                if metrics_file.exists():
                    with open(metrics_file) as f:
                        m = json.load(f)
                    score = m.get("task_scores", {}).get(subset, {}).get("string_match", 0.0)
                    row[subset] = round(score, 2)
                    scores.append(score)
                else:
                    row[subset] = "N/A"

            if scores:
                row["Average (%)"] = round(sum(scores) / len(scores), 2)
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    cols = ["Method", "Average (%)"] + RULER32K_HARD_SUBSETS
    df = df[cols]
    print(df.to_string(index=False))
    print("=" * 80)

    summary_path = base_out_dir / "low_density_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nSaved to {summary_path}")

    # Also include 10% results for reference
    existing = Path("results_uta_comparison/comparison_summary.csv")
    if existing.exists():
        print("\n--- 10% Density Results (for reference) ---")
        print(pd.read_csv(existing).to_string(index=False))


if __name__ == "__main__":
    main()
