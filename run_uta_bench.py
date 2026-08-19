#!/usr/bin/env python3
"""
Run RULER32K-HARD benchmarks comparing vAttention vs UTA (Unified Tail Aggregation).

Both methods share the same Sink + Local + OracleTopK masker pipeline at ~10% sparsity.
The difference is how they handle tail (unselected) tokens:
  - vAttention: Adaptive sampling of tail tokens
  - UTA: Mean-pooled proxy aggregation with log(N) compensation

Subsets: qa_1, qa_2, vt, fwe, niah_multikey_2, niah_multikey_3, niah_multivalue
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
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
    AdaptiveSamplingMaskerConfig,
)
from benchmark.ruler32k import Ruler32K


RULER32K_HARD_SUBSETS = [
    "qa_1", "qa_2", "vt", "fwe",
    "niah_multikey_2", "niah_multikey_3", "niah_multivalue",
]


def build_comparison_configs():
    """Build configs for vAttention and UTA with matched sparsity."""
    configs = {}

    # Baseline: Dense SDPA (no sparsity)
    configs["SDPA"] = None

    # vAttention(oracle-top-k): Sink + Local + OracleTopK(4.8%) + AdaptiveSampling(5.2%)
    # Total ~10% sparsity budget
    configs["vAttention(oracle-top-k)"] = ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=0.001),
            LocalMaskerConfig(window_size=0.001),
            OracleTopKConfig(heavy_size=0.048),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05, epsilon=0.25, delta=0.25,
                init_offset=0.001, local_offset=0.001,
            ),
        ]
    )

    # UTA(oracle-top-k): Sink + Local + OracleTopK(9.8%) + UTA tail aggregation
    # Same total mask budget as oracle-top-k baseline
    configs["UTA(oracle-top-k)"] = UTAAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=0.001),
            LocalMaskerConfig(window_size=0.001),
            OracleTopKConfig(heavy_size=0.098),
        ]
    )

    return configs


def safe_method_dir(name):
    """Convert method name to filesystem-safe directory name."""
    return name.replace("(", "_").replace(")", "")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare vAttention vs UTA on RULER32K-HARD")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--max-requests-per-subset", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="./results_uta_comparison")
    args = parser.parse_args()

    base_out_dir = Path(args.output_dir)
    configs = build_comparison_configs()

    print("=" * 80)
    print("vAttention vs UTA Comparison on RULER32K-HARD")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Subsets: {RULER32K_HARD_SUBSETS}")
    print("=" * 80)

    # ── Run each method on each subset ──
    for method_name, sparse_config in configs.items():
        print(f"\n{'─' * 60}")
        print(f"Method: {method_name}")
        print(f"{'─' * 60}")

        # Load model once per method
        adapter = ModelAdapterHF(
            model_name=args.model,
            sparse_attention_config=sparse_config,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=args.device,
        )

        for subset in RULER32K_HARD_SUBSETS:
            method_dir = base_out_dir / safe_method_dir(method_name)
            subset_dir = method_dir / subset

            # Skip if already completed
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

        # Free GPU memory
        del adapter
        torch.cuda.empty_cache()

    # ── Build comparison table ──
    print("\n" + "=" * 80)
    print("RESULTS: vAttention vs UTA (RULER32K-HARD @ 10% Sparsity)")
    print(f"Subsets: {RULER32K_HARD_SUBSETS}")
    print("=" * 80)

    all_rows = []
    for method_name in configs.keys():
        method_dir = base_out_dir / safe_method_dir(method_name)
        row = {"Method": method_name}
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

    # Save
    summary_path = base_out_dir / "comparison_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nSaved to {summary_path}")

    # Include existing results from results_table1_full for full context
    existing_csv = repo_root / "results_table1_full" / "table1_corrected_summary.csv"
    if existing_csv.exists():
        print("\n--- Existing Table 1 Results (for reference) ---")
        existing_df = pd.read_csv(existing_csv)
        print(existing_df.to_string(index=False))


if __name__ == "__main__":
    main()
