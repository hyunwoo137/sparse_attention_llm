#!/usr/bin/env python3
"""
Table 1 Reproduction Script (vAttention Paper)
RULER 32K-HARD Benchmark at 10% Sparsity
Target Model: Llama-3.1-8B-Instruct

Methods evaluated:
1. SDPA (Full Dense baseline)
2. oracle-top-k (10% sparsity)
3. vAttention(oracle-top-k) (10% sparsity: 5% Oracle Top-K + 5% Adaptive Sampling)
4. HAT / HashAttention (10% sparsity)
5. vAttention(HAT) (10% sparsity: 5% HAT Top-K + 5% Adaptive Sampling)
"""

import os
import sys
import json
import torch
import pandas as pd
from pathlib import Path

# Setup paths
repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

from sparse_attention_hub.adapters import ModelAdapterHF
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    SinkMaskerConfig,
    LocalMaskerConfig,
    OracleTopKConfig,
    HashAttentionTopKMaskerConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
    AdaptiveSamplingMaskerConfig,
)
from benchmark.ruler32k import Ruler32K

# Weight file for HashAttention (Llama-3.1-8B)
HAT_WEIGHT_FILE = "/database/hyunwoo/hf/HashAttention-1.0/repo/artifacts/llama3.1-8b-patch.64K.v1.hat_weights.pkl"

# RULER32K-HARD 7 Subsets
RULER32K_HARD_SUBSETS = [
    "cwe",
    "fwe",
    "vt",
    "qa_1",
    "qa_2",
    "niah_multikey_2",
    "niah_multikey_3",
]

def build_sparse_configs():
    configs = {}

    # 1. SDPA (Full Dense)
    configs["SDPA"] = None

    # 2. oracle-top-k (10% sparsity: 0.1% sink + 0.1% local + 9.8% oracle)
    configs["oracle-top-k"] = ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=0.001),
            LocalMaskerConfig(window_size=0.001),
            OracleTopKConfig(heavy_size=0.098),
        ]
    )

    # 3. vAttention(oracle-top-k) (10% sparsity: 5% oracle + 5% adaptive sampling)
    configs["vAttention(oracle-top-k)"] = ResearchAttentionConfig(
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

    # 4. HAT / HashAttention (10% sparsity: 0.1% sink + 0.1% local + 9.8% HAT)
    configs["HAT"] = ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=0.001),
            LocalMaskerConfig(window_size=0.001),
            HashAttentionTopKMaskerConfig(
                heavy_size=0.098,
                hat_bits=32,
                hat_mlp_layers=3,
                hat_mlp_hidden_size=128,
                hat_mlp_activation="silu",
                hat_weight_file=HAT_WEIGHT_FILE,
            ),
        ]
    )

    # 5. vAttention(HAT) (10% sparsity: 5% HAT + 5% adaptive sampling)
    configs["vAttention(HAT)"] = ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=0.001),
            LocalMaskerConfig(window_size=0.001),
            HashAttentionTopKMaskerConfig(
                heavy_size=0.048,
                hat_bits=32,
                hat_mlp_layers=3,
                hat_mlp_hidden_size=128,
                hat_mlp_activation="silu",
                hat_weight_file=HAT_WEIGHT_FILE,
            ),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,
                epsilon=0.4,
                delta=0.4,
                init_offset=0.001,
                local_offset=0.001,
            ),
        ]
    )

    return configs

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Reproduce Table 1 (RULER32K-HARD 10% Sparsity)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--max-requests-per-subset", type=int, default=50, help="Number of samples per subset (default: 50)")
    parser.add_argument("--output-dir", type=str, default="./results_table1_full")
    args = parser.parse_args()

    base_out_dir = Path(args.output_dir)
    base_out_dir.mkdir(exist_ok=True, parents=True)

    configs = build_sparse_configs()

    print("=" * 100)
    print(f"Table 1 Reproduction Suite | RULER32K-HARD (10% Sparsity)")
    print(f"Model: {args.model} | Device: {args.device}")
    print(f"Subsets ({len(RULER32K_HARD_SUBSETS)}): {RULER32K_HARD_SUBSETS}")
    print(f"Samples per subset: {args.max_requests_per_subset}")
    print("=" * 100)

    all_method_rows = []

    for method_name, sparse_config in configs.items():
        print(f"\n[{method_name}] Starting Evaluation across 7 RULER32K-HARD subsets...")
        method_out_dir = base_out_dir / method_name.replace("(", "_").replace(")", "")
        method_out_dir.mkdir(exist_ok=True, parents=True)

        adapter = ModelAdapterHF(
            model_name=args.model,
            sparse_attention_config=sparse_config,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=args.device,
        )

        subset_scores = {}
        all_raw_dfs = []

        for subset in RULER32K_HARD_SUBSETS:
            subset_out_dir = method_out_dir / subset
            subset_out_dir.mkdir(exist_ok=True, parents=True)

            print(f"  -> Running subset: {subset}")
            benchmark = Ruler32K(subsets_to_run=[subset])
            benchmark.run_benchmark(
                adapter,
                str(subset_out_dir),
                request_kwargs={
                    "max_requests": args.max_requests_per_subset,
                    "max_context_length": 32000,
                },
                generation_kwargs={
                    "max_new_tokens": 120,
                },
            )

            # Load metrics for subset
            metrics_file = subset_out_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    metrics_data = json.load(f)
                    task_scores = metrics_data.get("task_scores", {}).get(subset, {})
                    score = task_scores.get("string_match", 0.0)
                    subset_scores[subset] = round(score, 2)
                    print(f"     ✓ {subset}: {score:.2f}%")

            # Collect raw results
            raw_csv = subset_out_dir / "raw_results.csv"
            if raw_csv.exists():
                all_raw_dfs.append(pd.read_csv(raw_csv))

        # Merge raw results for method
        if all_raw_dfs:
            combined_raw = pd.concat(all_raw_dfs, ignore_index=True)
            combined_raw.to_csv(method_out_dir / "raw_results.csv", index=False)

        # Calculate average
        avg_score = round(sum(subset_scores.values()) / max(1, len(subset_scores)), 2)
        print(f"[{method_name}] RULER32K-HARD Average Score: {avg_score:.2f}% | Subsets: {subset_scores}")

        row = {"Method": method_name, "Average Score (%)": avg_score}
        row.update(subset_scores)
        all_method_rows.append(row)

    print("\n" + "=" * 100)
    print("FINAL REPRODUCED TABLE 1 RESULTS (Llama-3.1-8B-Instruct @ 10% Sparsity):")
    print("=" * 100)
    final_df = pd.DataFrame(all_method_rows)
    print(final_df.to_string(index=False))
    print("=" * 100)

    # Save summary table
    final_df.to_csv(base_out_dir / "table1_reproduction_summary.csv", index=False)
    print(f"Summary table saved to {base_out_dir / 'table1_reproduction_summary.csv'}")

if __name__ == "__main__":
    main()
