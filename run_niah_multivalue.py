#!/usr/bin/env python3
"""
Run niah_multivalue subset only for all 5 methods,
then combine with existing 6 HARD subsets to produce the corrected Table 1.
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

HAT_WEIGHT_FILE = "/database/hyunwoo/hf/HashAttention-1.0/repo/artifacts/llama3.1-8b-patch.64K.v1.hat_weights.pkl"

# Corrected RULER32K-HARD 7 subsets
RULER32K_HARD_SUBSETS = [
    "qa_1", "qa_2", "vt", "fwe",
    "niah_multikey_2", "niah_multikey_3", "niah_multivalue",
]

def build_sparse_configs():
    configs = {}
    configs["SDPA"] = None

    configs["oracle-top-k"] = ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=0.001),
            LocalMaskerConfig(window_size=0.001),
            OracleTopKConfig(heavy_size=0.098),
        ]
    )

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

    configs["HAT"] = ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=0.001),
            LocalMaskerConfig(window_size=0.001),
            HashAttentionTopKMaskerConfig(
                heavy_size=0.098, hat_bits=32, hat_mlp_layers=3,
                hat_mlp_hidden_size=128, hat_mlp_activation="silu",
                hat_weight_file=HAT_WEIGHT_FILE,
            ),
        ]
    )

    configs["vAttention(HAT)"] = ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=0.001),
            LocalMaskerConfig(window_size=0.001),
            HashAttentionTopKMaskerConfig(
                heavy_size=0.048, hat_bits=32, hat_mlp_layers=3,
                hat_mlp_hidden_size=128, hat_mlp_activation="silu",
                hat_weight_file=HAT_WEIGHT_FILE,
            ),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05, epsilon=0.4, delta=0.4,
                init_offset=0.001, local_offset=0.001,
            ),
        ]
    )
    return configs


def safe_method_dir(name):
    return name.replace("(", "_").replace(")", "")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--max-requests-per-subset", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="./results_table1_full")
    args = parser.parse_args()

    base_out_dir = Path(args.output_dir)
    configs = build_sparse_configs()

    # ── Step 1: Run niah_multivalue only ──
    missing_subset = "niah_multivalue"
    print("=" * 80)
    print(f"Running MISSING subset '{missing_subset}' for all 5 methods")
    print("=" * 80)

    for method_name, sparse_config in configs.items():
        method_dir = base_out_dir / safe_method_dir(method_name)
        subset_dir = method_dir / missing_subset

        # Skip if already exists
        if (subset_dir / "metrics.json").exists():
            print(f"  [SKIP] {method_name}/{missing_subset} already exists")
            continue

        print(f"\n  [{method_name}] Running {missing_subset}...")
        subset_dir.mkdir(exist_ok=True, parents=True)

        adapter = ModelAdapterHF(
            model_name=args.model,
            sparse_attention_config=sparse_config,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=args.device,
        )

        benchmark = Ruler32K(subsets_to_run=[missing_subset])
        benchmark.run_benchmark(
            adapter,
            str(subset_dir),
            request_kwargs={"max_requests": args.max_requests_per_subset, "max_context_length": 32000},
            generation_kwargs={"max_new_tokens": 120},
        )
        print(f"  [{method_name}] ✓ {missing_subset} done")

    # ── Step 2: Combine all 7 correct HARD subsets ──
    print("\n" + "=" * 80)
    print("CORRECTED TABLE 1 (RULER32K-HARD @ 10% Sparsity)")
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
    df.to_csv(base_out_dir / "table1_corrected_summary.csv", index=False)
    print(f"Saved to {base_out_dir / 'table1_corrected_summary.csv'}")

    # Print comparison with paper
    paper = {
        "SDPA": 88.74, "oracle-top-k": 87.18,
        "vAttention(oracle-top-k)": 88.61, "HAT": 81.94, "vAttention(HAT)": 86.56,
    }
    print("\n--- Paper vs Reproduced ---")
    for _, r in df.iterrows():
        m = r["Method"]
        ours = r["Average (%)"]
        theirs = paper.get(m, "?")
        print(f"  {m:30s}  Paper: {theirs}  |  Ours: {ours}  |  Δ = {float(ours) - float(theirs):+.2f}")


if __name__ == "__main__":
    main()
