#!/usr/bin/env python3
"""
Full benchmark: vAttention vs UTA variants at multiple density levels.

Runs on 2 GPUs in parallel:
  - GPU 0: oracle-top-k, UTA-naive, UTA-LogNormal
  - GPU 1: vAttention, CV-UTA

Uses ALL samples (200 per subset).
"""

import os
import sys
import json
import torch
import subprocess
import pandas as pd
from pathlib import Path
from dataclasses import asdict

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

from sparse_attention_hub.adapters import ModelAdapterHF
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.uta_attention import (
    UTAAttentionConfig,
    UTALogNormalConfig,
    CVUTAConfig,
)
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

# ============================================================================
# Configs for each method × density
# ============================================================================

def build_all_configs():
    """Build all method configs for each density level.

    For fair comparison, we match total density across methods.
    vAttention splits budget between top-k and adaptive sampling.
    UTA variants put all budget into top-k (since proxy is free).

    For vAttention: sink(0.1%) + local(0.1%) + top-k(half) + adaptive(half)
    For UTA/LN/CV: sink(0.1%) + local(0.1%) + top-k(full budget)
    """

    configs = {}

    # --- 10% density ---
    d = "10pct"

    configs[f"vAttention@{d}"] = ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.001),
        LocalMaskerConfig(window_size=0.001),
        OracleTopKConfig(heavy_size=0.048),
        AdaptiveSamplingMaskerConfig(
            base_rate_sampling=0.05, epsilon=0.25, delta=0.25,
            init_offset=0.001, local_offset=0.001,
        ),
    ])
    configs[f"oracle-top-k@{d}"] = ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.001),
        LocalMaskerConfig(window_size=0.001),
        OracleTopKConfig(heavy_size=0.098),
    ])
    configs[f"UTA@{d}"] = UTAAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.001),
        LocalMaskerConfig(window_size=0.001),
        OracleTopKConfig(heavy_size=0.098),
    ])
    configs[f"UTA-LN@{d}"] = UTALogNormalConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.001),
        LocalMaskerConfig(window_size=0.001),
        OracleTopKConfig(heavy_size=0.098),
    ])
    configs[f"CV-UTA@{d}"] = CVUTAConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=0.001),
            LocalMaskerConfig(window_size=0.001),
            OracleTopKConfig(heavy_size=0.048),
        ],
        cv_sample_rate=0.05,
        cv_min_samples=4,
    )

    # --- 5% density ---
    d = "5pct"

    configs[f"vAttention@{d}"] = ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.001),
        LocalMaskerConfig(window_size=0.001),
        OracleTopKConfig(heavy_size=0.023),
        AdaptiveSamplingMaskerConfig(
            base_rate_sampling=0.025, epsilon=0.25, delta=0.25,
            init_offset=0.001, local_offset=0.001,
        ),
    ])
    configs[f"oracle-top-k@{d}"] = ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.001),
        LocalMaskerConfig(window_size=0.001),
        OracleTopKConfig(heavy_size=0.048),
    ])
    configs[f"UTA@{d}"] = UTAAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.001),
        LocalMaskerConfig(window_size=0.001),
        OracleTopKConfig(heavy_size=0.048),
    ])
    configs[f"UTA-LN@{d}"] = UTALogNormalConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.001),
        LocalMaskerConfig(window_size=0.001),
        OracleTopKConfig(heavy_size=0.048),
    ])
    configs[f"CV-UTA@{d}"] = CVUTAConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=0.001),
            LocalMaskerConfig(window_size=0.001),
            OracleTopKConfig(heavy_size=0.023),
        ],
        cv_sample_rate=0.025,
        cv_min_samples=4,
    )

    # --- 3% density ---
    d = "3pct"

    configs[f"vAttention@{d}"] = ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.001),
        LocalMaskerConfig(window_size=0.001),
        OracleTopKConfig(heavy_size=0.013),
        AdaptiveSamplingMaskerConfig(
            base_rate_sampling=0.015, epsilon=0.25, delta=0.25,
            init_offset=0.001, local_offset=0.001,
        ),
    ])
    configs[f"oracle-top-k@{d}"] = ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.001),
        LocalMaskerConfig(window_size=0.001),
        OracleTopKConfig(heavy_size=0.028),
    ])
    configs[f"UTA@{d}"] = UTAAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.001),
        LocalMaskerConfig(window_size=0.001),
        OracleTopKConfig(heavy_size=0.028),
    ])
    configs[f"UTA-LN@{d}"] = UTALogNormalConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.001),
        LocalMaskerConfig(window_size=0.001),
        OracleTopKConfig(heavy_size=0.028),
    ])
    configs[f"CV-UTA@{d}"] = CVUTAConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=0.001),
            LocalMaskerConfig(window_size=0.001),
            OracleTopKConfig(heavy_size=0.013),
        ],
        cv_sample_rate=0.015,
        cv_min_samples=4,
    )

    return configs


def safe_name(name):
    return name.replace("@", "_at_").replace("-", "_").replace("(", "").replace(")", "")


def run_method(method_name, config, device, output_dir, max_requests=None):
    """Run a single method across all subsets."""
    method_dir = output_dir / safe_name(method_name)

    adapter = ModelAdapterHF(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        sparse_attention_config=config,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=device,
    )

    for subset in SUBSETS:
        subset_dir = method_dir / subset
        if (subset_dir / "metrics.json").exists():
            print(f"  [SKIP] {method_name}/{subset}")
            continue

        subset_dir.mkdir(exist_ok=True, parents=True)
        print(f"  [{device}] Running {method_name}/{subset}...")

        req_kwargs = {"max_context_length": 32000}
        if max_requests is not None:
            req_kwargs["max_requests"] = max_requests

        benchmark = Ruler32K(subsets_to_run=[subset])
        benchmark.run_benchmark(
            adapter, str(subset_dir),
            request_kwargs=req_kwargs,
            generation_kwargs={"max_new_tokens": 120},
        )
        print(f"  ✓ {method_name}/{subset} done")

    del adapter
    torch.cuda.empty_cache()


def collect_results(output_dir):
    """Collect all results into a summary table."""
    all_rows = []

    for method_dir in sorted(output_dir.iterdir()):
        if not method_dir.is_dir() or method_dir.name.startswith('.'):
            continue

        row = {"Method": method_dir.name}
        scores = []
        for subset in SUBSETS:
            mf = method_dir / subset / "metrics.json"
            if mf.exists():
                with open(mf) as f:
                    m = json.load(f)
                s = m.get("task_scores", {}).get(subset, {}).get("string_match", 0.0)
                row[subset] = round(s, 2)
                scores.append(s)
            else:
                row[subset] = "N/A"

        if scores:
            row["Average (%)"] = round(sum(scores) / len(scores), 2)
        all_rows.append(row)

    if all_rows:
        df = pd.DataFrame(all_rows)
        cols = ["Method", "Average (%)"] + SUBSETS
        df = df[[c for c in cols if c in df.columns]]
        return df
    return pd.DataFrame()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="GPU device to use")
    parser.add_argument("--methods", type=str, nargs="+", default=None,
                        help="Specific methods to run (e.g., 'vAttention@10pct')")
    parser.add_argument("--max-requests", type=int, default=None,
                        help="Max requests per subset (None=all)")
    parser.add_argument("--output-dir", type=str, default="./results_full_comparison")
    parser.add_argument("--collect-only", action="store_true",
                        help="Only collect and print results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if args.collect_only:
        df = collect_results(output_dir)
        if not df.empty:
            print(df.to_string(index=False))
            df.to_csv(output_dir / "summary.csv", index=False)
        return

    all_configs = build_all_configs()

    if args.methods:
        configs_to_run = {k: v for k, v in all_configs.items() if k in args.methods}
    else:
        configs_to_run = all_configs

    print("=" * 80)
    print(f"Full Benchmark on {args.device}")
    print(f"Methods: {list(configs_to_run.keys())}")
    print(f"Max requests: {args.max_requests or 'ALL'}")
    print("=" * 80)

    for method_name, config in configs_to_run.items():
        print(f"\n{'─' * 60}")
        print(f"Method: {method_name}")
        print(f"{'─' * 60}")
        run_method(method_name, config, args.device, output_dir, args.max_requests)

    # Collect results
    df = collect_results(output_dir)
    if not df.empty:
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(df.to_string(index=False))
        df.to_csv(output_dir / "summary.csv", index=False)
        print(f"\nSaved to {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
