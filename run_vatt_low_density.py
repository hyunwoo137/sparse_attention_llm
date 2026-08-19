#!/usr/bin/env python3
"""
Run vAttention at 5% and 3% density to compare with existing UTA results.
UTA results already exist in results_uta_low_density/.
"""

import os, sys, json, torch
import pandas as pd
from pathlib import Path

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

from sparse_attention_hub.adapters import ModelAdapterHF
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    SinkMaskerConfig, LocalMaskerConfig, OracleTopKConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
    AdaptiveSamplingMaskerConfig,
)
from benchmark.ruler32k import Ruler32K

SUBSETS = ["qa_1", "qa_2", "vt", "fwe", "niah_multikey_2", "niah_multikey_3", "niah_multivalue"]

# vAttention configs at different densities
# Total budget split: ~half top-k, ~half adaptive sampling
VATT_CONFIGS = {
    "5pct": ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.001),
        LocalMaskerConfig(window_size=0.001),
        OracleTopKConfig(heavy_size=0.023),
        AdaptiveSamplingMaskerConfig(
            base_rate_sampling=0.025, epsilon=0.25, delta=0.25,
            init_offset=0.001, local_offset=0.001,
        ),
    ]),
    "3pct": ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=0.001),
        LocalMaskerConfig(window_size=0.001),
        OracleTopKConfig(heavy_size=0.013),
        AdaptiveSamplingMaskerConfig(
            base_rate_sampling=0.015, epsilon=0.25, delta=0.25,
            init_offset=0.001, local_offset=0.001,
        ),
    ]),
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--max-requests-per-subset", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="./results_uta_low_density")
    args = parser.parse_args()

    base_out_dir = Path(args.output_dir)

    for density_name, config in VATT_CONFIGS.items():
        print(f"\n{'=' * 60}")
        print(f"vAttention @ {density_name}")
        print(f"{'=' * 60}")

        adapter = ModelAdapterHF(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            sparse_attention_config=config,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=args.device,
        )

        for subset in SUBSETS:
            subset_dir = base_out_dir / density_name / "vAttention" / subset
            if (subset_dir / "metrics.json").exists():
                print(f"  [SKIP] {subset}")
                continue
            subset_dir.mkdir(exist_ok=True, parents=True)
            print(f"  Running {subset}...")
            benchmark = Ruler32K(subsets_to_run=[subset])
            benchmark.run_benchmark(
                adapter, str(subset_dir),
                request_kwargs={"max_requests": args.max_requests_per_subset, "max_context_length": 32000},
                generation_kwargs={"max_new_tokens": 120},
            )
            print(f"  ✓ {subset} done")

        del adapter
        torch.cuda.empty_cache()

    # Build comparison table: vAttention vs UTA at each density
    print("\n" + "=" * 80)
    print("vAttention vs UTA at Low Density")
    print("=" * 80)

    all_rows = []
    for density_name, density_label in [("5pct", "~5%"), ("3pct", "~3%")]:
        for method in ["vAttention", "oracle-top-k", "UTA"]:
            method_dir = base_out_dir / density_name / method
            row = {"Method": f"{method}@{density_label}"}
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

    df = pd.DataFrame(all_rows)
    cols = ["Method", "Average (%)"] + SUBSETS
    df = df[cols]
    print(df.to_string(index=False))

    summary_path = base_out_dir / "vatt_vs_uta_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()
