#!/usr/bin/env python3
"""
Experiment 1: Density-axis sweep at EXTREME sparsity (1% / 2% / 3%).

Motivation
----------
At 10% density every method (oracle-top-k / UTA / vAttention / dense) is within
noise of each other, so the tail estimator cannot be evaluated there.  The value
of tail modelling shows up only at low density (oracle-top-k@3% = 78.8 vs
UTA@3% = 85.9).  This script re-measures the three baselines at 1/2/3% with the
full 200-sample protocol so that later method work has a resolvable signal.

Density accounting (total = sink + local + heavy [+ sampling])
    sink = local = 0.001
    oracle-top-k / UTA :  heavy = rho - 0.002
    vAttention         :  sampling = rho/2,  heavy = rho - 0.002 - rho/2
(the vAttention split reproduces the ratio used in run_damped_jensen_vs_vattention.py:
 rho=0.10 -> heavy 0.048 + sampling 0.05)

NOTE: vAttention's AdaptiveSamplingMasker allocates its budget adaptively, so its
realised density is *not* exactly rho.  Pass --log-density to record the actual
per-layer density into micro_metrics.jsonl and check it before claiming iso-density.

Usage (2 GPUs)
--------------
    python run_density_sweep.py --device cuda:0 --shard 0 --num-shards 2
    python run_density_sweep.py --device cuda:1 --shard 1 --num-shards 2

Resumable: any (method, density, subset) with metrics.json already present is skipped.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

from benchmark.ruler32k import Ruler32K  # noqa: E402
from sparse_attention_hub.adapters import ModelAdapterHF  # noqa: E402
from sparse_attention_hub.metric_logging.logger import MicroMetricLogger  # noqa: E402
from sparse_attention_hub.sparse_attention.research_attention import (  # noqa: E402
    ResearchAttentionConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (  # noqa: E402
    LocalMaskerConfig,
    OracleTopKConfig,
    SinkMaskerConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (  # noqa: E402
    AdaptiveSamplingMaskerConfig,
)
from sparse_attention_hub.sparse_attention.uta_attention import UTAAttentionConfig  # noqa: E402

SUBSETS = [
    "qa_1", "qa_2", "vt", "fwe",
    "niah_multikey_2", "niah_multikey_3", "niah_multivalue",
]

SINK = 0.001
LOCAL = 0.001


def build_config(method: str, rho: float):
    """Build a sparse-attention config for `method` at total density `rho`."""
    if method == "oracle-top-k":
        return ResearchAttentionConfig(masker_configs=[
            SinkMaskerConfig(sink_size=SINK),
            LocalMaskerConfig(window_size=LOCAL),
            OracleTopKConfig(heavy_size=round(rho - SINK - LOCAL, 6)),
        ])

    if method == "UTA":
        return UTAAttentionConfig(masker_configs=[
            SinkMaskerConfig(sink_size=SINK),
            LocalMaskerConfig(window_size=LOCAL),
            OracleTopKConfig(heavy_size=round(rho - SINK - LOCAL, 6)),
        ])

    if method == "vAttention":
        sampling = round(rho / 2, 6)
        heavy = round(rho - SINK - LOCAL - sampling, 6)
        if heavy <= 0:
            raise ValueError(f"rho={rho} too small for the vAttention 50/50 split")
        return ResearchAttentionConfig(masker_configs=[
            SinkMaskerConfig(sink_size=SINK),
            LocalMaskerConfig(window_size=LOCAL),
            OracleTopKConfig(heavy_size=heavy),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=sampling,
                epsilon=0.25,
                delta=0.25,
                init_offset=SINK,
                local_offset=LOCAL,
            ),
        ])

    raise ValueError(f"unknown method: {method}")


def density_tag(rho: float) -> str:
    return f"{rho * 100:g}pct"


def run_job(method: str, rho: float, args) -> None:
    tag = density_tag(rho)
    method_dir = Path(args.output_dir) / tag / method.replace("-", "_")

    pending = [s for s in SUBSETS if not (method_dir / s / "metrics.json").exists()]
    if not pending:
        print(f"[SKIP job] {method}@{tag} — all subsets done")
        return

    print(f"\n{'=' * 70}\n[{args.device}] {method} @ {tag} — {len(pending)} subset(s) to run\n{'=' * 70}")

    adapter = ModelAdapterHF(
        model_name=args.model,
        sparse_attention_config=build_config(method, rho),
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=args.device,
    )

    if args.log_density:
        MicroMetricLogger().configure_logging(
            log_path=str(method_dir),
            enabled_metrics=["research_attention_density"],
            sampling_factor=0.01,
        )

    try:
        for subset in pending:
            subset_dir = method_dir / subset
            subset_dir.mkdir(parents=True, exist_ok=True)
            print(f"  -> {method}@{tag}/{subset}", flush=True)

            Ruler32K(subsets_to_run=[subset]).run_benchmark(
                adapter,
                str(subset_dir),
                request_kwargs={
                    "max_requests": args.max_requests,
                    "max_context_length": 32000,
                },
                generation_kwargs={"max_new_tokens": 120},
            )

            mf = subset_dir / "metrics.json"
            if mf.exists():
                score = (json.loads(mf.read_text())
                         .get("task_scores", {}).get(subset, {}).get("string_match", 0.0))
                print(f"     done: string_match={score:.2f}", flush=True)
    finally:
        if args.log_density:
            MicroMetricLogger().flush()
        del adapter
        torch.cuda.empty_cache()


def summarize(output_dir: Path, densities, methods) -> None:
    rows = []
    for rho in densities:
        tag = density_tag(rho)
        for method in methods:
            method_dir = output_dir / tag / method.replace("-", "_")
            row = {"Method": method, "Density": tag}
            scores = []
            for subset in SUBSETS:
                mf = method_dir / subset / "metrics.json"
                if mf.exists():
                    s = (json.loads(mf.read_text())
                         .get("task_scores", {}).get(subset, {}).get("string_match", 0.0))
                    row[subset] = round(s, 2)
                    scores.append(s)
                else:
                    row[subset] = None
            row["Average"] = round(sum(scores) / len(scores), 2) if len(scores) == len(SUBSETS) else None
            rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)[["Density", "Method", "Average"] + SUBSETS]
    print("\n" + "=" * 100)
    print("DENSITY SWEEP SUMMARY")
    print("=" * 100)
    print(df.to_string(index=False))
    out = output_dir / "density_sweep_summary.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Extreme-sparsity density sweep")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", default="./results_density_sweep")
    p.add_argument("--max-requests", type=int, default=200)
    p.add_argument("--densities", default="0.01,0.02,0.03",
                   help="comma-separated total densities")
    p.add_argument("--methods", default="oracle-top-k,UTA,vAttention")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--log-density", action="store_true",
                   help="record realised per-layer density (needed to verify iso-density)")
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args()

    densities = [float(x) for x in args.densities.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]
    output_dir = Path(args.output_dir)

    if args.summarize_only:
        summarize(output_dir, densities, methods)
        return

    # Interleave (density, method) jobs so each shard gets a mix of cheap/expensive work.
    jobs = [(m, rho) for rho in densities for m in methods]
    mine = jobs[args.shard::args.num_shards]

    print(f"shard {args.shard}/{args.num_shards} on {args.device}: {len(mine)} job(s)")
    for m, rho in mine:
        print(f"   - {m} @ {density_tag(rho)}")

    for method, rho in mine:
        run_job(method, rho, args)

    summarize(output_dir, densities, methods)


if __name__ == "__main__":
    main()
