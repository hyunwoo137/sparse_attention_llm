#!/usr/bin/env python3
"""
CovUTA Benchmark: Compare CovUTA variants against UTA baseline and vAttention.

Methods evaluated:
  1. UTA (base, no Jensen correction)
  2. CovUTA (proxy_logit merge — σ²/2 denominator + M_KV numerator correction)
  3. CovUTA-Indep (independent scale — exploits exp(z̄) cancellation)
  4. vAttention (oracle top-k + adaptive sampling, reference)
  5. oracle-top-k (drop tail, lower bound)

All methods at 10%, 5%, 3% density.
Runs on 2 GPUs in parallel (GPU 0 and GPU 1).
Uses ALL samples per subset (no max_requests limit).
"""

import os
import sys
import json
import torch
import subprocess
import pandas as pd
from pathlib import Path

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

from sparse_attention_hub.adapters import ModelAdapterHF
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.uta_attention import (
    UTAAttentionConfig,
    CovUTAConfig,
    CovUTAIndependentConfig,
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
# Configs
# ============================================================================

def build_all_configs():
    """Build all method configs for each density level.

    Budget allocation:
    - UTA / CovUTA / CovUTA-Indep: sink(0.1%) + local(0.1%) + top-k(remaining)
      (proxy is free — all budget goes to top-k)
    - vAttention: sink(0.1%) + local(0.1%) + top-k(half) + adaptive_sampling(half)
    - oracle-top-k: sink(0.1%) + local(0.1%) + top-k(remaining) — drops tail
    """
    configs = {}

    for density_label, topk_full, topk_half, sampling_rate in [
        ("10pct", 0.098, 0.048, 0.05),
        ("5pct",  0.048, 0.023, 0.025),
        ("3pct",  0.028, 0.013, 0.015),
    ]:
        d = density_label

        # oracle-top-k (drop tail)
        configs[f"oracle-top-k@{d}"] = ResearchAttentionConfig(masker_configs=[
            SinkMaskerConfig(sink_size=0.001),
            LocalMaskerConfig(window_size=0.001),
            OracleTopKConfig(heavy_size=topk_full),
        ])

        # UTA (base, no Jensen correction)
        configs[f"UTA@{d}"] = UTAAttentionConfig(masker_configs=[
            SinkMaskerConfig(sink_size=0.001),
            LocalMaskerConfig(window_size=0.001),
            OracleTopKConfig(heavy_size=topk_full),
        ])

        # CovUTA (proxy_logit merge)
        configs[f"CovUTA@{d}"] = CovUTAConfig(masker_configs=[
            SinkMaskerConfig(sink_size=0.001),
            LocalMaskerConfig(window_size=0.001),
            OracleTopKConfig(heavy_size=topk_full),
        ])

        # CovUTA-Indep (independent scale)
        configs[f"CovUTA-Indep@{d}"] = CovUTAIndependentConfig(masker_configs=[
            SinkMaskerConfig(sink_size=0.001),
            LocalMaskerConfig(window_size=0.001),
            OracleTopKConfig(heavy_size=topk_full),
        ])

        # vAttention (oracle top-k + adaptive sampling)
        configs[f"vAttention@{d}"] = ResearchAttentionConfig(masker_configs=[
            SinkMaskerConfig(sink_size=0.001),
            LocalMaskerConfig(window_size=0.001),
            OracleTopKConfig(heavy_size=topk_half),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=sampling_rate,
                epsilon=0.25, delta=0.25,
                init_offset=0.001, local_offset=0.001,
            ),
        ])

    return configs


def safe_name(name):
    return name.replace("@", "_at_").replace("-", "_").replace("(", "").replace(")", "")


def run_method(method_name, config, device, output_dir):
    """Run a single method across all 7 RULER32K subsets with all samples."""
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
            print(f"  [SKIP] {method_name}/{subset} — already done")
            continue

        subset_dir.mkdir(exist_ok=True, parents=True)
        print(f"  [{device}] Running {method_name}/{subset}...")

        benchmark = Ruler32K(subsets_to_run=[subset])
        benchmark.run_benchmark(
            adapter,
            str(subset_dir),
            request_kwargs={"max_context_length": 32000},  # no max_requests → all samples
            generation_kwargs={"max_new_tokens": 120},
        )

    # Collect results
    subset_scores = {}
    for subset in SUBSETS:
        metrics_file = method_dir / subset / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                data = json.load(f)
                score = data.get("task_scores", {}).get(subset, {}).get("string_match", 0.0)
                subset_scores[subset] = round(score, 2)

    completed = len(subset_scores)
    avg = round(sum(subset_scores.values()) / max(1, completed), 2) if completed == len(SUBSETS) else "—"
    print(f"  [{method_name}] {completed}/{len(SUBSETS)} subsets done | avg={avg} | {subset_scores}")


def run_gpu_methods(device, methods_on_gpu, all_configs, output_dir):
    """Run a subset of methods on a specific GPU."""
    for method_name in methods_on_gpu:
        config = all_configs[method_name]
        print(f"\n{'='*80}")
        print(f"[{device}] Starting: {method_name}")
        print(f"{'='*80}")
        run_method(method_name, config, device, output_dir)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CovUTA Benchmark")
    parser.add_argument("--gpu", type=int, default=None,
                        help="Run only on this GPU (for subprocess mode)")
    parser.add_argument("--methods", type=str, nargs="+", default=None,
                        help="Run only these methods (for subprocess mode)")
    parser.add_argument("--output-dir", type=str, default="./results_covuta")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    all_configs = build_all_configs()

    if args.gpu is not None and args.methods is not None:
        # Subprocess mode: run specific methods on specific GPU
        device = f"cuda:{args.gpu}"
        run_gpu_methods(device, args.methods, all_configs, output_dir)
        return

    # Main mode: launch 2 subprocesses on GPU 0 and GPU 1
    # Split methods across GPUs, interleaving densities for balance
    all_methods = list(all_configs.keys())

    # Group by density
    methods_10 = [m for m in all_methods if "10pct" in m]
    methods_5 = [m for m in all_methods if "5pct" in m]
    methods_3 = [m for m in all_methods if "3pct" in m]

    # GPU 0: oracle-top-k, UTA, CovUTA at all densities
    # GPU 1: CovUTA-Indep, vAttention at all densities
    gpu0_methods = []
    gpu1_methods = []
    for methods_d in [methods_10, methods_5, methods_3]:
        for m in methods_d:
            if "oracle" in m or ("UTA@" in m and "Cov" not in m) or ("CovUTA@" in m and "Indep" not in m):
                gpu0_methods.append(m)
            else:
                gpu1_methods.append(m)

    print("=" * 100)
    print("CovUTA Benchmark Suite")
    print(f"Output: {output_dir}")
    print(f"GPU 0 methods ({len(gpu0_methods)}): {gpu0_methods}")
    print(f"GPU 1 methods ({len(gpu1_methods)}): {gpu1_methods}")
    print("=" * 100)

    # Launch subprocesses
    script = str(Path(__file__).resolve())
    procs = []

    for gpu_id, methods in [(0, gpu0_methods), (1, gpu1_methods)]:
        if not methods:
            continue
        cmd = [
            sys.executable, script,
            "--gpu", str(gpu_id),
            "--methods", *methods,
            "--output-dir", str(output_dir),
        ]
        print(f"\nLaunching GPU {gpu_id}: {' '.join(cmd[:6])}...")
        proc = subprocess.Popen(
            cmd,
            stdout=open(f"/tmp/covuta_gpu{gpu_id}.log", "w"),
            stderr=subprocess.STDOUT,
        )
        procs.append((gpu_id, proc))

    print("\nBoth GPUs launched. Monitor progress with:")
    print("  tail -f /tmp/covuta_gpu0.log")
    print("  tail -f /tmp/covuta_gpu1.log")

    # Wait for completion
    for gpu_id, proc in procs:
        proc.wait()
        print(f"GPU {gpu_id} finished with exit code {proc.returncode}")

    # Print final results table
    print("\n" + "=" * 100)
    print("FINAL RESULTS:")
    print("=" * 100)

    rows = []
    for method_name in all_configs:
        method_dir = output_dir / safe_name(method_name)
        subset_scores = {}
        for subset in SUBSETS:
            metrics_file = method_dir / subset / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    data = json.load(f)
                    score = data.get("task_scores", {}).get(subset, {}).get("string_match", 0.0)
                    subset_scores[subset] = round(score, 2)

        completed = len(subset_scores)
        avg = round(sum(subset_scores.values()) / max(1, completed), 2) if completed == len(SUBSETS) else None
        row = {"Method": method_name, "Avg": avg, **subset_scores}
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(output_dir / "covuta_results.csv", index=False)
    print(f"\nSaved to {output_dir / 'covuta_results.csv'}")


if __name__ == "__main__":
    main()
