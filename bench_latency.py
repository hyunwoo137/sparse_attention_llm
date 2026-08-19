#!/usr/bin/env python3
"""Tier-1 latency benchmark: the v2 ablation ladder vs vAttention, with a
per-stage breakdown.

What this measures
------------------
One attention layer, in isolation, on synthetic tensors shaped like
Llama-3.1-8B (H=32, H_kv=8, D=128), in the DECODE regime (Q=1) that the
adapter actually runs sparse attention in -- context prefill happens outside
`enable_sparse_mode()`, so every sparse call in a real run is a decode step.

Configs come from `run_multibin_hat_bench.build_config`, i.e. the exact same
builder that produced the accuracy numbers in results_v2/, so the latency and
the accuracy columns describe the same objects.

Two passes per point:
  1. clean pass  -- instrumentation OFF, cuda-event timing around the whole
                    `custom_attention` call.  This is the number to quote.
  2. breakdown   -- instrumentation ON (StageTimer), giving per-stage CUDA time.
                    Slightly perturbed by the events themselves; the report
                    prints both totals so the perturbation is visible.

READ THIS BEFORE QUOTING THE NUMBERS
------------------------------------
Every method here -- ours AND vAttention -- is a research prototype that
materialises the full (B,H,Q,K) score matrix.  The absolute milliseconds are
therefore dominated by an O(QK) matmul that a deployed kernel of *either*
method would not perform.  Tier 1 answers "what does the code that produced the
accuracy table cost, and where does the time go"; it does not answer "what
would a deployed kernel cost".  That is Tier 2.

Usage
-----
    python bench_latency.py --device cuda:1
    python bench_latency.py --device cuda:1 --suites main,seqlen
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

from run_multibin_hat_bench import LOCAL, SINK, _prefix, build_config  # noqa: E402
from sparse_attention_hub.sparse_attention.uta_attention import (  # noqa: E402
    UTAJensenConfig,
)
from sparse_attention_hub.metric_logging.stage_timer import StageTimer  # noqa: E402
from sparse_attention_hub.sparse_attention.base import SparseAttention  # noqa: E402
from sparse_attention_hub.sparse_attention.utils.kv_utils import repeat_kv  # noqa: E402
from sparse_attention_hub.sparse_attention.utils.mask import Mask  # noqa: E402

# Llama-3.1-8B attention geometry
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
DTYPE = torch.bfloat16


# --------------------------------------------------------------------- points
@dataclass
class Point:
    """One (method, shape) combination to measure."""

    label: str
    method: str
    rho: float = 0.05
    bin_size: int = 32
    kappa: str = "j2"
    eps: float = 0.25
    delta: float = 0.25
    seq_len: int = 32768
    batch: int = 1
    num_queries: int = 1
    bin_mode: str = "auto"
    extra: Dict[str, Any] = field(default_factory=dict)

    def config(self):
        if self.method == "dense":
            return None
        if self.method == "UTA+jensen-direct":
            # The ladder expresses "UTA + Jensen" as multi-bin with one giant bin,
            # which routes a single proxy through the scatter machinery.  This is
            # the same correction written directly (advanced.py), and is what
            # isolates the cost of the deviation stage from that detour.
            return UTAJensenConfig(
                alpha=1.0,
                masker_configs=_prefix(round(self.rho - SINK - LOCAL, 6), "hat"),
            )
        return build_config(
            self.method, self.rho, self.bin_size, self.kappa,
            eps=self.eps, delta=self.delta, selector="hat", bin_mode=self.bin_mode,
        )


# ------------------------------------------------------------------- fixtures
def make_tensors(batch: int, num_queries: int, seq_len: int, device: str):
    """Synthetic Q/K/V.  Keys are given per-position drift so that the tail is
    not i.i.d. -- a uniform tail would make position binning look artificially
    good and the Jensen correction artificially useless."""
    g = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(batch, NUM_HEADS, num_queries, HEAD_DIM,
                    generator=g, device=device, dtype=torch.float32)
    k = torch.randn(batch, NUM_KV_HEADS, seq_len, HEAD_DIM,
                    generator=g, device=device, dtype=torch.float32)
    v = torch.randn(batch, NUM_KV_HEADS, seq_len, HEAD_DIM,
                    generator=g, device=device, dtype=torch.float32)
    drift = torch.linspace(-1.0, 1.0, seq_len, device=device).view(1, 1, seq_len, 1)
    k = k + 0.5 * drift
    return (q.to(DTYPE) / math.sqrt(2), k.to(DTYPE), v.to(DTYPE))


def dense_attention(module, q, k, v, attention_mask, scaling, dropout, **kw):
    ngroups = q.shape[1] // k.shape[1]
    return (
        torch.nn.functional.scaled_dot_product_attention(
            q, repeat_kv(k, ngroups), repeat_kv(v, ngroups),
            attn_mask=None, dropout_p=0.0, is_causal=False, scale=scaling,
        ).transpose(1, 2).contiguous(),
        None,
    )


# -------------------------------------------------------------------- density
def measure_density(attn, q, k, v, scaling, meta) -> Optional[float]:
    """Realized density of the masker pipeline -- the equal-cost precondition.

    A latency table across methods is meaningless if they do not actually read
    the same number of KV entries, and vAttention's budget is data-dependent so
    its realized density is NOT its nominal one.
    """
    if not hasattr(attn, "maskers"):
        return 1.0
    shape = (q.shape[0], q.shape[1], q.shape[2], k.shape[2])
    mask = Mask.create_empty_mask(shape, dtype=q.dtype, device=q.device)
    for masker in attn.maskers:
        mask = masker.add_mask(
            keys=k, queries=q, values=v, attention_mask=None, scaling=scaling,
            dropout=0.0, sparse_meta_data=meta, previous_mask=mask, layer_idx=0,
        )
    d = mask.get_density()
    return float(d.item() if torch.is_tensor(d) else d)


# --------------------------------------------------------------------- timing
def time_calls(fn, reps: int, warmup: int) -> List[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
           for _ in range(reps)]
    for s, e in evs:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in evs]


def pct(xs: List[float], p: float) -> float:
    ys = sorted(xs)
    i = min(len(ys) - 1, max(0, int(round(p * (len(ys) - 1)))))
    return ys[i]


# ------------------------------------------------------------------ one point
def run_point(pt: Point, device: str, reps: int, warmup: int,
              bd_reps: int) -> Dict[str, Any]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    q, k, v = make_tensors(pt.batch, pt.num_queries, pt.seq_len, device)
    scaling = 1.0 / math.sqrt(HEAD_DIM)
    module = nn.Module().to(device).eval()
    meta: Dict[str, Any] = {}

    cfg = pt.config()
    if cfg is None:
        attn = None
        call = lambda: dense_attention(module, q, k, v, None, scaling, 0.0)  # noqa: E731
    else:
        attn = SparseAttention.create_from_config(cfg)
        call = lambda: attn.custom_attention(  # noqa: E731
            module=module, queries=q, keys=k, values=v, attention_mask=None,
            scaling=scaling, dropout=0.0, sparse_meta_data=meta, layer_idx=0,
        )

    row: Dict[str, Any] = {
        "label": pt.label, "method": pt.method, "rho": pt.rho,
        "bin_size": pt.bin_size, "eps": pt.eps, "delta": pt.delta,
        "seq_len": pt.seq_len, "batch": pt.batch, "num_queries": pt.num_queries,
    }

    try:
        # ---- pass 1: clean wall-clock, instrumentation off -------------------
        StageTimer.enable(False)
        ts = time_calls(call, reps, warmup)
        row.update(
            mean_ms=sum(ts) / len(ts),
            p50_ms=pct(ts, 0.50),
            p95_ms=pct(ts, 0.95),
            min_ms=min(ts),
            max_ms=max(ts),
            std_ms=(sum((t - sum(ts) / len(ts)) ** 2 for t in ts) / len(ts)) ** 0.5,
            peak_mem_MB=torch.cuda.max_memory_allocated(device) / 1e6,
        )

        # ---- realized density ------------------------------------------------
        if attn is not None:
            dens = [measure_density(attn, q, k, v, scaling, meta) for _ in range(5)]
            row["density"] = sum(dens) / len(dens)
            row["density_max"] = max(dens)
        else:
            row["density"] = 1.0
            row["density_max"] = 1.0

        # ---- pass 2: per-stage breakdown -------------------------------------
        if attn is not None:
            StageTimer.enable(True)
            # warm up with the timer already on, then drop those records -- the
            # warmup calls emit stage events too and would otherwise be folded
            # into the per-call averages.
            for _ in range(2):
                call()
            torch.cuda.synchronize()
            StageTimer.reset()
            bd_ts = time_calls(call, bd_reps, 0)
            summary = StageTimer.summary()
            StageTimer.enable(False)
            StageTimer.reset()
            row["stages"] = {
                name: {"ms_per_call": s["total_ms"] / bd_reps,
                       "regions_per_call": s["calls"] / bd_reps}
                for name, s in summary.items()
            }
            row["instrumented_mean_ms"] = sum(bd_ts) / len(bd_ts)
        else:
            row["stages"] = {}
            row["instrumented_mean_ms"] = None

        row["status"] = "ok"
    except torch.cuda.OutOfMemoryError as exc:
        row["status"] = f"OOM: {str(exc)[:120]}"
    except Exception as exc:  # noqa: BLE001 - one bad point must not kill the sweep
        row["status"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    finally:
        StageTimer.enable(False)
        StageTimer.reset()
        del q, k, v, meta
        torch.cuda.empty_cache()

    return row


# ---------------------------------------------------------------------- suites
LADDER = [
    ("dense",                "dense"),
    ("HAT",                  "selector"),
    ("UTA",                  "UTA"),
    ("UTA+multibin(b32)",    "UTA+multibin"),
    ("UTA+jensen",           "UTA+jensen"),
    ("UTA+multibin+jensen(b32)", "UTA+multibin+jensen"),
]


def suite_main() -> List[Point]:
    pts = [Point(label=lbl, method=m) for lbl, m in LADDER]
    # vAttention at the (eps,delta) that produced the results_v2 accuracy number,
    # and at the cheaper pair the sweep script defaults to.
    pts.append(Point(label="vAttention(HAT) e.25d.25", method="vAttention",
                     eps=0.25, delta=0.25))
    pts.append(Point(label="vAttention(HAT) e.4d.4", method="vAttention",
                     eps=0.4, delta=0.4))
    # Jensen written directly instead of as multi-bin-with-one-bin: isolates the
    # deviation stage from the scatter detour the ladder's rung takes.
    pts.append(Point(label="UTA+jensen (direct impl)", method="UTA+jensen-direct"))
    # the deployable bin partition -- position blocks, no scatter, cacheable stats
    pts.append(Point(label="UTA+multibin(b32,fixed)",
                     method="ours", bin_mode="fixed", kappa="none"))
    pts.append(Point(label="UTA+multibin+jensen(b32,fixed)",
                     method="ours", bin_mode="fixed", kappa="j2"))
    return pts


def _pair(**kw) -> List[Point]:
    """Ours in both partitions, plus the vAttention baseline, at one shape.

    `equalcount` is the partition the accuracy runs used; `fixed` is the same
    math over query-independent position blocks, i.e. the deployable one.  Both
    are carried through every sweep because they scale very differently.
    """
    return [
        Point(label="UTA+multibin+jensen(b32)", method="UTA+multibin+jensen", **kw),
        Point(label="UTA+multibin+jensen(b32,fixed)", method="ours",
              bin_mode="fixed", kappa="j2", **kw),
        Point(label="vAttention(HAT) e.25d.25", method="vAttention",
              eps=0.25, delta=0.25, **kw),
    ]


def suite_seqlen() -> List[Point]:
    out = []
    for K in (4096, 8192, 16384, 32768, 65536):
        out.append(Point(label="HAT", method="selector", seq_len=K))
        out.append(Point(label="UTA", method="UTA", seq_len=K))
        out += _pair(seq_len=K)
    return out


def suite_batch() -> List[Point]:
    out = []
    for B in (1, 2, 4, 8):
        out += _pair(batch=B)
    return out


def suite_binsize() -> List[Point]:
    out = []
    for m in (8, 16, 32, 64, 128):
        out.append(Point(label="UTA+multibin+jensen(equalcount)",
                         method="UTA+multibin+jensen", bin_size=m))
        out.append(Point(label="UTA+multibin+jensen(fixed)", method="ours",
                         bin_mode="fixed", kappa="j2", bin_size=m))
    return out


def suite_nq() -> List[Point]:
    """Q>1 happens once per request: the adapter feeds the question tokens in a
    single sparse forward before autoregressive decode begins."""
    out = []
    for Q in (1, 8, 32):
        out += _pair(num_queries=Q)
    return out


SUITES = {
    "main": suite_main,
    "seqlen": suite_seqlen,
    "batch": suite_batch,
    "binsize": suite_binsize,
    "nq": suite_nq,
}


# ------------------------------------------------------------------------ main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--suites", default="main,seqlen,batch,binsize,nq")
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--bd-reps", type=int, default=10)
    p.add_argument("--out", default="results_latency")
    a = p.parse_args()

    torch.cuda.set_device(a.device)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    name = torch.cuda.get_device_name(a.device)
    print(f"device {a.device} ({name})   torch {torch.__version__}")

    results: Dict[str, List[Dict[str, Any]]] = {}
    for suite in a.suites.split(","):
        suite = suite.strip()
        if suite not in SUITES:
            print(f"  !! unknown suite {suite}, skipping")
            continue
        pts = SUITES[suite]()
        print(f"\n=== suite {suite} ({len(pts)} points) ===", flush=True)
        rows = []
        for pt in pts:
            t0 = time.time()
            row = run_point(pt, a.device, a.reps, a.warmup, a.bd_reps)
            row["suite"] = suite
            rows.append(row)
            if row["status"] == "ok":
                print(f"  {pt.label:34s} K={pt.seq_len:6d} B={pt.batch} Q={pt.num_queries} "
                      f"| {row['mean_ms']:8.3f} ms  p95 {row['p95_ms']:8.3f}  "
                      f"rho {row['density']:.4f}  mem {row['peak_mem_MB']:7.0f}MB  "
                      f"[{time.time() - t0:.0f}s]", flush=True)
            else:
                print(f"  {pt.label:34s} K={pt.seq_len:6d} B={pt.batch} "
                      f"| {row['status']}", flush=True)
        results[suite] = rows

        with open(out_dir / "latency_raw.json", "w") as f:
            json.dump({"device": name, "torch": torch.__version__,
                       "results": results}, f, indent=2)

    print(f"\nsaved -> {out_dir / 'latency_raw.json'}")


if __name__ == "__main__":
    main()
