#!/usr/bin/env python3
"""Decode-step cost when the KV cache does NOT fit in VRAM.

The VRAM-resident measurement (bench_expand_latency.py) says every sparse method is
slower than dense at 32K on an H100: dense streams the KV cache at ~69% of peak HBM
bandwidth, while scattered reads reach ~8%, and 5% of the bytes at 8% efficiency is a
net loss.  That result is real, and it is also a statement about ONE regime.

The regime where reading less actually pays is the one where the cache has left VRAM:
host DRAM over PCIe Gen5 x16 is ~64 GB/s against HBM's ~3.9 TB/s, a 60x cliff.  There
the question stops being "how well does the access pattern coalesce" and becomes
"how many bytes cross the bus", which is the axis every sparse method is built for.

This script measures that crossing.  Host-side gather IS charged -- pulling scattered
rows out of pinned memory costs host bandwidth and it would be dishonest to bill only
the transfer.  Compute is excluded: at 64 GB/s the bus dominates by orders of magnitude
and including GPU math would only blur which term is being compared.

    python bench_offload_latency.py --batch 1 --device cuda:1
"""

import argparse
import json
import statistics
from typing import Callable, Dict, List

import torch


def timeit(fn: Callable[[], None], dev: torch.device,
           warmup: int = 5, iters: int = 30) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(dev)
    ts: List[float] = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize(dev)
        ts.append(s.elapsed_time(e))
    ts.sort()
    return {"median_ms": statistics.median(ts), "p90_ms": ts[int(0.9 * len(ts)) - 1]}


def main() -> None:
    p = argparse.ArgumentParser(description="Offloaded-KV decode cost")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=32768)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--density", type=float, default=0.05)
    p.add_argument("--expand-m", type=int, default=8)
    p.add_argument("--bin-size", type=int, default=32)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--json", default=None)
    a = p.parse_args()

    torch.manual_seed(0)
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    B, Hq, Hkv, K, D = a.batch, a.heads, a.kv_heads, a.seq_len, a.head_dim
    g, BH, BQ = Hq // Hkv, B * Hkv, B * Hq

    # the offloaded cache: pinned host memory, so the copy can use DMA
    kv = torch.randn(2, BH, K, D, dtype=torch.bfloat16).pin_memory()
    kvflat = kv.view(2 * BH * K, D)

    n_sel = int(round(K * (a.density - 0.002)))
    n_bins = (K - n_sel) // a.bin_size
    v_sel = int(round(K * (a.density / 2 - 0.002)))
    v_samp = int(round(K * a.density / 2))
    n_exp = a.expand_m * a.bin_size

    # bin statistics are query-independent, so under GQA one read serves g query
    # heads.  They are NOT shared across the batch: different sequences, different KV.
    stats = torch.randn(3, BH, n_bins, D, dtype=torch.bfloat16).pin_memory()

    def rows(n_per_qhead: int) -> torch.Tensor:
        return torch.randint(0, 2 * BH * K, (BQ * n_per_qhead,))

    r_sel, r_exp = rows(2 * n_sel), rows(2 * n_exp)
    r_vsel, r_vsamp = rows(2 * v_sel), rows(2 * v_samp)

    def move_all() -> None:
        kv.to(dev, non_blocking=True)

    def move_rows(idx: torch.Tensor, extra: torch.Tensor = None) -> Callable[[], None]:
        def run() -> None:
            buf = kvflat.index_select(0, idx)          # host-side gather, charged
            buf.to(dev, non_blocking=True)
            if extra is not None:
                extra.to(dev, non_blocking=True)
        return run

    mb = lambda n: n * D * 2 / 1e6                      # noqa: E731
    cases = {
        "dense (whole KV cache)": (move_all, kv.numel() * 2 / 1e6),
        "topk_only (selected KV)": (move_rows(r_sel), mb(r_sel.numel())),
        "vattention (topk rho/2 + samples)":
            (move_rows(torch.cat([r_vsel, r_vsamp])), mb(r_vsel.numel() + r_vsamp.numel())),
        "ours m=8 (selected + expanded + stats)":
            (move_rows(torch.cat([r_sel, r_exp]), stats),
             mb(r_sel.numel() + r_exp.numel()) + stats.numel() * 2 / 1e6),
    }

    print(f"\ndevice={torch.cuda.get_device_name(dev)}  batch={a.batch}  "
          f"heads={Hq}/{Hkv} (g={g})  K={K}  d={D}  density={a.density:.0%}")
    print("KV cache offloaded to pinned host DRAM; host-side gather charged, "
          "GPU compute excluded\n")
    print(f"{'case':40}{'bytes MB':>10}{'median ms':>11}{'vs dense':>10}")
    print("-" * 71)
    out, base = {}, None
    for name, (fn, mbytes) in cases.items():
        r = timeit(fn, dev, iters=a.iters)
        r["bytes_mb"] = mbytes
        out[name] = r
        if base is None:
            base = r["median_ms"]
        print(f"{name:40}{mbytes:>10.1f}{r['median_ms']:>11.3f}{r['median_ms']/base:>9.2f}x")
    print()
    if a.json:
        json.dump({"config": vars(a), "results": out}, open(a.json, "w"), indent=2)
        print(f"wrote {a.json}\n")


if __name__ == "__main__":
    main()
