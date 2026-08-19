#!/usr/bin/env python3
"""Does access GRANULARITY explain the sparse-attention bandwidth gap?

The stage profile says the decode step is 61% "read the selected KV", running at
382 GB/s while a dense stream of the same cache runs at 2690 GB/s -- a 7x penalty
that costs sparse attention its whole advantage.  The question this settles: is the
penalty caused by reading FEW bytes, or by reading them in SMALL SCATTERED pieces?

Same total bytes in every row below; only the shape of the access changes.  If
bandwidth climbs with block size, then granularity is the lever -- and bin_size,
which sets how many contiguous tokens an expanded bin contributes, is a knob we
already own.

    python bench_access_pattern.py --device cuda:1
"""

import argparse
import json
import statistics

import torch


def med(fn, dev, warmup=15, iters=60):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(dev)
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize(dev)
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--groups", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=32768)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--density", type=float, default=0.05)
    p.add_argument("--json", default=None)
    a = p.parse_args()

    torch.manual_seed(0)
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    BH = a.batch * a.kv_heads
    K, D, g = a.seq_len, a.head_dim, a.groups
    n = int(round(K * a.density))                      # rows read per query head
    src = torch.randn(BH * K, D, device=dev, dtype=torch.bfloat16)
    off = (torch.arange(BH, device=dev) * K).view(BH, 1)
    payload_mb = BH * g * n * D * 2 / 1e6

    def idx_blocks(block: int) -> torch.Tensor:
        """n rows per query head, drawn as n/block contiguous runs of `block` rows."""
        nblk = max(1, n // block)
        starts = torch.randint(0, K - block, (BH, g, nblk), device=dev)
        ids = (starts.unsqueeze(-1) + torch.arange(block, device=dev)).reshape(BH, g * nblk * block)
        return (ids + off).reshape(-1)

    rows = {b: idx_blocks(b) for b in (1, 8, 32, 128, 512)}
    print(f"\ndevice={torch.cuda.get_device_name(dev)}  BH={BH}  K={K}  d={D}")
    print(f"rows read per query head = {n} ({a.density:.0%})   payload = {payload_mb:.1f} MB\n")
    print(f"{'access pattern':30}{'ms':>9}{'GB/s':>10}{'vs stream':>11}")
    print("-" * 60)

    # reference: a fully contiguous stream of the same volume
    ncontig = BH * g * n
    t_stream = med(lambda: src.narrow(0, 0, min(ncontig, src.shape[0])).sum(), dev)
    bw_stream = (min(ncontig, src.shape[0]) * D * 2 / 1e9) / (t_stream / 1e3)
    out = {}
    for b, r in rows.items():
        t = med(lambda r=r: src.index_select(0, r), dev)
        bw = (payload_mb / 1e3) / (t / 1e3)
        out[f"block_{b}"] = {"ms": t, "gb_s": bw}
        print(f"{'random blocks of ' + str(b) + ' rows':30}{t:>9.3f}{bw:>10.1f}"
              f"{bw / bw_stream:>10.2f}x")
    print(f"{'contiguous stream (ref)':30}{t_stream:>9.3f}{bw_stream:>10.1f}{1.0:>10.2f}x")
    print()
    print("note: index_select also WRITES the gathered result, so each row above moves")
    print("      ~2x its payload.  A fused kernel would consume the rows in registers.")
    print()
    if a.json:
        out["stream"] = {"ms": t_stream, "gb_s": bw_stream}
        json.dump({"config": vars(a), "payload_mb": payload_mb, "results": out},
                  open(a.json, "w"), indent=2)


if __name__ == "__main__":
    main()
