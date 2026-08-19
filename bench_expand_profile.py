#!/usr/bin/env python3
"""Stage-level breakdown of the expansion decode step: where does the time go?

`bench_expand_latency.py` says expand_gather costs 2.14x dense at B=32.  That number
alone does not say what to fix.  This script times each stage in isolation, with its
inputs precomputed, so the cost of a stage is not confounded by what feeds it.

Stages, in the order the decode step runs them:

  S1  read selected KV        index_select of |S| rows out of the KV cache
  S2  selected logits         q . k for those rows
  S3  bin statistics          zbar_b and sigma_b^2 off the cached per-bin means
  S4  rank + topk             pick the m hottest bins
  S5  build expanded indices  m bins -> m*bin_size token ids
  S6  read expanded KV        index_select of those rows
  S7  expanded logits         q . k for them
  S8  merge                   softmax over selected + expanded + surviving bins

The sum of isolated stages will exceed the fused pipeline (no overlap, extra kernel
launches).  Both are reported: the parts say WHAT to attack, the whole says how much
is actually on the table.

    python bench_expand_profile.py --device cuda:1 --batch 32
"""

import argparse
import json
import statistics
from typing import Callable, Dict, List

import torch


def timeit(fn: Callable[[], object], dev, warmup: int = 20,
           iters: int = 100) -> float:
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
    return statistics.median(ts)


def main() -> None:
    p = argparse.ArgumentParser(description="Stage breakdown for the expansion step")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=32768)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--density", type=float, default=0.05)
    p.add_argument("--expand-m", type=int, default=8)
    p.add_argument("--bin-size", type=int, default=32)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--json", default=None)
    a = p.parse_args()

    torch.manual_seed(0)
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    dt = torch.bfloat16
    B, Hq, Hkv, K, D = a.batch, a.heads, a.kv_heads, a.seq_len, a.head_dim
    g, BH, BQ = Hq // Hkv, B * Hkv, B * Hq
    sc = D ** -0.5
    n_sel = int(round(K * (a.density - 0.002)))
    nb = (K - n_sel) // a.bin_size
    m, bs = a.expand_m, a.bin_size
    ne = m * bs

    q = torch.randn(BH, g, D, device=dev, dtype=dt)
    qq = q * q
    k = torch.randn(BH, K, D, device=dev, dtype=dt)
    v = torch.randn(BH, K, D, device=dev, dtype=dt)
    kflat, vflat = k.reshape(BH * K, D), v.reshape(BH * K, D)
    row_off = (torch.arange(BH, device=dev) * K).view(BH, 1)
    kbar = torch.randn(BH, nb, D, device=dev, dtype=dt)
    kvar = torch.rand(BH, nb, D, device=dev, dtype=dt)
    vbar = torch.randn(BH, nb, D, device=dev, dtype=dt)
    logn = torch.full((BH, 1, nb), float(bs), device=dev, dtype=dt).log()
    offs = torch.arange(bs, device=dev)
    sel = torch.stack([torch.randperm(K, device=dev)[:n_sel]
                       for _ in range(BQ)]).view(BH, g, n_sel)

    def read(idx):
        n = idx.shape[-1]
        rows = (idx.reshape(BH, g * n) + row_off).reshape(-1)
        return (kflat.index_select(0, rows).view(BH, g * n, D),
                vflat.index_select(0, rows).view(BH, g * n, D))

    # precomputed inputs so each stage is timed alone
    ks_, vs_ = read(sel)
    ks_ = ks_.view(BQ, n_sel, D); vs_ = vs_.view(BQ, n_sel, D)
    qf = q.reshape(BQ, 1, D)
    z_ = torch.bmm(qf, ks_.transpose(1, 2)) * sc
    zbar_ = torch.bmm(q, kbar.transpose(1, 2)) * sc
    s2_ = torch.bmm(qq, kvar.transpose(1, 2)) * (sc ** 2)
    score_ = zbar_ + 3.0 * s2_.clamp(min=0).sqrt()
    top_ = score_.topk(m, dim=-1).indices
    tok_ = (top_.unsqueeze(-1) * bs + offs).clamp(max=K - 1).reshape(BH, g, ne)
    ke_, ve_ = read(tok_)
    ke_ = ke_.view(BQ, ne, D); ve_ = ve_.view(BQ, ne, D)
    ze_ = torch.bmm(qf, ke_.transpose(1, 2)) * sc
    ell_ = (zbar_ + logn).scatter(-1, top_, float("-inf")).reshape(BQ, 1, nb)

    def s8():
        rmax = torch.maximum(torch.maximum(z_.amax(-1, keepdim=True),
                                           ze_.amax(-1, keepdim=True)),
                             ell_.amax(-1, keepdim=True))
        e = torch.exp(z_.float() - rmax.float()).to(dt)
        ee = torch.exp(ze_.float() - rmax.float()).to(dt)
        w = torch.exp(ell_.float() - rmax.float()).to(dt)
        num = (torch.bmm(e, vs_) + torch.bmm(ee, ve_)
               + torch.bmm(w.view(BH, g, nb), vbar).reshape(BQ, 1, D))
        den = (e.sum(-1, keepdim=True) + ee.sum(-1, keepdim=True)
               + w.sum(-1, keepdim=True))
        return num / den

    mb = lambda n: n * D * 2 / 1e6                                    # noqa: E731
    stages = [
        ("S1 read selected KV", lambda: read(sel), 2 * mb(BQ * n_sel)),
        ("S2 selected logits", lambda: torch.bmm(qf, ks_.transpose(1, 2)), 0.0),
        ("S3 bin statistics", lambda: (torch.bmm(q, kbar.transpose(1, 2)),
                                       torch.bmm(qq, kvar.transpose(1, 2))),
         2 * mb(BH * nb)),
        ("S4 rank + topk", lambda: (zbar_ + 3.0 * s2_.clamp(min=0).sqrt()
                                    ).topk(m, dim=-1), 0.0),
        ("S5 build expand idx", lambda: (top_.unsqueeze(-1) * bs + offs
                                         ).clamp(max=K - 1).reshape(BH, g, ne), 0.0),
        ("S6 read expanded KV", lambda: read(tok_), 2 * mb(BQ * ne)),
        ("S7 expanded logits", lambda: torch.bmm(qf, ke_.transpose(1, 2)), 0.0),
        ("S8 merge softmax", s8, mb(BH * nb)),
    ]

    print(f"\ndevice={torch.cuda.get_device_name(dev)}  B={B}  heads={Hq}/{Hkv} (g={g})  "
          f"K={K}  d={D}  density={a.density:.0%}  m={m}  bins={nb}")
    print(f"selected/head={n_sel}   expanded/head={ne}\n")
    print(f"{'stage':26}{'ms':>9}{'%':>7}{'MB moved':>11}{'GB/s':>9}")
    print("-" * 62)
    res: Dict[str, Dict[str, float]] = {}
    for name, fn, mbytes in stages:
        t = timeit(fn, dev, iters=a.iters)
        res[name] = {"ms": t, "mb": mbytes}
    tot = sum(r["ms"] for r in res.values())
    for name, r in res.items():
        bw = (r["mb"] / 1e3) / (r["ms"] / 1e3) if r["mb"] else 0.0
        print(f"{name:26}{r['ms']:>9.3f}{100 * r['ms'] / tot:>6.1f}%"
              f"{r['mb']:>11.1f}{bw:>9.1f}" if r["mb"] else
              f"{name:26}{r['ms']:>9.3f}{100 * r['ms'] / tot:>6.1f}%{'-':>11}{'-':>9}")
    print("-" * 62)
    print(f"{'sum of stages':26}{tot:>9.3f}")
    if a.json:
        json.dump({"config": vars(a), "sum_ms": tot, "stages": res},
                  open(a.json, "w"), indent=2)
        print(f"\nwrote {a.json}")
    print()


if __name__ == "__main__":
    main()
