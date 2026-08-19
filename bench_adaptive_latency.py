#!/usr/bin/env python3
"""What an ADAPTIVE expansion budget costs, in wall-clock, on top of a fixed one.

The fixed-m path already computes everything the adaptive rule needs:
    zbar_b = s * q.kbar_b            (bmm against the cached per-bin key mean)
    sigma_b^2 = s^2 * sum_d q_d^2 Var_b(k_d)   (bmm against the cached per-bin variance)
so the only NEW work is turning those into a per-row budget and living with the
raggedness that follows.  Three things get charged separately:

  (1) the decision itself     -- share, compare, count.  A few elementwise passes.
  (2) index materialisation   -- fixed m is one topk; adaptive is a sort (or a topk to
                                 the row maximum) plus a validity mask.
  (3) the kernel's tail       -- rows differ in m, so the launch has to pad every row
                                 to the batch maximum.  The kernel skips -1 slots, but
                                 the PROGRAM still runs the tile loop, so the cost is
                                 set by max(m), not by mean(m).  This is the real bill
                                 and it is a function of the budget distribution, not
                                 of the rule's arithmetic.

The m-distribution is taken from the measured one (run_softmax_decomp_diag.py) via
--m-quantiles, so (3) is charged against a distribution the method actually produces
rather than an invented one.
"""
import argparse
import json
import math
import statistics
import sys

import torch

sys.path.insert(0, "/home/hyunwoo/research/sparse-attention-hub")
from triton_expand_decode import decode_expand_attention  # noqa: E402


def med(fn, dev, w=25, it=150):
    for _ in range(w):
        fn()
    torch.cuda.synchronize(dev)
    ts = []
    for _ in range(it):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize(dev)
        ts.append(s.elapsed_time(e))
    return statistics.median(sorted(ts))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=32768)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--density", type=float, default=0.05)
    p.add_argument("--expand-m", type=int, default=8)
    p.add_argument("--bin-size", type=int, default=32)
    p.add_argument("--m-cap", type=int, default=64)
    p.add_argument("--iters", type=int, default=150)
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
    NB = (K - n_sel) // a.bin_size
    m0 = a.expand_m

    q = torch.randn(BQ, D, device=dev, dtype=dt)
    k = torch.randn(BH, K, D, device=dev, dtype=dt)
    v = torch.randn(BH, K, D, device=dev, dtype=dt)
    sel = torch.stack([torch.randperm(K, device=dev)[:n_sel] for _ in range(BQ)]).int()
    kbar = torch.randn(BH, NB, D, device=dev, dtype=dt)
    vbar = torch.randn(BH, NB, D, device=dev, dtype=dt)
    # cached second moment, same footprint as the mean
    kvar = torch.randn(BH, NB, D, device=dev, dtype=dt).abs()
    n_b = torch.full((BQ, NB), float(a.bin_size), device=dev)
    log_n = torch.log(n_b)
    den_sel = torch.rand(BQ, 1, device=dev) * 10 + 1.0

    qg = q.view(BH, g, D)
    q2g = (q.float() ** 2).to(dt).view(BH, g, D)

    # ---- stage 1: bin logits and variances from the cached statistics ----------
    def bin_moments():
        zbar = torch.bmm(qg, kbar.transpose(1, 2)).float().view(BQ, NB) * sc
        s2 = torch.bmm(q2g, kvar.transpose(1, 2)).float().view(BQ, NB) * (sc * sc)
        return zbar, s2

    zbar, s2 = bin_moments()
    sig = s2.clamp(min=0).sqrt()
    score = zbar + 3.0 * sig

    # ---- stage 2a: FIXED budget -- one topk -----------------------------------
    def sel_fixed():
        return score.topk(m0, dim=-1, sorted=False).indices

    # ---- stage 2b: ADAPTIVE budget -- share, threshold, count, ragged indices --
    row_max = score.amax(-1, keepdim=True)
    theta = 1e-3

    def decide():
        """(1) the decision arithmetic only."""
        M = torch.exp((score - row_max).clamp(min=-60.0))
        D_hat = den_sel + (n_b * torch.exp((zbar - row_max).clamp(min=-60.0))).sum(-1, keepdim=True)
        return ((M / D_hat) > theta).sum(-1)

    def sel_adaptive():
        """(1) + (2): decision plus the padded per-row index list."""
        M = torch.exp((score - row_max).clamp(min=-60.0))
        D_hat = den_sel + (n_b * torch.exp((zbar - row_max).clamp(min=-60.0))).sum(-1, keepdim=True)
        cnt = ((M / D_hat) > theta).sum(-1).clamp(min=1, max=a.m_cap)
        mmax = int(cnt.max())
        idx = score.topk(mmax, dim=-1).indices
        rank = torch.arange(mmax, device=dev).view(1, mmax)
        return torch.where(rank < cnt.unsqueeze(-1), idx, torch.full_like(idx, -1)), cnt

    # ---- stage 3: the kernel, at several padded widths -------------------------
    def make_exp(m: int, ragged_cnt: torch.Tensor = None):
        base = torch.stack([torch.randperm(K - a.bin_size, device=dev)[:m] for _ in range(BQ)])
        e = (base.unsqueeze(-1) + torch.arange(a.bin_size, device=dev)).reshape(BQ, m * a.bin_size)
        if ragged_cnt is not None:
            rank = torch.arange(m, device=dev).view(1, m, 1).expand(BQ, m, a.bin_size)
            e = torch.where(rank.reshape(BQ, -1) < ragged_cnt.view(BQ, 1),
                            e, torch.full_like(e, -1))
        return e.int()

    ln2 = log_n / math.log(2.0)
    _ = ln2

    def kernel(exp_idx):
        return lambda: decode_expand_attention(
            q, k, v, sel, exp_idx, kbar, vbar, log_n, sm_scale=sc)

    res = {"config": {"B": B, "Hq": Hq, "Hkv": Hkv, "K": K, "D": D,
                      "density": a.density, "bin_size": a.bin_size, "NB": NB,
                      "n_sel": n_sel, "m_fixed": m0, "m_cap": a.m_cap}}

    print(f"BQ={BQ} BH={BH} K={K} n_sel={n_sel} NB={NB} m_fixed={m0}\n")

    t = {}
    t["bin_moments (shared)"] = med(bin_moments, dev, it=a.iters)
    t["select fixed (topk m)"] = med(sel_fixed, dev, it=a.iters)
    t["decide only (adaptive)"] = med(decide, dev, it=a.iters)
    t["select adaptive (decide+ragged idx)"] = med(lambda: sel_adaptive(), dev, it=a.iters)

    print("--- per-decode-step CPU-side stages (ms) ---")
    for kk, vv in t.items():
        print(f"  {kk:40s} {vv:8.4f}")
    extra = t["select adaptive (decide+ragged idx)"] - t["select fixed (topk m)"]
    print(f"\n  adaptive decision overhead              {extra:+8.4f} ms")
    res["stages_ms"] = t
    res["decision_overhead_ms"] = extra

    # kernel cost as a function of the PADDED width -- this is what raggedness costs
    print("\n--- kernel cost vs padded expansion width (ms) ---")
    kt = {}
    for m in (0, 1, 2, 4, 8, 16, 32, 64):
        if m == 0:
            e = torch.full((BQ, a.bin_size), -1, device=dev, dtype=torch.int32)
        else:
            e = make_exp(m)
        kt[m] = med(kernel(e), dev, it=a.iters)
        print(f"  m_pad={m:<3d}  tokens={m * a.bin_size:<5d}  {kt[m]:8.4f}")
    res["kernel_ms_by_padded_m"] = kt

    # and with a REALISTIC ragged distribution: mean m0, but padded to the max
    print("\n--- ragged: mean budget m0 but padded to the batch max ---")
    rag = {}
    for mmax in (8, 16, 32, 64):
        # counts with mean ~m0 and maximum mmax (geometric-ish, matching the measured skew)
        cnt = torch.randint(1, max(2, 2 * m0), (BQ,), device=dev)
        cnt[torch.randperm(BQ, device=dev)[: max(1, BQ // 50)]] = mmax
        e = make_exp(mmax, ragged_cnt=cnt)
        ms = med(kernel(e), dev, it=a.iters)
        rag[mmax] = {"ms": ms, "mean_m": float(cnt.float().mean()),
                     "max_m": int(cnt.max())}
        print(f"  max_m={mmax:<3d}  mean_m={cnt.float().mean():5.2f}  {ms:8.4f} ms   "
              f"vs fixed m={m0}: {ms - kt[m0]:+.4f} ms ({ms / kt[m0]:.2f}x)")
    res["ragged"] = rag

    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\n[saved] {a.json}")


if __name__ == "__main__":
    main()
