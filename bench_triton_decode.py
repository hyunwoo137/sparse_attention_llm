#!/usr/bin/env python3
"""Fused Triton decode step vs the PyTorch gather path vs dense.

The PyTorch expansion path spends 61% of its time in `index_select`, which pays a
DRAM round trip the algorithm never asked for: the gathered K/V tile is written out
and read back.  A fused kernel gathers the same scattered rows straight into SRAM.
This measures whether that removes the penalty.
"""
import argparse, json, statistics, sys, math
import torch
sys.path.insert(0, "/home/hyunwoo/research/sparse-attention-hub")
from triton_expand_decode import decode_expand_attention


def med(fn, dev, w=25, it=150):
    for _ in range(w):
        fn()
    torch.cuda.synchronize(dev)
    ts = []
    for _ in range(it):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(dev)
        ts.append(s.elapsed_time(e))
    ts.sort()
    return statistics.median(ts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=32768)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--density", type=float, default=0.05)
    p.add_argument("--expand-m", type=int, default=8)
    p.add_argument("--bin-size", type=int, default=32)
    p.add_argument("--block-n", type=int, default=64)
    p.add_argument("--block-b", type=int, default=64)
    p.add_argument("--warps", type=int, default=4)
    p.add_argument("--iters", type=int, default=150)
    p.add_argument("--json", default=None)
    a = p.parse_args()

    torch.manual_seed(0)
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    dt = torch.bfloat16
    B, Hq, Hkv, K, D = a.batch, a.heads, a.kv_heads, a.seq_len, a.head_dim
    g, BH, BQ = Hq // Hkv, B * Hkv, B * Hq
    sc = D ** -0.5
    n_sel = int(round(K * (a.density - 0.002)))
    nb = (K - n_sel) // a.bin_size
    ne = a.expand_m * a.bin_size

    q = torch.randn(BQ, D, device=dev, dtype=dt)
    k = torch.randn(BH, K, D, device=dev, dtype=dt)
    v = torch.randn(BH, K, D, device=dev, dtype=dt)
    sel = torch.stack([torch.randperm(K, device=dev)[:n_sel] for _ in range(BQ)]).int()
    exp = torch.stack([torch.randperm(K - a.bin_size, device=dev)[:a.expand_m]
                       for _ in range(BQ)])
    exp = (exp.unsqueeze(-1) + torch.arange(a.bin_size, device=dev)).reshape(BQ, ne).int()
    kbar = torch.randn(BH, nb, D, device=dev, dtype=dt)
    vbar = torch.randn(BH, nb, D, device=dev, dtype=dt)
    ell = torch.full((BQ, nb), math.log2(float(a.bin_size)), device=dev, dtype=torch.float32)

    # dense reference, GQA-aware, no KV expansion
    qg = q.view(BH, g, D)
    def dense():
        raw = torch.bmm(qg, k.transpose(1, 2)) * sc
        torch.bmm(torch.softmax(raw.float(), -1).to(dt), v)

    # PyTorch gather path (what V7 measured)
    kflat, vflat = k.reshape(BH * K, D), v.reshape(BH * K, D)
    off = (torch.arange(BH, device=dev) * K).view(BH, 1)
    sel_r = (sel.view(BH, g * n_sel).long() + off).reshape(-1)
    exp_r = (exp.view(BH, g * ne).long() + off).reshape(-1)
    def torch_gather_path():
        ks = kflat.index_select(0, sel_r).view(BQ, n_sel, D)
        vs = vflat.index_select(0, sel_r).view(BQ, n_sel, D)
        ke = kflat.index_select(0, exp_r).view(BQ, ne, D)
        ve = vflat.index_select(0, exp_r).view(BQ, ne, D)
        qf = q.view(BQ, 1, D)
        z = torch.bmm(qf, ks.transpose(1, 2)) * sc
        ze = torch.bmm(qf, ke.transpose(1, 2)) * sc
        zb = torch.bmm(qg, kbar.transpose(1, 2)).view(BQ, 1, nb) * sc + ell.view(BQ, 1, nb)
        rm = torch.maximum(torch.maximum(z.amax(-1, True), ze.amax(-1, True)), zb.amax(-1, True))
        e, ee, w = [torch.exp(x.float() - rm.float()).to(dt) for x in (z, ze, zb)]
        num = (torch.bmm(e, vs) + torch.bmm(ee, ve)
               + torch.bmm(w.view(BH, g, nb), vbar).view(BQ, 1, D))
        return num / (e.sum(-1, True) + ee.sum(-1, True) + w.sum(-1, True))

    def triton_path():
        return decode_expand_attention(q, k, v, sel, exp, kbar, vbar, ell, sm_scale=sc,
                                       block_n=a.block_n, block_b=a.block_b,
                                       num_warps=a.warps)

    # correctness at this scale
    o_t = triton_path().float()
    o_p = torch_gather_path().view(BQ, D).float()
    rel = ((o_t - o_p).norm(dim=-1) / o_p.norm(dim=-1))
    kv_mb = 2 * BH * K * D * 2 / 1e6

    print(f"\ndevice={torch.cuda.get_device_name(dev)}  B={B}  heads={Hq}/{Hkv} (g={g})  "
          f"K={K}  d={D}  density={a.density:.0%}")
    print(f"selected/head={n_sel}  bins={nb}  expand m={a.expand_m} ({ne} tokens)  "
          f"BLOCK_N={a.block_n} BLOCK_B={a.block_b} warps={a.warps}")
    print(f"KV cache {kv_mb:.1f} MB   triton vs torch 상대오차 중앙 {rel.median():.2e}\n")

    res = {}
    print(f"{'variant':40}{'median ms':>11}{'vs dense':>10}")
    print("-" * 62)
    base = med(dense, dev, it=a.iters); res["dense"] = base
    print(f"{'dense (full attention)':40}{base:>11.3f}{1.0:>9.2f}x")
    for name, fn in (("expand, PyTorch gather", torch_gather_path),
                     ("expand, fused Triton", triton_path)):
        t = med(fn, dev, it=a.iters); res[name] = t
        print(f"{name:40}{t:>11.3f}{t / base:>9.2f}x")
    print()
    if a.json:
        json.dump({"config": vars(a), "rel_err": float(rel.median()), "ms": res},
                  open(a.json, "w"), indent=2)


if __name__ == "__main__":
    main()
