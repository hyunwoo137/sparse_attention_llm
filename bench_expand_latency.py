#!/usr/bin/env python3
"""Wall-clock for the V5 expansion method's decode step, with and without full QK.

Why a micro-benchmark rather than the end-to-end run: the accuracy harness is an
O(QK) research prototype, so timing it measures the prototype, not the method.  What
we want is the cost of ONE decode step's attention over a 32K context, isolated from
tokenisation, sampling and the rest of the model.

The ladder.  Everything shares the same top-k SELECTION cost, which is not charged to
anyone here; the variants differ only in what they do afterwards:

  dense          full QK + softmax + AV                     -- the thing we replace
  topk_only      read the selected KV, attend to them       -- HAT, tail discarded
  uta            + one cached tail proxy                    -- V3's winner
  expand_fullqk  V5 as currently implemented (builds the whole QK matrix)
  expand_gather  V5 as it would DEPLOY: logits only where they are needed

`expand_gather` is the number that matters.  The full QK in the current code is
prototype convenience, not algorithm: `raw` is consumed at exactly three places --
`valid` (derivable from attention_mask), `row_max` (a shift, obtainable from the kept
terms), and `exp(raw)*exact` (needed only at selected+expanded positions).  So the
deployable step evaluates ~5.6% of the logits and this variant does exactly that.

Two implementation rules, because breaking either turns the comparison into fiction:

  1. The KV cache is NEVER expanded to query-head layout.  Under GQA a real kernel
     reads the (B,Hkv,K,D) tensor through the group id; materialising (B,Hq,K,D)
     would copy hundreds of MB per call and penalise exactly the gather variants.
  2. Bin statistics are CACHED, not rebuilt.  kbar_b / Var_b / vbar_b depend only on
     K and V, so a decode step reads them.  Their read and the O(d)-per-bin dot
     products are charged; building them is amortised over the whole generation.

Everything is expressed as batched GEMMs (bmm) on 3-D tensors so the shapes map onto
cuBLAS directly rather than falling into einsum's permute-and-copy path.

Usage:
    python bench_expand_latency.py --device cuda:1
    python bench_expand_latency.py --device cuda:1 --batch 32 --json out.json
"""

import argparse
import json
import statistics
from typing import Callable, Dict, List, Tuple

import torch


def timeit(fn: Callable[[], None], dev: torch.device,
           warmup: int = 25, iters: int = 200) -> Dict[str, float]:
    """Median / p90 / min ms per call, timed with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(dev)
    times: List[float] = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize(dev)
        times.append(s.elapsed_time(e))
    times.sort()
    return {"median_ms": statistics.median(times),
            "p90_ms": times[int(0.9 * len(times)) - 1],
            "min_ms": times[0]}


class Ctx:
    """One decode step's tensors, shaped like Llama-3.1-8B at 32K.

    Layout convention:
      BH  = B * Hkv     -- KV tensors are (BH, K, D), never expanded to query heads
      g   = Hq // Hkv   -- query heads per KV head; the query is (BH, g, D)
      BQ  = B * Hq      -- flattened query heads, for per-query-head tensors
    """

    def __init__(self, a: argparse.Namespace) -> None:
        d = torch.device(a.device)
        self.dev, self.dtype = d, torch.bfloat16
        B, Hq, Hkv, K, D = a.batch, a.heads, a.kv_heads, a.seq_len, a.head_dim
        self.B, self.Hq, self.Hkv, self.K, self.D = B, Hq, Hkv, K, D
        self.g = Hq // Hkv
        self.BH, self.BQ = B * Hkv, B * Hq
        self.scaling = D ** -0.5

        self.q = torch.randn(self.BH, self.g, D, device=d, dtype=self.dtype)
        self.k = torch.randn(self.BH, K, D, device=d, dtype=self.dtype)
        self.v = torch.randn(self.BH, K, D, device=d, dtype=self.dtype)
        self.qq = self.q * self.q                        # for the diagonal variance

        self.n_sel = int(round(K * (a.density - 0.002)))
        # vAttention splits the SAME total density: half to the selector, half to
        # random sampling, and the sampling half runs as two dependent phases.
        self.v_sel = int(round(K * (a.density / 2 - 0.002)))
        self.v_samp = int(round(K * a.density / 2))
        self.v_base = max(2, int(0.25 * self.v_samp))       # base phase
        self.v_adapt = self.v_samp - self.v_base            # adaptive phase
        self.m, self.bin_size = a.expand_m, a.bin_size
        self.n_bins = (K - self.n_sel) // a.bin_size

        # what the selector hands back: distinct indices per QUERY head
        self.sel = torch.stack([torch.randperm(K, device=d)[:self.n_sel]
                                for _ in range(self.BQ)]).view(self.BH, self.g, self.n_sel)
        self.vsel = self.sel[:, :, :self.v_sel].contiguous()

        # cached, query-independent bin statistics -- per KV head
        nb = self.n_bins
        self.kbar = torch.randn(self.BH, nb, D, device=d, dtype=self.dtype)
        self.kvar = torch.rand(self.BH, nb, D, device=d, dtype=self.dtype)
        self.vbar = torch.randn(self.BH, nb, D, device=d, dtype=self.dtype)
        self.logn = torch.full((self.BH, 1, nb), float(a.bin_size),
                               device=d, dtype=self.dtype).log()
        self.offs = torch.arange(a.bin_size, device=d)

        # flat views + per-(batch,kv-head) row offsets, for row-granular index_select
        self.kflat = self.k.reshape(self.BH * K, D)
        self.vflat = self.v.reshape(self.BH * K, D)
        self.row_off = (torch.arange(self.BH, device=d) * K).view(self.BH, 1)

        # BLOCK view: one row == one whole bin of `bin_size` tokens.  A 256-byte token
        # row is too small to amortise the gather's per-row indirection (measured:
        # 766 GB/s vs 3428 GB/s for a contiguous copy of the same bytes), and the
        # penalty vanishes once a row reaches ~1 KB.  Our expanded bins are contiguous
        # runs by construction, so they can be read at block granularity for free.
        self.nblk = K // a.bin_size
        self.kblk = self.k.reshape(self.BH * self.nblk, D * a.bin_size)
        self.vblk = self.v.reshape(self.BH * self.nblk, D * a.bin_size)
        self.blk_off = (torch.arange(self.BH, device=d) * self.nblk).view(self.BH, 1)

    def read_bins(self, bins: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """bins:(BH,g,m) -> K,V of those whole bins, (BH, g*m*bin_size, D).

        One index per BIN instead of one per token: 8 KB rows instead of 256 B, which
        is what recovers full bandwidth.  Bit-identical to the token-wise gather.
        """
        mm = bins.shape[-1]
        rows = (bins.reshape(self.BH, self.g * mm) + self.blk_off).reshape(-1)
        n = self.g * mm * self.bin_size
        ks = self.kblk.index_select(0, rows).view(self.BH, n, self.D)
        vs = self.vblk.index_select(0, rows).view(self.BH, n, self.D)
        return ks, vs

    def read_kv(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """idx:(BH,g,n) -> K,V slices (BH, g*n, D).  This IS the sparse KV read.

        Row-granular `index_select`, not `gather` with a D-expanded index: a real
        kernel issues ONE index per 256-byte row.  Feeding gather an expanded int64
        index makes it read an index per element -- 8x the bytes of the payload --
        which would make every sparse variant look far worse than it is.
        """
        n = idx.shape[-1]
        rows = (idx.reshape(self.BH, self.g * n) + self.row_off).reshape(-1)
        ks = self.kflat.index_select(0, rows).view(self.BH, self.g * n, self.D)
        vs = self.vflat.index_select(0, rows).view(self.BH, self.g * n, self.D)
        return ks, vs


# --------------------------------------------------------------------- variants --
def make_dense(c: Ctx) -> Callable[[], None]:
    def run() -> None:
        raw = torch.bmm(c.q, c.k.transpose(1, 2)) * c.scaling            # (BH,g,K)
        torch.bmm(torch.softmax(raw.float(), -1).to(c.dtype), c.v)
    return run


def _sel_logits(c: Ctx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ks, vs = c.read_kv(c.sel)                                            # (BH,g*n,D)
    ks = ks.view(c.BQ, c.n_sel, c.D)
    vs = vs.view(c.BQ, c.n_sel, c.D)
    qf = c.q.reshape(c.BQ, 1, c.D)
    z = torch.bmm(qf, ks.transpose(1, 2)) * c.scaling                    # (BQ,1,n_sel)
    return z, vs, qf


def make_topk_only(c: Ctx) -> Callable[[], None]:
    def run() -> None:
        z, vs, _ = _sel_logits(c)
        torch.bmm(torch.softmax(z.float(), -1).to(c.dtype), vs)
    return run


def _bin_terms(c: Ctx) -> Tuple[torch.Tensor, torch.Tensor]:
    """zbar_b and sigma_b^2, (BH,g,nb).  One read of the stats serves all g heads."""
    zbar = torch.bmm(c.q, c.kbar.transpose(1, 2)) * c.scaling
    s2 = torch.bmm(c.qq, c.kvar.transpose(1, 2)) * (c.scaling ** 2)
    return zbar, s2


def make_uta(c: Ctx) -> Callable[[], None]:
    def run() -> None:
        z, vs, _ = _sel_logits(c)
        zbar, _ = _bin_terms(c)
        zt = zbar.mean(-1, keepdim=True).reshape(c.BQ, 1, 1)             # one proxy
        vt = c.vbar.mean(1, keepdim=True).repeat_interleave(c.g, 0)      # (BQ,1,D)
        rmax = torch.maximum(z.amax(-1, keepdim=True), zt)
        e = torch.exp(z.float() - rmax.float()).to(c.dtype)
        w = (torch.exp(zt.float() - rmax.float()) * (c.K - c.n_sel)).to(c.dtype)
        num = torch.bmm(e, vs) + w * vt
        num / (e.sum(-1, keepdim=True) + w)
    return run


def make_expand_fullqk(c: Ctx) -> Callable[[], None]:
    """V5 exactly as the accuracy runs execute it: the whole QK matrix is built."""
    def run() -> None:
        raw = torch.bmm(c.q, c.k.transpose(1, 2)) * c.scaling            # <-- full QK
        zbar, s2 = _bin_terms(c)
        score = zbar + 3.0 * s2.clamp(min=0).sqrt()
        top = score.topk(c.m, dim=-1).indices                            # (BH,g,m)
        rmax = raw.amax(-1, keepdim=True)
        e = torch.exp(raw.float() - rmax.float()).to(c.dtype)
        num = torch.bmm(e, c.v)
        gath = torch.gather(c.vbar, 1,
                            top.reshape(c.BH, c.g * c.m, 1).expand(-1, -1, c.D))
        num + gath.view(c.BH, c.g, c.m, c.D).sum(2)
    return run


def make_expand_gather(c: Ctx) -> Callable[[], None]:
    """V5 as it deploys: logits only at selected + expanded positions."""
    def run() -> None:
        z, vs, qf = _sel_logits(c)                                       # selected only
        zbar, s2 = _bin_terms(c)

        # rank bins by the key-side bound, take the top m
        score = zbar + 3.0 * s2.clamp(min=0).sqrt()                      # (BH,g,nb)
        top = score.topk(c.m, dim=-1).indices                            # (BH,g,m)

        # expanded bins -> read whole bins at block granularity (see read_bins)
        ne = c.m * c.bin_size
        ke, ve = c.read_bins(top)
        ke = ke.view(c.BQ, ne, c.D)
        ve = ve.view(c.BQ, ne, c.D)
        ze = torch.bmm(qf, ke.transpose(1, 2)) * c.scaling               # expanded only

        # surviving bins keep their proxy; expanded bins drop out
        ell = (zbar + c.logn).scatter(-1, top, float("-inf"))            # (BH,g,nb)
        ellq = ell.reshape(c.BQ, 1, c.n_bins)

        rmax = torch.maximum(torch.maximum(z.amax(-1, keepdim=True),
                                           ze.amax(-1, keepdim=True)),
                             ellq.amax(-1, keepdim=True))
        e = torch.exp(z.float() - rmax.float()).to(c.dtype)
        ee = torch.exp(ze.float() - rmax.float()).to(c.dtype)
        w = torch.exp(ellq.float() - rmax.float()).to(c.dtype)           # (BQ,1,nb)
        num = (torch.bmm(e, vs) + torch.bmm(ee, ve)
               + torch.bmm(w.view(c.BH, c.g, c.n_bins), c.vbar).reshape(c.BQ, 1, c.D))
        den = (e.sum(-1, keepdim=True) + ee.sum(-1, keepdim=True)
               + w.sum(-1, keepdim=True))
        num / den
    return run


def make_vattention(c: Ctx) -> Callable[[], None]:
    """vAttention: top-k at rho/2, then adaptive random sampling for the other rho/2.

    Two things here are structural, not incidental, and both cost latency:
      * the phases are SEQUENTIAL -- the adaptive budget is a function of the base
        phase's standard deviation, so the second gather cannot be issued until the
        first has been reduced;
      * every sampled index is drawn per QUERY, so nothing is shared across the batch
        or the GQA group, unlike our query-independent bin statistics.
    """
    def run() -> None:
        # --- selector half -----------------------------------------------------
        ks, vs = c.read_kv(c.vsel)
        ks = ks.view(c.BQ, c.v_sel, c.D)
        vs = vs.view(c.BQ, c.v_sel, c.D)
        qf = c.q.reshape(c.BQ, 1, c.D)
        z = torch.bmm(qf, ks.transpose(1, 2)) * c.scaling

        # --- sampling phase 1: base samples -> std estimate ---------------------
        ib = torch.randint(c.K, (c.BH, c.g, c.v_base), device=c.dev)
        kb, vb = c.read_kv(ib)
        kb = kb.view(c.BQ, c.v_base, c.D)
        vb = vb.view(c.BQ, c.v_base, c.D)
        zb = torch.bmm(qf, kb.transpose(1, 2)) * c.scaling
        rmax = torch.maximum(z.amax(-1, keepdim=True), zb.amax(-1, keepdim=True))
        eb = torch.exp(zb.float() - rmax.float())
        std = eb.std(-1, keepdim=True)                      # <-- the dependency

        # --- sampling phase 2: adaptive budget, drawn AFTER the reduction -------
        budget = (0.674 * std * c.K / (0.25 * eb.sum(-1, keepdim=True).clamp(min=1e-8)))
        budget = budget.pow(2).clamp(1, float(c.K))         # per-row, ragged in a kernel
        ia = torch.randint(c.K, (c.BH, c.g, c.v_adapt), device=c.dev)
        ka, va = c.read_kv(ia)
        ka = ka.view(c.BQ, c.v_adapt, c.D)
        va = va.view(c.BQ, c.v_adapt, c.D)
        za = torch.bmm(qf, ka.transpose(1, 2)) * c.scaling

        e = torch.exp(z.float() - rmax.float()).to(c.dtype)
        ea = torch.exp(za.float() - rmax.float()).to(c.dtype)
        scale = (budget / max(c.v_adapt, 1)).to(c.dtype)
        num = torch.bmm(e, vs) + torch.bmm(ea * scale, va) + torch.bmm(eb.to(c.dtype), vb)
        den = (e.sum(-1, keepdim=True) + (ea * scale).sum(-1, keepdim=True)
               + eb.to(c.dtype).sum(-1, keepdim=True))
        num / den
    return run


VARIANTS = {
    "dense (full attention)": make_dense,
    "topk_only (HAT, tail discarded)": make_topk_only,
    "uta (topk + 1 cached proxy)": make_uta,
    "expand_fullqk (V5 as implemented)": make_expand_fullqk,
    "expand_gather (V5 as deployed)": make_expand_gather,
    "vattention (topk rho/2 + sampling)": make_vattention,
}


def main() -> None:
    p = argparse.ArgumentParser(description="Decode-step wall clock for V5 expansion")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=32768)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--density", type=float, default=0.05)
    p.add_argument("--expand-m", type=int, default=8)
    p.add_argument("--bin-size", type=int, default=32)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--json", default=None)
    a = p.parse_args()

    torch.manual_seed(0)
    if a.device.startswith("cuda"):
        # CUDA events belong to the current device's context; pin it before creating any
        torch.cuda.set_device(torch.device(a.device))
    c = Ctx(a)

    kv_mb = 2 * c.BH * c.K * c.D * 2 / 1e6
    sel_mb = 2 * c.BQ * c.n_sel * c.D * 2 / 1e6
    stat_mb = 3 * c.BH * c.n_bins * c.D * 2 / 1e6
    print(f"\ndevice={torch.cuda.get_device_name(c.dev)}  batch={a.batch}  "
          f"heads={a.heads}/{a.kv_heads} (g={c.g})  K={a.seq_len}  d={a.head_dim}")
    print(f"density={a.density:.0%}  selected/head={c.n_sel}  bins={c.n_bins}  "
          f"expand m={a.expand_m} ({a.expand_m * a.bin_size} tokens)")
    print(f"KV cache {kv_mb:.1f} MB | selected KV {sel_mb:.1f} MB | "
          f"bin stats {stat_mb:.1f} MB (shared by g={c.g} heads)\n")

    out: Dict[str, Dict[str, float]] = {}
    base = None
    print(f"{'variant':38}{'median ms':>11}{'p90 ms':>9}{'vs dense':>10}")
    print("-" * 68)
    for name, mk in VARIANTS.items():
        r = timeit(mk(c), c.dev, iters=a.iters)
        out[name] = r
        if base is None:
            base = r["median_ms"]
        print(f"{name:38}{r['median_ms']:>11.3f}{r['p90_ms']:>9.3f}"
              f"{r['median_ms'] / base:>9.2f}x")
    print()

    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"config": vars(a), "kv_cache_mb": kv_mb,
                       "selected_kv_mb": sel_mb, "bin_stats_mb": stat_mb,
                       "results": out}, fh, indent=2)
        print(f"wrote {a.json}\n")


if __name__ == "__main__":
    main()
