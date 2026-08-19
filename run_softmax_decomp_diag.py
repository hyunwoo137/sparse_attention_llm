#!/usr/bin/env python3
"""
V9 diagnostic: WHY aiming beats correcting, in the algebra of the softmax itself.

Every tail estimator in this literature -- ours, vAttention, and every centroid
method (RetroInfer, ClusterKV, CentroidKV, Double-P) -- is ultimately a pair

    (S_hat, mu_hat)      tail mass, tail direction

folded into one softmax with the exactly-computed heavy set:

    o_hat = (num_H + S_hat * mu_hat) / (S_H + S_hat)

Writing w = S_T/(S_H+S_T) and w_hat = S_hat/(S_H+S_hat), the output error splits
EXACTLY (no approximation, verified numerically per row) into

    o_hat - o = (w_hat - w)(mu_T - mu_H)   +   w_hat (mu_hat - mu_T)
                \_________ A _________/       \_______ B _______/
                 denominator / mass            numerator / direction

Both terms are recorded per (layer, head, query row) for every estimator, which
settles three things the previous reports could only guess at:

Q1  V1 concluded "the numerator needs no correction" from the observation that
    handing UTA the exact mu_T changes nothing.  The decomposition says that is a
    MEASUREMENT ARTEFACT: UTA underestimates so badly that w_hat ~ 0, and B is
    multiplied by w_hat.  Fixing the direction of a term that carries no weight
    cannot show up.  Prediction: B is tiny under UTA and LARGE the moment the mass
    is fixed.  If so, "denominator-only" methods have a hard floor at
        ||w (v_bar_T - mu_T)|| / ||o||
    and no Jensen/moment/top-p correction can go below it.

Q2  That floor, computed per bin, is the floor of EVERY centroid-proxy method --
    including one with perfect clustering (bins by true logit) and perfect mass.
    `floor_centroid_*` measures it.  Our expansion is not a better proxy; it
    removes bins from the proxy set, so it is not bounded by that floor at all.

Q3  Which ranking statistic finds the bins worth removing.  The centroid-mass rule
    zbar_b + log n_b (Double-P eq.5, RetroInfer, ClusterKV) versus our upper
    confidence bound zbar_b + c*sigma_b.  A bin holding one 13-nat outlier among 31
    ordinary tokens has an ORDINARY mean, so the mass rule ranks it mid-table; the
    UCB rule ranks it first.  Measured as output error AND as recall of the single
    heaviest tail token.

Q4  (Task 3) An adaptive budget.  Given the same key-side statistics, the residual
    risk of stopping after m bins is bounded by
        R(m) = sum_{b not expanded} M_hat_b / D_hat,     M_hat_b = exp(zbar_b + c sigma_b)
    so m*(tau) = min{m : R(m) <= tau} is computable with a sort and a cumsum.  This
    records the distribution of m*(tau) and the error achieved, so that adaptive and
    uniform allocation can be compared AT EQUAL AVERAGE BUDGET.

Runs the model with the DENSE output, so the generation trajectory is untouched
and every number is measured on the trajectory the model actually takes.

Usage
-----
    python run_softmax_decomp_diag.py --device cuda:1 --heavy-size 0.048 --selector hat
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

from benchmark.ruler32k import Ruler32K  # noqa: E402
from sparse_attention_hub.adapters import ModelAdapterHF  # noqa: E402
from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (  # noqa: E402
    ResearchMasker,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (  # noqa: E402
    HashAttentionTopKMaskerConfig,
    LocalMaskerConfig,
    OracleTopKConfig,
    SinkMaskerConfig,
)
from sparse_attention_hub.sparse_attention.uta_attention import (  # noqa: E402
    UTAAttention,
    UTAAttentionConfig,
    UTAMultiBinAttention,
    UTAMultiBinConfig,
)
from sparse_attention_hub.sparse_attention.utils.kv_utils import (  # noqa: E402
    _get_num_key_value_groups,
    repeat_kv,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask  # noqa: E402
from sparse_attention_hub.sparse_attention.utils.mask_attention_utils import (  # noqa: E402
    get_true_attention_output,
)

NEG = -1e4

RECORDS: List[Dict[str, Any]] = []
SETTINGS: Dict[str, Any] = {
    "bin_size": 32,
    "max_qrows": 4,
    "max_records": 400_000,
    "expand_m": [1, 2, 4, 8, 16, 32],
    "rank_c": [0.0, 1.0, 2.0, 3.0, 4.0],
    "score_bins": [16, 64, 256],       # idealised clustering for the centroid floor
    "tau": [1e0, 3e-1, 1e-1, 3e-2, 1e-2],
    "theta": [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4],
    "m_cap": 64,
    "gamma": [0.999, 0.99, 0.9, 0.7, 0.5, 0.3, 1e-1, 1e-2],
    "kmeans_clusters": [1024],
    "kmeans_iters": 4,
    "adaptive_c": 3.0,
}

# spherical k-means over the key vectors, cached per layer.  This is the partition a
# centroid method builds; it is query-independent by construction, which is exactly
# why it cannot follow any particular query's logit ordering.
_KM_CACHE: Dict[tuple, torch.Tensor] = {}


@torch.no_grad()
def _kmeans_assign(k: torch.Tensor, layer_idx: int, C: int) -> torch.Tensor:
    """(B,H,K,D) keys -> (B,H,K) cluster ids.  Spherical k-means, Lloyd iterations."""
    B, H, K, D = k.shape
    key = (layer_idx, B, H, K, C)
    if key in _KM_CACHE:
        return _KM_CACHE[key]
    kn = torch.nn.functional.normalize(k, dim=-1)
    g = torch.Generator(device="cpu").manual_seed(1234 + layer_idx)
    init = torch.randperm(K, generator=g)[:C].to(k.device)
    cent = kn.index_select(2, init).clone()                       # (B,H,C,D)
    asg = torch.zeros((B, H, K), dtype=torch.long, device=k.device)
    # cap the (B,h,K,C) similarity buffer at ~512 MB
    hstep = max(1, min(H, int(512e6 / max(1, B * K * C * 4))))
    for _ in range(SETTINGS["kmeans_iters"]):
        for h0 in range(0, H, hstep):
            h1 = min(H, h0 + hstep)
            sim = torch.matmul(kn[:, h0:h1], cent[:, h0:h1].transpose(-1, -2))
            asg[:, h0:h1] = sim.argmax(-1)
            del sim
        acc = torch.zeros((B, H, C, D), dtype=k.dtype, device=k.device)
        cnt = torch.zeros((B, H, C, 1), dtype=k.dtype, device=k.device)
        acc.scatter_add_(2, asg.unsqueeze(-1).expand(-1, -1, -1, D), kn)
        cnt.scatter_add_(2, asg.unsqueeze(-1),
                         torch.ones((B, H, K, 1), dtype=k.dtype, device=k.device))
        cent = torch.nn.functional.normalize(acc / cnt.clamp(min=1), dim=-1)
        cent = torch.where(cnt > 0, cent, kn.index_select(2, init))
        del acc, cnt
    if len(_KM_CACHE) > 4:     # only the layers still in flight are ever needed
        _KM_CACHE.clear()
    _KM_CACHE[key] = asg
    return asg

HAT_WEIGHT_FILE = (
    "/database/hyunwoo/hf/HashAttention-1.0/repo/artifacts/"
    "llama3.1-8b-patch.64K.v1.hat_weights.pkl"
)

_BINNER: Dict[tuple, UTAMultiBinAttention] = {}


def _binner(bin_size: int = 32, num_bins: int = 16,
            bin_mode: str = "equalcount") -> UTAMultiBinAttention:
    """Production binning object, so the diagnostic cannot drift from the method."""
    key = (bin_size, num_bins, bin_mode)
    if key not in _BINNER:
        cfg = UTAMultiBinConfig(masker_configs=[], bin_size=bin_size, num_bins=num_bins,
                                bin_mode=bin_mode, kappa_mode="none", var_mode="diag")
        _BINNER[key] = UTAMultiBinAttention(cfg, [])
    return _BINNER[key]


def _relerr(approx: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    return (approx - exact).norm(dim=-1) / exact.norm(dim=-1).clamp(min=1e-12)


class DecompDiagnostic(UTAAttention):
    """Builds the sparse mask, decomposes every estimator's error, returns DENSE."""

    def custom_attention(
        self, module: nn.Module, queries: torch.Tensor, keys: torch.Tensor,
        values: torch.Tensor, attention_mask: Optional[torch.Tensor],
        scaling: float, dropout: float, **kwargs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        sparse_meta_data = kwargs.pop("sparse_meta_data")
        layer_idx = int(kwargs.get("layer_idx", -1) or -1)

        shape = (queries.shape[0], queries.shape[1], queries.shape[2], keys.shape[2])
        sparse_mask = Mask.create_empty_mask(shape, dtype=queries.dtype, device=queries.device)
        for masker in self.maskers:
            sparse_mask = masker.add_mask(
                keys=keys, queries=queries, values=values, attention_mask=attention_mask,
                scaling=scaling, dropout=dropout, sparse_meta_data=sparse_meta_data,
                previous_mask=sparse_mask, **kwargs,
            )

        if (not sparse_mask.is_full_mask()) and len(RECORDS) < SETTINGS["max_records"]:
            try:
                self._measure(queries, keys, values, attention_mask, scaling,
                              sparse_mask, layer_idx)
            except Exception as exc:
                print(f"[diag] layer {layer_idx} skipped: {type(exc).__name__}: {exc}",
                      flush=True)

        return get_true_attention_output(
            module, queries, keys, values, attention_mask, scaling, dropout, **kwargs)

    # ------------------------------------------------------------------ core --
    @torch.no_grad()
    def _measure(self, queries, keys, values, attention_mask, scaling,
                 sparse_mask: Mask, layer_idx: int) -> None:
        qr = min(SETTINGS["max_qrows"], queries.shape[2])
        ngroups = _get_num_key_value_groups(queries, keys)
        k = repeat_kv(keys, ngroups).to(torch.float32)
        v = repeat_kv(values, ngroups).to(torch.float32)
        q = queries[:, :, -qr:, :].to(torch.float32)

        raw = torch.matmul(q, k.transpose(2, 3)) * scaling
        if attention_mask is not None:
            raw = raw + attention_mask[:, :, -qr:, : k.shape[-2]].to(torch.float32)

        valid = raw > NEG
        sel = (sparse_mask.get_dense_mask()[:, :, -qr:, :] != 0) & valid
        tail = (~sel) & valid
        tf = tail.to(torch.float32)

        n_tail = tf.sum(-1)
        ok = n_tail >= 2
        if not bool(ok.any()):
            return

        # ---------------- exact reference in shifted space --------------------
        row_max = raw.masked_fill(~valid, float("-inf")).amax(-1, keepdim=True)
        E = torch.exp(raw - row_max) * valid
        eH, eT = E * sel, E * tf
        S_H = eH.sum(-1, keepdim=True)
        S_T = eT.sum(-1, keepdim=True)
        num_H = torch.matmul(eH, v)
        num_T = torch.matmul(eT, v)
        mu_H = num_H / S_H.clamp(min=1e-30)
        mu_T = num_T / S_T.clamp(min=1e-30)
        o_exact = (num_H + num_T) / (S_H + S_T).clamp(min=1e-30)
        onorm = o_exact.norm(dim=-1).clamp(min=1e-12)
        w = (S_T / (S_H + S_T).clamp(min=1e-30))                       # (B,H,q,1)

        rec: Dict[str, torch.Tensor] = {}

        # ---------------- tail concentration ----------------------------------
        # participation ratio: how many tail tokens the tail MASS is really made of.
        # (sum e)^2 / sum e^2 == 1 for a single token, == n for a flat tail.
        pr = (S_T.squeeze(-1) ** 2) / (eT * eT).sum(-1).clamp(min=1e-30)
        e_top, i_top = eT.topk(min(8, eT.shape[-1]), dim=-1)
        rec["n_tail"] = n_tail
        rec["tail_weight_w"] = w.squeeze(-1)
        rec["participation_ratio"] = pr
        rec["pr_over_ntail"] = pr / n_tail.clamp(min=1)
        rec["top1_mass_share"] = e_top[..., 0] / S_T.squeeze(-1).clamp(min=1e-30)
        rec["top8_mass_share"] = e_top.sum(-1) / S_T.squeeze(-1).clamp(min=1e-30)

        # ---------------- global tail moments (UTA's view) ---------------------
        Nt = n_tail.unsqueeze(-1).clamp(min=1)
        zbar = (raw * tf).sum(-1, keepdim=True) / Nt
        zsq = (raw * raw * tf).sum(-1, keepdim=True) / Nt
        sigma2 = (zsq - zbar * zbar).clamp(min=0)
        v_bar_T = torch.matmul(tf, v) / Nt
        S_uta = Nt * torch.exp(zbar - row_max)
        rec["sigma2_tail"] = sigma2.squeeze(-1)

        # ---------------- THE decomposition ------------------------------------
        def decomp(name: str, S_hat: torch.Tensor, mu_hat: torch.Tensor) -> None:
            """Record ||A||, ||B||, ||A+B|| and the identity residual for one estimator."""
            w_hat = S_hat / (S_H + S_hat).clamp(min=1e-30)
            o_hat = (num_H + S_hat * mu_hat) / (S_H + S_hat).clamp(min=1e-30)
            A = (w_hat - w) * (mu_T - mu_H)          # denominator / mass term
            Bt = w_hat * (mu_hat - mu_T)             # numerator / direction term
            rec[f"err_{name}"] = _relerr(o_hat, o_exact)
            rec[f"A_{name}"] = A.norm(dim=-1) / onorm
            rec[f"B_{name}"] = Bt.norm(dim=-1) / onorm
            # identity check: (o_hat - o) - (A + B) must be ~0 to fp32 precision
            rec[f"resid_{name}"] = ((o_hat - o_exact) - (A + Bt)).norm(dim=-1) / onorm
            rec[f"what_{name}"] = w_hat.squeeze(-1)

        # base UTA and the two mass corrections
        decomp("uta", S_uta, v_bar_T)
        decomp("uta_j1", Nt * torch.exp(zbar + sigma2 / 2 - row_max), v_bar_T)
        decomp("uta_j2", Nt * torch.exp(zbar + torch.log1p(sigma2 / 2) - row_max), v_bar_T)
        # the two one-sided oracles: this is the pair that makes Q1 decidable
        decomp("oracleS", S_T, v_bar_T)          # perfect mass, mean value  -> B unmasked
        decomp("oraclemu", S_uta, mu_T)          # UTA mass, perfect value   -> A alone

        # ---------------- adjacency bins (the production partition) ------------
        m_bin = SETTINGS["bin_size"]
        mb = _binner(m_bin)
        bid, nb = mb._bin_ids_equalcount(tf)
        shp = tf.shape[:-1] + (nb + 1,)
        n_b = torch.zeros(shp, dtype=tf.dtype, device=tf.device)
        n_b = n_b.scatter_add_(-1, bid, tf)[..., :nb]
        den_b = n_b.clamp(min=1)
        vbar_b = mb._scatter_mean(v, tf, bid, nb, den_b)                # (B,H,q,nb,D)
        zbar_b, s2_b = mb._bin_moments_scatter(q, k, tf, bid, nb, den_b, scaling)
        sig_b = s2_b.clamp(min=0).sqrt()
        live_b = n_b > 0

        # true per-bin mass and true per-bin direction (reference only)
        S_b = torch.zeros(shp, dtype=tf.dtype, device=tf.device)
        S_b = S_b.scatter_add_(-1, bid, eT)[..., :nb]                   # (B,H,q,nb)
        num_b = mb._scatter_mean(v, eT, bid, nb, torch.ones_like(den_b))  # sum e_i v_i
        mu_b = num_b / S_b.clamp(min=1e-30).unsqueeze(-1)
        zmax_b = torch.full(shp, float("-inf"), dtype=raw.dtype, device=raw.device)
        zmax_b = zmax_b.scatter_reduce_(
            -1, bid, raw.masked_fill(~tail, float("-inf")),
            reduce="amax", include_self=True)[..., :nb]

        rec["n_bins_live"] = live_b.sum(-1).float()
        rec["headroom"] = (zmax_b.masked_fill(~live_b, float("-inf")).amax(-1)
                           - zbar.squeeze(-1))

        def agg(mass_b: torch.Tensor, val_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Collapse per-bin (mass, value) into the tail's effective (S_hat, mu_hat)."""
            S_hat = mass_b.sum(-1, keepdim=True)
            mu_hat = (torch.einsum("bhqn,bhqnd->bhqd", mass_b, val_b)
                      / S_hat.clamp(min=1e-30))
            return S_hat, mu_hat

        w_plain = (n_b * torch.exp(zbar_b - row_max)) * live_b
        decomp("mb32", *agg(w_plain, vbar_b))
        w_j2 = (n_b * torch.exp(zbar_b + torch.log1p(s2_b / 2) - row_max)) * live_b
        decomp("mb32_j2", *agg(w_j2, vbar_b))
        # Q2: the centroid floor -- PERFECT per-bin mass, mean value.  No estimator
        # that keeps a mean-value proxy per bin can beat this, however it gets its mass.
        decomp("floor_centroid_adj32", *agg(S_b * live_b, vbar_b))

        # ...and with an IDEALISED clustering: bins by true logit, i.e. the best any
        # k-means-in-key-space could hope to be for this particular query.
        for NB in SETTINGS["score_bins"]:
            sb = _binner(num_bins=NB, bin_mode="score")
            bid_s, nbs = sb._bin_ids_score(raw, tf, None)
            shp_s = tf.shape[:-1] + (nbs + 1,)
            n_s = torch.zeros(shp_s, dtype=tf.dtype, device=tf.device)
            n_s = n_s.scatter_add_(-1, bid_s, tf)[..., :nbs]
            S_s = torch.zeros(shp_s, dtype=tf.dtype, device=tf.device)
            S_s = S_s.scatter_add_(-1, bid_s, eT)[..., :nbs]
            vbar_s = sb._scatter_mean(v, tf, bid_s, nbs, n_s.clamp(min=1))
            decomp(f"floor_centroid_score{NB}", *agg(S_s * (n_s > 0), vbar_s))

        # ...and with a REAL key-space clustering, which is what RetroInfer /
        # ClusterKV / CentroidKV / Double-P actually build: spherical k-means over the
        # key vectors, query-independent and cached.  Adjacency (0.34) and true-logit
        # bins (0.006) bracket the answer; this says where a real centroid method
        # lands inside that bracket, still given PERFECT mass for every cluster.
        for C in SETTINGS["kmeans_clusters"]:
            asg = _kmeans_assign(k, layer_idx, C)                       # (B,H,K)
            bid_c = torch.where(tail, asg.unsqueeze(2).expand_as(bid),
                                torch.full_like(bid, C))
            shp_c = tf.shape[:-1] + (C + 1,)
            n_c = torch.zeros(shp_c, dtype=tf.dtype, device=tf.device)
            n_c = n_c.scatter_add_(-1, bid_c, tf)[..., :C]
            S_c = torch.zeros(shp_c, dtype=tf.dtype, device=tf.device)
            S_c = S_c.scatter_add_(-1, bid_c, eT)[..., :C]
            vbar_c = mb._scatter_mean(v, tf, bid_c, C, n_c.clamp(min=1))
            decomp(f"floor_centroid_kmeans{C}", *agg(S_c * (n_c > 0), vbar_c))
            # and what that clustering achieves with a DEPLOYABLE mass estimate,
            # i.e. the actual Double-P / RetroInfer approximate-zone contribution
            zc, s2c = mb._bin_moments_scatter(q, k, tf, bid_c, C, n_c.clamp(min=1), scaling)
            decomp(f"kmeans{C}_massest",
                   *agg((n_c * torch.exp(zc - row_max)) * (n_c > 0), vbar_c))

        # ---------------- Q3: ranking rules ------------------------------------
        e_all = torch.exp(raw - row_max)
        num_sel = torch.matmul(e_all * sel, v)
        den_sel = (e_all * sel).sum(-1, keepdim=True)
        ell_b = zbar_b + torch.log(den_b)                # centroid-mass logit
        pad0 = torch.zeros_like(live_b[..., :1])
        # which bin holds the single heaviest tail token (ranking-recall target)
        top_bin = bid.gather(-1, i_top[..., :1]).squeeze(-1).clamp(max=nb - 1)

        def expand_stats(score_b: torch.Tensor, m: int
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """(S_hat, mu_hat, hit) when the top-m bins by `score_b` are made exact."""
            mm = min(m, nb)
            idx = score_b.masked_fill(~live_b, float("-inf")).topk(mm, dim=-1).indices
            is_exp = torch.zeros_like(live_b).scatter_(-1, idx, True) & live_b
            tok = torch.cat([is_exp, pad0], -1).gather(-1, bid) & tail
            e_exp = e_all * tok
            keep = live_b & ~is_exp
            wk = torch.exp(ell_b.masked_fill(~keep, float("-inf")) - row_max)
            wk = torch.where(torch.isfinite(wk), wk, torch.zeros_like(wk))
            S_hat = e_exp.sum(-1, keepdim=True) + wk.sum(-1, keepdim=True)
            mu_hat = ((torch.matmul(e_exp, v)
                       + torch.einsum("bhqn,bhqnd->bhqd", wk, vbar_b))
                      / S_hat.clamp(min=1e-30))
            hit = is_exp.gather(-1, top_bin.unsqueeze(-1)).squeeze(-1).float()
            return S_hat, mu_hat, hit

        gen = torch.Generator(device=raw.device).manual_seed(77)
        rules: Dict[str, torch.Tensor] = {
            # what every centroid method ranks by: size-weighted centroid logit
            "mass": ell_b,
            "massj2": ell_b + torch.log1p(s2_b / 2),
            # ours: upper confidence bound on the bin's MAX logit
            **{f"c{c:g}": zbar_b + c * sig_b for c in SETTINGS["rank_c"]},
            # c chosen by the sub-Gaussian maximal bound instead of by hand
            "ucbln": zbar_b + sig_b * torch.sqrt(2 * torch.log(den_b.clamp(min=1.5))),
            # ceiling and control
            "oracle": zmax_b,
            "rand": torch.rand(n_b.shape, generator=gen, device=raw.device),
        }
        for rname, score_b in rules.items():
            for m in SETTINGS["expand_m"]:
                S_hat, mu_hat, hit = expand_stats(score_b, m)
                decomp(f"exp{m}_{rname}", S_hat, mu_hat)
                rec[f"hit{m}_{rname}"] = hit

        # --- faithful centroid-method reconstruction ---------------------------
        # RetroInfer / ClusterKV / Double-P do not stop at proxies: they also make the
        # top clusters exact.  So the honest head-to-head is
        #     key-space k-means  +  rank by size-weighted centroid logit  +  expand
        # against
        #     adjacency bins     +  rank by zbar_b + c*sigma_b            +  expand
        # equalised on the TOKEN budget, not the cluster count, because k-means
        # clusters have unequal sizes.  Both then read the same number of KV rows.
        for C in SETTINGS["kmeans_clusters"]:
            asg = _kmeans_assign(k, layer_idx, C)
            bid_c = torch.where(tail, asg.unsqueeze(2).expand_as(bid),
                                torch.full_like(bid, C))
            shp_c = tf.shape[:-1] + (C + 1,)
            n_c = torch.zeros(shp_c, dtype=tf.dtype, device=tf.device)
            n_c = n_c.scatter_add_(-1, bid_c, tf)[..., :C]
            live_c = n_c > 0
            vbar_c = mb._scatter_mean(v, tf, bid_c, C, n_c.clamp(min=1))
            zc, s2c = mb._bin_moments_scatter(q, k, tf, bid_c, C, n_c.clamp(min=1), scaling)
            ell_c = zc + torch.log(n_c.clamp(min=1))
            pad0c = torch.zeros_like(live_c[..., :1])
            for rname, sc in (("mass", ell_c),
                              ("c3", zc + 3.0 * s2c.clamp(min=0).sqrt())):
                order_c = sc.masked_fill(~live_c, float("-inf")).argsort(-1, descending=True)
                # expand clusters in rank order until the token budget runs out
                took = n_c.gather(-1, order_c).cumsum(-1)
                for m in SETTINGS["expand_m"]:
                    budget = float(m * m_bin)
                    within = (took - n_c.gather(-1, order_c)) < budget
                    is_exp = torch.zeros_like(live_c).scatter_(
                        -1, order_c, within.contiguous()) & live_c
                    tok = torch.cat([is_exp, pad0c], -1).gather(-1, bid_c) & tail
                    e_exp = e_all * tok
                    keep = live_c & ~is_exp
                    wk = torch.exp(ell_c.masked_fill(~keep, float("-inf")) - row_max)
                    wk = torch.where(torch.isfinite(wk), wk, torch.zeros_like(wk))
                    S_hat = e_exp.sum(-1, keepdim=True) + wk.sum(-1, keepdim=True)
                    mu_hat = ((torch.matmul(e_exp, v)
                               + torch.einsum("bhqn,bhqnd->bhqd", wk, vbar_c))
                              / S_hat.clamp(min=1e-30))
                    decomp(f"km{C}exp{m}_{rname}", S_hat, mu_hat)
                    rec[f"kmtok{C}exp{m}_{rname}"] = tok.sum(-1).float()

        # ---------------- Q4: adaptive budget ----------------------------------
        # Allocation model.  Expanding bin b converts its proxy contribution into an
        # exact one, so the error it removes is at most its own share of the softmax
        # denominator.  With a sub-Gaussian bound on the bin's largest logit,
        #     M_hat_b = exp(zbar_b + c sigma_b)          (one dominant token)
        # is an upper bound on that share's numerator; D_hat is the denominator we
        # already have.  Minimising total error under a total budget is then a
        # water-filling problem, whose stationarity condition is a SINGLE GLOBAL
        # threshold on the marginal:
        #     expand b  <=>  M_hat_b / D_hat_row  >  theta
        # -- not a per-row error target.  Both rules are measured; `cumul` is the
        # per-row-target version and `thresh` the water-filling one.  All inputs are
        # already computed for the ranking, so the only new work is a compare (thresh)
        # or a sort + cumsum (cumul).
        c_ad = SETTINGS["adaptive_c"]
        M_hat = torch.exp((zbar_b + c_ad * sig_b - row_max).clamp(max=60.0))
        M_hat = torch.where(live_b, M_hat, torch.zeros_like(M_hat))
        D_hat = den_sel + (n_b * torch.exp(zbar_b - row_max) * live_b).sum(-1, keepdim=True)
        share = M_hat / D_hat.clamp(min=1e-30)                     # (B,H,q,nb)
        cap = SETTINGS["m_cap"]

        def eval_alloc(name: str, is_exp: torch.Tensor) -> None:
            """Error and budget for an arbitrary per-row set of expanded bins."""
            is_exp = is_exp & live_b
            tok = torch.cat([is_exp, pad0], -1).gather(-1, bid) & tail
            e_exp = e_all * tok
            keep = live_b & ~is_exp
            wk = torch.exp(ell_b.masked_fill(~keep, float("-inf")) - row_max)
            wk = torch.where(torch.isfinite(wk), wk, torch.zeros_like(wk))
            S_hat = e_exp.sum(-1, keepdim=True) + wk.sum(-1, keepdim=True)
            mu_hat = ((torch.matmul(e_exp, v)
                       + torch.einsum("bhqn,bhqnd->bhqd", wk, vbar_b))
                      / S_hat.clamp(min=1e-30))
            decomp(name, S_hat, mu_hat)
            rec[f"m_{name}"] = is_exp.sum(-1).float()

        def topm_mask(score: torch.Tensor, m_row: torch.Tensor) -> torch.Tensor:
            """Per-row top-m_row mask, m_row an integer tensor (B,H,q)."""
            order = score.masked_fill(~live_b, float("-inf")).argsort(dim=-1, descending=True)
            rank = torch.arange(nb, device=raw.device).view(1, 1, 1, nb)
            return torch.zeros_like(live_b).scatter_(
                -1, order, (rank < m_row.unsqueeze(-1)).contiguous())

        # (a) water-filling: one global threshold on the per-bin share
        for th in SETTINGS["theta"]:
            m_row = ((share > th) & live_b).sum(-1).clamp(max=cap)
            eval_alloc(f"thresh{th:g}", topm_mask(share, m_row))

        # (a2) the same water-filling rule with a CALIBRATED mass estimate instead
        # of a bound.  exp(zbar+c*sigma) is exponential in sigma, so its slack varies
        # by orders of magnitude ACROSS rows -- fine for ranking inside a row, useless
        # for comparing rows.  The j2 moment estimate is calibrated in absolute terms.
        M_j2 = (n_b * torch.exp(zbar_b + torch.log1p(s2_b / 2) - row_max)) * live_b
        D_j2 = den_sel + M_j2.sum(-1, keepdim=True)
        share_j2 = M_j2 / D_j2.clamp(min=1e-30)
        for th in SETTINGS["theta"]:
            m_row = ((share_j2 > th) & live_b).sum(-1).clamp(max=cap)
            eval_alloc(f"j2thresh{th:g}", topm_mask(share_j2, m_row))

        # (a3) per-row RELATIVE rule: expand while a bin is within gamma of the row's
        # hottest bin.  Scale-free per row, so the surrogate's cross-row miscalibration
        # cancels exactly -- it only has to get the ORDER right, which is what the UCB
        # is good at.
        top1 = M_hat.amax(-1, keepdim=True).clamp(min=1e-30)
        for gam in SETTINGS["gamma"]:
            m_row = ((M_hat > gam * top1) & live_b).sum(-1).clamp(min=1, max=cap)
            eval_alloc(f"rel{gam:g}", topm_mask(share, m_row))

        # (b) per-row error target: expand until the residual risk drops under tau
        srt, _ = share.sort(dim=-1, descending=True)
        suffix = srt.sum(-1, keepdim=True) - srt.cumsum(-1)
        for tau in SETTINGS["tau"]:
            below = suffix <= tau
            m_row = torch.where(below.any(-1), below.float().argmax(-1) + 1,
                                torch.full_like(n_b[..., 0], float(nb)).long()).clamp(max=cap)
            eval_alloc(f"cumul{tau:g}", topm_mask(share, m_row))

        # (c) ceiling: the same threshold rule with the TRUE per-bin share.  The gap
        # to (a) is what a better surrogate could still buy; the gap from (a) to
        # uniform is what the surrogate already buys.
        share_true = S_b / (S_H + S_T).clamp(min=1e-30)
        for th in SETTINGS["theta"]:
            m_row = ((share_true > th) & live_b).sum(-1).clamp(max=cap)
            eval_alloc(f"othresh{th:g}", topm_mask(share_true, m_row))

        # ---------------- flatten ---------------------------------------------
        keep_row = ok & (S_T.squeeze(-1) > 0)
        idx = keep_row.nonzero(as_tuple=False)
        if idx.numel() == 0:
            return
        heads = idx[:, 1].cpu().numpy()
        flat = {name: t[keep_row].float().cpu().numpy() for name, t in rec.items()}
        for i in range(len(heads)):
            row: Dict[str, Any] = {"layer": layer_idx, "head_idx": int(heads[i])}
            row.update({name: float(arr[i]) for name, arr in flat.items()})
            RECORDS.append(row)


def _masker_configs(heavy_size: float, selector: str):
    top = (OracleTopKConfig(heavy_size=heavy_size) if selector == "oracle"
           else HashAttentionTopKMaskerConfig(
               heavy_size=heavy_size, hat_bits=32, hat_mlp_layers=3,
               hat_mlp_hidden_size=128, hat_mlp_activation="silu",
               hat_weight_file=HAT_WEIGHT_FILE))
    return [SinkMaskerConfig(sink_size=0.001), LocalMaskerConfig(window_size=0.001), top]


# ------------------------------------------------------------------ analysis --
def analyse(df: pd.DataFrame, out_dir: Path, tag: str) -> None:
    line = "=" * 92
    med = df.median(numeric_only=True)
    print(f"\n{line}\nSOFTMAX DECOMPOSITION  [{tag}]  rows={len(df)}  "
          f"layers={df.layer.nunique()}  heads={df.head_idx.nunique()}\n{line}")

    resid = [c for c in df.columns if c.startswith("resid_")]
    print(f"\n[identity check]  max median residual over {len(resid)} estimators: "
          f"{max(med[c] for c in resid):.3e}   (must be ~1e-7)")

    print("\n--- tail structure ---")
    for c in ("n_tail", "tail_weight_w", "participation_ratio", "pr_over_ntail",
              "top1_mass_share", "top8_mass_share", "sigma2_tail", "headroom",
              "n_bins_live"):
        if c in med:
            print(f"  {c:24s} median {med[c]:12.5g}   mean {df[c].mean():12.5g}")

    print("\n--- Q1: where the error lives (median, relative to ||o||) ---")
    print(f"  {'estimator':26s} {'err':>10s} {'A(mass)':>10s} {'B(dir)':>10s} {'w_hat':>10s}")
    for nm in ("uta", "uta_j1", "uta_j2", "oracleS", "oraclemu", "mb32", "mb32_j2",
               "floor_centroid_adj32",
               *[f"floor_centroid_score{n}" for n in SETTINGS["score_bins"]],
               "exp1_c3", "exp4_c3", "exp8_c3", "exp16_c3", "exp32_c3"):
        if f"err_{nm}" in med:
            print(f"  {nm:26s} {med[f'err_{nm}']:10.5f} {med[f'A_{nm}']:10.5f} "
                  f"{med[f'B_{nm}']:10.5f} {med[f'what_{nm}']:10.5f}")
    if "B_uta" in med and "B_oracleS" in med:
        print(f"\n  B under UTA {med['B_uta']:.5f}  ->  B once the mass is fixed "
              f"{med['B_oracleS']:.5f}   ({med['B_oracleS'] / max(med['B_uta'], 1e-12):.1f}x)")
        print(f"  => denominator-only floor = {med['err_oracleS']:.5f} "
              f"({100 * med['err_oracleS'] / med['err_uta']:.1f}% of UTA's error remains)")

    print("\n--- Q3b: faithful centroid reconstruction, matched TOKEN budget ---")
    print(f"  {'method':34s}" + "  ".join(f"{f'm={m}':>9s}" for m in SETTINGS["expand_m"]))
    for C in SETTINGS["kmeans_clusters"]:
        for r in ("mass", "c3"):
            cells = "  ".join(f"{med.get(f'err_km{C}exp{m}_{r}', float('nan')):9.5f}"
                              for m in SETTINGS["expand_m"])
            print(f"  kmeans{C} + {r:<8s} rank        {cells}")
            tk = "  ".join(f"{df.get(f'kmtok{C}exp{m}_{r}', pd.Series([np.nan])).mean():9.0f}"
                           for m in SETTINGS["expand_m"])
            print(f"    (tokens read)                {tk}")
    for r in ("mass", "c3"):
        cells = "  ".join(f"{med.get(f'err_exp{m}_{r}', float('nan')):9.5f}"
                          for m in SETTINGS["expand_m"])
        print(f"  adjacency32 + {r:<8s} rank     {cells}")

    print("\n--- Q3: ranking rules (median output error) ---")
    rules = ["mass", "massj2", "c0", "c1", "c2", "c3", "c4", "ucbln", "oracle", "rand"]
    hdr = "  ".join(f"{f'm={m}':>9s}" for m in SETTINGS["expand_m"])
    print(f"  {'rule':10s} {hdr}")
    for r in rules:
        cells = "  ".join(f"{med.get(f'err_exp{m}_{r}', float('nan')):9.5f}"
                          for m in SETTINGS["expand_m"])
        print(f"  {r:10s} {cells}")
    print(f"\n  recall of the single heaviest tail token's bin (mean)")
    print(f"  {'rule':10s} {hdr}")
    for r in rules:
        cells = "  ".join(f"{df.get(f'hit{m}_{r}', pd.Series([np.nan])).mean():9.4f}"
                          for m in SETTINGS["expand_m"])
        print(f"  {r:10s} {cells}")

    print("\n--- Q2: centroid floors (perfect per-cluster mass, mean value) ---")
    for c in [c for c in med.index if c.startswith("err_floor_centroid_")
              or c.startswith("err_kmeans")]:
        print(f"  {c[4:]:30s} {med[c]:10.5f}")

    print("\n--- Q4: error vs budget (median err | mean err | mean m) ---")
    print("  uniform allocation")
    for m in SETTINGS["expand_m"]:
        c = f"exp{m}_c3"
        print(f"    m={m:<4d}            {med[f'err_{c}']:9.5f} | "
              f"{df[f'err_{c}'].mean():9.5f} | {float(m):7.2f}")
    for label, keys in (("water-filling threshold, UCB mass  (expand iff share > theta)",
                         [f"thresh{t:g}" for t in SETTINGS["theta"]]),
                        ("water-filling threshold, j2 mass",
                         [f"j2thresh{t:g}" for t in SETTINGS["theta"]]),
                        ("per-row relative rule    (expand while M_b > gamma * M_max)",
                         [f"rel{g:g}" for g in SETTINGS["gamma"]]),
                        ("per-row risk target      (expand until suffix risk < tau)",
                         [f"cumul{t:g}" for t in SETTINGS["tau"]]),
                        ("ORACLE threshold         (true per-bin share)",
                         [f"othresh{t:g}" for t in SETTINGS["theta"]])):
        print(f"  {label}")
        for kk in keys:
            if f"err_{kk}" not in med:
                continue
            mm = df[f"m_{kk}"]
            print(f"    {kk:<18s} {med[f'err_{kk}']:9.5f} | "
                  f"{df[f'err_{kk}'].mean():9.5f} | {mm.mean():7.2f}"
                  f"   (p50 {mm.quantile(.5):.0f}, p90 {mm.quantile(.9):.0f}, "
                  f"p99 {mm.quantile(.99):.0f}, max {mm.max():.0f})")

    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"decomp_{tag}.parquet"
    df.to_parquet(p)
    print(f"\n[saved] {p}  ({len(df)} rows, {len(df.columns)} columns)")


def main() -> None:
    p = argparse.ArgumentParser(description="softmax numerator/denominator decomposition")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--heavy-size", type=float, default=0.048)
    p.add_argument("--selector", default="hat", choices=["oracle", "hat"])
    p.add_argument("--tag", default=None)
    p.add_argument("--subsets", default="qa_1,niah_multikey_3")
    p.add_argument("--samples-per-subset", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=3)
    p.add_argument("--max-qrows", type=int, default=2)
    p.add_argument("--bin-size", type=int, default=32)
    p.add_argument("--max-records", type=int, default=400_000)
    p.add_argument("--output-dir", default="./results_v9_decomp")
    p.add_argument("--analyse", default=None)
    args = p.parse_args()

    tag = args.tag or f"{(args.heavy_size + 0.002) * 100:g}pct_{args.selector}"
    out_dir = Path(args.output_dir)
    SETTINGS["bin_size"] = args.bin_size
    SETTINGS["max_qrows"] = args.max_qrows
    SETTINGS["max_records"] = args.max_records

    if args.analyse:
        analyse(pd.read_parquet(args.analyse), out_dir, tag)
        return

    print(f"[diag] density={(args.heavy_size + 0.002) * 100:g}%  device={args.device}  "
          f"selector={args.selector}  bin_size={args.bin_size}", flush=True)

    adapter = ModelAdapterHF(
        model_name=args.model,
        sparse_attention_config=UTAAttentionConfig(
            masker_configs=_masker_configs(args.heavy_size, args.selector)),
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=args.device,
    )
    cfg = UTAAttentionConfig(masker_configs=_masker_configs(args.heavy_size, args.selector))
    adapter.sparse_attention = DecompDiagnostic(
        cfg, [ResearchMasker.create_masker_from_config(mc) for mc in cfg.masker_configs])
    if getattr(adapter, "_registered_attention_name", None) is not None:
        adapter._cleanup_attention_registration()

    scratch = out_dir / f"_scratch_{tag}"
    for subset in [s.strip() for s in args.subsets.split(",")]:
        d = scratch / subset
        d.mkdir(parents=True, exist_ok=True)
        print(f"[diag] collecting on {subset} ...", flush=True)
        Ruler32K(subsets_to_run=[subset]).run_benchmark(
            adapter, str(d),
            request_kwargs={"max_requests": args.samples_per_subset,
                            "max_context_length": 32000},
            generation_kwargs={"max_new_tokens": args.max_new_tokens},
        )
        print(f"[diag]   records so far: {len(RECORDS)}", flush=True)
        if len(RECORDS) >= SETTINGS["max_records"]:
            break

    del adapter
    torch.cuda.empty_cache()
    if not RECORDS:
        print("no records collected")
        return
    analyse(pd.DataFrame(RECORDS), out_dir, tag)


if __name__ == "__main__":
    main()
