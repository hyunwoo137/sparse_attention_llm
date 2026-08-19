#!/usr/bin/env python3
"""
Experiment 2: Tail-structure diagnostic  (no benchmark score, pure measurement).

Runs the model with the *dense* attention output (so the generation trajectory is
never perturbed) while, on every sparse-attention call, measuring the exact
properties of the tail set that UTA has to approximate.

It answers four questions in a single pass:

Q1  Why did UTA+Jensen(alpha=1) blow up?
    The tail is right-truncated by top-k selection, so
        log S_T <= zmax_tail + log N        (exact, no assumption)
    whereas the log-normal correction predicts zbar + sigma^2/2 + log N.
    -> `viol_lognormal` = fraction of rows where sigma^2/2 > (zmax_tail - zbar),
       i.e. the correction is provably impossible.  Also reports the exact Jensen
       gap K1_exact = log S_T - zbar - log N against the sigma^2/2 prediction.

Q2  Denominator or numerator -- which one should we fix first?
    `err_oracle_S`  : exact S_T, but still v_bar for the value  (numerator-limited)
    `err_oracle_mu` : UTA's S_T, but exact mu_T                 (denominator-limited)
    Compare both against `err_uta`.

Q3  GO/NO-GO for block-wise multi-proxy:
    `var_ratio_B` = (n_b-weighted mean of within-block logit variance) / sigma^2_tail
    for several block sizes B.  If this does not fall well below 1, partitioning the
    tail buys nothing and the whole block-proxy direction is dead.

Q4  Does the block proxy actually reduce output error, and does a *bounded* Jensen
    correction help on top of it?
    `err_block_{B}_{plain|j2|j3}` where the per-block correction kappa_b is
        plain : 0
        j2    : log(1 + sigma_b^2/2)        (moment expansion, never explodes)
        j3    : min(sigma_b^2/2, zmax_b - zbar_b)   (hard feasibility clamp)

All errors are relative L2 error of the merged attention output for that row.

Usage
-----
    python run_block_tail_diag.py --device cuda:0 --heavy-size 0.008 --tag 1pct
    python run_block_tail_diag.py --device cuda:1 --heavy-size 0.028 --tag 3pct
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
    LocalMaskerConfig,
    OracleTopKConfig,
    SinkMaskerConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (  # noqa: E402
    HashAttentionTopKMaskerConfig,
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

NEG = -1e4  # anything below this counts as "masked out"

# ---------------------------------------------------------------- collector ---
RECORDS: List[Dict[str, Any]] = []
SETTINGS: Dict[str, Any] = {"block_sizes": [32, 64, 128, 256, 512],
                            "num_bins_list": [16, 32],
                            "max_qrows": 4,
                            "max_records": 400_000,
                            "random_control": True,
                            "selector": "oracle",
                            "deploy_grid": True,
                            "deploy_bin_sizes": [32],
                            "rand_bin_counts": [1, 2, 4, 8, 16, 64, 256, 1024, 4096, 16384],
                            "expand_grid": True,
                            "expand_bin_size": 32,
                            "expand_m": [1, 2, 4, 8, 16, 32, 64],
                            "expand_c": [0, 1, 2, 3]}

# Reuses the production binning code so the diagnostic cannot drift from the method.
_SCORE_BINNERS: Dict[int, UTAMultiBinAttention] = {}


_DEPLOY_BINNERS: Dict[tuple, UTAMultiBinAttention] = {}


class _SlicedMask:
    """Exposes only the last `qr` query rows of a Mask, matching the sliced tensors."""

    def __init__(self, mask: Mask, qr: int) -> None:
        self._d = mask.get_dense_mask()[:, :, -qr:, :]

    def get_dense_mask(self) -> torch.Tensor:
        return self._d


def _deploy_binner(bin_size: int, kappa: str, var: str) -> UTAMultiBinAttention:
    """Production attention object for the deployable config (adjacent-tail-index bins)."""
    key = (bin_size, kappa, var)
    if key not in _DEPLOY_BINNERS:
        cfg = UTAMultiBinConfig(masker_configs=[], bin_size=bin_size,
                                bin_mode="equalcount", kappa_mode=kappa, var_mode=var)
        _DEPLOY_BINNERS[key] = UTAMultiBinAttention(cfg, [])
    return _DEPLOY_BINNERS[key]


_RAND_BINNERS: Dict[int, UTAMultiBinAttention] = {}


def _rand_binner(num_bins: int) -> UTAMultiBinAttention:
    """V4 control: tail partitioned uniformly at random into `num_bins` bins."""
    if num_bins not in _RAND_BINNERS:
        cfg = UTAMultiBinConfig(masker_configs=[], num_bins=num_bins,
                                bin_mode="random", kappa_mode="none", var_mode="diag")
        _RAND_BINNERS[num_bins] = UTAMultiBinAttention(cfg, [])
    return _RAND_BINNERS[num_bins]


def _score_binner(num_bins: int) -> UTAMultiBinAttention:
    if num_bins not in _SCORE_BINNERS:
        cfg = UTAMultiBinConfig(masker_configs=[], num_bins=num_bins, bin_mode="score")
        _SCORE_BINNERS[num_bins] = UTAMultiBinAttention(cfg, [])
    return _SCORE_BINNERS[num_bins]


def _relerr(approx: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    """Row-wise relative L2 error;  shapes (..., D) -> (...)"""
    return (approx - exact).norm(dim=-1) / exact.norm(dim=-1).clamp(min=1e-12)


class TailDiagnosticAttention(UTAAttention):
    """Builds the sparse mask, measures the tail, then returns the DENSE output."""

    def custom_attention(
        self,
        module: nn.Module,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float,
        **kwargs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        sparse_meta_data = kwargs.pop("sparse_meta_data")
        layer_idx = int(kwargs.get("layer_idx", -1) or -1)

        mask_shape = (queries.shape[0], queries.shape[1], queries.shape[2], keys.shape[2])
        sparse_mask = Mask.create_empty_mask(mask_shape, dtype=queries.dtype, device=queries.device)
        for masker in self.maskers:
            sparse_mask = masker.add_mask(
                keys=keys, queries=queries, values=values,
                attention_mask=attention_mask, scaling=scaling, dropout=dropout,
                sparse_meta_data=sparse_meta_data, previous_mask=sparse_mask, **kwargs,
            )

        hat_scores = sparse_meta_data.get("hat_scores", {}).get(layer_idx)

        if (not sparse_mask.is_full_mask()) and len(RECORDS) < SETTINGS["max_records"]:
            try:
                self._measure(queries, keys, values, attention_mask, scaling,
                              sparse_mask, layer_idx, hat_scores)
            except Exception as exc:  # never let the diagnostic kill the run
                print(f"[diag] layer {layer_idx} skipped: {type(exc).__name__}: {exc}", flush=True)

        return get_true_attention_output(
            module, queries, keys, values, attention_mask, scaling, dropout, **kwargs
        )

    # ------------------------------------------------------------------ core --
    @torch.no_grad()
    def _measure(self, queries, keys, values, attention_mask, scaling,
                 sparse_mask: Mask, layer_idx: int, hat_scores=None) -> None:
        qr = min(SETTINGS["max_qrows"], queries.shape[2])
        ngroups = _get_num_key_value_groups(queries, keys)
        k = repeat_kv(keys, ngroups).to(torch.float32)          # (B,H,K,D)
        v = repeat_kv(values, ngroups).to(torch.float32)        # (B,H,K,D)
        q = queries[:, :, -qr:, :].to(torch.float32)            # (B,H,q,D)

        raw = torch.matmul(q, k.transpose(2, 3)) * scaling      # (B,H,q,K)
        if attention_mask is not None:
            raw = raw + attention_mask[:, :, -qr:, : k.shape[-2]].to(torch.float32)

        valid = raw > NEG
        sel = (sparse_mask.get_dense_mask()[:, :, -qr:, :] != 0) & valid
        tail = (~sel) & valid
        tf = tail.to(torch.float32)

        n_tail = tf.sum(-1)                                     # (B,H,q)
        ok = n_tail >= 2
        if not bool(ok.any()):
            return

        row_max = raw.masked_fill(~valid, float("-inf")).amax(-1, keepdim=True)   # (B,H,q,1)
        E = torch.exp(raw - row_max) * valid

        # --- exact split ------------------------------------------------------
        eH, eT = E * sel, E * tf
        S_H, S_T = eH.sum(-1, keepdim=True), eT.sum(-1, keepdim=True)            # (B,H,q,1)
        num_H, num_T = torch.matmul(eH, v), torch.matmul(eT, v)                  # (B,H,q,D)
        o_exact = (num_H + num_T) / (S_H + S_T).clamp(min=1e-30)
        mu_T = num_T / S_T.clamp(min=1e-30)

        # --- global tail moments ---------------------------------------------
        Nt = n_tail.unsqueeze(-1).clamp(min=1)
        zbar = (raw * tf).sum(-1, keepdim=True) / Nt
        zsq = (raw * raw * tf).sum(-1, keepdim=True) / Nt
        sigma2 = (zsq - zbar * zbar).clamp(min=0)
        zmax_t = raw.masked_fill(~tail, float("-inf")).amax(-1, keepdim=True)
        tau_sel = raw.masked_fill(~sel, float("inf")).amin(-1, keepdim=True)
        v_bar = torch.matmul(tf, v) / Nt

        S_uta = Nt * torch.exp(zbar - row_max)                                   # shifted space
        K1_exact = torch.log(S_T.clamp(min=1e-30)) + row_max - zbar - torch.log(Nt)
        headroom = zmax_t - zbar

        def merged(mass: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
            return (num_H + mass * val) / (S_H + mass).clamp(min=1e-30)

        rec: Dict[str, torch.Tensor] = {
            "n_tail": n_tail,
            "tail_mass_frac": (S_T / (S_H + S_T).clamp(min=1e-30)).squeeze(-1),
            "zbar": zbar.squeeze(-1),
            "sigma2": sigma2.squeeze(-1),
            "headroom": headroom.squeeze(-1),
            "tau_sel_minus_zbar": (tau_sel - zbar).squeeze(-1),
            "K1_exact": K1_exact.squeeze(-1),
            "K1_lognormal": (sigma2 / 2).squeeze(-1),
            "viol_lognormal": ((sigma2 / 2) > headroom).to(torch.float32).squeeze(-1),
            "err_uta": _relerr(merged(S_uta, v_bar), o_exact),
            "err_oracle_S": _relerr(merged(S_T, v_bar), o_exact),
            "err_oracle_mu": _relerr(merged(S_uta, mu_T), o_exact),
        }

        # --- E0: the tie group at HAT's top-k boundary -------------------------
        # HAT scores are dot products of two 32-bit +/-1 sign vectors, so they take
        # only 33 distinct values across ~32K keys.  `torch.topk` therefore cuts an
        # enormous tie group at the boundary level in arbitrary index order.  What
        # matters is whether the tail mass HAT loses sits INSIDE that tie group: if it
        # does, rescoring just that group with exact logits recovers it for
        # n_at_tau * d MACs, and the tail estimator is solving the wrong problem.
        if hat_scores is not None:
            hs = hat_scores[:, :, -qr:, : k.shape[-2]].to(torch.float32)
            # sink/local positions were set to finfo.min before the top-k, so this
            # isolates exactly the tokens HAT itself ranked and selected.
            hs_ok = valid & (hs > -1e4)
            hat_sel = sel & hs_ok
            n_hat_sel = hat_sel.sum(-1)
            tau = hs.masked_fill(~hat_sel, float("inf")).amin(-1, keepdim=True)
            tau = torch.where(n_hat_sel.unsqueeze(-1) > 0, tau,
                              torch.full_like(tau, float("nan")))
            at_tau = hs_ok & (hs == tau)
            sel_at_tau = (at_tau & hat_sel).sum(-1).float()

            rec.update({
                "hat_n_sel": n_hat_sel.float(),
                "hat_tau": tau.squeeze(-1),
                "hat_score_max": hs.masked_fill(~hs_ok, float("-inf")).amax(-1),
                # size of the tie group, and how much of the budget it swallowed
                "hat_n_at_tau": at_tau.sum(-1).float(),
                "hat_n_sel_at_tau": sel_at_tau,
                "hat_tie_frac_of_budget": sel_at_tau / n_hat_sel.clamp(min=1).float(),
                # THE metric: share of missed tail mass sitting in the tie group
                "hat_tail_mass_at_tau": ((E * (at_tau & tail)).sum(-1)
                                         / S_T.squeeze(-1).clamp(min=1e-30)),
            })

            # and where the heaviest individually-missed tokens actually sit
            J = 8
            tv, ti = raw.masked_fill(~tail, float("-inf")).topk(J, dim=-1)
            hs_top = hs.gather(-1, ti)
            live = (tv > NEG).float()
            nlive = live.sum(-1).clamp(min=1)
            rec["hat_topJ_at_tau"] = (((hs_top == tau).float() * live).sum(-1) / nlive)
            rec["hat_topJ_below_tau"] = (((hs_top < tau).float() * live).sum(-1) / nlive)

        # --- block-wise multi-proxy ------------------------------------------
        # Pad ONCE to a multiple of the largest block size; every (power-of-two)
        # block size then reshapes cleanly off the same padded tensors.
        K = raw.shape[-1]
        stride = max(SETTINGS["block_sizes"])
        pad = (-K) % stride
        if pad:
            rawp = torch.nn.functional.pad(raw, (0, pad), value=NEG - 1.0)
            tfp = torch.nn.functional.pad(tf, (0, pad), value=0.0)
            vp = torch.nn.functional.pad(v, (0, 0, 0, pad), value=0.0)
        else:
            rawp, tfp, vp = raw, tf, v
        Kp = K + pad

        # `perm` is the CONTROL: same number of proxies, but the partition ignores
        # position.  If contiguous blocks beat it, locality is what buys the gain;
        # if they tie, only the proxy count matters (-> clustering would be better).
        partitions: List[Tuple[str, Optional[torch.Tensor]]] = [("block", None)]
        if SETTINGS["random_control"]:
            g = torch.Generator(device=raw.device).manual_seed(1234)
            partitions.append(("rand", torch.randperm(Kp, generator=g, device=raw.device)))

        for pname, perm in partitions:
            rp, tp, vpp = (rawp, tfp, vp) if perm is None else (
                rawp[..., perm], tfp[..., perm], vp[..., perm, :])

            for B in SETTINGS["block_sizes"]:
                nb = Kp // B
                shp = rp.shape[:-1] + (nb, B)                                    # (B,H,q,nb,B)
                rb, tb = rp.reshape(shp), tp.reshape(shp)
                n_b = tb.sum(-1)                                                 # (B,H,q,nb)
                nb_c = n_b.clamp(min=1)
                zbar_b = (rb * tb).sum(-1) / nb_c
                s2_b = ((rb * rb * tb).sum(-1) / nb_c - zbar_b * zbar_b).clamp(min=0)
                s2_b = torch.where(n_b >= 2, s2_b, torch.zeros_like(s2_b))
                zmax_b = rb.masked_fill(tb == 0, float("-inf")).amax(-1)

                # (B,H,q,nb,D) mean value per block over tail positions
                v_view = vpp.reshape(vpp.shape[:-2] + (nb, B, vpp.shape[-1]))    # (B,H,nb,B,D)
                v_bar_b = torch.einsum("bhqns,bhnsd->bhqnd", tb, v_view) / nb_c.unsqueeze(-1)

                live = (n_b > 0).to(torch.float32)
                rec[f"var_ratio_{pname}_{B}"] = (
                    ((n_b * s2_b).sum(-1) / Nt.squeeze(-1)) / sigma2.squeeze(-1).clamp(min=1e-12)
                )

                variants = (("plain", torch.zeros_like(s2_b)),
                            ("j2", torch.log1p(s2_b / 2)),
                            ("j3", torch.minimum(s2_b / 2, (zmax_b - zbar_b).clamp(min=0))))
                if pname == "rand":     # control only needs the winning variant
                    variants = (variants[1],)

                for name, kappa in variants:
                    mass_b = n_b * torch.exp(zbar_b + kappa - row_max) * live    # (B,H,q,nb)
                    mass = mass_b.sum(-1, keepdim=True)
                    num = torch.einsum("bhqn,bhqnd->bhqd", mass_b, v_bar_b)
                    o_hat = (num_H + num) / (S_H + mass).clamp(min=1e-30)
                    rec[f"err_{pname}_{B}_{name}"] = _relerr(o_hat, o_exact)

        # --- score-based bins (the reference implementation's criterion) -------
        # Equal-width bins over a score, restricted to the tail.  Two score sources:
        #   "score" = the TRUE logit          -> free under an oracle-top-k selector
        #   "hat"   = HashAttention's approx  -> free under a HAT selector
        # In both cases the bin LOGIT is still the mean of true logits; the score only
        # decides membership.  Uses the production binner so the two cannot diverge.
        score_srcs = [("score", raw)]
        if hat_scores is not None:
            score_srcs.append(("hat", hat_scores[:, :, -qr:, :].to(torch.float32)))

        for sname, src in score_srcs:
            for NB in SETTINGS["num_bins_list"]:
                binner = _score_binner(NB)
                n_b, zbar_b, s2_b, vbar_b = binner._bins_score(raw, tf, v, src)

                rec[f"var_ratio_{sname}_nb{NB}"] = (
                    ((n_b * s2_b).sum(-1) / Nt.squeeze(-1)) / sigma2.squeeze(-1).clamp(min=1e-12)
                )
                live = n_b > 0
                for name, kappa in (("plain", torch.zeros_like(s2_b)),
                                    ("j2", torch.log1p(s2_b / 2))):
                    ell = zbar_b + torch.log(n_b.clamp(min=1)) + kappa
                    w_b = torch.exp(ell - row_max) * live
                    o_hat = ((num_H + torch.einsum("bhqn,bhqnd->bhqd", w_b, vbar_b))
                             / (S_H + w_b.sum(-1, keepdim=True)).clamp(min=1e-30))
                    rec[f"err_{sname}_nb{NB}_{name}"] = _relerr(o_hat, o_exact)

        # --- deployable grid: adjacent-tail-index bins x kappa x var ----------
        # Bins are runs of adjacent TAIL indices (the complement of HAT's top-k), which
        # needs no logits at all -- only cached block sums plus subtracting the selected
        # tokens we already load.  This is the configuration that can actually ship, so
        # the (kappa, var_mode) decision has to be made here, not on the score-bin
        # ceiling where sigma_b^2 is so small that every variant looks the same.
        if SETTINGS["deploy_grid"]:
            for m in SETTINGS["deploy_bin_sizes"]:
                for km in ("none", "j1", "j2"):
                    for vm in ("exact", "diag"):
                        att = _deploy_binner(m, km, vm)
                        try:
                            o = att._multibin_output(
                                queries[:, :, -qr:, :], keys, values,
                                None if attention_mask is None else attention_mask[:, :, -qr:, :],
                                scaling, _SlicedMask(sparse_mask, qr),
                            ).transpose(1, 2)
                        except Exception:
                            continue
                        rec[f"err_adj{m}_{km}_{vm}"] = _relerr(o, o_exact)

        # --- E1/E2: expand the hottest bins exactly instead of approximating them --
        # E0 showed the tail's damage is a few enormous-logit tokens HAT missed, and
        # that no proxy shape (jensen, multibin) can represent them.  So: keep the
        # adjacency partition, pick m bins, and compute those bins' tokens EXACTLY --
        # pulling the outlier out of its bin instead of correcting the bin's mass.
        #   E1 ranks bins by their true max logit  -> ceiling of the strategy
        #   E2 ranks by zbar_b + c*sigma_b, both KEY-SIDE  -> deployable, no logits
        if SETTINGS["expand_grid"]:
            mb = _deploy_binner(SETTINGS["expand_bin_size"], "none", "diag")
            bidE, nbE = mb._bin_ids_equalcount(tf)
            shpE = tf.shape[:-1] + (nbE + 1,)
            n_bE = torch.zeros(shpE, dtype=tf.dtype, device=tf.device)
            n_bE = n_bE.scatter_add_(-1, bidE, tf)[..., :nbE]
            denE = n_bE.clamp(min=1)
            vbarE = mb._scatter_mean(v, tf, bidE, nbE, denE)
            zbarE, s2E = mb._bin_moments_scatter(q, k, tf, bidE, nbE, denE, scaling)
            liveE = n_bE > 0

            # oracle ranking: the true max logit inside each bin
            zmaxE = torch.full(shpE, float("-inf"), dtype=raw.dtype, device=raw.device)
            zmaxE = zmaxE.scatter_reduce_(
                -1, bidE, raw.masked_fill(~tail, float("-inf")),
                reduce="amax", include_self=True)[..., :nbE]

            e_all = torch.exp(raw - row_max)
            num_sel, den_sel = torch.matmul(e_all * sel, v), (e_all * sel).sum(-1, keepdim=True)
            ell_all = zbarE + torch.log(n_bE.clamp(min=1))

            def _expanded(score_b: torch.Tensor, m: int) -> torch.Tensor:
                """Output when the top-m bins by `score_b` are computed exactly."""
                mm = min(m, nbE)
                sc = score_b.masked_fill(~liveE, float("-inf"))
                idx = sc.topk(mm, dim=-1).indices
                is_exp = torch.zeros_like(liveE)
                is_exp.scatter_(-1, idx, True)
                is_exp = is_exp & liveE
                exp_tok = torch.cat(
                    [is_exp, torch.zeros_like(is_exp[..., :1])], -1).gather(-1, bidE) & tail
                num = num_sel + torch.matmul(e_all * exp_tok, v)
                den = den_sel + (e_all * exp_tok).sum(-1, keepdim=True)
                keep_b = liveE & ~is_exp
                w = torch.exp(ell_all.masked_fill(~keep_b, float("-inf")) - row_max)
                num = num + torch.einsum("bhqn,bhqnd->bhqd", w, vbarE)
                den = den + w.sum(-1, keepdim=True)
                return num / den.clamp(min=1e-30)

            for m in SETTINGS["expand_m"]:
                rec[f"err_exp{m}_oracle"] = _relerr(_expanded(zmaxE, m), o_exact)
            sig = s2E.clamp(min=0).sqrt()
            for c in SETTINGS["expand_c"]:
                bound = zbarE + c * sig
                for m in SETTINGS["expand_m"]:
                    rec[f"err_exp{m}_bound_c{c}"] = _relerr(_expanded(bound, m), o_exact)
            # control: does the RANKING matter, or just spending the extra budget?
            gexp = torch.Generator(device=raw.device).manual_seed(77)
            rnd = torch.rand(shpE[:-1] + (nbE,), generator=gexp, device=raw.device)
            for m in SETTINGS["expand_m"]:
                rec[f"err_exp{m}_random"] = _relerr(_expanded(rnd, m), o_exact)

        # --- V4: random-partition bin-count sweep ------------------------------
        # Random membership keeps every bin an unbiased sample of the same tail, so
        # within-bin variance never shrinks; only Var(zbar_b) ~ sigma^2 * B / n grows.
        # The predicted curve is therefore flat until B approaches n, which is exactly
        # what the benchmark's B <= 8 cannot show -- hence this sweep to B >> 8.
        for nb in SETTINGS["rand_bin_counts"]:
            att = _rand_binner(nb)
            try:
                o = att._multibin_output(
                    queries[:, :, -qr:, :], keys, values,
                    None if attention_mask is None else attention_mask[:, :, -qr:, :],
                    scaling, _SlicedMask(sparse_mask, qr),
                ).transpose(1, 2)
            except Exception:
                continue
            rec[f"err_randbin_{nb}"] = _relerr(o, o_exact)

        # --- flatten valid rows ----------------------------------------------
        keep = ok & (S_T.squeeze(-1) > 0)
        idx = keep.nonzero(as_tuple=False)
        if idx.numel() == 0:
            return
        heads = idx[:, 1].cpu().numpy()
        flat = {name: t[keep].float().cpu().numpy() for name, t in rec.items()}
        for i in range(len(heads)):
            row: Dict[str, Any] = {"layer": layer_idx, "head_idx": int(heads[i])}
            row.update({name: float(arr[i]) for name, arr in flat.items()})
            RECORDS.append(row)


HAT_WEIGHT_FILE = (
    "/database/hyunwoo/hf/HashAttention-1.0/repo/artifacts/"
    "llama3.1-8b-patch.64K.v1.hat_weights.pkl"
)


def _diag_masker_configs(heavy_size: float, selector: str):
    top = (OracleTopKConfig(heavy_size=heavy_size) if selector == "oracle"
           else HashAttentionTopKMaskerConfig(
               heavy_size=heavy_size, hat_bits=32, hat_mlp_layers=3,
               hat_mlp_hidden_size=128, hat_mlp_activation="silu",
               hat_weight_file=HAT_WEIGHT_FILE))
    return [SinkMaskerConfig(sink_size=0.001),
            LocalMaskerConfig(window_size=0.001), top]


def build_diag_attention(heavy_size: float, selector: str) -> TailDiagnosticAttention:
    cfg = UTAAttentionConfig(masker_configs=_diag_masker_configs(heavy_size, selector))
    maskers = [ResearchMasker.create_masker_from_config(mc) for mc in cfg.masker_configs]
    return TailDiagnosticAttention(cfg, maskers)


# ----------------------------------------------------------------- analysis ---
def analyse(df: pd.DataFrame, out_dir: Path, tag: str) -> None:
    bs = SETTINGS["block_sizes"]
    line = "=" * 88

    print(f"\n{line}\nTAIL DIAGNOSTIC  [{tag}]   rows={len(df)}  "
          f"layers={df['layer'].nunique()}  heads={df['head_idx'].nunique()}\n{line}")

    print("\n--- Q1: is the log-normal Jensen correction feasible? ---")
    print(f"  P(sigma^2/2 > zmax_tail - zbar)   : {df.viol_lognormal.mean():.3f}   <-- violation rate")
    print(f"  exact Jensen gap  K1   median/mean: {df.K1_exact.median():.3f} / {df.K1_exact.mean():.3f}")
    print(f"  log-normal pred sigma^2/2         : {df.K1_lognormal.median():.3f} / {df.K1_lognormal.mean():.3f}")
    print(f"  hard bound (zmax_tail - zbar)     : {df.headroom.median():.3f} / {df.headroom.mean():.3f}")
    over = (df.K1_lognormal - df.K1_exact)
    print(f"  overshoot in nats  median/p90     : {over.median():.3f} / {over.quantile(0.9):.3f}"
          f"   (= {np.exp(over.median()):.1f}x / {np.exp(over.quantile(0.9)):.1f}x tail mass)")

    print("\n--- Q2: denominator vs numerator (relative output error, median) ---")
    print(f"  UTA (both approximate)   : {df.err_uta.median():.5f}")
    print(f"  + exact S_T only         : {df.err_oracle_S.median():.5f}"
          f"   ({100 * (1 - df.err_oracle_S.median() / df.err_uta.median()):+.1f}%)")
    print(f"  + exact mu_T only        : {df.err_oracle_mu.median():.5f}"
          f"   ({100 * (1 - df.err_oracle_mu.median() / df.err_uta.median()):+.1f}%)")
    print(f"  tail mass fraction       : median {df.tail_mass_frac.median():.4f}  "
          f"p90 {df.tail_mass_frac.quantile(0.9):.4f}")

    has_rand = any(c.startswith("var_ratio_rand_") for c in df.columns)

    print("\n--- Q3: GO/NO-GO — within-partition logit variance / global tail variance ---")
    for B in bs:
        c = f"var_ratio_block_{B}"
        msg = f"  block {B:4d} : median {df[c].median():.4f}   p90 {df[c].quantile(0.9):.4f}"
        if has_rand:
            msg += f"   [random control: {df[f'var_ratio_rand_{B}'].median():.4f}]"
        print(msg)
    print("  (need << 1 — and clearly below the random control — for locality to be the reason)")

    adj = sorted({c for c in df.columns if c.startswith("err_adj")})
    if adj:
        print("\n--- DEPLOYABLE GRID: adjacent-tail-index bins (median relative error) ---")
        print(f"  {'bin':>6}{'kappa':>8}{'exact var':>12}{'diag var':>12}{'diag/exact':>12}")
        sizes = sorted({int(c.split("_")[1][3:]) for c in adj})
        for m in sizes:
            for km in ("none", "j1", "j2"):
                e = df.get(f"err_adj{m}_{km}_exact")
                d = df.get(f"err_adj{m}_{km}_diag")
                if e is None or d is None:
                    continue
                em, dm = e.median(), d.median()
                print(f"  {m:>6}{km:>8}{em:>12.5f}{dm:>12.5f}{dm / em:>12.3f}")

    print("\n--- Q4: relative output error by estimator (median / p90) ---")
    print(f"  {'estimator':<28}{'median':>10}{'p90':>10}{'vs UTA':>10}")
    base = df.err_uta.median()
    print(f"  {'UTA (single proxy)':<28}{base:>10.5f}{df.err_uta.quantile(0.9):>10.5f}{'-':>10}")
    print(f"  {'oracle S_T (lower bound)':<28}{df.err_oracle_S.median():>10.5f}"
          f"{df.err_oracle_S.quantile(0.9):>10.5f}{100 * (1 - df.err_oracle_S.median() / base):>9.1f}%")
    for B in bs:
        names = ["plain", "j2", "j3"]
        for name in names:
            c = f"err_block_{B}_{name}"
            m = df[c].median()
            print(f"  {f'block {B} / {name}':<28}{m:>10.5f}{df[c].quantile(0.9):>10.5f}"
                  f"{100 * (1 - m / base):>9.1f}%")
        if has_rand:
            c = f"err_rand_{B}_j2"
            m = df[c].median()
            print(f"  {f'  [control] rand {B} / j2':<28}{m:>10.5f}{df[c].quantile(0.9):>10.5f}"
                  f"{100 * (1 - m / base):>9.1f}%")

    nbs = sorted({int(c.split("nb")[1]) for c in df.columns if "_nb" in c and c.startswith("var_ratio_")})
    for sname in ("score", "hat"):
        if not any(c.startswith(f"var_ratio_{sname}_nb") for c in df.columns):
            continue
        print(f"\n  -- {sname}-based bins (equal-width over "
              f"{'true logit' if sname == 'score' else 'HAT approx score'}) --")
        for NB in nbs:
            vr = df[f"var_ratio_{sname}_nb{NB}"].median()
            print(f"  {f'{sname} {NB} bins':<28}{'var_ratio':>10}{vr:>10.4f}")
            for name in ("plain", "j2"):
                c = f"err_{sname}_nb{NB}_{name}"
                if c not in df.columns:
                    continue
                m = df[c].median()
                print(f"  {f'{sname} {NB} bins / {name}':<28}{m:>10.5f}"
                      f"{df[c].quantile(0.9):>10.5f}{100 * (1 - m / base):>9.1f}%")

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"tail_diag_{tag}.parquet"
    try:
        df.to_parquet(raw_path)
    except Exception:
        raw_path = out_dir / f"tail_diag_{tag}.csv.gz"
        df.to_csv(raw_path, index=False)
    print(f"\nraw rows      -> {raw_path}")

    agg = {"viol_lognormal": "mean", "K1_exact": "median", "K1_lognormal": "median",
           "sigma2": "median", "headroom": "median", "tail_mass_frac": "median",
           "err_uta": "median", "err_oracle_S": "median", "err_oracle_mu": "median"}
    agg.update({c: "median" for c in df.columns
                if c.startswith(("var_ratio_", "err_block_", "err_rand_"))})
    per_layer = df.groupby("layer").agg(agg).reset_index()
    lp = out_dir / f"tail_diag_{tag}_per_layer.csv"
    per_layer.to_csv(lp, index=False)
    print(f"per-layer     -> {lp}")
    print(line)


def main() -> None:
    p = argparse.ArgumentParser(description="Tail-structure diagnostic")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--heavy-size", type=float, default=0.008,
                   help="top-k fraction; total density = heavy + 0.002")
    p.add_argument("--tag", default=None, help="label for output files (default from heavy-size)")
    p.add_argument("--subsets", default="qa_1,niah_multikey_3,fwe")
    p.add_argument("--samples-per-subset", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--max-qrows", type=int, default=4)
    p.add_argument("--block-sizes", default="32,64,128,256,512")
    p.add_argument("--num-bins", default="16,32",
                   help="bin counts for the score/hat binning modes")
    p.add_argument("--selector", default="oracle", choices=["oracle", "hat"],
                   help="top-k selector; 'hat' also enables the hat-score binning arm")
    p.add_argument("--max-records", type=int, default=400_000)
    p.add_argument("--no-random-control", action="store_true",
                   help="skip the position-agnostic random partition control")
    p.add_argument("--output-dir", default="./results_tail_diag")
    p.add_argument("--analyse", default=None, help="skip the run, re-analyse an existing parquet/csv")
    args = p.parse_args()

    tag = args.tag or f"{(args.heavy_size + 0.002) * 100:g}pct"
    out_dir = Path(args.output_dir)

    if args.analyse:
        path = Path(args.analyse)
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        SETTINGS["block_sizes"] = sorted({int(c.split("_")[2]) for c in df.columns
                                          if c.startswith("err_block_")})
        print(f"[analyse] block sizes recovered from columns: {SETTINGS['block_sizes']}")
        analyse(df, out_dir, tag)
        return

    SETTINGS["block_sizes"] = [int(x) for x in args.block_sizes.split(",")]
    SETTINGS["num_bins_list"] = [int(x) for x in args.num_bins.split(",")]
    SETTINGS["max_qrows"] = args.max_qrows
    SETTINGS["max_records"] = args.max_records
    SETTINGS["random_control"] = not args.no_random_control

    SETTINGS["selector"] = args.selector
    print(f"[diag] density={(args.heavy_size + 0.002) * 100:g}%  device={args.device}  "
          f"selector={args.selector}  blocks={SETTINGS['block_sizes']}  "
          f"num_bins={SETTINGS['num_bins_list']}")

    adapter = ModelAdapterHF(
        model_name=args.model,
        sparse_attention_config=UTAAttentionConfig(
            masker_configs=_diag_masker_configs(args.heavy_size, args.selector)),
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=args.device,
    )
    # swap in the diagnostic implementation (same masker pipeline, dense output).
    # Registration is lazy and closes over adapter.sparse_attention, so reset any
    # cached registration to be safe.
    adapter.sparse_attention = build_diag_attention(args.heavy_size, args.selector)
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
        print("no records collected — check that the sparse path was actually taken")
        return
    analyse(pd.DataFrame(RECORDS), out_dir, tag)


if __name__ == "__main__":
    main()
