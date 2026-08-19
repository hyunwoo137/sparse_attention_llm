"""Regression: key-side bin statistics must match the full-score-matrix ones.

`_multibin_output` used to derive every per-bin statistic from the materialised
(B,H,Q,K) score matrix.  Under var_mode="diag" it now derives them from the bin's
own key moments instead, which is what makes them query-independent and cacheable
with the KV pages.

The two routes are algebraically identical:

    zbar_b = mean_{i in bin, tail} (s * q . k_i)  ==  s * q . k_bar_b

so this file pins that identity down -- at the level of the statistics AND at the
level of the attention output -- so the change cannot silently alter the accuracy
numbers in results_v2/.
"""

import math

import pytest
import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig,
    OracleTopKConfig,
    SinkMaskerConfig,
)
from sparse_attention_hub.sparse_attention.uta_attention import (
    UTAMultiBinAttention,
    UTAMultiBinConfig,
)
from sparse_attention_hub.sparse_attention.utils.kv_utils import (
    _get_num_key_value_groups,
    repeat_kv,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask

B, H, H_KV, D, K, Q = 1, 4, 2, 32, 512, 1
SCALING = 1.0 / math.sqrt(D)


def _config(bin_mode: str, bin_size: int, kappa: str = "j2") -> UTAMultiBinConfig:
    """Same shape as the results_v2 ladder: sink + local + top-k, var_mode='diag'."""
    return UTAMultiBinConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=8),
            LocalMaskerConfig(window_size=8),
            OracleTopKConfig(heavy_size=0.05),
        ],
        bin_mode=bin_mode,
        bin_size=bin_size,
        kappa_mode=kappa,
        var_mode="diag",
    )


def _tensors(device, dtype=torch.float32, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, H, Q, D, generator=g, device=device, dtype=dtype)
    k = torch.randn(B, H_KV, K, D, generator=g, device=device, dtype=dtype)
    v = torch.randn(B, H_KV, K, D, generator=g, device=device, dtype=dtype)
    # per-position drift so the tail is not i.i.d.; a uniform tail would make the
    # identity hold for uninteresting reasons
    drift = torch.linspace(-1.0, 1.0, K, device=device, dtype=dtype).view(1, 1, K, 1)
    return q, k + 0.5 * drift, v


def _selection(attn, q, k, v):
    """Run the masker pipeline and return (raw, sel, tail_f, k_rep, v_rep)."""
    mask = Mask.create_empty_mask((B, H, Q, K), dtype=q.dtype, device=q.device)
    meta = {}
    for masker in attn.maskers:
        mask = masker.add_mask(
            keys=k, queries=q, values=v, attention_mask=None, scaling=SCALING,
            dropout=0.0, sparse_meta_data=meta, previous_mask=mask, layer_idx=0,
        )
    ngroups = _get_num_key_value_groups(q, k)
    k_rep = repeat_kv(k, ngroups).to(torch.float32)
    v_rep = repeat_kv(v, ngroups).to(torch.float32)
    q32 = q.to(torch.float32)
    raw = torch.matmul(q32, k_rep.transpose(2, 3)) * SCALING
    valid = raw > -1e4
    sel = (mask.get_dense_mask() != 0) & valid
    tail_f = ((~sel) & valid).to(torch.float32)
    assert 0 < int(tail_f.sum()) < B * H * Q * K, "need a non-trivial tail"
    assert int(sel.sum()) > 0, "need a non-empty selection"
    return q32, raw, sel, tail_f, k_rep, v_rep


@pytest.mark.parametrize("bin_mode,bin_size", [
    ("equalcount", 32),      # the results_v2 UTA+multibin / +jensen rung
    ("equalcount", 10 ** 9),  # the results_v2 UTA+jensen rung (one bin over the tail)
    ("fixed", 32),           # the deployable position-block partition
    ("fixed", 8),
])
def test_bin_statistics_match_full_score_matrix(bin_mode, bin_size):
    """zbar_b/n_b/vbar_b from key moments == the same read off the score matrix."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    attn = UTAMultiBinAttention.create_from_config(_config(bin_mode, bin_size))
    q, k, v = _tensors(device)
    q32, raw, sel, tail_f, k_rep, v_rep = _selection(attn, q, k, v)

    if bin_mode == "fixed":
        n_ref, z_ref, _, vbar_ref = attn._bins_fixed(raw, tail_f, v_rep)
        n_new, z_new, _, vbar_new = attn._bins_fixed_keyside(
            q32, k_rep, v_rep, tail_f, SCALING)
    else:
        bid, nbins = attn._bin_ids_equalcount(tail_f)
        n_ref, z_ref, _, vbar_ref = attn._aggregate_by_bin_id(
            raw, tail_f, v_rep, bid, nbins)
        n_new, z_new, _, vbar_new = attn._bins_scatter_keyside(
            q32, k_rep, v_rep, tail_f, bid, nbins, SCALING)

    # Counts are sums of 1.0 -- exact in float32 regardless of summation order.
    torch.testing.assert_close(n_new, n_ref, rtol=0, atol=0)

    # Value means: the 'fixed' route is an einsum and is deterministic, so it must be
    # bit-identical.  The scatter route goes through scatter_add_, whose CUDA atomics
    # accumulate in nondeterministic order -- calling it twice on identical inputs
    # already disagrees at ~1e-7 -- so it only gets float32 tolerance.
    if bin_mode == "fixed":
        torch.testing.assert_close(vbar_new, vbar_ref, rtol=0, atol=0)
    else:
        torch.testing.assert_close(vbar_new, vbar_ref, rtol=1e-4, atol=1e-6)

    # zbar_b is the same number reached by a different summation order -> float32 noise
    # only, and only on bins that hold tail tokens (empty bins are dropped downstream)
    live = n_ref > 0
    torch.testing.assert_close(z_new[live], z_ref[live], rtol=2e-3, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA atomics only")
def test_scatter_add_nondeterminism_is_the_tolerance_floor():
    """Documents why the scatter route above cannot be asserted bit-exact.

    If this ever starts passing with zero difference, the tolerances in
    `test_bin_statistics_match_full_score_matrix` can be tightened to exact.
    """
    device = "cuda"
    attn = UTAMultiBinAttention.create_from_config(_config("equalcount", 32))
    q, k, v = _tensors(device)
    _, _, _, tail_f, _, v_rep = _selection(attn, q, k, v)
    bid, nbins = attn._bin_ids_equalcount(tail_f)
    shp = tail_f.shape[:-1] + (nbins + 1,)
    n_b = torch.zeros(shp, dtype=tail_f.dtype, device=device)
    den = n_b.scatter_add_(-1, bid, tail_f)[..., :nbins].clamp(min=1)

    a = attn._scatter_mean(v_rep, tail_f, bid, nbins, den)
    b = attn._scatter_mean(v_rep, tail_f, bid, nbins, den)
    drift = (a - b).abs().max().item()
    assert drift > 0.0, "scatter_add_ became deterministic; tighten the tolerances"
    assert drift < 1e-5, f"nondeterminism far larger than expected: {drift}"


@pytest.mark.parametrize("bin_mode,bin_size,kappa", [
    ("equalcount", 32, "none"),      # UTA+multibin
    ("equalcount", 32, "j2"),        # UTA+multibin+jensen
    ("equalcount", 10 ** 9, "j2"),   # UTA+jensen
    ("fixed", 32, "j2"),
])
def test_attention_output_unchanged(bin_mode, bin_size, kappa):
    """End-to-end: the attention output is the same as the raw-derived route.

    Reconstructs what `_multibin_output` produced before the change by feeding the
    score-matrix statistics through the identical merge, and compares against the
    live implementation.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    attn = UTAMultiBinAttention.create_from_config(_config(bin_mode, bin_size, kappa))
    q, k, v = _tensors(device, seed=1)
    q32, raw, sel, tail_f, k_rep, v_rep = _selection(attn, q, k, v)

    # ---- reference: statistics off the full score matrix, then the same merge ----
    if bin_mode == "fixed":
        n_b, zbar_b, _, vbar_b = attn._bins_fixed(raw, tail_f, v_rep)
        m = attn.bin_size
        pad = (-K) % m
        tfp = torch.nn.functional.pad(tail_f, (0, pad)) if pad else tail_f
        kp = torch.nn.functional.pad(k_rep, (0, 0, 0, pad)) if pad else k_rep
        nbk = (K + pad) // m
        tb = tfp.reshape(tfp.shape[:-1] + (nbk, m))
        s2_b = attn._diag_var_fixed(q32, kp, tb, nbk, m, n_b.clamp(min=1), SCALING)
    else:
        bid, nbins = attn._bin_ids_equalcount(tail_f)
        n_b, zbar_b, _, vbar_b = attn._aggregate_by_bin_id(raw, tail_f, v_rep, bid, nbins)
        s2_b = attn._diag_var_scatter(
            q32, k_rep, tail_f, bid, nbins, n_b.clamp(min=1), SCALING)
        s2_b = torch.where(n_b >= 2, s2_b, torch.zeros_like(s2_b))

    live = n_b > 0
    ell_b = (zbar_b + torch.log(n_b.clamp(min=1)) + attn._kappa(s2_b))
    ell_b = ell_b.masked_fill(~live, float("-inf"))
    valid = raw > -1e4
    row_max = torch.maximum(
        raw.masked_fill(~valid, float("-inf")).amax(-1, keepdim=True),
        ell_b.amax(-1, keepdim=True))
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    e_sel = torch.exp(raw - row_max) * sel
    num = torch.matmul(e_sel, v_rep)
    den = e_sel.sum(-1, keepdim=True)
    w_b = torch.exp(ell_b - row_max)
    w_b = torch.where(live, w_b, torch.zeros_like(w_b))
    num = num + torch.einsum("bhqn,bhqnd->bhqd", w_b, vbar_b)
    den = den + w_b.sum(-1, keepdim=True)
    expected = (num / den.clamp(min=1e-30)).to(q.dtype).transpose(1, 2).contiguous()

    # ---- live implementation ----
    mask = Mask.create_empty_mask((B, H, Q, K), dtype=q.dtype, device=q.device)
    meta = {}
    for masker in attn.maskers:
        mask = masker.add_mask(
            keys=k, queries=q, values=v, attention_mask=None, scaling=SCALING,
            dropout=0.0, sparse_meta_data=meta, previous_mask=mask, layer_idx=0,
        )
    got = attn._multibin_output(q, k, v, None, SCALING, mask)

    assert got.shape == expected.shape
    torch.testing.assert_close(got, expected, rtol=2e-3, atol=1e-5)


@pytest.mark.parametrize("bin_mode,bin_size", [
    ("equalcount", 32),
    ("equalcount", 10 ** 9),
    ("fixed", 32),
])
def test_tail_statistics_do_not_read_the_score_matrix(bin_mode, bin_size):
    """Corrupt the logits at TAIL positions; the output must not move.

    This is the behavioural statement of the whole change.  Under var_mode="diag"
    every tail statistic comes from K/V, so the score matrix matters only at the
    SELECTED positions -- the ones a sparse kernel computes anyway.  If any tail
    statistic still read a logit, pushing the tail logits down by ~177 would move
    zbar_b by the same amount and the output would change drastically.

    The shift is chosen to stay above the NEG validity threshold (so `valid` and
    hence the tail mask are untouched) and below the selected tokens' logits (so
    the row-max used for softmax stability is untouched).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    attn = UTAMultiBinAttention.create_from_config(_config(bin_mode, bin_size))
    q, k, v = _tensors(device, seed=3)

    mask = Mask.create_empty_mask((B, H, Q, K), dtype=q.dtype, device=device)
    meta = {}
    for masker in attn.maskers:
        mask = masker.add_mask(
            keys=k, queries=q, values=v, attention_mask=None, scaling=SCALING,
            dropout=0.0, sparse_meta_data=meta, previous_mask=mask, layer_idx=0,
        )
    tail = (mask.get_dense_mask() == 0)          # (B,H,Q,K) bool

    clean = attn._multibin_output(q, k, v, None, SCALING, mask)

    orig_matmul = torch.matmul

    def corrupting_matmul(a, b, **kw):
        out = orig_matmul(a, b, **kw)
        # the (B,H,Q,K) score matrix, pre-scaling
        if out.shape == tail.shape and out.dtype == torch.float32 and a.shape[-1] == D:
            out = out - 1000.0 * tail.to(out.dtype)
        return out

    torch.matmul = corrupting_matmul
    try:
        corrupted = attn._multibin_output(q, k, v, None, SCALING, mask)
    finally:
        torch.matmul = orig_matmul

    torch.testing.assert_close(corrupted, clean, rtol=1e-4, atol=1e-6)


def test_exact_var_mode_still_uses_true_logit_variance():
    """var_mode='exact' must NOT be rerouted through key-side statistics.

    Its sigma_b^2 is the true within-bin logit variance, which no key-side moment
    can reproduce (dropping cross-dimension covariance is exactly what 'diag' does).
    If these two ever agree, the 'exact' path has been silently replaced.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = _config("equalcount", 32)
    cfg.var_mode = "exact"
    attn = UTAMultiBinAttention.create_from_config(cfg)
    q, k, v = _tensors(device, seed=2)
    q32, raw, sel, tail_f, k_rep, v_rep = _selection(attn, q, k, v)

    bid, nbins = attn._bin_ids_equalcount(tail_f)
    n_b, _, s2_exact, _ = attn._aggregate_by_bin_id(raw, tail_f, v_rep, bid, nbins)
    s2_diag = attn._diag_var_scatter(
        q32, k_rep, tail_f, bid, nbins, n_b.clamp(min=1), SCALING)

    live = n_b >= 2
    assert live.any(), "need bins with at least two tail tokens"
    # both are variances of the same logits, so they are close but not equal
    assert not torch.allclose(s2_exact[live], s2_diag[live], rtol=1e-4, atol=1e-8), (
        "exact and diag variances are identical -- the 'exact' path lost its "
        "true-logit variance"
    )
