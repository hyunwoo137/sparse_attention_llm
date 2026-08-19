"""Correctness tests for multi-bin UTA tail aggregation.

Checks the invariants that the maths depends on:
  1. sum_b n_b == N_tail            (per-bin counts partition the tail)
  2. bin_size == 1  =>  EXACT attention (every bin holds one token, log 1 = 0)
  3. bin_size >= K  =>  reduces to base UTA (one proxy over the whole tail)
  4. error decreases monotonically as bin_size shrinks
  5. "fixed" and "equalcount" agree when nothing is selected inside a block
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sparse_attention_hub.sparse_attention.uta_attention.multibin import (  # noqa: E402
    UTAMultiBinAttention,
    UTAMultiBinConfig,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask  # noqa: E402


def _setup(B=1, H=2, Q=3, K=64, D=16, n_sel=8, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, H, Q, D, dtype=torch.float32)
    k = torch.randn(B, H, K, D, dtype=torch.float32)
    v = torch.randn(B, H, K, D, dtype=torch.float32)
    scaling = D ** -0.5

    dense = torch.zeros(B, H, Q, K, dtype=torch.float32)
    scores = torch.matmul(q, k.transpose(2, 3)) * scaling
    top = scores.topk(n_sel, dim=-1).indices
    dense.scatter_(-1, top, 1.0)
    mask = Mask.create_mask_from_dense_mask((B, H, Q, K), dense, dtype=torch.float32)

    exact = torch.softmax(scores, dim=-1) @ v          # (B,H,Q,D)
    return q, k, v, scaling, mask, exact


def _make(bin_size, bin_mode="fixed", kappa_mode="none", num_bins=16):
    cfg = UTAMultiBinConfig(masker_configs=[], bin_size=bin_size, num_bins=num_bins,
                            bin_mode=bin_mode, kappa_mode=kappa_mode)
    return UTAMultiBinAttention(cfg, [])


def _run(att, q, k, v, scaling, mask, key_scores=None):
    # (B,Q,H,D) -> (B,H,Q,D)
    return att._multibin_output(q, k, v, None, scaling, mask, key_scores).transpose(1, 2)


@pytest.mark.parametrize("mode", ["fixed", "equalcount"])
def test_bin_counts_partition_the_tail(mode):
    q, k, v, scaling, mask, _ = _setup()
    att = _make(8, bin_mode=mode)

    raw = torch.matmul(q, k.transpose(2, 3)) * scaling
    sel = mask.get_dense_mask() != 0
    tail_f = (~sel).float()

    binner = att._bins_fixed if mode == "fixed" else att._bins_equalcount
    n_b, _, _, _ = binner(raw, tail_f, v)
    torch.testing.assert_close(n_b.sum(-1), tail_f.sum(-1))


@pytest.mark.parametrize("mode", ["fixed", "equalcount"])
def test_bin_size_one_is_exact(mode):
    """One token per bin => k_bar_b = k_i, v_bar_b = v_i, log(1) = 0 => no approximation."""
    q, k, v, scaling, mask, exact = _setup()
    out = _run(_make(1, bin_mode=mode), q, k, v, scaling, mask)
    torch.testing.assert_close(out, exact, rtol=1e-4, atol=1e-5)


def test_single_bin_matches_base_uta():
    """A bin larger than the sequence must reproduce base UTA's single proxy."""
    from sparse_attention_hub.sparse_attention.uta_attention.base import (
        UTAAttention, UTAAttentionConfig,
    )
    q, k, v, scaling, mask, _ = _setup()
    multibin = _run(_make(4096), q, k, v, scaling, mask)

    base = UTAAttention(UTAAttentionConfig(masker_configs=[]), [])
    vm, pl, tc = base._compute_uta_proxy(q, k, v, scaling, mask)
    ref, _ = base._merge_sparse_and_proxy(
        queries=q, keys=k, values=v, attention_mask=None, scaling=scaling,
        dropout=0.0, sparse_attention_mask=mask, module=None,
        proxy_v_mean=vm, proxy_logit=pl, tail_count=tc,
    )
    torch.testing.assert_close(multibin, ref.transpose(1, 2), rtol=1e-4, atol=1e-5)


def _setup_local(B=1, H=2, Q=3, K=256, D=16, n_sel=16, block=16, spread=0.15, seed=1):
    """Keys/values with LOCALITY: tokens inside a block share a centroid.

    Position-based binning can only help when nearby keys are similar, which is the
    property the diagnostic measured on real models (within-block logit variance
    ~0.5x the global tail variance).  Uniform random keys have no such structure,
    so monotonicity in bin size is only expected here.
    """
    torch.manual_seed(seed)
    nb = K // block
    kc = torch.randn(B, H, nb, 1, D)
    vc = torch.randn(B, H, nb, 1, D)
    k = (kc + spread * torch.randn(B, H, nb, block, D)).reshape(B, H, K, D)
    v = (vc + spread * torch.randn(B, H, nb, block, D)).reshape(B, H, K, D)
    q = torch.randn(B, H, Q, D)
    scaling = D ** -0.5

    scores = torch.matmul(q, k.transpose(2, 3)) * scaling
    dense = torch.zeros(B, H, Q, K)
    dense.scatter_(-1, scores.topk(n_sel, dim=-1).indices, 1.0)
    mask = Mask.create_mask_from_dense_mask((B, H, Q, K), dense, dtype=torch.float32)
    return q, k, v, scaling, mask, torch.softmax(scores, dim=-1) @ v


def test_error_decreases_with_smaller_bins():
    q, k, v, scaling, mask, exact = _setup_local()
    errs = []
    for m in (256, 64, 16, 4, 1):
        out = _run(_make(m), q, k, v, scaling, mask)
        errs.append(((out - exact).norm() / exact.norm()).item())
    assert all(a >= b - 1e-9 for a, b in zip(errs, errs[1:])), errs
    assert errs[-1] < 1e-5, f"bin_size=1 should be exact, got {errs[-1]}"
    assert errs[0] > 10 * errs[-2], errs


def test_locality_is_what_multibin_exploits():
    """Same bin count, but bins built on shuffled positions must do worse."""
    q, k, v, scaling, mask, exact = _setup_local()
    att = _make(16)
    err_local = ((_run(att, q, k, v, scaling, mask) - exact).norm() / exact.norm()).item()

    perm = torch.randperm(k.shape[2])
    kp, vp = k[:, :, perm], v[:, :, perm]
    scores = torch.matmul(q, kp.transpose(2, 3)) * scaling
    dense = torch.zeros_like(scores)
    dense.scatter_(-1, scores.topk(16, dim=-1).indices, 1.0)
    mask_p = Mask.create_mask_from_dense_mask(dense.shape, dense, dtype=torch.float32)
    exact_p = torch.softmax(scores, dim=-1) @ vp
    err_shuf = ((_run(att, q, kp, vp, scaling, mask_p) - exact_p).norm() / exact_p.norm()).item()

    assert err_local < err_shuf, (err_local, err_shuf)


def test_j2_changes_output_but_stays_finite():
    q, k, v, scaling, mask, exact = _setup(K=256, n_sel=16)
    plain = _run(_make(32, kappa_mode="none"), q, k, v, scaling, mask)
    j2 = _run(_make(32, kappa_mode="j2"), q, k, v, scaling, mask)
    assert torch.isfinite(j2).all()
    assert not torch.allclose(plain, j2)


@pytest.mark.parametrize("mode", ["score", "hat"])
def test_score_bins_partition_the_tail(mode):
    q, k, v, scaling, mask, _ = _setup(K=256, n_sel=16)
    att = _make(32, bin_mode=mode, num_bins=16)

    raw = torch.matmul(q, k.transpose(2, 3)) * scaling
    tail_f = (mask.get_dense_mask() == 0).float()
    scores = raw if mode == "score" else torch.randn_like(raw)

    n_b, _, _, _ = att._bins_score(raw, tail_f, v, scores)
    torch.testing.assert_close(n_b.sum(-1), tail_f.sum(-1))


def test_score_bins_converge_to_exact():
    """With more bins than distinct tail logits every bin holds one token -> exact."""
    q, k, v, scaling, mask, exact = _setup(K=64, n_sel=8)
    n_tail = 64 - 8
    out = _run(_make(32, bin_mode="score", num_bins=200 * n_tail), q, k, v, scaling, mask)
    assert ((out - exact).norm() / exact.norm()).item() < 1e-3


def _within_bin_var_ratio(att, binner, raw, tail_f, v, key_scores=None):
    """n_b-weighted mean within-bin logit variance / global tail logit variance."""
    n_b, _, s2_b, _ = binner(raw, tail_f, v, key_scores)
    n = tail_f.sum(-1)
    zbar = (raw * tail_f).sum(-1) / n.clamp(min=1)
    var = ((raw * raw * tail_f).sum(-1) / n.clamp(min=1) - zbar ** 2).clamp(min=0)
    return ((n_b * s2_b).sum(-1) / n.clamp(min=1) / var.clamp(min=1e-12)).median().item()


@pytest.mark.parametrize("setup", ["random", "local"])
def test_score_bins_cut_within_bin_variance(setup):
    """The mechanism: equal-width bins on the logit shrink within-bin logit variance.

    This holds by construction regardless of how values are laid out, so it is the
    claim worth pinning in a unit test.  Whether that translates into a lower OUTPUT
    error depends on the value structure of the real model, which the position-locality
    synthetic fixture below deliberately biases -- that horse race is settled by the
    run_block_tail_diag.py gate on real activations, not here.
    """
    q, k, v, scaling, mask, _ = (_setup(K=256, n_sel=16) if setup == "random"
                                 else _setup_local(K=256, n_sel=16))
    att = _make(32, bin_mode="score", num_bins=16)
    raw = torch.matmul(q, k.transpose(2, 3)) * scaling
    tail_f = (mask.get_dense_mask() == 0).float()

    r_score = _within_bin_var_ratio(att, att._bins_score, raw, tail_f, v, raw)
    r_pos = _within_bin_var_ratio(_make(16), _make(16)._bins_fixed, raw, tail_f, v)
    assert r_score < r_pos, (r_score, r_pos)
    assert r_score < 0.05, r_score


def test_score_bins_beat_position_bins_without_positional_locality():
    """With no positional structure, position blocks carry no signal but score bins do."""
    q, k, v, scaling, mask, exact = _setup(K=256, n_sel=16)
    n_bins = 16
    err_score = ((_run(_make(32, bin_mode="score", num_bins=n_bins), q, k, v, scaling, mask)
                  - exact).norm() / exact.norm()).item()
    # same budget of proxies, but split by position instead of by score
    err_pos = ((_run(_make(256 // n_bins, bin_mode="fixed"), q, k, v, scaling, mask)
                - exact).norm() / exact.norm()).item()
    assert err_score < err_pos, (err_score, err_pos)


def test_binning_quantity_does_not_leak_into_bin_logit():
    """`hat` bins by an APPROXIMATE score; the bin logit must still use true logits.

    Grouping by a permuted/noisy score changes which tokens share a bin, but each
    bin's zbar must remain the mean of the real logits of its members.
    """
    q, k, v, scaling, mask, _ = _setup(K=128, n_sel=16)
    att = _make(32, bin_mode="hat", num_bins=8)
    raw = torch.matmul(q, k.transpose(2, 3)) * scaling
    tail_f = (mask.get_dense_mask() == 0).float()

    noisy = raw + 3.0 * torch.randn_like(raw)
    n_b, zbar_b, _, _ = att._bins_score(raw, tail_f, v, noisy)

    # recompute zbar independently from the bin assignment implied by `noisy`
    tail = tail_f > 0
    lo = noisy.masked_fill(~tail, float("inf")).amin(-1, keepdim=True)
    hi = noisy.masked_fill(~tail, float("-inf")).amax(-1, keepdim=True)
    bid = ((noisy - lo) / (hi - lo).clamp(min=1e-6) * 8).floor().long().clamp(0, 7)
    for b in range(8):
        member = tail & (bid == b)
        cnt = member.sum(-1)
        want = (raw * member).sum(-1) / cnt.clamp(min=1)
        got = torch.where(cnt > 0, zbar_b[..., b], torch.zeros_like(want))
        torch.testing.assert_close(torch.where(cnt > 0, want, torch.zeros_like(want)), got,
                                   rtol=1e-4, atol=1e-5)


def test_no_nan_when_a_whole_block_is_selected():
    """Blocks fully covered by the top-k mask have n_b == 0 and must be skipped."""
    B, H, Q, K, D = 1, 1, 1, 32, 8
    torch.manual_seed(3)
    q = torch.randn(B, H, Q, D)
    k = torch.randn(B, H, K, D)
    v = torch.randn(B, H, K, D)
    dense = torch.zeros(B, H, Q, K)
    dense[..., :8] = 1.0                      # first block entirely selected
    mask = Mask.create_mask_from_dense_mask((B, H, Q, K), dense, dtype=torch.float32)
    out = _run(_make(8), q, k, v, D ** -0.5, mask)
    assert torch.isfinite(out).all()
