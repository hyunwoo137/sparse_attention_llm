"""UTA with multi-bin tail aggregation.

Base UTA collapses the whole tail into ONE proxy:

    S_T ~= N * exp(s * q.k_mean),      mu_T ~= v_mean

which is exact only if every tail logit is identical.  Its error is driven by the
within-tail logit variance.  Multi-bin UTA instead partitions the tail into bins
of adjacent key positions and gives each bin its own proxy:

    logit_b = s * (q . k_bar_b) + log(n_b)          <-- n_b is the bin's OWN count
    S_T     ~= sum_b exp(logit_b)
    num_T   ~= sum_b exp(logit_b) * v_bar_b

Key properties
--------------
* n_b, not N.  Each bin carries its own log-count; sum_b n_b == N is preserved.
* One global softmax.  Bin logits join the heavy-token logits in a single
  numerator/denominator merge -- never a per-bin softmax.
* Bins with n_b == 1 are EXACT (k_bar_b = k_i, v_bar_b = v_i, log 1 = 0), so bin
  size is a clean accuracy dial whose limit is exact attention.
* Bins with n_b == 0 contribute nothing (skipped, no log(0)).
* Residual Jensen bias per bin is ~sigma_b^2/2, which shrinks with bin size --
  partitioning attacks the bias at its source rather than correcting it.
  `kappa_mode="j2"` additionally adds log(1 + sigma_b^2/2) (moment form, bounded);
  it is OFF by default.

Prototype vs deployable
-----------------------
Under the default var_mode="diag" the bin statistics are computed the deployable
way: from the bin's own key moments, never from the score matrix.

    z_bar_b   = s * q . k_bar_b          <-- identical to the mean of the true logits
                                             over the bin's tail members, since
                                             mean_i s*q.k_i = s*q.mean_i k_i
    sigma_b^2 = s^2 * sum_d q_d^2 Var_b(k_d)

k_bar_b and Var_b(k) are query-independent, so a kernel gets them from per-block
sums cached once with the KV pages,
    k_bar_b = (sum_k[b] - sum_{i in b, selected} k_i) / n_b
at O(n_bins*d + k*d) per query.  The math in this file is the math that kernel
implements, and now so is the data flow.

The score matrix is still materialised in this file, but only for the parts a
sparse kernel computes anyway: the logits of the SELECTED tokens, which enter the
softmax exactly.  Restricting that to a gather over the selected indices is a
kernel change, not a math change.

var_mode="exact" is the exception and is deliberately circular: the true
within-bin logit variance cannot be recovered from key-side statistics, so that
mode reads the full score matrix.  It exists as an accuracy ceiling, not as a
deployable configuration.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from ..base import SparseAttentionConfig
from ..research_attention.maskers.base import ResearchMasker
from ..utils.kv_utils import _get_num_key_value_groups, repeat_kv
from ..utils.mask import Mask
from .base import UTAAttention, UTAAttentionConfig

from sparse_attention_hub.metric_logging.logger import MicroMetricLogger
from sparse_attention_hub.metric_logging.stage_timer import stage

NEG = -1e4

# How the tail is partitioned.  "fixed"/"equalcount" are parameterised by `bin_size`
# (tokens per bin); "score"/"hat"/"random" are parameterised by `num_bins` (bins per row).
BIN_MODES = ("fixed", "equalcount", "score", "hat", "topblock", "random")

# How sigma_b^2 is obtained.  This is the (2) axis: the true within-bin logit variance
# needs every tail logit (circular), whereas
#     sigma_b^2 = s^2 * q^T Cov_b(K) q
# needs only key-side statistics, which are query-independent and cacheable.
#   "exact" : variance of the true logits            -- prototype ceiling, circular
#   "diag"  : s^2 * sum_d q_d^2 Var_b(k_d)           -- drops cross-dim covariance,
#             O(d) per bin, storage d per bin (same footprint as the bin mean)
# Full covariance is deliberately absent: B*d^2 per query is ~10x vAttention's sampling
# cost at B=256, d=128, and B*d^2 floats per head is far past any KV-cache budget.
VAR_MODES = ("exact", "diag")

# Jensen correction K_b(1) = log E[exp(z - zbar_b)]:
#   "none" : 0                      -- always under-estimates the tail mass
#   "j1"   : sigma^2/2              -- exact iff z is Gaussian; unbounded in sigma
#   "j2"   : log(1 + sigma^2/2)     -- moment expansion; agrees with j1 to O(sigma^2),
#                                      grows only logarithmically afterwards
KAPPA_MODES = ("none", "j1", "j2")


@dataclass
class UTAMultiBinConfig(UTAAttentionConfig):
    """Config for multi-bin UTA.

    Attributes:
        bin_size: target number of key positions per bin.  Smaller = more proxies
            = more accurate = more per-query work.  bin_size=1 is exact attention.
        num_bins: number of bins per row for the "score"/"hat" modes.
        bin_mode: how the tail is partitioned.
            "fixed"      contiguous key-position blocks intersected with the tail.
                         Boundaries are query-independent, so bin statistics are
                         cacheable alongside the KV pages.
            "equalcount" runs of exactly bin_size consecutive tail tokens.
            "score"      equal-width bins over the true tail logit.  Directly
                         minimises within-bin logit variance -- the quantity that
                         drives UTA's error.  Needs every tail logit, so it is only
                         cost-neutral under a selector that already computes them
                         (oracle-top-k).
            "hat"        same, but binned by HashAttention's *approximate* score,
                         which the HAT masker already computes for selection -- so
                         it is cost-neutral under a HAT selector.
        kappa_mode: "none" (default) or "j2" = add log(1 + sigma_b^2/2) per bin.
    """

    bin_size: int = 32
    num_bins: int = 16
    bin_mode: str = "fixed"
    kappa_mode: str = "none"
    var_mode: str = "exact"
    refine_blocks: int = 0
    random_seed: int = 1234
    # Second stage (E1/E2): compute the top `expand_m` bins EXACTLY instead of
    # representing them by their mean.  Bins are ranked by zbar_b + expand_c*sigma_b,
    # both key-side, so the ranking needs no logits.  expand_rank="random" is the
    # control that spends the same budget without the ranking.
    expand_m: int = 0
    expand_c: float = 3.0
    expand_rank: str = "bound"
    # Per-(layer, head) expansion budget, calibrated offline: {layer: {head: m}}.
    # 65% of the per-row error variance sits BETWEEN heads rather than within them,
    # so the head is the unit at which a non-uniform budget actually pays.  Being
    # static, it costs nothing at run time -- no per-step decision, no ragged launch --
    # unlike a per-row rule, which makes the kernel pad every row to the batch maximum.
    expand_m_table: Optional[Dict[int, Dict[int, int]]] = None
    # Route the DECODE step through the fused Triton kernel.  Prefill keeps the torch
    # path (the kernel is a one-query-per-program decode kernel).  Numerically this is
    # the same computation; the point of running it end-to-end is to prove that.
    use_triton_decode: bool = False

    def __post_init__(self) -> None:
        if self.bin_size < 1:
            raise ValueError(f"bin_size must be >= 1, got {self.bin_size}")
        if self.num_bins < 1:
            raise ValueError(f"num_bins must be >= 1, got {self.num_bins}")
        if self.bin_mode not in BIN_MODES:
            raise ValueError(f"bin_mode must be one of {sorted(BIN_MODES)}, got {self.bin_mode}")
        if self.kappa_mode not in KAPPA_MODES:
            raise ValueError(f"kappa_mode must be one of {sorted(KAPPA_MODES)}, "
                             f"got {self.kappa_mode}")
        if self.var_mode not in VAR_MODES:
            raise ValueError(f"var_mode must be one of {sorted(VAR_MODES)}, got {self.var_mode}")
        if self.bin_mode == "topblock" and self.refine_blocks < 1:
            raise ValueError("bin_mode='topblock' needs refine_blocks >= 1")


class UTAMultiBinAttention(UTAAttention):
    """UTA whose tail is represented by many local proxies instead of one."""

    def __init__(self, config: UTAMultiBinConfig, maskers: List[ResearchMasker]) -> None:
        super().__init__(config, maskers)
        self.bin_size = int(config.bin_size)
        self.num_bins = int(config.num_bins)
        self.bin_mode = config.bin_mode
        self.kappa_mode = config.kappa_mode
        self.var_mode = getattr(config, "var_mode", "exact")
        self.refine_blocks = int(getattr(config, "refine_blocks", 0))
        self.random_seed = int(getattr(config, "random_seed", 1234))
        self.expand_m = int(getattr(config, "expand_m", 0))
        self.expand_c = float(getattr(config, "expand_c", 3.0))
        self.expand_rank = getattr(config, "expand_rank", "bound")
        self.use_triton_decode = bool(getattr(config, "use_triton_decode", False))
        self.expand_m_table = getattr(config, "expand_m_table", None)
        self._mtab: Dict[Any, torch.Tensor] = {}     # per-layer (H,) budget, cached
        self._cur_layer: int = -1
        self._rand_bid: Optional[torch.Tensor] = None       # cache, see _bin_ids_random
        self._exp_rnd: Optional[torch.Tensor] = None        # cache, random-rank control

    def _head_budget(self, n_heads: int, device: torch.device) -> Optional[torch.Tensor]:
        """(H,) expansion budget for the current layer, or None for a uniform budget."""
        if self.expand_m_table is None:
            return None
        key = (self._cur_layer, n_heads, device)
        if key not in self._mtab:
            per_head = self.expand_m_table.get(self._cur_layer, {})
            vals = [int(per_head.get(h, self.expand_m)) for h in range(n_heads)]
            self._mtab[key] = torch.tensor(vals, dtype=torch.long, device=device)
        return self._mtab[key]

    def _kappa(self, s2: torch.Tensor) -> torch.Tensor:
        if self.kappa_mode == "j1":
            return s2 / 2
        if self.kappa_mode == "j2":
            return torch.log1p(s2 / 2)
        return torch.zeros_like(s2)

    def _blockwise(self, x: torch.Tensor, tb: torch.Tensor, nb: int, m: int,
                   den: torch.Tensor) -> torch.Tensor:
        """Per-block tail mean of a key/value-side tensor. x: (B,H,K,D) -> (B,H,Q,nb,D)."""
        xv = x.reshape(x.shape[:-2] + (nb, m, x.shape[-1]))
        return torch.einsum("bhqnm,bhnmd->bhqnd", tb, xv) / den.unsqueeze(-1)

    @staticmethod
    def _moments_from_key_stats(q: torch.Tensor, k_mean: torch.Tensor,
                                k_sq: torch.Tensor,
                                scaling: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """(zbar_b, sigma_b^2) from a bin's first two key moments.

            zbar_b    = s * q . k_bar_b
            sigma_b^2 = s^2 * sum_d q_d^2 Var_b(k_d)

        zbar_b here is *algebraically identical* to the mean of the true logits over the
        bin's tail members -- mean_i s*q.k_i = s*q.mean_i k_i -- so reading it off the
        full score matrix, as the prototype used to, computes the same number the
        expensive way.  Going through k_bar_b instead is what makes the quantity
        query-independent and therefore cacheable with the KV pages.

        sigma_b^2 drops cross-dimension covariance, which is what keeps it O(d) per bin
        instead of O(d^2).
        """
        k_var = (k_sq - k_mean * k_mean).clamp(min=0)
        qe = q.unsqueeze(-2)                                              # (B,H,Q,1,D)
        zbar_b = scaling * (qe * k_mean).sum(-1)                          # (B,H,Q,nb)
        s2_b = (scaling ** 2) * ((q * q).unsqueeze(-2) * k_var).sum(-1)
        return zbar_b, s2_b

    def _bin_moments_fixed(self, q: torch.Tensor, k: torch.Tensor, tb: torch.Tensor,
                           nb: int, m: int, den: torch.Tensor,
                           scaling: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """(zbar_b, sigma_b^2) for position blocks, from key-side statistics only.

        Var_b(k_d) can be cached with the KV pages (d floats per block, the same
        footprint as the block mean), so nothing here ever needs the logits.
        """
        with stage("mb/T4a_bin_k_mean"):
            k_mean = self._blockwise(k, tb, nb, m, den)                   # (B,H,Q,nb,D)
        with stage("mb/T4b_bin_k_sq_mean"):
            k_sq = self._blockwise(k * k, tb, nb, m, den)
        with stage("mb/T4c_zbar_and_sigma"):
            return self._moments_from_key_stats(q, k_mean, k_sq, scaling)

    def _diag_var_fixed(self, q: torch.Tensor, k: torch.Tensor, tb: torch.Tensor,
                        nb: int, m: int, den: torch.Tensor, scaling: float) -> torch.Tensor:
        """sigma_b^2 only, for callers that already have zbar_b (the topblock path)."""
        return self._bin_moments_fixed(q, k, tb, nb, m, den, scaling)[1]

    # -------------------------------------------------- key-side bin statistics --
    # The two functions below produce the same (n_b, zbar_b, s2_b, vbar_b) as
    # `_bins_fixed` / `_aggregate_by_bin_id`, but WITHOUT reading the full score
    # matrix.  Every quantity is a reduction over K, V and the tail mask, i.e. over
    # the tokens the selector left behind -- which is the form a deployable kernel
    # can serve from per-block sums cached with the KV pages.  Used whenever
    # var_mode="diag"; var_mode="exact" still needs the true logits by construction.

    def _bins_fixed_keyside(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        tail_f: torch.Tensor, scaling: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """`fixed` position blocks, statistics from K/V only."""
        m = self.bin_size
        K = tail_f.shape[-1]
        pad = (-K) % m
        if pad:
            tail_f = torch.nn.functional.pad(tail_f, (0, pad), value=0.0)
            v = torch.nn.functional.pad(v, (0, 0, 0, pad), value=0.0)
            k = torch.nn.functional.pad(k, (0, 0, 0, pad), value=0.0)
        nb = (K + pad) // m

        with stage("mb/T3_bin_stats"):
            tb = tail_f.reshape(tail_f.shape[:-1] + (nb, m))
            n_b = tb.sum(-1)
            den = n_b.clamp(min=1)
            vbar_b = self._blockwise(v, tb, nb, m, den)
        with stage("mb/T4_bin_moments"):
            zbar_b, s2_b = self._bin_moments_fixed(q, k, tb, nb, m, den, scaling)
        return n_b, zbar_b, s2_b, vbar_b

    def _bins_scatter_keyside(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        tail_f: torch.Tensor, bid: torch.Tensor, nbins: int, scaling: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Irregular (adjacency / score) bins, statistics from K/V only."""
        with stage("mb/T3_bin_stats"):
            shp = tail_f.shape[:-1] + (nbins + 1,)
            n_b = torch.zeros(shp, dtype=tail_f.dtype, device=tail_f.device)
            n_b = n_b.scatter_add_(-1, bid, tail_f)[..., :nbins]
            den = n_b.clamp(min=1)
            vbar_b = self._scatter_mean(v, tail_f, bid, nbins, den)
        with stage("mb/T4_bin_moments"):
            zbar_b, s2_b = self._bin_moments_scatter(
                q, k, tail_f, bid, nbins, den, scaling)
            s2_b = torch.where(n_b >= 2, s2_b, torch.zeros_like(s2_b))
        return n_b, zbar_b, s2_b, vbar_b

    # ------------------------------------------------------------------ bins --
    def _bins_fixed(
        self, raw: torch.Tensor, tail_f: torch.Tensor, v: torch.Tensor,
        key_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Contiguous position blocks intersected with the tail.

        Returns (n_b, zbar_b, s2_b, vbar_b) with bin axis last-but-one.
        """
        m = self.bin_size
        K = raw.shape[-1]
        pad = (-K) % m
        if pad:
            raw = torch.nn.functional.pad(raw, (0, pad), value=NEG - 1.0)
            tail_f = torch.nn.functional.pad(tail_f, (0, pad), value=0.0)
            v = torch.nn.functional.pad(v, (0, 0, 0, pad), value=0.0)
        nb = (K + pad) // m

        shp = raw.shape[:-1] + (nb, m)                       # (B,H,Q,nb,m)
        rb, tb = raw.reshape(shp), tail_f.reshape(shp)
        n_b = tb.sum(-1)                                     # (B,H,Q,nb)
        den = n_b.clamp(min=1)
        zbar_b = (rb * tb).sum(-1) / den
        s2_b = ((rb * rb * tb).sum(-1) / den - zbar_b * zbar_b).clamp(min=0)
        s2_b = torch.where(n_b >= 2, s2_b, torch.zeros_like(s2_b))

        v_view = v.reshape(v.shape[:-2] + (nb, m, v.shape[-1]))   # (B,H,nb,m,D)
        vbar_b = torch.einsum("bhqnm,bhnmd->bhqnd", tb, v_view) / den.unsqueeze(-1)
        return n_b, zbar_b, s2_b, vbar_b

    def _scatter_mean(self, x: torch.Tensor, tail_f: torch.Tensor, bid: torch.Tensor,
                      nbins: int, den: torch.Tensor) -> torch.Tensor:
        """Per-bin tail mean of a key/value-side tensor, for irregular bin ids."""
        B, H, Q, _ = tail_f.shape
        D = x.shape[-1]
        acc = torch.zeros((B, H, Q, nbins + 1, D), dtype=tail_f.dtype, device=tail_f.device)
        hstep = max(1, 8 // max(1, Q))
        for h0 in range(0, H, hstep):
            h1 = min(H, h0 + hstep)
            src = x[:, h0:h1].unsqueeze(2) * tail_f[:, h0:h1].unsqueeze(-1)
            idx = bid[:, h0:h1].unsqueeze(-1).expand(-1, -1, -1, -1, D)
            acc[:, h0:h1].scatter_add_(3, idx, src)
            del src, idx
        return acc[..., :nbins, :] / den.unsqueeze(-1)

    def _bin_moments_scatter(self, q: torch.Tensor, k: torch.Tensor,
                             tail_f: torch.Tensor, bid: torch.Tensor, nbins: int,
                             den: torch.Tensor,
                             scaling: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """(zbar_b, sigma_b^2) for irregular (adjacency) bins, key-side only."""
        with stage("mb/T4a_bin_k_mean"):
            k_mean = self._scatter_mean(k, tail_f, bid, nbins, den)
        with stage("mb/T4b_bin_k_sq_mean"):
            k_sq = self._scatter_mean(k * k, tail_f, bid, nbins, den)
        with stage("mb/T4c_zbar_and_sigma"):
            return self._moments_from_key_stats(q, k_mean, k_sq, scaling)

    def _diag_var_scatter(self, q: torch.Tensor, k: torch.Tensor, tail_f: torch.Tensor,
                          bid: torch.Tensor, nbins: int, den: torch.Tensor,
                          scaling: float) -> torch.Tensor:
        """sigma_b^2 only, for callers that already have zbar_b."""
        return self._bin_moments_scatter(q, k, tail_f, bid, nbins, den, scaling)[1]

    def _aggregate_by_bin_id(
        self, raw: torch.Tensor, tail_f: torch.Tensor, v: torch.Tensor,
        bid: torch.Tensor, nbins: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scatter-reduce tail statistics into `nbins` bins given per-token bin ids.

        `bid` must already route every non-tail position to index `nbins`, a discard
        bin that is dropped on the way out.

        Statistics are always taken over the TRUE logits in `raw`, never over whatever
        quantity produced `bid`.  That separation is what lets the "hat" mode group by
        an *approximate* score without letting the approximation into the bin logit.

        Prototype only: materialises a (B,h_chunk,Q,K,D) buffer, so it is chunked over
        heads and is considerably slower than "fixed".
        """
        B, H, Q, _ = raw.shape
        D = v.shape[-1]

        shp = (B, H, Q, nbins + 1)
        z = torch.zeros(shp, dtype=raw.dtype, device=raw.device)
        n_b = z.clone().scatter_add_(-1, bid, tail_f)
        zsum = z.clone().scatter_add_(-1, bid, raw * tail_f)
        z2sum = z.clone().scatter_add_(-1, bid, raw * raw * tail_f)

        vsum = torch.zeros((B, H, Q, nbins + 1, D), dtype=raw.dtype, device=raw.device)
        hstep = max(1, 8 // max(1, Q))          # keep the temp buffer ~O(100MB)
        for h0 in range(0, H, hstep):
            h1 = min(H, h0 + hstep)
            src = v[:, h0:h1].unsqueeze(2) * tail_f[:, h0:h1].unsqueeze(-1)   # (B,h,Q,K,D)
            idx = bid[:, h0:h1].unsqueeze(-1).expand(-1, -1, -1, -1, D)
            vsum[:, h0:h1].scatter_add_(3, idx, src)
            del src, idx

        n_b, zsum = n_b[..., :nbins], zsum[..., :nbins]
        z2sum, vsum = z2sum[..., :nbins], vsum[..., :nbins, :]
        den = n_b.clamp(min=1)
        zbar_b = zsum / den
        s2_b = (z2sum / den - zbar_b * zbar_b).clamp(min=0)
        s2_b = torch.where(n_b >= 2, s2_b, torch.zeros_like(s2_b))
        return n_b, zbar_b, s2_b, vsum / den.unsqueeze(-1)

    def _bin_ids_equalcount(self, tail_f: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Runs of exactly `bin_size` ADJACENT tail indices (the complement of top-k).

        Membership depends only on positions, never on logits, so it escapes the
        circularity: a deployable kernel gets each run's sums from cached block prefix
        sums minus the selected tokens it is already loading.
        """
        m = self.bin_size
        K = tail_f.shape[-1]
        nbins = (K + m - 1) // m
        rank = (tail_f.cumsum(-1) - 1).clamp(min=0).long()
        bid = (rank // m).clamp(max=nbins - 1)
        return torch.where(tail_f > 0, bid, torch.full_like(bid, nbins)), nbins

    def _bins_equalcount(
        self, raw: torch.Tensor, tail_f: torch.Tensor, v: torch.Tensor,
        key_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bid, nbins = self._bin_ids_equalcount(tail_f)
        return self._aggregate_by_bin_id(raw, tail_f, v, bid, nbins)

    def _bin_ids_random(self, tail_f: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Uniformly random assignment of tail tokens to `num_bins` bins.

        Membership ignores position AND logits.  The assignment is drawn once per key
        POSITION and shared by every query row, which keeps the two properties that
        make a bin statistic deployable: it needs no logits (no circularity), and
        k_bar_b / Cov_b(K) stay query-independent, so one read serves the whole batch.

        What this isolates: every bin is an unbiased sample of the same tail, so the
        within-bin logit variance stays at the global tail variance for any bin count.
        Only the spread of the bin MEANS grows -- Var(zbar_b) ~ sigma^2 * B / n -- and
        that is the whole mechanism by which random binning recovers Jensen mass.
        """
        nbins = self.num_bins
        K = tail_f.shape[-1]
        # Cache across layers/steps: the partition must not be re-drawn per call, or
        # the tail proxy would jitter between decode steps for no reason.
        if (self._rand_bid is None or self._rand_bid.numel() < K
                or self._rand_bid.device != tail_f.device):
            gen = torch.Generator(device="cpu").manual_seed(self.random_seed)
            self._rand_bid = torch.randint(
                nbins, (max(K, 1),), generator=gen, dtype=torch.long).to(tail_f.device)
        bid = self._rand_bid[:K].expand(tail_f.shape).contiguous()
        return torch.where(tail_f > 0, bid, torch.full_like(bid, nbins)), nbins

    def _bins_random(
        self, raw: torch.Tensor, tail_f: torch.Tensor, v: torch.Tensor,
        key_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bid, nbins = self._bin_ids_random(tail_f)
        return self._aggregate_by_bin_id(raw, tail_f, v, bid, nbins)

    def _bins_score(
        self, raw: torch.Tensor, tail_f: torch.Tensor, v: torch.Tensor,
        key_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Equal-width bins over `key_scores`, taken across the tail's observed range.

        Grouping by score is what actually attacks UTA's error term: within-bin logit
        variance falls roughly as (range/num_bins)^2, whereas position blocks bottom
        out around half the global tail variance no matter how small they get.
        """
        bid, nbins = self._bin_ids_score(raw, tail_f, key_scores)
        return self._aggregate_by_bin_id(raw, tail_f, v, bid, nbins)

    def _bin_ids_score(self, raw: torch.Tensor, tail_f: torch.Tensor,
                       key_scores: Optional[torch.Tensor]) -> Tuple[torch.Tensor, int]:
        if key_scores is None:
            key_scores = raw
        nbins = self.num_bins
        tail = tail_f > 0
        s = key_scores.to(raw.dtype)

        lo = s.masked_fill(~tail, float("inf")).amin(-1, keepdim=True)
        hi = s.masked_fill(~tail, float("-inf")).amax(-1, keepdim=True)
        width = (hi - lo).clamp(min=1e-6)
        bid = ((s - lo) / width * nbins).floor().long().clamp(0, nbins - 1)
        return torch.where(tail, bid, torch.full_like(bid, nbins)), nbins

    # ------------------------------------------------------- two-stage path --
    @torch.no_grad()
    def _topblock_output(self, q, k, v, raw, sel, tail_f, scaling) -> torch.Tensor:
        """Cached block proxies everywhere + exact logits for the top-J blocks only.

        This is the deployable answer to the circularity: binning the tail by its own
        logits needs the full QK we are trying to avoid, but ranking BLOCKS needs only
        zbar_b = s * q . kbar_b, which is O(d) per block off cached means.  We then spend
        a bounded refinement budget on the few blocks that ranking says carry the mass,
        computing their tokens' true logits exactly -- the same kind of budget vAttention
        spends, except aimed rather than random.

        Cost per query/head:  3*B*d  (means, diagonal variance, value accumulation)
                            + J*m*d  (refinement)
        vs vAttention's      2*m_s*d  with m_s random gathers.
        """
        m, J = self.bin_size, self.refine_blocks
        K = raw.shape[-1]
        pad = (-K) % m
        if pad:
            rawp = torch.nn.functional.pad(raw, (0, pad), value=NEG - 1.0)
            tfp = torch.nn.functional.pad(tail_f, (0, pad), value=0.0)
            vp = torch.nn.functional.pad(v, (0, 0, 0, pad), value=0.0)
            kp = torch.nn.functional.pad(k, (0, 0, 0, pad), value=0.0)
        else:
            rawp, tfp, vp, kp = raw, tail_f, v, k
        nb = (K + pad) // m

        shp = rawp.shape[:-1] + (nb, m)
        rb, tb = rawp.reshape(shp), tfp.reshape(shp)
        n_b = tb.sum(-1)
        den = n_b.clamp(min=1)
        zbar_b = (rb * tb).sum(-1) / den                     # = s * q . kbar_b
        vbar_b = self._blockwise(vp, tb, nb, m, den)

        if self.var_mode == "diag":
            s2_b = self._diag_var_fixed(q, kp, tb, nb, m, den, scaling)
        else:
            s2_b = ((rb * rb * tb).sum(-1) / den - zbar_b * zbar_b).clamp(min=0)
        s2_b = torch.where(n_b >= 2, s2_b, torch.zeros_like(s2_b))

        live = n_b > 0
        ell_b = (zbar_b + torch.log(den) + self._kappa(s2_b)).masked_fill(~live, float("-inf"))

        # rank blocks by the mass their cached statistics predict, refine the top J
        j = min(J, nb)
        top = ell_b.topk(j, dim=-1).indices                                  # (B,H,Q,j)
        refine_blk = torch.zeros_like(n_b, dtype=torch.bool).scatter_(-1, top, True)
        promoted = (refine_blk.unsqueeze(-1) & (tb > 0)).reshape(rawp.shape)[..., :K]

        exact = sel | promoted                       # tokens whose logit we truly compute
        ell_b = ell_b.masked_fill(refine_blk, float("-inf"))   # their blocks drop out

        row_max = torch.maximum(
            raw.masked_fill(~exact, float("-inf")).amax(-1, keepdim=True),
            ell_b.amax(-1, keepdim=True))
        row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))

        e = torch.exp(raw - row_max) * exact
        num, den_ = torch.matmul(e, v), e.sum(-1, keepdim=True)

        w_b = torch.exp(ell_b - row_max)
        w_b = torch.where(torch.isfinite(w_b), w_b, torch.zeros_like(w_b))
        num = num + torch.einsum("bhqn,bhqnd->bhqd", w_b, vbar_b)
        den_ = den_ + w_b.sum(-1, keepdim=True)

        if MicroMetricLogger().is_metric_enabled("uta_refined_tokens"):
            MicroMetricLogger().log("uta_refined_tokens",
                                    float(promoted.sum(-1).float().mean().item()))

        return (num / den_.clamp(min=1e-30)).to(q.dtype).transpose(1, 2).contiguous()

    # ----------------------------------------------------------------- merge --
    def _multibin_output(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        sparse_mask: Mask,
        key_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with stage("mb/T2_scores_fullQK"):
            ngroups = _get_num_key_value_groups(queries, keys)
            k = repeat_kv(keys, ngroups).to(torch.float32)
            v = repeat_kv(values, ngroups).to(torch.float32)
            q = queries.to(torch.float32)

            raw = torch.matmul(q, k.transpose(2, 3)) * scaling        # (B,H,Q,K)
            if attention_mask is not None:
                raw = raw + attention_mask[:, :, :, : k.shape[-2]].to(torch.float32)

        with stage("mb/T1_tail_mask"):
            valid = raw > NEG
            sel = (sparse_mask.get_dense_mask() != 0) & valid
            tail_f = ((~sel) & valid).to(torch.float32)

        if self.bin_mode == "topblock":
            return self._topblock_output(q, k, v, raw, sel, tail_f, scaling)

        # var_mode="diag": every bin statistic is a reduction over K/V and the tail
        # mask, so the score matrix is never read here -- zbar_b comes from the bin's
        # own key mean.  var_mode="exact" wants the true within-bin logit variance,
        # which no key-side statistic can reproduce, so it keeps the raw path.
        bid = None
        if self.bin_mode == "fixed":
            if self.var_mode == "diag":
                n_b, zbar_b, s2_b, vbar_b = self._bins_fixed_keyside(
                    q, k, v, tail_f, scaling)
            else:
                with stage("mb/T3_bin_stats"):
                    n_b, zbar_b, s2_b, vbar_b = self._bins_fixed(raw, tail_f, v)
        else:
            with stage("mb/T2b_bin_ids"):
                if self.bin_mode == "equalcount":
                    bid, nbins = self._bin_ids_equalcount(tail_f)
                elif self.bin_mode == "random":
                    bid, nbins = self._bin_ids_random(tail_f)
                else:
                    bid, nbins = self._bin_ids_score(raw, tail_f, key_scores)
            if self.var_mode == "diag":
                n_b, zbar_b, s2_b, vbar_b = self._bins_scatter_keyside(
                    q, k, v, tail_f, bid, nbins, scaling)
            else:
                with stage("mb/T3_bin_stats"):
                    n_b, zbar_b, s2_b, vbar_b = self._aggregate_by_bin_id(
                        raw, tail_f, v, bid, nbins)

        # --- second stage: pull the hottest bins out and compute them exactly -----
        # E0 established that what the tail loses is a few huge-logit tokens, which no
        # proxy SHAPE can carry.  So instead of correcting a bin's mass, remove the bin
        # from the proxy set and evaluate its tokens directly.  Ranking uses only
        # zbar_b = s*q.kbar_b and sigma_b, both key-side, so no logits are consulted.
        expand_tok: Optional[torch.Tensor] = None
        if self.expand_m > 0 and bid is not None:
            with stage("mb/T4b_expand_rank"):
                live0 = n_b > 0
                if self.expand_rank == "random":
                    if self._exp_rnd is None or self._exp_rnd.numel() < nbins:
                        g = torch.Generator(device="cpu").manual_seed(self.random_seed + 7)
                        self._exp_rnd = torch.rand(max(nbins, 1), generator=g).to(n_b.device)
                    score_b = self._exp_rnd[:nbins].expand_as(n_b)
                elif self.expand_rank == "mass":
                    # The criterion every centroid method ranks by: the bin's own
                    # estimated attention mass, size-weighted centroid logit
                    # (Double-P eq.5, RetroInfer's cluster estimate, ClusterKV's
                    # centroid inner product).  It is a FIRST-moment statistic, so a
                    # bin holding one outlier among many ordinary tokens looks
                    # ordinary -- which is exactly the bin worth expanding.
                    score_b = zbar_b + torch.log(n_b.clamp(min=1))
                elif self.expand_rank == "massj2":
                    # ...the same, with the bounded Jensen correction folded in, so
                    # the comparison is not merely "they forgot the second moment".
                    score_b = (zbar_b + torch.log(n_b.clamp(min=1))
                               + torch.log1p(s2_b.clamp(min=0) / 2))
                else:
                    # ours: an upper confidence bound on the bin's LARGEST logit.
                    # Under a sub-Gaussian model for the within-bin logits,
                    # max_i z_i <~ zbar_b + sigma_b*sqrt(2 log n_b), so expand_c plays
                    # the role of sqrt(2 log n_b) (2.63 at n_b=32).
                    score_b = zbar_b + self.expand_c * s2_b.clamp(min=0).sqrt()
                m_h = self._head_budget(score_b.shape[1], score_b.device)
                if m_h is None:
                    mm = min(self.expand_m, nbins)
                    idx = score_b.masked_fill(~live0, float("-inf")).topk(mm, dim=-1).indices
                    is_exp = torch.zeros_like(live0).scatter_(-1, idx, True) & live0
                else:
                    # per-head budget: one topk to the layer's maximum, then keep only
                    # each head's own prefix
                    mm = min(int(m_h.max()), nbins)
                    idx = score_b.masked_fill(~live0, float("-inf")).topk(mm, dim=-1).indices
                    rank = torch.arange(mm, device=score_b.device).view(1, 1, 1, mm)
                    keep = (rank < m_h.view(1, -1, 1, 1)).expand_as(idx).contiguous()
                    is_exp = torch.zeros_like(live0).scatter_(-1, idx, keep) & live0
                pad_col = torch.zeros_like(is_exp[..., :1])
                expand_tok = (torch.cat([is_exp, pad_col], -1).gather(-1, bid)
                              & (tail_f > 0))
                n_b = n_b.masked_fill(is_exp, 0)     # expanded bins leave the proxy set

        # Fused decode path: hand the merge to the Triton kernel.  Everything above --
        # bin statistics, ranking, expansion choice -- is unchanged, so any score
        # difference would be the kernel's doing and nothing else.
        if (self.use_triton_decode and expand_tok is not None
                and queries.shape[2] == 1 and bid is not None):
            fused = self._triton_decode(q, k, v, sel, expand_tok, tail_f, bid,
                                        nbins, n_b, vbar_b, scaling)
            if fused is not None:
                return fused.to(queries.dtype).transpose(1, 2).contiguous()

        # per-bin logit: mean logit + log of the bin's OWN token count
        with stage("mb/T5_kappa_bin_logit"):
            live = n_b > 0
            ell_b = zbar_b + torch.log(n_b.clamp(min=1)) + self._kappa(s2_b)
            ell_b = ell_b.masked_fill(~live, float("-inf"))

        # single global softmax over {heavy tokens} U {bins}
        with stage("mb/T6_merge_softmax"):
            raw_max = raw.masked_fill(~valid, float("-inf")).amax(-1, keepdim=True)
            bin_max = ell_b.amax(-1, keepdim=True)
            row_max = torch.maximum(raw_max, bin_max)
            row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))

            exact = sel if expand_tok is None else (sel | expand_tok)
            e_sel = torch.exp(raw - row_max) * exact
            num = torch.matmul(e_sel, v)
            den = e_sel.sum(-1, keepdim=True)

            w_b = torch.exp(ell_b - row_max)                           # (B,H,Q,nb)
            w_b = torch.where(live, w_b, torch.zeros_like(w_b))
            num = num + torch.einsum("bhqn,bhqnd->bhqd", w_b, vbar_b)
            den = den + w_b.sum(-1, keepdim=True)

        if MicroMetricLogger().is_metric_enabled("uta_bin_count"):
            MicroMetricLogger().log("uta_bin_count", float(live.sum(-1).float().mean().item()))

        return (num / den.clamp(min=1e-30)).to(queries.dtype).transpose(1, 2).contiguous()

    @torch.no_grad()
    def _triton_decode(self, q, k, v, sel, expand_tok, tail_f, bid, nbins,
                       n_b, vbar_b, scaling):
        """Decode-step merge via the fused Triton kernel; None if the shapes do not fit.

        The kernel wants explicit token-index lists while the masker pipeline produces
        boolean masks.  At decode the per-row counts are constant (fixed sink/local/
        heavy budgets, fixed m), so the conversion is a topk rather than a ragged
        gather; if that ever fails to hold we fall back to the torch path rather than
        silently truncating.
        """
        try:
            from triton_expand_decode import decode_expand_attention
        except Exception:
            return None

        B, H, Q, K = sel.shape
        if Q != 1:
            return None
        selc, expc = sel[:, :, 0, :], expand_tok[:, :, 0, :]
        n_sel, n_exp = selc.sum(-1), expc.sum(-1)
        ns, ne = int(n_sel.max()), int(n_exp.max())
        if ns == 0 or ne == 0:
            return None

        def _idx(flags, n, counts):
            """Boolean row mask -> (B*H, n) int32 index list, -1 in the padding slots.

            Rows can legitimately hold different counts (a bin clipped by the end of
            the sequence), so the list is padded rather than truncated -- truncating
            would silently drop tokens from the attention.
            """
            idx = flags.float().topk(n, dim=-1).indices
            rank = torch.arange(n, device=idx.device).view(1, 1, n)
            idx = torch.where(rank < counts.unsqueeze(-1), idx, torch.full_like(idx, -1))
            return idx.int().reshape(B * H, n)

        sel_idx = _idx(selc, ns, n_sel)
        exp_idx = _idx(expc, ne, n_exp)

        # per-bin key mean: the query-independent statistic the kernel consumes, and
        # exactly what zbar_b was derived from (zbar_b = s * q . kbar_b)
        den = n_b.clamp(min=1)
        kbar = self._scatter_mean(k, tail_f, bid, nbins, den)[:, :, 0, :, :]

        BQ, D = B * H, q.shape[-1]
        log_n = torch.log(n_b[:, :, 0, :].clamp(min=1)).reshape(BQ, nbins).float()
        log_n = log_n.masked_fill(n_b[:, :, 0, :].reshape(BQ, nbins) <= 0, float("-inf"))

        out = decode_expand_attention(
            q[:, :, 0, :].reshape(BQ, D).contiguous(),
            k.reshape(BQ, K, D).contiguous(), v.reshape(BQ, K, D).contiguous(),
            sel_idx, exp_idx,
            kbar.reshape(BQ, nbins, D).contiguous(),
            vbar_b[:, :, 0, :, :].reshape(BQ, nbins, D).contiguous(),
            log_n, sm_scale=scaling,
        )
        return out.view(B, H, 1, D)

    # ------------------------------------------------------------ entrypoint --
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
        if "sparse_meta_data" not in kwargs:
            raise ValueError("sparse_meta_data must be provided in kwargs")
        sparse_meta_data: Dict[Any, Any] = kwargs.pop("sparse_meta_data")

        mask_shape = (queries.shape[0], queries.shape[1], queries.shape[2], keys.shape[2])
        sparse_mask = Mask.create_empty_mask(mask_shape, dtype=queries.dtype, device=queries.device)
        for masker in self.maskers:
            with stage(f"sel/{type(masker).__name__}"):
                sparse_mask = masker.add_mask(
                    keys=keys, queries=queries, values=values,
                    attention_mask=attention_mask, scaling=scaling, dropout=dropout,
                    sparse_meta_data=sparse_meta_data, previous_mask=sparse_mask, **kwargs,
                )

        if MicroMetricLogger().is_metric_enabled("research_attention_density"):
            MicroMetricLogger().log(
                "research_attention_density", sparse_mask.get_density(),
                metadata={"layer_idx": kwargs.get("layer_idx")},
            )

        if sparse_mask.is_full_mask():
            from ..utils.mask_attention_utils import get_true_attention_output
            return get_true_attention_output(
                module, queries, keys, values, attention_mask, scaling, dropout, **kwargs
            )

        key_scores: Optional[torch.Tensor] = None
        if self.bin_mode == "hat":
            layer_idx = kwargs.get("layer_idx")
            key_scores = sparse_meta_data.get("hat_scores", {}).get(layer_idx)
            if key_scores is None:
                raise ValueError(
                    "bin_mode='hat' needs HashAttention's approximate scores, but none "
                    f"were published for layer {layer_idx}. Put a HashAttentionTopKMasker "
                    "in the masker pipeline, or use bin_mode='score'/'fixed'."
                )

        self._cur_layer = int(kwargs.get("layer_idx") or 0)
        output = self._multibin_output(
            queries, keys, values, attention_mask, scaling, sparse_mask, key_scores
        )

        if MicroMetricLogger().is_metric_enabled("research_attention_output_error"):
            from ..utils.mask_attention_utils import get_true_attention_output
            true_output, _ = get_true_attention_output(
                module, queries, keys, values, attention_mask, scaling, dropout, **kwargs
            )
            err = torch.norm(true_output - output) / torch.norm(true_output)
            MicroMetricLogger().log(
                "research_attention_output_error", float(err.item()),
                metadata={"layer_idx": kwargs.get("layer_idx")},
            )

        return output, None

    @classmethod
    def create_from_config(cls, config: SparseAttentionConfig) -> "UTAMultiBinAttention":
        if not isinstance(config, UTAMultiBinConfig):
            raise TypeError(f"Expected UTAMultiBinConfig, got {type(config)}")
        maskers = [ResearchMasker.create_masker_from_config(mc) for mc in config.masker_configs]
        return cls(config, maskers)
