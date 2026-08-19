#!/usr/bin/env python3
"""
Experiment 3: multi-bin UTA on top of a HashAttention (HAT) selector.

Why HAT rather than oracle-top-k
--------------------------------
oracle-top-k needs the full QK product to pick its indices, so nothing built on
it can ever be fast.  HAT produces approximate top-k indices from learned hash
signatures -- it is the selector vAttention actually deploys.  Everything after
selection (the tail) is then genuinely free budget, which is where multi-bin UTA
lives.  HAT's selection is also *imperfect*: some high-logit tokens leak into the
tail, which raises the tail's logit variance.  Prediction: multi-bin should help
MORE under HAT than under oracle-top-k, because it isolates those leaked tokens
into their own bins instead of averaging them away.

Methods (all at matched total density rho)
------------------------------------------
  HAT              sink + local + HAT top-k                 (no tail recovery)
  UTA-HAT          + single mean-pooled tail proxy          (base UTA)
  MultiBin-HAT     + per-bin proxies, bin_size sweep        (this work)
  vAttention-HAT   HAT top-k at rho/2 + adaptive sampling   (baseline to beat)

Density accounting: sink 0.1% + local 0.1% + heavy (+ sampling for vAttention).
The tail proxy itself costs no *tokens*, so all methods read the same number of
KV entries; the multi-bin overhead is bin metadata, accounted separately.

Usage
-----
    python run_multibin_hat_bench.py --device cuda:2 --densities 0.03
    python run_multibin_hat_bench.py --device cuda:2 --summarize-only
"""

import argparse
import json
import os
import sys
from typing import Any, Dict
from pathlib import Path

import pandas as pd
import torch

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

from benchmark.ruler32k import Ruler32K  # noqa: E402
from sparse_attention_hub.adapters import ModelAdapterHF  # noqa: E402
from sparse_attention_hub.sparse_attention.research_attention import (  # noqa: E402
    ResearchAttentionConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (  # noqa: E402
    HashAttentionTopKMaskerConfig,
    LocalMaskerConfig,
    OracleTopKConfig,
    SinkMaskerConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (  # noqa: E402
    AdaptiveSamplingMaskerConfig,
)
from sparse_attention_hub.sparse_attention.uta_attention import (  # noqa: E402
    UTAAttentionConfig,
    UTAMultiBinConfig,
)

SUBSETS = [
    "qa_1", "qa_2", "vt", "fwe",
    "niah_multikey_2", "niah_multikey_3", "niah_multivalue",
]

HAT_WEIGHT_FILE = (
    "/database/hyunwoo/hf/HashAttention-1.0/repo/artifacts/"
    "llama3.1-8b-patch.64K.v1.hat_weights.pkl"
)
HAT_KW = dict(hat_bits=32, hat_mlp_layers=3, hat_mlp_hidden_size=128,
              hat_mlp_activation="silu", hat_weight_file=HAT_WEIGHT_FILE)

SINK = 0.001
LOCAL = 0.001

# Context length the density fractions are resolved against.  The expansion budget is
# a token COUNT (expand_m * bin_size), so turning it into a density needs this; RULER32K
# is fixed at 32768, and the density-matched variant is only defined for that benchmark.
CTX_LEN = 32768
EXPAND_M_TABLE = "results_v9_decomp/expand_m_table_mean8.json"


def _selector(heavy: float, selector: str):
    """Top-k selector.  'hat' is the deployable one; 'oracle' needs the full QK."""
    h = round(heavy, 6)
    if selector == "hat":
        return HashAttentionTopKMaskerConfig(heavy_size=h, **HAT_KW)
    if selector == "oracle":
        return OracleTopKConfig(heavy_size=h)
    raise ValueError(f"unknown selector: {selector}")


def _prefix(heavy: float, selector: str = "hat"):
    return [SinkMaskerConfig(sink_size=SINK), LocalMaskerConfig(window_size=LOCAL),
            _selector(heavy, selector)]


# Default: bin the tail by whatever score the selector already computed, which costs
# nothing extra.  MEASURED CAVEAT (run_block_tail_diag, 5%): HAT's approximate score
# is useless for this -- within-bin logit variance ratio 0.72 vs 0.47 for plain
# position blocks and 0.97 for a random partition.  Its sign-vector signatures are
# trained to surface the top tokens, not to rank the bulk.  So under a HAT selector
# pass --bin-mode explicitly: "score" for the accuracy ceiling (needs the full QK, so
# not cost-consistent) or "fixed" for the cost-consistent position-block point.
BIN_MODE_FOR = {"hat": "hat", "oracle": "score"}
POSITION_MODES = ("fixed", "equalcount")


def build_config(method: str, rho: float, num_bins: int, kappa: str,
                 vatt_split: float = 0.5, eps: float = 0.4, delta: float = 0.4,
                 selector: str = "hat", bin_mode: str = "auto", var_mode: str = "diag",
                 expand_c: float = 3.0):
    heavy = round(rho - SINK - LOCAL, 6)
    _prefix_ = lambda h: _prefix(h, selector)  # noqa: E731
    mode = BIN_MODE_FOR[selector] if bin_mode == "auto" else bin_mode

    # ---- ablation ladder (all at the same density, same selector) ------------
    # dense -> no sparsity at all;  selector -> top-k only, tail discarded;
    # UTA -> one mean-pooled tail proxy;  +jensen -> that proxy gets log(1+s2/2);
    # +multibin -> proxy per run of adjacent tail indices;  ours -> both.
    if method == "dense":
        return None

    if method == "selector":
        return ResearchAttentionConfig(masker_configs=_prefix_(heavy))

    if method == "UTA":
        return UTAAttentionConfig(masker_configs=_prefix_(heavy))

    if method == "UTA+jensen":
        # one bin spanning the whole tail == base UTA, plus the correction
        return UTAMultiBinConfig(masker_configs=_prefix_(heavy), bin_mode="equalcount",
                                 bin_size=10 ** 9, kappa_mode="j2", var_mode="diag")

    if method == "UTA+multibin":
        return UTAMultiBinConfig(masker_configs=_prefix_(heavy), bin_mode="equalcount",
                                 bin_size=num_bins, kappa_mode="none", var_mode="diag")

    if method == "UTA+expand+matched":
        # Density-MATCHED expansion: the m*bin_size tokens the second stage reads are
        # taken OUT of the selector's budget instead of being added on top, so the
        # total tokens touched equal vAttention's and HAT's exactly.  Without this the
        # comparison gives us 0.78% more density than the baselines.
        exp_frac = num_bins * 32 / CTX_LEN
        heavy_m = round(rho - SINK - LOCAL - exp_frac, 6)
        if heavy_m <= 0:
            raise ValueError(f"rho={rho} too small for m={num_bins} expansion")
        return UTAMultiBinConfig(
            masker_configs=_prefix_(heavy_m), bin_mode="equalcount", bin_size=32,
            kappa_mode="none", var_mode="diag", expand_m=num_bins, expand_c=expand_c,
            expand_rank="bound")

    if method == "UTA+expand+triton":
        # identical configuration to UTA+expand; the decode step is routed through the
        # fused Triton kernel so the benchmark verifies end-to-end equivalence
        return UTAMultiBinConfig(
            masker_configs=_prefix_(heavy), bin_mode="equalcount", bin_size=32,
            kappa_mode="none", var_mode="diag", expand_m=num_bins, expand_c=expand_c,
            expand_rank="bound", use_triton_decode=True)

    if method == "UTA+expandhead":
        # Non-uniform but STATIC per-(layer,head) budget with the same MEAN m, so the
        # token count matches UTA+expand exactly on average and the run-time cost is
        # identical (no per-step decision, no ragged launch).
        import json as _json
        tbl = _json.load(open(EXPAND_M_TABLE))
        table = {int(l): {int(h): int(m) for h, m in hh.items()}
                 for l, hh in tbl["table"].items()}
        return UTAMultiBinConfig(
            masker_configs=_prefix_(heavy), bin_mode="equalcount", bin_size=32,
            kappa_mode="none", var_mode="diag", expand_m=num_bins, expand_c=expand_c,
            expand_rank="bound", expand_m_table=table)

    if method in ("UTA+expandmass", "UTA+expandmassj2"):
        # The centroid-mass ranking every prior cluster/page method uses: pick the
        # bins with the largest estimated mass, zbar_b + log n_b.  Same partition,
        # same budget, same merge -- ONLY the ranking statistic differs from
        # UTA+expand, so the gap isolates the second moment's contribution.
        return UTAMultiBinConfig(
            masker_configs=_prefix_(heavy), bin_mode="equalcount", bin_size=32,
            kappa_mode="none", var_mode="diag", expand_m=num_bins, expand_c=expand_c,
            expand_rank="mass" if method == "UTA+expandmass" else "massj2")

    if method in ("UTA+expand", "UTA+expandrand"):
        # E1/E2: adjacency bins of 32, then the top `num_bins` bins by the key-side
        # bound are computed exactly.  `num_bins` is the EXPANSION COUNT m here.
        return UTAMultiBinConfig(
            masker_configs=_prefix_(heavy), bin_mode="equalcount", bin_size=32,
            kappa_mode="none", var_mode="diag", expand_m=num_bins, expand_c=expand_c,
            expand_rank="random" if method == "UTA+expandrand" else "bound")

    if method == "UTA+randbin":
        # V4: partition the tail (= everything sink/local/HAT did NOT take) uniformly
        # at random into `num_bins` bins.  num_bins is a BIN COUNT here, not a size.
        return UTAMultiBinConfig(masker_configs=_prefix_(heavy), bin_mode="random",
                                 num_bins=num_bins, kappa_mode="none", var_mode="diag")

    if method in ("ours", "UTA+multibin+jensen"):
        if method != "ours":
            return UTAMultiBinConfig(masker_configs=_prefix_(heavy), bin_mode="equalcount",
                                     bin_size=num_bins, kappa_mode="j2", var_mode="diag")
        # position/adjacency modes are parameterised by tokens-per-bin, score modes by
        # bin count
        cfg: Dict[str, Any] = ({"bin_size": num_bins} if mode in POSITION_MODES
                               else {"num_bins": num_bins})
        return UTAMultiBinConfig(masker_configs=_prefix_(heavy), bin_mode=mode,
                                 kappa_mode=kappa, var_mode=var_mode, **cfg)

    # ---- legacy names kept so already-completed 1%/3% runs stay resumable ----
    if method == "HAT":
        return ResearchAttentionConfig(masker_configs=_prefix(heavy, "hat"))

    if method == "UTA-HAT":
        return UTAAttentionConfig(masker_configs=_prefix(heavy, "hat"))

    if method == "MultiBin-HAT":
        return UTAMultiBinConfig(masker_configs=_prefix(heavy, "hat"),
                                 bin_size=num_bins, bin_mode="fixed", kappa_mode=kappa)

    if method in ("vAttention", "vAttention-HAT"):
        # `vatt_split` is the fraction of the total density spent on random sampling.
        # 0.5 reproduces the paper's 50/50 setting; lower values hand budget back to
        # the HAT selector, which matters at low rho where a starved selector — not
        # the sampling estimator — is what breaks retrieval.
        sampling = round(rho * vatt_split, 6)
        h = round(rho - SINK - LOCAL - sampling, 6)
        if h <= 0:
            raise ValueError(f"rho={rho} too small for vatt_split={vatt_split}")
        # (eps, delta) drive the adaptive budget as ~(ppf(1-delta)/eps)^2, so 0.4/0.4
        # allocates ~18x fewer samples than 0.25/0.25.  Any comparison against the
        # oracle-top-k density sweep must use the same pair (0.25/0.25).
        return ResearchAttentionConfig(masker_configs=_prefix_(h) + [
            AdaptiveSamplingMaskerConfig(base_rate_sampling=sampling, epsilon=eps, delta=delta,
                                         init_offset=SINK, local_offset=LOCAL),
        ])

    raise ValueError(f"unknown method: {method}")


def job_name(method: str, rho: float, num_bins: int, kappa: str,
             vatt_split: float = 0.5, eps: float = 0.4, delta: float = 0.4,
             selector: str = "hat", bin_mode: str = "auto",
             expand_c: float = 3.0) -> str:
    tag = f"{rho * 100:g}pct"
    sel = "HAT" if selector == "hat" else "oracle"
    mode = BIN_MODE_FOR[selector] if bin_mode == "auto" else bin_mode

    # --- new selector-aware names ---
    if method == "ours":
        suffix = "" if kappa == "j2" else f"_{kappa}"
        # name the bin criterion whenever it is not the selector's free default
        tail = "" if mode == BIN_MODE_FOR[selector] else f"-{mode}"
        return f"ours-b{num_bins}{tail}{suffix}({sel})@{tag}"
    if method in ("selector", "UTA"):
        label = sel if method == "selector" else f"UTA({sel})"
        return f"{label}@{tag}"
    if method in ("UTA+multibin", "UTA+multibin+jensen"):
        return f"{method}(b{num_bins})@{tag}"
    if method == "UTA+randbin":
        return f"UTA+randbin(n{num_bins})@{tag}"
    if method == "UTA+expand":
        return f"UTA+expand(m{num_bins}c{expand_c:g})@{tag}"
    if method == "UTA+expand+triton":
        return f"UTA+expand-triton(m{num_bins}c{expand_c:g})@{tag}"
    if method == "UTA+expand+matched":
        return f"UTA+expand-matched(m{num_bins}c{expand_c:g})@{tag}"
    if method == "UTA+expandrand":
        return f"UTA+expandrand(m{num_bins})@{tag}"
    if method == "UTA+expandhead":
        return f"UTA+expand-perhead(mbar{num_bins})@{tag}"
    if method in ("UTA+expandmass", "UTA+expandmassj2"):
        rule = "mass" if method == "UTA+expandmass" else "massj2"
        return f"UTA+expand-{rule}(m{num_bins})@{tag}"
    if method == "vAttention":
        parts = "" if (eps, delta) == (0.25, 0.25) else f"-e{eps:g}d{delta:g}"
        if vatt_split != 0.5:
            parts += f"-s{vatt_split:g}"
        return f"vAttention({sel}){parts}@{tag}"

    # --- legacy names: keep byte-identical so finished 1%/3% dirs still resume ---
    if method == "MultiBin-HAT":
        suffix = "" if kappa == "none" else f"_{kappa}"
        return f"MultiBin{num_bins}{suffix}-HAT@{tag}"
    if method == "vAttention-HAT":
        parts = ""
        if vatt_split != 0.5:
            parts += f"-s{vatt_split:g}"
        if (eps, delta) != (0.4, 0.4):
            parts += f"-e{eps:g}d{delta:g}"
        return f"vAttention-HAT{parts}@{tag}"
    return f"{method}@{tag}"


def run_job(name: str, config, args) -> None:
    method_dir = Path(args.output_dir) / name.replace("@", "_at_")
    pending = [s for s in SUBSETS if not (method_dir / s / "metrics.json").exists()]
    if not pending:
        print(f"[SKIP] {name} — complete")
        return

    print(f"\n{'=' * 72}\n[{args.device}] {name} — {len(pending)} subset(s)\n{'=' * 72}", flush=True)
    adapter = ModelAdapterHF(
        model_name=args.model,
        sparse_attention_config=config,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=args.device,
    )
    try:
        for subset in pending:
            d = method_dir / subset
            d.mkdir(parents=True, exist_ok=True)
            print(f"  -> {name}/{subset}", flush=True)
            Ruler32K(subsets_to_run=[subset]).run_benchmark(
                adapter, str(d),
                request_kwargs={"max_requests": args.max_requests, "max_context_length": 32000},
                generation_kwargs={"max_new_tokens": 120},
            )
            mf = d / "metrics.json"
            if mf.exists():
                s = json.loads(mf.read_text()).get("task_scores", {}).get(subset, {}).get("string_match", 0.0)
                print(f"     {subset}: {s:.2f}", flush=True)
    finally:
        del adapter
        torch.cuda.empty_cache()


# Finished runs elsewhere in the repo that use the SAME protocol (sink/local 0.001,
# eps=delta=0.25, 200 samples) and can therefore be quoted next to the new ones.
REUSABLE = {
    "vAttention(oracle)@5pct": "results_covuta/vAttention_at_5pct",
    "vAttention(oracle)@10pct": "results_covuta/vAttention_at_10pct",
    "UTA(oracle)@5pct": "results_covuta/UTA_at_5pct",
    "UTA(oracle)@10pct": "results_covuta/UTA_at_10pct",
    "oracle@5pct": "results_covuta/oracle_top_k_at_5pct",
    "oracle@10pct": "results_covuta/oracle_top_k_at_10pct",
}


def summarize(output_dir: Path, include_reusable: bool = True) -> None:
    dirs = [(d.name.replace("_at_", "@"), d) for d in sorted(output_dir.glob("*_at_*"))]
    if include_reusable:
        dirs += [(label, Path(p)) for label, p in REUSABLE.items() if Path(p).is_dir()]

    rows = []
    for name, d in sorted(dirs):
        row = {"Method": name}
        scores = []
        for subset in SUBSETS:
            mf = d / subset / "metrics.json"
            if mf.exists():
                s = json.loads(mf.read_text()).get("task_scores", {}).get(subset, {}).get("string_match", 0.0)
                row[subset] = round(s, 2)
                scores.append(s)
            else:
                row[subset] = None
        row["Average"] = round(sum(scores) / len(scores), 2) if len(scores) == len(SUBSETS) else None
        rows.append(row)

    if not rows:
        print("no results yet")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)[["Method", "Average"] + SUBSETS]
    print("\n" + "=" * 104)
    print("MULTI-BIN UTA on HAT — RULER32K")
    print("=" * 104)
    print(df.to_string(index=False))
    out = output_dir / "multibin_hat_summary.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-bin UTA with a HAT selector")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--output-dir", default="./results_multibin_hat")
    p.add_argument("--max-requests", type=int, default=200)
    p.add_argument("--densities", default="0.05,0.10")
    p.add_argument("--bin-mode", default="auto",
                   choices=["auto", "score", "hat", "fixed", "equalcount"],
                   help="tail bin criterion; 'auto' uses the selector's free score "
                        "(see BIN_MODE_FOR). Under a HAT selector 'hat' is measurably "
                        "bad -- prefer 'score' (accuracy ceiling) or 'fixed'.")
    p.add_argument("--selector", default="hat", choices=["hat", "oracle"],
                   help="top-k selector; also picks the tail bin criterion "
                        "(hat -> HAT approx scores, oracle -> true logits)")
    p.add_argument("--methods", default="ours,vAttention",
                   help="ours | vAttention | UTA | selector (legacy: MultiBin-HAT, ...)")
    p.add_argument("--bin-sizes", default="16,32",
                   help="comma-separated num_bins for 'ours' (bin_size for legacy MultiBin-HAT)")
    p.add_argument("--no-reusable", action="store_true",
                   help="omit already-finished protocol-compatible runs from the summary")
    p.add_argument("--kappa", default="j2", choices=["none", "j2"],
                   help="per-bin Jensen correction on the bin logit (denominator side)")
    p.add_argument("--eps", type=float, default=0.4,
                   help="vAttention adaptive-sampling error bound (0.25 matches the "
                        "oracle-top-k density sweep; 0.4 matches reproduce_table1)")
    p.add_argument("--delta", type=float, default=0.4,
                   help="vAttention adaptive-sampling confidence bound")
    p.add_argument("--vatt-split", type=float, default=0.5,
                   help="fraction of total density vAttention spends on random sampling "
                        "(0.5 = paper setting; lower gives the HAT selector more budget)")
    p.add_argument("--expand-c", type=float, default=3.0,
                   help="bins are ranked by zbar_b + c*sigma_b for the expansion stage")
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    if args.summarize_only:
        summarize(output_dir, include_reusable=not args.no_reusable)
        return

    densities = [float(x) for x in args.densities.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]
    bin_sizes = [int(x) for x in args.bin_sizes.split(",")]

    if args.selector == "hat" and methods != ["dense"] and not Path(HAT_WEIGHT_FILE).exists():
        raise FileNotFoundError(f"HAT weights not found: {HAT_WEIGHT_FILE}")

    jobs = []
    for rho in densities:
        for m in methods:
            sizes = bin_sizes if m in ("ours", "MultiBin-HAT", "UTA+multibin",
                                       "UTA+multibin+jensen", "UTA+randbin",
                                       "UTA+expand", "UTA+expandrand", "UTA+expandmass", "UTA+expandmassj2", "UTA+expandhead",
                                       "UTA+expand+triton",
                                       "UTA+expand+matched") else [0]
            for bs in sizes:
                common = (m, rho, bs, args.kappa, args.vatt_split,
                          args.eps, args.delta, args.selector, args.bin_mode,
                          args.expand_c)
                jobs.append((job_name(*common), build_config(*common)))

    print(f"[{args.device}] {len(jobs)} job(s):")
    for n, _ in jobs:
        print(f"   - {n}")

    for name, cfg in jobs:
        run_job(name, cfg, args)

    summarize(output_dir)


if __name__ == "__main__":
    main()
