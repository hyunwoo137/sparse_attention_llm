#!/usr/bin/env python3
"""Phase 3: cost accounting for the tail estimator, against vAttention's sampling.

The constraint this project is under: whatever we spend recovering the tail must not
exceed what vAttention spends on its random samples.  This script makes that budget
explicit and auditable instead of asserted.

It counts the work done PER DECODE QUERY PER ATTENTION HEAD, on top of the shared
top-k selection that both methods pay.  Two quantities matter and they are counted
separately, because on a memory-bound decode the second one usually decides:

  MACs   - multiply-accumulates
  bytes  - HBM traffic, split into per-query (cannot be shared) and shared
           (query-independent, so one read serves the whole batch and, under GQA,
           every query head in the group -- these land in L2 and are effectively free
           once amortised)

Assumptions are printed with the result; change them with flags.
"""

import argparse


def fmt(n: float) -> str:
    for unit, div in (("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:.0f}"


def main() -> None:
    p = argparse.ArgumentParser(description="Tail-estimator cost vs vAttention")
    p.add_argument("--seq-len", type=int, default=32768)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--density", type=float, default=0.05, help="total density rho")
    p.add_argument("--bin-size", type=int, default=32, help="tokens per adjacency bin")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--gqa", type=int, default=4, help="query heads per KV head")
    p.add_argument("--bytes-per-elem", type=int, default=2, help="bf16 KV cache")
    p.add_argument("--stat-bytes", type=int, default=1, help="int8 bin statistics")
    a = p.parse_args()

    K, d, rho, m = a.seq_len, a.head_dim, a.density, a.bin_size
    sink_local = 0.002

    # vAttention splits the budget: half to the selector, half to random sampling.
    # Only the sampling half is *extra* over a plain top-k method at the same density.
    n_samp = int(round(K * rho / 2))
    # per sample: q.k_i  (d MAC) and the weighted value accumulation (d MAC)
    v_mac = 2 * n_samp * d
    # reads K and V for each sampled position, at random offsets, different per query
    v_bytes_pq = n_samp * 2 * d * a.bytes_per_elem

    # Ours: every tail token belongs to a bin; bins are runs of adjacent tail indices.
    n_tail = K * (1 - rho)
    n_bins = int(round(n_tail / m))
    #   zbar_b = s * q.kbar_b            -> d MAC per bin
    #   sigma_b^2 (diagonal)             -> d MAC per bin   (q^2 . var_b)
    #   numerator  sum_b w_b * vbar_b    -> d MAC per bin
    o_mac = 3 * n_bins * d
    # per bin we read kbar_b, var_b, vbar_b: 3d elements, and they are query-independent
    o_bytes_shared = n_bins * 3 * d * a.stat_bytes
    # correcting each bin for the selected tokens inside it reuses tokens the sparse
    # attention already loaded, so it adds MACs but no new traffic
    n_sel = int(round(K * (rho - sink_local)))
    o_mac += 2 * n_sel * d

    share = a.batch * a.gqa
    o_bytes_eff = o_bytes_shared / share

    print(f"\nassumptions: K={K}  d={d}  rho={rho:.0%}  bin={m}  batch={a.batch}  "
          f"GQA={a.gqa}  KV={a.bytes_per_elem}B  bin-stats={a.stat_bytes}B")
    print(f"             vAttention samples={n_samp}   ours bins={n_bins}   "
          f"selected={n_sel}\n")

    rows = [
        ("vAttention sampling (extra over top-k)", v_mac, v_bytes_pq, 0.0),
        ("ours: adjacency bins + diag variance", o_mac, 0.0, o_bytes_shared),
    ]
    print(f"{'':40s}{'MAC':>10}{'bytes/query':>14}{'bytes shared':>14}")
    for name, mac, bpq, bsh in rows:
        print(f"{name:40s}{fmt(mac):>10}{fmt(bpq):>14}{fmt(bsh):>14}")

    print(f"\nratio ours/vAttention   MAC {o_mac / v_mac:6.2f}x"
          f"   traffic {(o_bytes_shared / share) / v_bytes_pq:6.2f}x"
          f"  (shared read amortised over batch x GQA = {share})")
    print(f"                        traffic un-amortised "
          f"{o_bytes_shared / v_bytes_pq:6.2f}x")

    verdict = "WITHIN budget" if (o_mac <= v_mac and o_bytes_eff <= v_bytes_pq) else \
              "OVER budget"
    print(f"\n=> {verdict}\n")

    # What the alternatives would have cost, for the record.
    full_cov = n_bins * d * d
    print("rejected alternatives, same units:")
    print(f"  full covariance variance (B*d^2)      {fmt(full_cov):>10} MAC "
          f"= {full_cov / v_mac:.1f}x vAttention  -> rejected")
    print(f"  score binning (needs every tail logit) {fmt(2 * int(n_tail) * d):>9} MAC "
          f"= {2 * n_tail * d / v_mac:.1f}x vAttention  -> ceiling only\n")


if __name__ == "__main__":
    main()
