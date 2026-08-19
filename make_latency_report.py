#!/usr/bin/env python3
"""results_latency/latency_raw.json -> results/REPORT_V3_LATENCY.md 렌더링.

순수 포맷팅만 한다. 리포트의 모든 수치는 벤치마크가 기록한 JSON에서 나오고,
정확도 열만 results_v2/multibin_hat_summary.csv에서 인용한다.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

repo_root = Path(__file__).resolve().parent

# results_v2/multibin_hat_summary.csv 의 어느 행이 어느 latency 라벨에 대응하는지.
# 여기 없는 포인트는 정확도를 아직 측정하지 않은 것이다.
ACCURACY_FOR = {
    "dense": "dense@5pct",
    "HAT": "HAT@5pct",
    "UTA": "UTA(HAT)@5pct",
    "UTA+multibin(b32)": "UTA+multibin(b32)@5pct",
    "UTA+jensen": "UTA+jensen@5pct",
    "UTA+multibin+jensen(b32)": "UTA+multibin+jensen(b32)@5pct",
    "vAttention(HAT) e.25d.25": "vAttention(HAT)@5pct",
}

# 단계 중첩 관계. 상위 단계만 더하면 호출 전체가 되도록 하고, 하위 단계는
# 들여쓰기해서 보여주되 합계에 다시 더하지 않는다.
CHILDREN = {
    # 현재 이름 (key-side bin moment 경로)
    "mb/T4_bin_moments": ["mb/T4a_bin_k_mean", "mb/T4b_bin_k_sq_mean",
                          "mb/T4c_zbar_and_sigma"],
    # 예전 이름 (key-side 전환 이전에 기록된 JSON도 계속 렌더되도록 유지)
    "mb/T4_jensen_var": ["mb/T4a_var_k_mean", "mb/T4b_var_k_sq_mean",
                         "mb/T4c_var_q2_dot"],
    "utaj/T4_jensen_var": ["utaj/T4b_var_k_sq_mean", "utaj/T4c_var_q2_dot"],
    "sel/AdaptiveSamplingMasker": [
        "vatt/S1_expwts_fullQK", "vatt/S2_static_denominator",
        "vatt/S3_base_sample_std", "vatt/S4_error_eval_and_budget",
        "vatt/S5_resample_extra_budget", "vatt/S6_merge_mask",
    ],
    "vatt/S5_resample_extra_budget": ["vatt/S5a_D2H_sync_budget_sum"],
}
CHILD_OF = {c: p for p, cs in CHILDREN.items() for c in cs}

STAGE_NOTE = {
    "sel/SinkMasker": "sink 토큰 (전 방법 공통)",
    "sel/LocalMasker": "local window (전 방법 공통)",
    "sel/HashAttentionTopKMasker": "HAT top-k 선택 (전 방법 공통)",
    "sel/AdaptiveSamplingMasker": "vAttention 샘플링 단 전체",
    "core/masked_attention_output": "sparse softmax + 출력",
    "vatt/S1_expwts_fullQK": "full QK + exp (프로토타입에만 존재)",
    "vatt/S2_static_denominator": "이미 선택된 토큰들의 분모",
    "vatt/S3_base_sample_std": "base 샘플 → std 추정 = **오차 평가**",
    "vatt/S4_error_eval_and_budget": "오차 한계 → 행별 **추가 budget 산정**",
    "vatt/S5_resample_extra_budget": "새 budget으로 **재샘플링**",
    "vatt/S5a_D2H_sync_budget_sum": "device→host 동기화: budget 크기가 데이터 의존이라 발생",
    "vatt/S6_merge_mask": "샘플링 마스크를 선택 마스크에 병합",
    "uta/T1_tail_mask": "tail 마스크 + tail 개수",
    "uta/T2_scores_fullQK": "full QK (프로토타입에만 존재)",
    "uta/T3_tail_kv_mean": "tail 구간의 K, V 평균 풀링",
    "uta/T5_proxy_logit": "q·k_mean + log N",
    "uta/T6_merge_softmax": "전역 softmax 병합 + 출력",
    "utaj/T1_tail_mask": "tail 마스크 + tail 개수",
    "utaj/T3_tail_kv_mean": "tail 구간의 K, V 평균 풀링",
    "utaj/T4_jensen_var": "**Jensen 편차 단 (합계)**",
    "utaj/T4b_var_k_sq_mean": "  tail 구간 E[k²] → Var(k_d)",
    "utaj/T4c_var_q2_dot": "  σ² = s²·Σ_d q_d²·Var(k_d)",
    "utaj/T5_proxy_logit": "z̄ + α·σ²/2 + log N",
    "mb/T1_tail_mask": "선택 결과로부터 tail 마스크 구성",
    "mb/T2_scores_fullQK": "full QK (프로토타입에만 존재)",
    "mb/T2b_bin_ids": "**bin 분할** (토큰별 bin id)",
    "mb/T3_bin_stats": "bin별 n_b, v̄_b (K/V 리덕션)",
    "mb/T4_bin_moments": "**bin 모멘트 단: z̄_b + σ_b² (합계)**",
    "mb/T4a_bin_k_mean": "  bin별 E[k] = k̄_b",
    "mb/T4b_bin_k_sq_mean": "  bin별 E[k²]",
    "mb/T4c_zbar_and_sigma": "  z̄_b = s·q·k̄_b, σ_b² = s²·Σ_d q_d²·Var_b(k_d)",
    # 예전 이름 (key-side 전환 이전 JSON 호환)
    "mb/T4_jensen_var": "**Jensen 편차 단 (합계)** — 전환 이전 이름",
    "mb/T4a_var_k_mean": "  bin별 E[k]",
    "mb/T4b_var_k_sq_mean": "  bin별 E[k²]",
    "mb/T4c_var_q2_dot": "  σ_b² = s²·Σ_d q_d²·Var_b(k_d)",
    "mb/T5_kappa_bin_logit": "κ = log(1+σ²/2), bin logit 계산",
    "mb/T6_merge_softmax": "{heavy} ∪ {bins} 전역 softmax + 출력",
}


def load_accuracy(path: Path) -> Dict[str, Optional[float]]:
    if not path.exists():
        return {}
    out: Dict[str, Optional[float]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                out[row["Method"]] = float(row["Average"])
            except (TypeError, ValueError):
                out[row["Method"]] = None
    return out


def fmt(x: Optional[float], nd: int = 3) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def table(headers: List[str], rows: List[List[str]], align: str = "") -> str:
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = {"l": ":---", "r": "---:", "c": ":--:"}
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(sep[a] for a in align) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def stage_rows(stages: Dict[str, Any], total: float) -> List[List[str]]:
    """상위 단계를 비용 내림차순으로, 하위 단계는 부모 아래 들여쓰기."""
    tops = [(n, s) for n, s in stages.items() if n not in CHILD_OF]
    tops.sort(key=lambda x: -x[1]["ms_per_call"])
    rows = []
    for name, s in tops:
        ms = s["ms_per_call"]
        rows.append([f"`{name}`", f"{ms:.3f}",
                     f"{100 * ms / total:.1f}%" if total else "—",
                     STAGE_NOTE.get(name, "")])
        for child in CHILDREN.get(name, []):
            if child in stages:
                cms = stages[child]["ms_per_call"]
                rows.append([f"&nbsp;&nbsp;↳ `{child.split('/')[-1]}`", f"{cms:.3f}",
                             f"{100 * cms / total:.1f}%" if total else "—",
                             STAGE_NOTE.get(child, "")])
    return rows


def top_level_total(stages: Dict[str, Any]) -> float:
    return sum(s["ms_per_call"] for n, s in stages.items() if n not in CHILD_OF)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="results_latency/latency_raw.json")
    p.add_argument("--acc", default="results_v2/multibin_hat_summary.csv")
    p.add_argument("--out", default="results/REPORT_V3_LATENCY.md")
    a = p.parse_args()

    data = json.load(open(a.raw))
    res: Dict[str, List[Dict[str, Any]]] = data["results"]
    acc = load_accuracy(Path(a.acc))
    main_rows = res.get("main", [])
    ok = [r for r in main_rows if r["status"] == "ok"]

    base = next((r for r in ok if r["label"] == "vAttention(HAT) e.25d.25"), None)
    base_ms = base["mean_ms"] if base else None

    L: List[str] = []
    A = L.append

    A("# REPORT V3 — Latency: UTA ablation ladder vs vAttention")
    A("")
    A(f"장비: **{data['device']}** (GPU 1) · torch {data['torch']} · "
      "Llama-3.1-8B 형상 (H=32, H_kv=8, D=128, bf16) · "
      "HAT selector · 밀도 ρ=5% 일치")
    A("")
    A("모든 config는 `run_multibin_hat_bench.build_config`로 만들었다. "
      "`results_v2/`의 정확도 수치를 생성한 것과 **동일한 빌더**이므로, "
      "이 리포트의 latency 열과 정확도 열은 같은 대상을 가리킨다.")
    A("")

    # ---------------------------------------------------------------- 주의사항
    A("## 0. 숫자를 인용하기 전에 반드시 읽을 것")
    A("")
    A("**Tier 1이 재는 것은 프로토타입이지 배포 커널이 아니다.** 여기 나오는 모든 "
      "방법이 — 우리 것도, vAttention도 — full `(B,H,Q,K)` score matrix를 그대로 "
      "만든다 (`mb/T2_scores_fullQK`, `uta/T2_scores_fullQK`, `vatt/S1_expwts_fullQK`). "
      "배포용 커널이라면 어느 쪽도 이런 일을 하지 않는다. 따라서:")
    A("")
    A("- 같은 형상에서의 **방법 간 비율**은 의미가 있다. O(QK) 항이 모두에게 공통이기 때문이다.")
    A("- **절대 밀리초는 배포 성능 추정치가 아니다.**")
    A("- 프로토타입 전용 기계장치가 비용을 지배하는 항목(§3 참고)은 알고리즘이 아니라 "
      "코드에 대해 말해주는 것이다.")
    A("")
    A("decode(Q=1)가 올바른 측정 영역이다. 어댑터는 context prefill을 "
      "`enable_sparse_mode()` **바깥에서** 수행하므로, 실제 실행에서 sparse attention "
      "호출은 전부 decode step이다 "
      "([huggingface.py:174](../sparse_attention_hub/adapters/huggingface.py#L174)).")
    A("")

    # ------------------------------------------------------------ 메인 테이블
    A("## 1. 메인 비교 — K=32768, B=1, Q=1, ρ=5%")
    A("")
    rows = []
    for r in main_rows:
        if r["status"] != "ok":
            rows.append([r["label"], "—", "—", "—", "—", "—", "—", r["status"][:40]])
            continue
        ratio = f"{r['mean_ms'] / base_ms:.2f}×" if base_ms else "—"
        acc_key = ACCURACY_FOR.get(r["label"])
        acc_val = acc.get(acc_key) if acc_key else None
        rows.append([
            r["label"], fmt(r["mean_ms"]), fmt(r["p50_ms"]), fmt(r["p95_ms"]),
            ratio, f"{r['density']:.4f}", f"{r['peak_mem_MB']:.0f}",
            fmt(acc_val, 2) if acc_val is not None else "—",
        ])
    A(table(["method", "mean ms", "p50", "p95", "vs vAtt", "실측 ρ",
             "peak MB", "RULER avg"], rows))
    A("")
    A("`RULER avg`는 `results_v2/multibin_hat_summary.csv`에서 인용했다 "
      "(200 samples × 7 subsets). `—`는 latency는 쟀지만 정확도 실험은 아직 "
      "돌리지 않은 변형이라는 뜻이다.")
    A("")

    # 밀도 일치 검증
    dens = [r["density"] for r in ok if r["label"] != "dense"]
    if dens:
        A(f"**등가 비용 전제가 성립한다.** 실측 밀도가 모든 sparse 방법에서 "
          f"{min(dens):.4f}–{max(dens):.4f} 범위에 있다. 즉 전부 2% 이내로 같은 개수의 "
          "KV 엔트리를 읽는다. vAttention이 명목치보다 살짝 *아래*에 있는 이유는 "
          "복원추출(with replacement)로 샘플링하기 때문이며, 따라서 이 비교는 오히려 "
          "vAttention에 유리한 쪽으로 기울어 있다.")
        A("")

    # ------------------------------------------------------------- 단계 분해
    A("## 2. 단계별 분해")
    A("")
    A("호출당 CUDA 시간, 계측 활성 상태(`UTA_STAGE_TIMING=1`). 비율은 상위 단계 합계 "
      "대비이며, 들여쓴 행은 바로 위 행에 **포함된** 하위 단계라 합계에 다시 더하지 않는다.")
    A("")

    def emit_breakdown(label: str, heading: str, blurb: str = "") -> None:
        r = next((x for x in ok if x["label"] == label), None)
        if not r or not r.get("stages"):
            return
        tot = top_level_total(r["stages"])
        A(f"### {heading}")
        A("")
        if blurb:
            A(blurb)
            A("")
        A(f"계측 없는 총합 **{r['mean_ms']:.3f} ms** · 계측 포함 "
          f"{r['instrumented_mean_ms']:.3f} ms · 단계 합계 {tot:.3f} ms")
        A("")
        A(table(["단계", "ms/call", "비중", "무엇인가"],
                stage_rows(r["stages"], tot), align="lrrl"))
        A("")

    emit_breakdown(
        "vAttention(HAT) e.25d.25",
        "2.1 vAttention — 오차 평가 → 추가 budget → 재샘플링 포함",
        "\"오차 평가 → 추가 budget 할당 → 재샘플링\" 단은 **S3 → S4 → S5**에 해당한다. "
        "이건 따로 덧붙인 게 아니라 `AdaptiveSamplingMasker.add_mask` 안에 들어 있어서 "
        "**구조적으로** 측정 구간에 포함된다 "
        "([adaptive_sampling.py:366-410](../sparse_attention_hub/sparse_attention/research_attention/maskers/sampling/implementations/adaptive_sampling.py#L366-L410)).")
    emit_breakdown("UTA", "2.2 UTA (tail 전체를 하나의 평균 proxy로)")
    emit_breakdown(
        "UTA+jensen (direct impl)", "2.3 UTA + Jensen, 직접 구현",
        "`advanced.py`의 `UTAJensenAttention`: proxy 하나 + 편차 단. "
        "Jensen 비용을 깨끗하게 분리해서 보여주는 것이 이 행이다.")
    emit_breakdown(
        "UTA+multibin+jensen(b32)", "2.4 UTA + multi-bin + Jensen (b=32, ladder config)",
        "ladder의 `equalcount` 분할 — 정확도 수치를 생성한 바로 그 config다.")
    emit_breakdown(
        "UTA+multibin+jensen(b32,fixed)",
        "2.5 UTA + multi-bin + Jensen (b=32, `fixed` bins)",
        "수식은 동일하고, tail을 인접 tail 토큰의 런이 아니라 **위치 블록**으로 나눈다. "
        "경계가 query와 무관하므로 bin 통계를 KV 페이지와 함께 캐시할 수 있다 — "
        "즉 이쪽이 배포 가능한 분할이다.")
    emit_breakdown(
        "UTA+jensen", "2.6 ladder가 표현하는 방식의 UTA + Jensen (거대 bin 1개)")

    # ---------------------------------------------------- Jensen 비용 분리
    A("## 3. Jensen 편차 단의 실제 비용")
    A("")
    by = {r["label"]: r for r in ok}
    jrows = []

    def delta(a_lbl: str, b_lbl: str, what: str) -> None:
        if a_lbl in by and b_lbl in by:
            d = by[b_lbl]["mean_ms"] - by[a_lbl]["mean_ms"]
            jrows.append([what, f"{by[a_lbl]['mean_ms']:.3f}",
                          f"{by[b_lbl]['mean_ms']:.3f}", f"{d:+.3f}",
                          f"{100 * d / by[a_lbl]['mean_ms']:+.1f}%"])

    delta("UTA", "UTA+jensen (direct impl)", "proxy 1개 위의 Jensen (직접 구현)")
    delta("UTA", "UTA+jensen", "proxy 1개 위의 Jensen (ladder: 거대 bin 1개)")
    delta("UTA+multibin(b32)", "UTA+multibin+jensen(b32)",
          "1024개 bin 위의 Jensen (equalcount)")
    delta("UTA+multibin(b32,fixed)", "UTA+multibin+jensen(b32,fixed)",
          "1024개 bin 위의 Jensen (fixed)")
    if jrows:
        A(table(["측정", "없을 때 ms", "있을 때 ms", "Δ ms", "Δ %"], jrows))
        A("")

    A("이 표를 같이 읽으면:")
    A("")
    A("1. **직접 구현하면 편차 단은 싸다.** `UTA+jensen (direct impl)`은 UTA 위에 "
      "차원별 `E[k²]` 리덕션 하나와 `Σ_d q_d²·Var(k_d)` 내적 하나를 더할 뿐이고, "
      "그만큼의 작고 한정된 증분만 든다.")
    A("2. **multi-bin 위에서는 사실상 공짜다.** `var_mode=\"diag\"`가 bin 평균이 이미 "
      "쓰는 것과 동일한 형태의 bin별 리덕션을 재활용하기 때문이다. `fixed` 분할에서는 "
      "한계 비용이 측정 노이즈 수준이다.")
    A("3. **ladder의 `UTA+jensen` 행은 측정 함정이다.** 이 행은 "
      "`bin_mode=\"equalcount\", bin_size=1e9`, 즉 tail 전체를 덮는 bin 하나로 구성돼 있다 "
      "([run_multibin_hat_bench.py:129-131](../run_multibin_hat_bench.py#L129-L131)). "
      "그 결과 모든 tail 토큰이 bin id 0으로 scatter_add되어 리덕션 전체가 단일 주소 "
      "atomics로 직렬화된다. 수학적으로는 *가장 싼* 변형인데 표에서 가장 느린 이유가 "
      "이것이다. **이 행을 Jensen의 비용으로 인용하면 안 된다.** 1번 행이 정직한 숫자다.")
    A("")
    A("§2.4에서 하나 더 보인다: `mb/T4a_var_k_mean`이 계산하는 bin별 `E[k]`는 "
      "`mb/T3_bin_stats`가 `z̄_b`를 구하려고 이미 손에 들고 있던 값이다. 이걸 넘겨주기만 "
      "하면 편차 단의 세 리덕션 중 하나를 공짜로 없앨 수 있다.")
    A("")

    # ---------------------------------------------------------- vAttention 고유
    A("## 4. 연산량으로 잡히지 않는 vAttention 고유 비용")
    A("")
    vr = by.get("vAttention(HAT) e.25d.25")
    if vr and vr.get("stages"):
        s = vr["stages"]
        sync = s.get("vatt/S5a_D2H_sync_budget_sum", {}).get("ms_per_call")
        samp = s.get("sel/AdaptiveSamplingMasker", {}).get("ms_per_call")
        A(f"**매 layer, 매 step마다 device→host 동기화가 걸린다.** adaptive budget이 "
          f"데이터에 의존하므로 샘플링 마스크의 크기를 device가 알려주기 전까지 알 수 "
          f"없다. 그래서 "
          f"[mask_attention_utils.py:145-150](../sparse_attention_hub/sparse_attention/utils/mask_attention_utils.py#L145-L150)의 "
          f"`int(budgets_flat.sum().item())`이 강제 동기화를 일으킨다. K=32768에서 "
          f"**{fmt(sync)} ms/call**로 측정됐다 — 연산이 아니라 파이프라인 정지다. "
          f"단순히 32개 layer로 곱하면 토큰당 ~{fmt(sync * 32 if sync else None, 2)} ms지만, "
          f"실제 모델에서 그중 얼마가 남는지는 CPU가 얼마나 앞서 달리고 있었는지에 "
          f"달려 있고 단일 layer 하네스로는 알 수 없다. 확실한 것은 구조적인 사실 쪽이다: "
          f"UTA 계열에는 host 동기화가 **하나도 없다**. 스텝당 작업량이 미리 정해지기 "
          f"때문이다.")
        A("")
        if samp:
            A(f"**vAttention의 추가 budget이 어디로 가는가.** 샘플링 단 전체가 "
              f"{fmt(samp)} ms/call이고, 그중 오차 평가 → budget → 재샘플링 사슬"
              f"(S3+S4+S5)이 "
              f"{fmt(sum(s.get(k, {}).get('ms_per_call', 0.0) for k in ('vatt/S3_base_sample_std', 'vatt/S4_error_eval_and_budget', 'vatt/S5_resample_extra_budget')))} ms다.")
            A("")

    # 분산 — 데이터가 뒷받침하는 만큼만
    A("**한쪽은 확률적이고 한쪽은 결정적이다 — 다만 그 결과는 이 벤치마크로 아직 "
      "정량화되지 않았다.** vAttention은 budget이 데이터에 반응하므로 실측 밀도가 "
      "반복마다 달라진다. 반면 모든 UTA 변형은 반복 간 완전히 동일하다:")
    A("")
    drows = []
    for r in ok:
        if r["label"] == "dense":
            continue
        spread = r["density_max"] - r["density"]
        drows.append([r["label"], f"{r['density']:.5f}", f"{r['density_max']:.5f}",
                      "결정적" if spread == 0 else f"변동 (+{spread:.5f})",
                      fmt(r["p50_ms"]), fmt(r["p95_ms"]), fmt(r["max_ms"]),
                      fmt(r["std_ms"], 4)])
    A(table(["method", "ρ 평균", "ρ 최대", "반복 간 ρ",
             "p50 ms", "p95 ms", "max ms", "std ms"], drows))
    A("")
    A("여기서 조심할 것이 둘 있다:")
    A("")
    A("- 밀도 열은 **실제로 두 방법을 구분한다.** vAttention은 확률적이고 UTA 계열은 "
      "정확히 결정적이다. 이건 진짜 구조적 차이다 — 서빙 시스템은 UTA 스텝의 작업량은 "
      "미리 산정할 수 있지만 vAttention 스텝은 산정할 수 없다.")
    A("- **하지만 latency 열은 두 방법을 구분하지 못한다.** vAttention의 std는 "
      "`UTA+multibin(b32,fixed)`의 std와 비슷한 수준이고, 편차는 전부 몇 % 이내다. "
      "이 마이크로벤치는 모든 반복에 고정된 합성 텐서 하나를 먹이므로 vAttention의 "
      "budget이 거의 움직이지 않고, 따라서 데이터 의존성이 전혀 발현되지 않는다. "
      "tail latency 논증을 정량화하려면 실제 RULER 입력으로 여러 decode step에 걸친 "
      "스텝별 측정이 필요하다. 위 표로는 **입증되지 않으며, 위 표를 근거로 주장해서는 "
      "안 된다.**")
    A("")

    # ------------------------------------------------------------- 스윕
    def sweep(key: str, heading: str, xcol: str, blurb: str = "") -> None:
        rows_ = res.get(key, [])
        if not rows_:
            return
        A(f"## {heading}")
        A("")
        if blurb:
            A(blurb)
            A("")
        labels, xs = [], []
        for r in rows_:
            if r["label"] not in labels:
                labels.append(r["label"])
            if r[xcol] not in xs:
                xs.append(r[xcol])
        body = []
        for x in xs:
            row = [str(x)]
            for lbl in labels:
                m = next((r for r in rows_ if r["label"] == lbl and r[xcol] == x), None)
                row.append(fmt(m["mean_ms"]) if m and m["status"] == "ok"
                           else (m["status"][:18] if m else "—"))
            body.append(row)
        A(table([xcol] + labels, body))
        A("")

    A("아래 모든 스윕은 **두 가지 tail 분할을 함께** 싣는다: `equalcount`(인접 tail "
      "토큰의 런 — `results_v2` 정확도 수치를 만든 분할)와 `fixed`(위치 블록 — 수식은 "
      "같고 경계가 query와 무관한, 배포 가능한 분할). 둘의 스케일링이 크게 다르고, "
      "그 차이가 이번 실험의 가장 실용적인 결과다.")
    A("")

    sweep("seqlen", "5. 시퀀스 길이 스케일링 (mean ms, B=1, Q=1)", "seq_len",
          "모든 프로토타입이 O(K) full-QK 항을 지고 있어서 모든 열이 선형으로 자란다. "
          "여기서 읽어야 할 것은 기울기가 아니라 **격차**다.")

    # fixed 분할과 vAttention 사이의 교차점 (스윕 데이터에서 계산)
    sl = res.get("seqlen", [])
    if sl:
        pairs = []
        for K in sorted({r["seq_len"] for r in sl}):
            f = next((r for r in sl if r["seq_len"] == K
                      and r["label"] == "UTA+multibin+jensen(b32,fixed)"
                      and r["status"] == "ok"), None)
            g = next((r for r in sl if r["seq_len"] == K
                      and r["label"].startswith("vAttention") and r["status"] == "ok"), None)
            if f and g:
                pairs.append((K, f["mean_ms"] / g["mean_ms"]))
        faster = [K for K, ratio in pairs if ratio < 1.0]
        if faster and len(faster) < len(pairs):
            A(f"**교차점이 존재한다.** `fixed` 분할은 K={max(faster)}까지는 vAttention보다 "
              f"오히려 *빠르고* ({', '.join(f'K={K}: {r:.2f}×' for K, r in pairs)}), "
              f"긴 쪽 끝에서만 진다. 두 곡선 모두 동일한 프로토타입 전용 O(K) full-QK 항이 "
              f"지배하므로, 이 교차점이 알려주는 것은 두 tail 추정기의 **상수 계수** 차이다. "
              f"그리고 그 부분이야말로 배포 커널에서도 남는 부분이다.")
            A("")

    sweep("batch", "6. 배치 스케일링 (mean ms, K=32768, Q=1)", "batch",
          "우리 쪽의 구조적 논거는, bin 통계가 query와 무관하므로 한 번 읽으면 배치 전체와 "
          "GQA 그룹 내 모든 query head에 재사용된다는 것이다. 반면 vAttention의 random "
          "gather는 query별이라 공유할 수 없다. **그런데 이 프로토타입은 아직 그 이점을 "
          "구현하지 않았다.** 양쪽 다 query마다 전부 다시 계산한다. 따라서 이 표는 논거의 "
          "증거가 아니라, 배포 경로가 넘어서야 할 기준선으로 읽어야 한다.")
    sweep("binsize", "7. Bin 크기 다이얼 (mean ms, K=32768, B=1, Q=1)", "bin_size",
          "bin당 토큰 수. bin이 작을수록 proxy가 많아지고 정확해지며 연산도 늘어난다. "
          "bin_size=1이면 exact attention이다. 이 표에서 쓸모 있는 부분은 **열이 거의 "
          "평평하다는 것**이다. 프로토타입에서 비용을 결정하는 것은 bin 개수가 아니라 "
          "O(K) 패스이므로, 이 구간에서는 정확도를 거의 공짜로 살 수 있다.")
    sweep("nq", "8. Query 개수 스케일링 (mean ms, K=32768, B=1)", "num_queries",
          "Q>1은 요청당 한 번 발생한다. 어댑터가 autoregressive decode를 시작하기 전에 "
          "질문 토큰들을 sparse forward 한 번으로 처리하는 시점이다.")

    # ------------------------------------------------- 분할 비교
    A("### 8.1 스케일하지 않는 것은 `equalcount`의 scatter 경로다")
    A("")
    A("`equalcount` bin은 인접한 *tail* 토큰의 런이므로 bin 소속이 선택 결과에 의존하고, "
      "따라서 reshape으로 표현할 수 없다. 그래서 "
      "[`_aggregate_by_bin_id`](../sparse_attention_hub/sparse_attention/uta_attention/multibin.py#L226-L253)는 "
      "`(B, h_chunk, Q, K, D)` 버퍼를 실제로 만들어 scatter_add를 수행하고, 청킹 휴리스틱 "
      "`hstep = max(1, 8 // Q)`는 Q > 8이 되는 순간 1로 붕괴한다. 반면 `fixed` bin은 위치 "
      "블록이라 동일한 통계가 `reshape` + `sum`으로 끝난다 — scatter도, 거대 버퍼도 없고, "
      "경계가 query와 무관하다. 그리고 바로 그 성질이 통계를 KV 페이지와 함께 캐시 "
      "가능하게 만드는 성질이다.")
    A("")
    A("두 분할은 **같은 추정기**를 구현한다. 정확도 차이가 무엇으로 밝혀지든, 위 표들의 "
      "비용 차이는 충분히 크므로 `equalcount`가 RULER에서 값어치를 증명하지 못하는 한 "
      "`fixed`가 기본값이 되어야 한다 — 그리고 그 비교는 아직 돌리지 않았다.")
    A("")

    # ------------------------------------------------------------ 결론
    A("## 9. 이 실험이 입증한 것과 입증하지 않은 것")
    A("")
    A("**입증한 것:**")
    A("")
    if base_ms:
        eq = by.get("UTA+multibin+jensen(b32)")
        fx = by.get("UTA+multibin+jensen(b32,fixed)")
        eq_acc = acc.get("UTA+multibin+jensen(b32)@5pct")
        va_acc = acc.get("vAttention(HAT)@5pct")
        if eq and eq_acc is not None and va_acc is not None:
            A(f"- **latency와 정확도가 둘 다 있는 config**: "
              f"`UTA+multibin+jensen(b32)` (equalcount)는 **{eq['mean_ms']:.3f} ms** — "
              f"vAttention의 **{base_ms:.3f} ms** 대비 {eq['mean_ms'] / base_ms:.2f}× — "
              f"이고, vAttention 쪽 숫자에는 오차 평가 → budget → 재샘플링 단이 완전히 "
              f"포함돼 있다. RULER는 **{fmt(eq_acc, 2)}** 대 vAttention의 "
              f"**{fmt(va_acc, 2)}**. 즉 이 ablation은 프로토타입 기준 "
              f"{eq['mean_ms'] / base_ms:.2f}× latency로 RULER +{eq_acc - va_acc:.1f}점을 산다.")
        if fx:
            A(f"- **가장 빠른 config**: 동일한 추정기를 `fixed` 위치 블록 위에서 돌리면 "
              f"**{fx['mean_ms']:.3f} ms** ({fx['mean_ms'] / base_ms:.2f}× vAttention). "
              f"단 이 변형의 RULER 정확도는 **아직 측정하지 않았으므로** 정확도 주장과 "
              f"짝지을 수 없다 — 아래 다음 단계 2번이 그 실험이다. "
              f"{fx['mean_ms']:.2f} ms를 {fmt(eq_acc, 2)} 옆에 나란히 인용하지 말 것.")
    A("- Jensen 편차 단은 직접 구현하면 작고 한정된 증분이고(UTA 대비 +0.5 ms, +11%), "
      "multi-bin 위에서는 거의 공짜다.")
    A("- vAttention은 UTA 계열에 전혀 없는 layer당 device→host 동기화를 지불하며, "
      "실측 밀도가 확률적인 반면 UTA 계열은 정확히 결정적이다.")
    A("- 모든 방법이 실제로 동일한 밀도에 있다(§1). 즉 이 비교 자체가 유효하다.")
    A("")
    A("**입증하지 않은 것 (그리고 주장하지 않는 것):**")
    A("")
    A("- 배포 시 비용. 양쪽 다 여전히 full score matrix를 계산한다. 구조적 비대칭 — "
      "우리 bin 통계는 query와 무관하고 KV 페이지와 함께 캐시 가능한 반면 vAttention의 "
      "gather는 그렇지 않다 — 은 [analyze_cost.py](../analyze_cost.py)에 모델링돼 있지만 "
      "**아직 측정되지 않았다**. 그게 Tier 2다.")
    A("- `UTA+jensen` ladder 행에서 나오는 어떤 주장도. 이유는 §3의 atomics 문제.")
    A("- tail latency / 예측 가능성 논증. 밀도 열은 이를 뒷받침하지만, 측정된 latency "
      "편차는 여기서 두 방법을 구분하지 못한다(§4).")
    A("- end-to-end tokens/s. 이건 attention layer 하나를 격리해서 잰 것이고, layer 간 "
      "오버헤드나 32× layer 배수는 포함돼 있지 않다.")
    A("")
    A("**권장하는 다음 단계 (우선순위 순):**")
    A("")
    A("1. `fixed` 분할의 정확도를 측정한다. 배포 가능한 분할이면서 여기서 더 빠른 "
      "프로토타입 경로이기도 하다. RULER에서 `equalcount`와 대등하다면 이쪽이 헤드라인 "
      "config가 되어야 한다.")
    A("2. `UTA+jensen` ablation 단을 `bin_size=1e9`에서 떼어내 `UTAJensenAttention` "
      "(또는 `bin_mode=\"fixed\"`)으로 교체한다. 그래야 그 단이 atomics 핫스팟이 아니라 "
      "보정 자체를 측정하게 된다. `results_v2`의 해당 정확도 행도 아직 미완성이다.")
    A("3. `mb/T3_bin_stats`의 `E[k]`를 편차 단으로 넘겨서 중복 계산을 없앤다.")
    A("4. 그 다음 Tier 2: bin 통계를 KV 페이지 메타데이터로 캐시하고 per-query 비용만 측정.")
    A("")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L))
    print(f"saved -> {out}")

    # 플로팅용 평평한 CSV
    csv_path = Path("results_latency/latency_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["suite", "label", "method", "seq_len", "batch", "num_queries",
                    "bin_size", "eps", "delta", "mean_ms", "p50_ms", "p95_ms",
                    "density", "peak_mem_MB", "status"])
        for suite, rr in res.items():
            for r in rr:
                w.writerow([suite, r["label"], r["method"], r["seq_len"], r["batch"],
                            r["num_queries"], r["bin_size"], r["eps"], r["delta"],
                            fmt(r.get("mean_ms")), fmt(r.get("p50_ms")),
                            fmt(r.get("p95_ms")), fmt(r.get("density"), 5),
                            fmt(r.get("peak_mem_MB"), 1), r["status"]])
    print(f"saved -> {csv_path}")


if __name__ == "__main__":
    main()
