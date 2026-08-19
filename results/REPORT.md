# Multi-bin UTA — 누적 실험 보고서

모델: `meta-llama/Llama-3.1-8B-Instruct`, bf16 · 벤치마크: RULER 32K, subset당 200샘플
subset: `qa_1, qa_2, vt, fwe, niah_multikey_2, niah_multikey_3, niah_multivalue`

> **이 파일의 구조.** `<!-- AUTO:... -->` 마커 사이 블록은 `python make_report.py`가
> 원자료에서 재생성하므로 **직접 수정하지 마세요.** 그 밖의 모든 서술(결론, 무효 처리
> 이력, 미확립 사항)은 수기이며 재생성해도 보존됩니다. 실험을 돌린 뒤에는
> `python make_report.py`를 실행하세요.

**통계적 해상도:** 200개 이진 샘플에서 subset 점수의 SE ≈ 3.5%p, 7-subset 평균의
SE ≈ 1.3%p입니다. 단일 비교에서 약 1.5%p 미만 차이는 분해되지 않습니다. 그 미만에
근거한 주장은 본문에 명시했습니다.

---

## 1. 결론

### 1.1 확립된 것

**tail 오차는 전부 분모에 있고 분자는 보정이 필요 없다.**
tail value proxy `mu_T`를 정확한 값으로 바꿔도 출력오차가 1/3/10% density에서 각각
−0.6% / +0.2% / +0.9%(즉 변화 없음)인데, tail 질량 `S_T`를 정확한 값으로 바꾸면
48% / 60% / 70%가 사라집니다. density당 78,848 row에서 측정.
→ key별 value 재가중(원래의 "분자 Jensen gap" 계획)은 **폐기**. 과거 CovUTA 실패도
같은 이유입니다 — 고장나지 않은 항을 고쳤습니다.

**tail 전체를 proxy 하나로 대표하는 것이 진짜 한계다 — proxy의 정확도가 아니라.**
multi-bin은 *완벽한* 단일 proxy 분모가 도달할 수 있는 한계를 넘습니다. bin이 질량과
value 구조를 동시에 쪼개기 때문입니다. 3% 기준: 정확한 `S_T` + 단일 `v̄`가 상대오차
0.0259인데, 32토큰 위치 bin은 0.0234입니다.

**multi-bin을 작동하게 만드는 것은 위치가 아니라 score 기준 binning이다.**
오차 동인은 bin 내부 로짓 분산입니다. 전역 tail 분산 대비 비율(5%, oracle 셀렉터):
위치 블록은 **0.49**에서 바닥을 치고 랜덤 분할은 **0.97**인데, 실제 로짓 등간격 bin은
**0.0120**(16개) / **0.0030**(32개)에 도달합니다. 출력오차도 따라옵니다:
UTA 0.0407 → 위치bin32 0.0112 → **score bin 16개 0.00057** → **score bin 32개 0.00025**.
**score bin 16개가 위치 bin 1000개보다 20배 정확합니다.** 사용자가 제공한 Wan/VideoSys
참조 구현에서 온 통찰로, 그 코드는 합성 구조가 우리와 동일하지만 bin 기준이 score입니다.

**multi-bin + score binning이 5%와 10%에서 vAttention을 이긴다.** 4/4 비교 모두 우리
쪽이 앞서며 일관되게 약 +0.3입니다(§2). 개별 격차는 1.3%p 노이즈 하한 아래이므로,
근거는 특정 수치가 아니라 **방향의 일관성**입니다.

**이 밀도에서 bin 개수는 포화됐다.** 벤치마크상 b16 ≈ b32(5%에서 87.00/87.05,
10%에서 87.03/87.01)입니다. 진단에서는 b32가 출력오차 기준 2.3배 정확한데도 그렇습니다.
tail 추정은 사실상 해결됐고 남은 여지는 다른 곳에 있습니다. → 적응형 bin 개수는 **후순위**.

**score binning은 약한 셀렉터를 oracle 수준으로 끌어올린다.** score bin 32개에서 HAT
셀렉터가 86.83(5%) / 86.95(10%)에 도달합니다. oracle 셀렉터의 87.05 / 87.01 대비
0.22 / 0.06%p 차이로 노이즈 안쪽입니다. `vAttention(HAT)` 대비로는 **5%에서 +17.96%p**,
**10%에서 +5.46%p**이며, `vAttention(oracle)`(86.72 / 86.70)마저 근소하게 앞섭니다.
기전은 niah_multikey_3에서 드러납니다: 5%에서 33.5 → 99.5. **HAT이 선택에 실패한 needle
토큰이 tail에 남아도 score bin에서는 최상위 bin에 들어가 제 가중치를 거의 회복합니다.**
단일 proxy나 위치 bin에서는 평균에 묻힙니다. 단, §1.3의 비용 정합성 단서를 참조.

**Jensen 보정에 쓰는 분산은 bin 내 실제 로짓 분산이며, fp64 정답과 일치한다.**
`raw = q·kᵀ·scale`에서 직접 계산합니다([multibin.py:115](../sparse_attention_hub/sparse_attention/uta_attention/multibin.py#L115),
[:181](../sparse_attention_hub/sparse_attention/uta_attention/multibin.py#L181)).
검증: score bin 16개에서 최대 편차 1.16e-06(해당 bin 실제 분산 0.0043), 위치 bin에서
1.02e-07. 자세한 유도와 과거 구현과의 차이는 §3.4.

### 1.2 이전 결론에 대한 정정

- **"저밀도 병목은 tail이 아니라 셀렉터" — 뒤집힘.** score binning을 쓰면 HAT 셀렉터가
  oracle 대비 0.22%p 이내로 붙습니다(§1.1). 셀렉터의 실수는 tail에서 회복 가능하며
  확정 손실이 아닙니다.
- **"σ²/2가 하드 바운드 τ를 넘어서 폭발한다" — 절반만 맞음.** 위반율은 0.5~2%에
  불과합니다. 과대추정 자체는 실재하지만(중앙값 1.5배, p90 4.6배), 해법은 truncation
  clamp가 아니라 **적률 형태** `log(1 + σ²/2)`였습니다. clamp 변형(j3)은 셋 중 가장
  약했습니다.
- **"정확한 S_T가 달성 가능 오차의 하한" — 잘못된 프레이밍.** 그것은 단일 value proxy를
  유지하는 방법에 대해서만 하한입니다. score bin은 약 23배 넘어섭니다.

### 1.3 아직 확립되지 않은 것

- **속도 측정이 전무하다.** 여기 모든 구현은 O(Q·K) 연구용 프로토타입이며 가속이 0입니다.
  어떤 wall-clock 주장도 근거가 없습니다.
- **`ours(HAT)` + score bin은 비용 정합이 아니다** — tail 토큰 전체의 실제 로짓이
  필요하고, 그건 HAT이 회피하려던 full QK입니다. 정확도 상한일 뿐입니다.
  비용 정합 기준점은 `ours-*-fixed(HAT)`(위치 bin).
- **σ²의 배포 가능한 계산법이 없다.** 현재는 로짓 행렬 전체가 있어 참값이 공짜지만,
  커널에서는 bin별 2차 모멘트(bin당 d×d) 캐싱이 필요합니다. 대각 근사로 낮추면
  §3.4의 0.84~1.47배 오차가 들어옵니다.
- vAttention 대비 `+0.3` 우위는 단일 비교의 노이즈 하한 아래입니다.

---

## 2. 결과

<!-- AUTO:results START -->
_2026-08-17 01:29 KST `make_report.py` 자동 생성. `_숫자_*` 표기는 7개 미만 subset의 부분 평균이며 7/7 평균과 비교 불가._

### `results_multibin_hat`

Multi-bin UTA vs vAttention (5%/10%, 두 셀렉터)  
_17/22개 완료(7/7 subset), 최종 기록 08-17 01:06_

| 방법 | 평균 | 완료 | qa_1 | qa_2 | vt | fwe | nk_2 | nk_3 | nmv |
|---|---|---|---|---|---|---|---|---|---|
| ours-b32(oracle)@5pct | **87.05** | 7/7 | 79.0 | 50.5 | 95.2 | 93.17 | 95.0 | 100.0 | 96.5 |
| ours-b16(oracle)@10pct | **87.03** | 7/7 | 78.5 | 50.5 | 95.3 | 93.17 | 95.0 | 100.0 | 96.75 |
| ours-b32(oracle)@10pct | **87.01** | 7/7 | 79.0 | 50.0 | 95.1 | 93.33 | 95.0 | 100.0 | 96.62 |
| ours-b16(oracle)@5pct | **87.0** | 7/7 | 79.0 | 50.0 | 95.1 | 93.17 | 95.0 | 100.0 | 96.75 |
| ours-b32-score(HAT)@10pct | **86.95** | 7/7 | 78.5 | 49.5 | 95.2 | 93.67 | 95.0 | 100.0 | 96.75 |
| ours-b32-score(HAT)@5pct | **86.83** | 7/7 | 78.0 | 50.0 | 95.4 | 93.17 | 95.0 | 99.5 | 96.75 |
| vAttention(HAT)@10pct | **81.49** | 7/7 | 71.0 | 47.5 | 90.7 | 92.83 | 85.0 | 87.5 | 95.88 |
| MultiBin16-HAT@3pct | **76.19** | 7/7 | 73.0 | 47.0 | 88.2 | 85.5 | 75.5 | 69.0 | 95.12 |
| MultiBin32-HAT@3pct | **75.65** | 7/7 | 72.5 | 47.0 | 88.5 | 83.17 | 74.5 | 68.0 | 95.88 |
| MultiBin64-HAT@3pct | **75.13** | 7/7 | 73.0 | 48.5 | 89.7 | 82.33 | 75.0 | 61.5 | 95.88 |
| UTA-HAT@3pct | **74.35** | 7/7 | 74.0 | 46.0 | 88.9 | 83.33 | 74.5 | 58.0 | 95.75 |
| vAttention(HAT)@5pct | **68.87** | 7/7 | 62.0 | 44.5 | 86.3 | 91.17 | 70.0 | 33.5 | 94.62 |
| HAT@3pct | **63.48** | 7/7 | 64.0 | 43.0 | 80.9 | 86.83 | 58.5 | 16.0 | 95.12 |
| MultiBin32-HAT@1pct | **56.97** | 7/7 | 59.5 | 46.0 | 80.9 | 81.17 | 40.0 | 1.5 | 89.75 |
| vAttention-HAT-s0.25@3pct | **55.49** | 7/7 | 55.0 | 41.5 | 79.5 | 84.17 | 31.0 | 5.0 | 92.25 |
| UTA-HAT@1pct | **55.27** | 7/7 | 60.0 | 44.0 | 82.1 | 81.17 | 27.5 | 1.0 | 91.12 |
| vAttention-HAT@3pct | **54.58** | 7/7 | 55.0 | 37.5 | 79.0 | 86.17 | 34.0 | 1.5 | 88.88 |
| HAT@1pct | _59.33_* | 4/7 | 44.0 | 41.5 | 73.3 | 78.5 | — | — | — |
| ours-b32-fixed(HAT)@10pct | _77.41_* | 5/7 | 74.0 | 44.5 | 91.9 | 84.17 | 92.5 | — | — |
| ours-b32-fixed(HAT)@5pct | _74.36_* | 5/7 | 72.5 | 45.0 | 89.3 | 81.5 | 83.5 | — | — |
| vAttention-HAT-e0.25d0.25@3pct | _60.33_* | 3/7 | 58.0 | 41.0 | 82.0 | — | — | — | — |
| vAttention-HAT@1pct | _36.21_* | 5/7 | 21.5 | 24.0 | 59.7 | 74.83 | 1.0 | — | — |

<details><summary>config.json에서 검증한 설정</summary>

| 방법 | selector | heavy | density | bin_mode | bin_size | kappa_mode | num_bins | sampling | eps/delta |
|---|---|---|---|---|---|---|---|---|---|
| HAT@1pct | HAT | 0.008 | 1% | — | — | — | — | — | — |
| HAT@3pct | HAT | 0.028 | 3% | — | — | — | — | — | — |
| MultiBin16-HAT@3pct | HAT | 0.028 | 3% | fixed | 16 | none | — | — | — |
| MultiBin32-HAT@1pct | HAT | 0.008 | 1% | fixed | 32 | none | — | — | — |
| MultiBin32-HAT@3pct | HAT | 0.028 | 3% | fixed | 32 | none | — | — | — |
| MultiBin64-HAT@3pct | HAT | 0.028 | 3% | fixed | 64 | none | — | — | — |
| UTA-HAT@1pct | HAT | 0.008 | 1% | — | — | — | — | — | — |
| UTA-HAT@3pct | HAT | 0.028 | 3% | — | — | — | — | — | — |
| ours-b16(oracle)@10pct | oracle-top-k | 0.098 | 10% | score | — | j2 | 16 | — | — |
| ours-b16(oracle)@5pct | oracle-top-k | 0.048 | 5% | score | — | j2 | 16 | — | — |
| ours-b32(oracle)@10pct | oracle-top-k | 0.098 | 10% | score | — | j2 | 32 | — | — |
| ours-b32(oracle)@5pct | oracle-top-k | 0.048 | 5% | score | — | j2 | 32 | — | — |
| ours-b32-fixed(HAT)@10pct | HAT | 0.098 | 10% | fixed | 32 | j2 | — | — | — |
| ours-b32-fixed(HAT)@5pct | HAT | 0.048 | 5% | fixed | 32 | j2 | — | — | — |
| ours-b32-score(HAT)@10pct | HAT | 0.098 | 10% | score | — | j2 | 32 | — | — |
| ours-b32-score(HAT)@5pct | HAT | 0.048 | 5% | score | — | j2 | 32 | — | — |
| vAttention(HAT)@10pct | HAT | 0.048 | 10% | — | — | — | — | 0.05 | 0.25/0.25 |
| vAttention(HAT)@5pct | HAT | 0.023 | 5% | — | — | — | — | 0.025 | 0.25/0.25 |
| vAttention-HAT-e0.25d0.25@3pct | HAT | 0.013 | 3% | — | — | — | — | 0.015 | 0.25/0.25 |
| vAttention-HAT-s0.25@3pct | HAT | 0.0205 | 3% | — | — | — | — | 0.0075 | 0.4/0.4 |
| vAttention-HAT@1pct | HAT | 0.003 | 1% | — | — | — | — | 0.005 | 0.4/0.4 |
| vAttention-HAT@3pct | HAT | 0.013 | 3% | — | — | — | — | 0.015 | 0.4/0.4 |

</details>

### `results_density_sweep`

Density sweep 1/2/3% (oracle-top-k 셀렉터)  
_9/9개 완료(7/7 subset), 최종 기록 08-14 10:45_

| 방법 | 평균 | 완료 | qa_1 | qa_2 | vt | fwe | nk_2 | nk_3 | nmv |
|---|---|---|---|---|---|---|---|---|---|
| 3pct/vAttention | **86.69** | 7/7 | 78.0 | 50.0 | 95.0 | 92.33 | 95.5 | 100.0 | 96.0 |
| 2pct/vAttention | **86.35** | 7/7 | 75.5 | 50.0 | 95.4 | 91.67 | 95.5 | 100.0 | 96.38 |
| 2pct/UTA | **85.9** | 7/7 | 77.5 | 52.5 | 94.6 | 85.67 | 95.0 | 99.5 | 96.5 |
| 3pct/UTA | **85.81** | 7/7 | 76.5 | 51.0 | 94.5 | 86.67 | 95.5 | 100.0 | 96.5 |
| 1pct/UTA | **84.92** | 7/7 | 76.0 | 51.0 | 94.6 | 81.83 | 94.5 | 100.0 | 96.5 |
| 1pct/vAttention | **82.93** | 7/7 | 73.5 | 47.5 | 91.2 | 84.67 | 91.5 | 97.0 | 95.12 |
| 3pct/oracle_top_k | **80.85** | 7/7 | 70.0 | 47.0 | 92.0 | 87.67 | 83.5 | 89.5 | 96.25 |
| 2pct/oracle_top_k | **76.43** | 7/7 | 66.5 | 45.5 | 90.6 | 84.5 | 76.0 | 76.0 | 95.88 |
| 1pct/oracle_top_k | **66.52** | 7/7 | 62.0 | 41.5 | 87.4 | 77.83 | 60.0 | 41.5 | 95.38 |

<details><summary>config.json에서 검증한 설정</summary>

| 방법 | selector | heavy | density | sampling | eps/delta |
|---|---|---|---|---|---|
| 1pct/UTA | oracle-top-k | 0.008 | 1% | — | — |
| 1pct/oracle_top_k | oracle-top-k | 0.008 | 1% | — | — |
| 1pct/vAttention | oracle-top-k | 0.003 | 1% | 0.005 | 0.25/0.25 |
| 2pct/UTA | oracle-top-k | 0.018 | 2% | — | — |
| 2pct/oracle_top_k | oracle-top-k | 0.018 | 2% | — | — |
| 2pct/vAttention | oracle-top-k | 0.008 | 2% | 0.01 | 0.25/0.25 |
| 3pct/UTA | oracle-top-k | 0.028 | 3% | — | — |
| 3pct/oracle_top_k | oracle-top-k | 0.028 | 3% | — | — |
| 3pct/vAttention | oracle-top-k | 0.013 | 3% | 0.015 | 0.25/0.25 |

</details>

### `results_covuta`

이전 3/5/10% 실행 — 프로토콜 일치, baseline으로 재사용  
_10/12개 완료(7/7 subset), 최종 기록 08-13 12:32_

| 방법 | 평균 | 완료 | qa_1 | qa_2 | vt | fwe | nk_2 | nk_3 | nmv |
|---|---|---|---|---|---|---|---|---|---|
| vAttention@5pct | **86.72** | 7/7 | 77.5 | 49.5 | 95.3 | 93.5 | 95.0 | 99.5 | 96.75 |
| vAttention@10pct | **86.7** | 7/7 | 78.0 | 50.0 | 95.2 | 93.17 | 94.5 | 99.5 | 96.5 |
| UTA@10pct | **86.43** | 7/7 | 77.0 | 49.0 | 95.2 | 92.33 | 95.0 | 100.0 | 96.5 |
| UTA@5pct | **86.04** | 7/7 | 76.5 | 50.5 | 94.9 | 89.5 | 94.5 | 100.0 | 96.38 |
| oracle_top_k@10pct | **85.28** | 7/7 | 71.5 | 47.5 | 94.9 | 93.33 | 94.0 | 99.5 | 96.25 |
| oracle_top_k@5pct | **83.5** | 7/7 | 71.0 | 47.5 | 92.4 | 90.0 | 90.0 | 97.5 | 96.12 |
| CovUTA_Indep@10pct | **77.34** | 7/7 | 73.5 | 47.5 | 95.1 | 91.5 | 87.0 | 50.5 | 96.25 |
| CovUTA@10pct | **77.1** | 7/7 | 73.0 | 47.5 | 95.1 | 91.5 | 87.0 | 49.0 | 96.62 |
| CovUTA_Indep@5pct | **70.51** | 7/7 | 69.0 | 45.0 | 95.1 | 90.33 | 76.5 | 21.5 | 96.12 |
| CovUTA_Indep@3pct | **67.77** | 7/7 | 65.5 | 45.0 | 95.0 | 89.0 | 71.0 | 12.5 | 96.38 |
| CovUTA@5pct | _56.5_* | 2/7 | 68.0 | 45.0 | — | — | — | — | — |
| vAttention@3pct | _62.5_* | 2/7 | 75.5 | 49.5 | — | — | — | — | — |

<details><summary>config.json에서 검증한 설정</summary>

| 방법 | selector | heavy | density | sampling | eps/delta |
|---|---|---|---|---|---|
| CovUTA@10pct | oracle-top-k | 0.098 | 10% | — | — |
| CovUTA@5pct | oracle-top-k | 0.048 | 5% | — | — |
| CovUTA_Indep@10pct | oracle-top-k | 0.098 | 10% | — | — |
| CovUTA_Indep@3pct | oracle-top-k | 0.028 | 3% | — | — |
| CovUTA_Indep@5pct | oracle-top-k | 0.048 | 5% | — | — |
| UTA@10pct | oracle-top-k | 0.098 | 10% | — | — |
| UTA@5pct | oracle-top-k | 0.048 | 5% | — | — |
| oracle_top_k@10pct | oracle-top-k | 0.098 | 10% | — | — |
| oracle_top_k@5pct | oracle-top-k | 0.048 | 5% | — | — |
| vAttention@10pct | oracle-top-k | 0.048 | 10% | 0.05 | 0.25/0.25 |
| vAttention@3pct | oracle-top-k | 0.013 | 3% | 0.015 | 0.25/0.25 |
| vAttention@5pct | oracle-top-k | 0.023 | 5% | 0.025 | 0.25/0.25 |

</details>

### `results_full_comparison`

이전 5/10% 비교 (CV_UTA 포함)  
_7/9개 완료(7/7 subset), 최종 기록 08-13 00:54_

| 방법 | 평균 | 완료 | qa_1 | qa_2 | vt | fwe | nk_2 | nk_3 | nmv |
|---|---|---|---|---|---|---|---|---|---|
| vAttention@10pct | **87.07** | 7/7 | 78.0 | 50.5 | 95.3 | 93.67 | 95.5 | 100.0 | 96.5 |
| CV_UTA@10pct | **86.59** | 7/7 | 77.5 | 50.0 | 95.2 | 91.33 | 95.5 | 100.0 | 96.62 |
| vAttention@5pct | **86.55** | 7/7 | 77.5 | 48.5 | 95.4 | 93.33 | 94.5 | 100.0 | 96.62 |
| UTA@10pct | **86.43** | 7/7 | 77.0 | 49.0 | 95.2 | 92.33 | 95.0 | 100.0 | 96.5 |
| oracle_top_k@10pct | **85.28** | 7/7 | 71.5 | 47.5 | 94.9 | 93.33 | 94.0 | 99.5 | 96.25 |
| oracle_top_k@5pct | **83.5** | 7/7 | 71.0 | 47.5 | 92.4 | 90.0 | 90.0 | 97.5 | 96.12 |
| UTA_LN@10pct | **81.85** | 7/7 | 65.5 | 42.5 | 94.5 | 91.33 | 92.5 | 90.5 | 96.12 |
| CV_UTA@5pct | _84.17_* | 6/7 | 75.0 | 48.0 | 96.0 | 91.0 | 95.5 | 99.5 | — |
| UTA@5pct | _63.5_* | 2/7 | 76.5 | 50.5 | — | — | — | — | — |

<details><summary>config.json에서 검증한 설정</summary>

| 방법 | selector | heavy | density | sampling | eps/delta |
|---|---|---|---|---|---|
| CV_UTA@10pct | oracle-top-k | 0.048 | 5% | — | — |
| CV_UTA@5pct | oracle-top-k | 0.023 | 2.5% | — | — |
| UTA@10pct | oracle-top-k | 0.098 | 10% | — | — |
| UTA@5pct | oracle-top-k | 0.048 | 5% | — | — |
| UTA_LN@10pct | oracle-top-k | 0.098 | 10% | — | — |
| oracle_top_k@10pct | oracle-top-k | 0.098 | 10% | — | — |
| oracle_top_k@5pct | oracle-top-k | 0.048 | 5% | — | — |
| vAttention@10pct | oracle-top-k | 0.048 | 10% | 0.05 | 0.25/0.25 |
| vAttention@5pct | oracle-top-k | 0.023 | 5% | 0.025 | 0.25/0.25 |

</details>

### `results_uta_low_density`

초기 3/5% 탐색 (50샘플 — 검정력 부족)  
_6/6개 완료(7/7 subset), 최종 기록 08-11 10:33_

| 방법 | 평균 | 완료 | qa_1 | qa_2 | vt | fwe | nk_2 | nk_3 | nmv |
|---|---|---|---|---|---|---|---|---|---|
| 5pct/UTA | **86.48** | 7/7 | 80.0 | 52.0 | 94.0 | 89.33 | 94.0 | 100.0 | 96.0 |
| 5pct/vAttention | **86.37** | 7/7 | 76.0 | 52.0 | 94.4 | 92.67 | 94.0 | 100.0 | 95.5 |
| 3pct/UTA | **85.9** | 7/7 | 80.0 | 52.0 | 94.0 | 85.33 | 94.0 | 100.0 | 96.0 |
| 3pct/vAttention | **85.8** | 7/7 | 76.0 | 50.0 | 94.4 | 90.67 | 94.0 | 100.0 | 95.5 |
| 5pct/oracle-top-k | **84.29** | 7/7 | 76.0 | 52.0 | 92.0 | 90.0 | 86.0 | 98.0 | 96.0 |
| 3pct/oracle-top-k | **78.81** | 7/7 | 70.0 | 50.0 | 91.2 | 86.0 | 80.0 | 78.0 | 96.5 |

<details><summary>config.json에서 검증한 설정</summary>

| 방법 | selector | heavy | density | sampling | eps/delta |
|---|---|---|---|---|---|
| 3pct/UTA | oracle-top-k | 0.028 | 3% | — | — |
| 3pct/oracle-top-k | oracle-top-k | 0.028 | 3% | — | — |
| 3pct/vAttention | oracle-top-k | 0.013 | 3% | 0.015 | 0.25/0.25 |
| 5pct/UTA | oracle-top-k | 0.048 | 5% | — | — |
| 5pct/oracle-top-k | oracle-top-k | 0.048 | 5% | — | — |
| 5pct/vAttention | oracle-top-k | 0.023 | 5% | 0.025 | 0.25/0.25 |

</details>

### `results_damped_jensen_full`

Damped Jensen (alpha=0.25) vs vAttention @10%  
_2/2개 완료(7/7 subset), 최종 기록 08-13 21:48_

| 방법 | 평균 | 완료 | qa_1 | qa_2 | vt | fwe | nk_2 | nk_3 | nmv |
|---|---|---|---|---|---|---|---|---|---|
| vAttention_10pct | **86.6** | 7/7 | 78.0 | 49.0 | 95.1 | 92.83 | 95.0 | 99.5 | 96.75 |
| UTAJensen_Damped_10pct | **86.41** | 7/7 | 78.0 | 49.5 | 94.9 | 91.5 | 95.0 | 100.0 | 96.0 |

<details><summary>config.json에서 검증한 설정</summary>

| 방법 | selector | heavy | density | sampling | eps/delta |
|---|---|---|---|---|---|
| UTAJensen_Damped_10pct | oracle-top-k | 0.098 | 10% | — | — |
| vAttention_10pct | oracle-top-k | 0.048 | 10% | 0.05 | 0.25/0.25 |

</details>

### `results_jensen_diagnostic`

Jensen alpha 스윕 진단  
_0/3개 완료(7/7 subset), 최종 기록 08-13 17:49_

| 방법 | 평균 | 완료 | qa_1 | qa_2 | vt | fwe | nk_2 | nk_3 | nmv |
|---|---|---|---|---|---|---|---|---|---|
| UTAJensen_Damped_0.25 | _78.67_* | 4/7 | 80.0 | 50.0 | — | 90.67 | 94.0 | — | — |
| UTAJensen_Full_1.0 | _74.0_* | 4/7 | 66.0 | 46.0 | — | 92.0 | 92.0 | — | — |
| UTA_Base | _79.0_* | 4/7 | 80.0 | 50.0 | — | 92.0 | 94.0 | — | — |

<details><summary>config.json에서 검증한 설정</summary>

| 방법 | selector | heavy | density |
|---|---|---|---|
| UTAJensen_Damped_0.25 | oracle-top-k | 0.098 | 10% |
| UTAJensen_Full_1.0 | oracle-top-k | 0.098 | 10% |
| UTA_Base | oracle-top-k | 0.098 | 10% |

</details>

### `results_covuta_v2`

CovUTA 분자 보정 (폐기됨)  
_0/2개 완료(7/7 subset), 최종 기록 08-13 14:48_

| 방법 | 평균 | 완료 | qa_1 | qa_2 | vt | fwe | nk_2 | nk_3 | nmv |
|---|---|---|---|---|---|---|---|---|---|
| CovUTA_Indep_v2@10pct | _77.02_* | 5/7 | 64.5 | 42.0 | 95.1 | 92.0 | 91.5 | — | — |
| CovUTA_v2@10pct | _77.19_* | 5/7 | 64.5 | 42.5 | 95.1 | 92.33 | 91.5 | — | — |

<details><summary>config.json에서 검증한 설정</summary>

| 방법 | selector | heavy | density |
|---|---|---|---|
| CovUTA_Indep_v2@10pct | oracle-top-k | 0.098 | 10% |
| CovUTA_v2@10pct | oracle-top-k | 0.098 | 10% |

</details>
<!-- AUTO:results END -->

---

## 3. 진단

`run_block_tail_diag.py`로 측정합니다. 이 스크립트는 *dense* attention 출력을 반환하므로
생성 궤적을 전혀 교란하지 않으면서 (layer, head, query-row)별 tail 성질을 기록합니다.
원자료는 `results_tail_diag*/**.parquet`.

### 3.1 분모 vs 분자 (oracle-top-k 셀렉터, density당 78,848 row)

| 출력 상대오차 (중앙값) | 1% | 3% | 10% |
|---|---|---|---|
| UTA (단일 proxy) | 0.1120 | 0.0646 | 0.0251 |
| + 정확한 `S_T`만 | 0.0576 (−48.5%) | 0.0259 (−59.9%) | 0.0076 (−69.8%) |
| + 정확한 `mu_T`만 | 0.1113 (−0.6%) | 0.0647 (+0.2%) | 0.0253 (+0.9%) |
| tail 질량 비중 (중앙값) | 0.048 | 0.029 | 0.012 |

### 3.2 Jensen gap

| | 1% | 3% | 10% |
|---|---|---|---|
| P(σ²/2 > z_max,tail − z̄) | 0.005 | 0.011 | 0.020 |
| 실제 gap K₁ (중앙값) | 2.165 | 1.808 | 1.351 |
| log-normal 예측 σ²/2 | 2.491 | 2.280 | 1.860 |
| 과대추정 배율, 중앙값 / p90 | 1.5× / 4.6× | 1.6× / 5.2× | 1.7× / 4.8× |

보정 순위는 density 전반에서 일관됩니다: `log(1+σ²/2)`(j2) ≫ clamp(j3) > 무보정.

### 3.3 bin 기준 게이트 (5% density) — `results_tail_diag_gate/`

| oracle-top-k 셀렉터 | var_ratio | 출력오차 (j2) | vs UTA |
|---|---|---|---|
| UTA 단일 proxy | 1.0 | 0.04069 | — |
| 정확한 `S_T` + 단일 `v̄` | — | 0.01317 | −67.6% |
| 위치 bin, 32토큰 (~1000개) | 0.4931 | 0.01122 | −72.4% |
| 랜덤 분할 (대조군) | 0.9672 | 0.02192 | −46.1% |
| **score bin, 16개** | **0.0120** | **0.00057** | **−98.6%** |
| **score bin, 32개** | **0.0030** | **0.00025** | **−99.4%** |

| HAT 셀렉터 | var_ratio | 출력오차 (j2) |
|---|---|---|
| UTA 단일 proxy | 1.0 | 0.21318 |
| 정확한 `S_T` + 단일 `v̄` | — | 0.32595 *(UTA보다 나쁨)* |
| 위치 bin, 32토큰 | 0.4697 | 0.17098 |
| **hat-score bin, 16 / 32개** | **0.7200 / 0.7196** | 0.17414 / 0.17412 |
| 실제 로짓 score bin, 16 / 32개 | 0.0293 / 0.0075 | 0.00479 / 0.00195 |

→ hat-score binning은 **게이트 탈락**. 16개와 32개가 동일하다는 것은 bin을 더 쪼개도
아무것도 분리되지 않는다는 뜻입니다. HAT signature는 32비트 부호 벡터라 정수 65단계뿐이고
*상위* 토큰을 골라내도록 학습되어 하위 95%의 순서 정보를 거의 갖고 있지 않습니다.
같은 tail을 실제 로짓으로 나누면 0.0293이 나옵니다. HAT arm은 이에 따라 재설계했습니다.

### 3.4 Jensen 보정식과 분산의 정의

tail 질량은 정확히 다음과 같이 분해됩니다:

```
log S_b = z̄_b + log n_b + K_b(1),     K_b(1) = log E[exp(z − z̄_b)]
```

`K_b(1)`을 무엇으로 근사하느냐가 변형을 가릅니다. `j2`는 그중 채택된 것의 이름표입니다.

| 이름 | K_b(1) 근사 | 유래 | 결과 |
|---|---|---|---|
| plain | `0` | 보정 없음 (기본 UTA) | 항상 과소추정 |
| j1 | `σ_b²/2` | z가 정규분포라 가정, `E[e^X] = e^{σ²/2}` | **폭발** (α=1에서 qa_1 77→65.5, nk_3 100→49) |
| **j2** | `log(1 + σ_b²/2)` | 적률 전개 `E[e^X] ≈ 1 + E[X] + E[X²]/2` | **채택** |
| j3 | `min(σ_b²/2, z_max,b − z̄_b)` | 하드 클램프 | 셋 중 최약 |

j1과 j2는 σ가 작을 때 2차까지 일치하지만, 커지면 j1은 지수적으로 발산하고 j2는
로그로만 자랍니다. 이것이 과거 붕괴의 직접 원인입니다.

**σ_b²의 정의 — bin 안 tail 토큰들의 `q·k·scale` 로짓 분산입니다.** 로짓 행렬 `raw`에서
직접 계산하며, fp64 독립 계산과 대조 검증했습니다:

| bin 모드 | \|코드값 − fp64 참값\| 최대 | 해당 bin 실제 분산 |
|---|---|---|
| score, 16개 | 1.16e-06 | 0.0043 |
| fixed, 64토큰 | 1.02e-07 | 1.0654 |

**과거 실패한 구현은 이 값을 다르게 계산했습니다.**
[advanced.py:70-77](../sparse_attention_hub/sparse_attention/uta_attention/advanced.py#L70-L77)의
`UTAJensenAttention`은 `σ² = s²·Σ_d q_d²·Var(k_d)` — key 차원별 분산만 쓰고 **교차
공분산을 버린 대각 근사**입니다. 참값은 `s²·qᵀ·Cov(K)·q`이므로 교차항이 빠지면 안 됩니다.
실측하면 중앙값 비는 1.016으로 나쁘지 않으나 **행별로 0.84~1.47까지 흔들립니다.**
σ²≈5에서 47% 과대추정이면 j1 형태에서 `e^{1.2}≈3.3`배 질량이 더 붙으므로, 대각 근사가
j1의 폭발을 악화시켰습니다. 현재 구현은 이 근사를 쓰지 않습니다.

**보정은 분모와 분자 양쪽에 들어가며, 그래야 맞습니다.** 보정이 추정하는 대상은 bin의
tail 질량 `S_b` 스칼라 하나인데, 정확식에서 이미 양쪽에 나타납니다:

```
out = (Σ_H e^{z_i} v_i + Σ_b S_b·μ_b) / (Σ_H e^{z_i} + Σ_b S_b)
```

분모에만 넣으면 가중치 합이 1이 되지 않아 볼록결합이 깨지고, κ→∞에서 출력이 0으로
수축합니다. 우리가 하지 *않는* 것은 `μ_b`를 보정하는 것입니다 — `v̄_b`는 단순 평균입니다.
즉 "분모 term 보정"과 "분자에 안 들어감"은 서로 다른 얘기입니다.

**수치 정밀도 주의.** `E[X²] − E[X]²` 방식은 상쇄오차에 취약한데, score bin에서
σ_b²가 0.004 수준까지 작아집니다. 현재 상대오차는 0.03%로 문제없으나 bin을 더 쪼개면
감시가 필요합니다.

---

## 4. 무효 / 보류 결과 — 인용 금지

| 결과 | 위치 | 사유 |
|---|---|---|
| `vAttention-HAT@3pct` = 54.58 | `results_multibin_hat/` | `eps=delta=0.4`로 실행(`reproduce_table1_ruler32k.py`에서 복사). 다른 모든 실행은 `0.25`. adaptive budget이 ~(ppf(1−δ)/ε)²로 스케일하므로 샘플이 약 18배 적음. 어떤 것과도 비교 불가 |
| `vAttention-HAT-s0.25@3pct` = 55.49 | 동일 | 같은 eps/delta 문제. split 대조군이 엉뚱한 질문에 답하고 있었음 |
| `vAttention-HAT-e0.25d0.25@3pct` | 동일 | 조건 정정 재실행이나 3/7 subset에서 중단. 미완 |
| 모든 `@1pct` HAT 셀렉터 실행 | 동일 | HAT은 1%에서 retrieval 불가(전 method nk_3 ≈ 1). tail 품질이 아니라 셀렉터 붕괴를 측정한 것 |
| 1/2/3% density sweep 결론 | `results_density_sweep/` | 데이터는 유효하나 결정에 의해 보류 — UTA가 vAttention을 이긴 density(1%)가 배포 가능한 셀렉터가 동작하지 않는 density임 |
| `CovUTA*` | `results_covuta*/` | 분자 보정. §1.1로 대체됨(분자 기여 ≈ 0) |
| `UTA_LN`, `UTA_Jensen@alpha=1` | `results_full_comparison/`, `results_jensen/` | 감쇠 없는 log-normal 보정(j1). 붕괴(qa_1 77→65.5, nk_3 100→49) |
| 50샘플 실행 | `results_uta_low_density/` | subset당 SE ≈ 7%p — 검정력 부족 |

---

## 5. 재현

```bash
ENV=~/miniconda3/envs/sah/bin/python

# 주 비교, 5%/10%, 두 셀렉터
$ENV run_multibin_hat_bench.py --device cuda:2 --densities 0.05 \
    --selector oracle --methods ours --bin-sizes 16,32
$ENV run_multibin_hat_bench.py --device cuda:2 --densities 0.05 \
    --selector hat --methods ours --bin-mode score --bin-sizes 32
$ENV run_multibin_hat_bench.py --device cuda:2 --densities 0.05 \
    --selector hat --methods vAttention --eps 0.25 --delta 0.25

# tail 진단 (생성 없음, dense 출력 반환)
$ENV run_block_tail_diag.py --device cuda:2 --heavy-size 0.048 \
    --selector oracle --num-bins 16,32 --output-dir ./results_tail_diag_gate

# 표 생성
$ENV run_multibin_hat_bench.py --summarize-only --output-dir ./results_multibin_hat
$ENV make_report.py            # 이 파일 갱신
```

`vAttention(oracle)@5pct/@10pct`, `UTA(oracle)`, `oracle-top-k` 5/10%는 **재실행하지
않습니다** — `results_covuta/`에 동일 프로토콜(sink/local 0.001, eps=delta=0.25,
200샘플)로 이미 있습니다. `make_report.py`는 디렉터리 이름이 아니라 `config.json`에서
이를 검증합니다.

단위 테스트: `pytest tests/test_uta_multibin.py -q` (16개). 고정하는 불변식 —
`Σ n_b == N_tail`, `bin_size=1` ⇒ exact attention, bin ≥ K ⇒ 기본 UTA,
그리고 **bin을 나누는 데 쓴 값이 bin logit에 절대 새어 들어가지 않을 것**.

---

## 6. 실행 인벤토리

<!-- AUTO:inventory START -->
| 결과 디렉터리 | 실행 수 | 완료 | 완료 subset 합 | 최종 기록 |
|---|---|---|---|---|
| `results_multibin_hat` | 22 | 17 | 141 | 08-17 01:06 |
| `results_density_sweep` | 9 | 9 | 63 | 08-14 10:45 |
| `results_covuta` | 12 | 10 | 74 | 08-13 12:32 |
| `results_full_comparison` | 9 | 7 | 57 | 08-13 00:54 |
| `results_uta_low_density` | 6 | 6 | 42 | 08-11 10:33 |
| `results_damped_jensen_full` | 2 | 2 | 14 | 08-13 21:48 |
| `results_jensen_diagnostic` | 3 | 0 | 12 | 08-13 17:49 |
| `results_covuta_v2` | 2 | 0 | 10 | 08-13 14:48 |
<!-- AUTO:inventory END -->

---

## 7. 다음

1. **비용 정합 score binning** — 핵심 공학 과제. score bin은 토큰별 tail 로짓이 필요하고
   HAT은 그 순서 정보를 제공하지 못합니다(§3.3). 후보: key 공간 클러스터링(query 무관,
   캐싱 가능), 또는 위치 블록 위의 저비용 2단계 score. σ_b²의 배포 가능한 계산법(§1.3)도
   같은 지점에서 만납니다.
2. **wall-clock 측정** — 아직 전무하며 모든 효율 주장을 막고 있습니다.
3. **kappa=none arm** — 현재 모든 실행이 `j2`입니다. score binning이 이미 bin 내 분산을
   0.003~0.012로 눌러놓아 j2의 한계효용이 작을 수 있으므로, 실제 기여도 확인 필요
   (1 job, 약 1.7h).
4. **셀렉터 연구** — §1.2의 정정에 따라 후순위로 내림.
