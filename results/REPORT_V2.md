# Multi-bin UTA V2 — Jensen 보정 확정 + Ablation

모델 `meta-llama/Llama-3.1-8B-Instruct` (bf16) · RULER 32K, subset당 200샘플
**density 5% 고정 · 셀렉터는 HashAttention(HAT) 전용** — vAttention도 우리 방법도 모두 HAT 기반

> `<!-- AUTO:... -->` 블록은 `python make_report.py --report results/REPORT_V2.md`가
> 원자료에서 재생성합니다. 나머지 서술은 수기이며 재생성해도 보존됩니다.

**통계적 해상도:** subset 점수 SE ≈ 3.5%p, 7-subset 평균 SE ≈ 1.3%p.
단일 비교에서 1.5%p 미만 차이는 분해되지 않습니다.

---

## 1. V2에서 확정한 것

### 1.1 순환 문제와 그 범위

로짓으로 bin을 나누려면 로짓이 필요합니다 — 회피하려던 full QK입니다. 무엇이 순환이고
무엇이 아닌지 분리하면:

| 필요한 것 | 순환? | 근거 |
|---|---|---|
| 토큰별 로짓으로 **bin 나누기** | **걸림** | full QK 필요 |
| bin 내 **참 로짓 분산** | **걸림** | 동일 |
| bin 평균 로짓 $\bar z_b = s\,q\cdot\bar k_b$ | 안 걸림 | 캐시된 $\bar k_b$와 $O(d)$ 내적 |
| bin 로짓 분산 $s^2 q^\top\!\mathrm{Cov}_b(K)\,q$ | 안 걸림 | $\mathrm{Cov}_b(K)$는 **query 무관**, 캐싱 가능 |

따라서 **분산은 순환이 아니고, bin 멤버십만 순환입니다.** 그래서 V2의 bin은
**HAT top-k의 여집합(tail)에서 인접한 인덱스끼리 묶은 것**입니다. 위치 인접성은 로짓을
전혀 요구하지 않고, 블록 누적합을 캐싱하면 구간합이 $O(d)$이며 선택 토큰 차감은 어차피
읽는 토큰이라 새 트래픽이 없습니다.

이에 따라 V1의 score-bin 결과(86.83 / 86.95)는 **도달 불가능한 상한**으로 재분류합니다.

### 1.2 (1) Jensen 보정 형태 — **j2 확정**

tail 질량은 정확히 $\log S_b = \bar z_b + \log n_b + K_b(1)$이고, $K_b(1)=\log\mathbb E[e^{z-\bar z_b}]$의
근사가 변형을 가릅니다.

| 이름 | $K_b(1)$ 근사 | 유래 |
|---|---|---|
| none | $0$ | 보정 없음 — 항상 과소추정 |
| j1 | $\sigma_b^2/2$ | $z$ 정규분포 가정, $\mathbb E[e^X]=e^{\sigma^2/2}$ |
| **j2** | $\log(1+\sigma_b^2/2)$ | 적률 전개 $\mathbb E[e^X]\approx1+\mathbb E[X]+\mathbb E[X^2]/2$ |

**측정 (5%, HAT, 인접 인덱스 bin 32, 출력 상대오차 중앙값):**

| kappa | exact var | diag var |
|---|---|---|
| none | 0.18771 | 0.18771 |
| **j1** | **0.28862** | 0.19158 |
| **j2** | **0.17710** | **0.17730** |

**j1은 참 분산을 줘도 무보정보다 나쁩니다**(0.289 vs 0.188, +54%). 이로써 과거 붕괴
(α=1에서 qa_1 77→65.5, nk_3 100→49)가 단일 proxy나 대각 근사 탓이 아니라 **j1 형태
자체**의 문제였음이 분리 확인됐습니다. j2는 무보정 대비 −5.7%.

### 1.3 (2) 분산 계산 방식 — **대각 근사 확정**

$\sigma_b^2 \approx s^2\sum_d q_d^2\,\mathrm{Var}_b(k_d)$ — 교차 공분산을 버리는 대신
bin당 $O(d)$, 저장은 bin당 $d$ (bin 평균과 같은 발자국).

**j2에서 대각과 참 분산이 구별되지 않습니다 (0.17730 vs 0.17710, 비 1.001).**
전체 tail에서 쟀을 때 나왔던 0.84~1.47배 오차는 bin 내부에서 사라집니다 — bin이
조밀할수록 공분산이 등방에 가까워지기 때문입니다.

full covariance는 **비용으로 이미 탈락**입니다(§4: bin당 $O(d^2)$, vAttention의 76배).
j1에서 대각이 오히려 나은 것(0.192 vs 0.289)은 대각이 분산을 과소추정해 j1의 과대보정을
우연히 상쇄한 것으로, 채택 근거가 아닙니다.

### 1.4 (3) 토큰별 vs 집계 보정 — V1에서 이미 확정

3개 density × 78,848 row: 정확한 `mu_T`를 줘도 −0.6%/+0.2%/+0.9%(변화 없음),
정확한 `S_T`는 48~70% 제거. **토큰별 value 재가중은 폐기.**

### 1.5 최종 방법 (현 단계 확정본)

```
ours = HAT top-k 선택
     + tail을 인접 인덱스 runs(32 토큰)로 분할
     + bin별 logit  ell_b = s·q·k̄_b + log n_b + log(1 + σ_b²/2)
     + σ_b² = s²·Σ_d q_d²·Var_b(k_d)            (대각, query 무관, 캐싱 가능)
     + heavy 토큰과 bin을 하나의 softmax로 합성
```

보정은 **분모와 분자 양쪽에** 들어갑니다. 보정 대상이 bin 질량 $S_b$ 스칼라 하나이고
그것이 정확식에서 이미 양쪽에 나타나기 때문입니다. 분모에만 넣으면 볼록결합이 깨져
$\kappa\to\infty$에서 출력이 0으로 수축합니다. 하지 *않는* 것은 $\mu_b$ 보정입니다
($\bar v_b$는 단순 평균).

---

## 2. Ablation 결과 (5%, HAT)

<!-- AUTO:results START -->
_2026-08-17 15:28 UTC `make_report.py` 자동 생성. `_숫자_*` 표기는 7개 미만 subset의 부분 평균이며 7/7 평균과 비교 불가._

### `results_v2`

V2 ablation ladder @5%, HAT selector (dense → ours)  
_7/7개 완료(7/7 subset), 최종 기록 08-17 15:24_

| 방법 | 평균 | 완료 | qa_1 | qa_2 | vt | fwe | nk_2 | nk_3 | nmv |
|---|---|---|---|---|---|---|---|---|---|
| dense@5pct | **87.04** | 7/7 | 79.0 | 50.0 | 95.2 | 93.33 | 95.0 | 100.0 | 96.75 |
| UTA+multibin(b32)@5pct | **81.44** | 7/7 | 78.0 | 48.0 | 90.5 | 85.83 | 86.5 | 85.5 | 95.75 |
| UTA+multibin+jensen(b32)@5pct | **81.43** | 7/7 | 76.0 | 51.0 | 90.2 | 86.17 | 85.0 | 86.0 | 95.62 |
| UTA(HAT)@5pct | **81.25** | 7/7 | 74.5 | 48.0 | 90.7 | 85.67 | 85.5 | 88.5 | 95.88 |
| UTA+jensen@5pct | **78.33** | 7/7 | 73.5 | 48.0 | 90.7 | 84.5 | 78.0 | 78.5 | 95.12 |
| HAT@5pct | **71.55** | 7/7 | 68.0 | 44.5 | 81.9 | 87.33 | 72.5 | 50.5 | 96.12 |
| vAttention(HAT)@5pct | **68.7** | 7/7 | 66.0 | 46.5 | 85.1 | 91.33 | 64.5 | 34.5 | 93.0 |

<details><summary>config.json에서 검증한 설정</summary>

| 방법 | selector | heavy | density | bin_mode | bin_size | kappa_mode | sampling | eps/delta |
|---|---|---|---|---|---|---|---|---|
| HAT@5pct | HAT | 0.048 | 5% | — | — | — | — | — |
| UTA(HAT)@5pct | HAT | 0.048 | 5% | — | — | — | — | — |
| UTA+jensen@5pct | HAT | 0.048 | 5% | equalcount | 1000000000 | j2 | — | — |
| UTA+multibin(b32)@5pct | HAT | 0.048 | 5% | equalcount | 32 | none | — | — |
| UTA+multibin+jensen(b32)@5pct | HAT | 0.048 | 5% | equalcount | 32 | j2 | — | — |
| dense@5pct | — | — | 100% | — | — | — | — | — |
| vAttention(HAT)@5pct | HAT | 0.023 | 5% | — | — | — | 0.025 | 0.25/0.25 |

</details>

### `results_multibin_hat`

Multi-bin UTA vs vAttention (5%/10%, 두 셀렉터)  
_17/22개 완료(7/7 subset), 최종 기록 08-17 06:51_

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
_9/9개 완료(7/7 subset), 최종 기록 08-17 06:52_

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
_10/12개 완료(7/7 subset), 최종 기록 08-17 06:51_

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
_7/9개 완료(7/7 subset), 최종 기록 08-17 06:51_

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
_6/6개 완료(7/7 subset), 최종 기록 08-17 06:52_

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
_2/2개 완료(7/7 subset), 최종 기록 08-17 06:51_

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
_0/3개 완료(7/7 subset), 최종 기록 08-17 06:51_

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
_0/2개 완료(7/7 subset), 최종 기록 08-17 06:51_

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

## 3. 아직 확립되지 않은 것

- **wall-clock 미측정.** 구현은 여전히 $O(QK)$ 프로토타입입니다. §4는 **해석적 회계**이지
  실측이 아닙니다.
- **§4 기준 현재 구성은 MAC 예산 초과**입니다. 경로는 있으나 미검증(§4 하단).
- j1/j2·exact/diag 판정은 2샘플 × 2subset(진단 격자) 근거입니다. Ablation이 j2 구성만
  돌고 있으므로 벤치마크 수준의 재확인은 아직 없습니다.

---

## 4. Phase 3 — 비용 회계 (`analyze_cost.py`)

decode query·head당, 두 방법이 공통으로 내는 top-k 비용 **위에 추가되는** 양.
K=32768, d=128, ρ=5%, batch=32, GQA=4, KV bf16, bin 통계 int8.

| | MAC | bytes/query | bytes shared |
|---|---|---|---|
| vAttention 샘플링 (819 샘플) | 209.7K | 419.3K (랜덤) | 0 |
| ours, 인접 bin 32 + 대각 분산 | 776.3K | 0 | 373.6K (연속) |

**트래픽은 압도적으로 유리합니다.** 우리 bin 통계는 query와 무관해 batch×GQA=128개
query가 같은 바이트를 공유(L2 상주) → 실효 **0.01배**. 비상각 기준으로도 0.89배.
vAttention의 샘플은 query마다 다른 랜덤 접근이라 공유가 불가능합니다.
**디코딩이 메모리 바운드라는 점을 감안하면 이쪽이 결정적입니다.**

**MAC은 초과합니다 (3.70배).** 분해하면:

| 항목 | MAC | bin 크기 의존 |
|---|---|---|
| bin 통계 3·B·d (평균·분산·value) | 373K | ∝ 1/bin |
| **선택 토큰 차감 2·k·d** | **403K** | **고정** |

bin 크기를 키워도 32→256에서 3.70→2.14배로 포화합니다. **지배항이 선택 토큰 차감이고
이건 bin 크기와 무관**하기 때문입니다.

기각된 대안 (같은 단위): full covariance 분산 15.9M MAC = **76배**,
score binning 8.0M MAC = **38배** (상한 참고선 전용).

**예산 안으로 들어가는 경로 (미검증):** 선택 토큰 차감을 생략하면 373K(bin 32) →
187K(bin 64)로 vAttention 아래로 내려갑니다. 대신 선택 토큰이 자기 bin 평균에도
포함돼 이중 계산되는 편향이 생깁니다. 5%에서 bin 32당 약 1.5개이지만 **그것이 bin 내
최고 로짓 토큰들**이라 $\bar z_b$를 올릴 수 있어 크기를 실측해야 합니다.

---

## 5. 재현

```bash
ENV=~/miniconda3/envs/sparse-attn/bin/python

# 이 export는 생략 불가: env의 python을 직접 호출하면 conda activate.d 훅이 돌지 않아
# HF_HOME이 unset이 되고, 8B 모델과 RULER 데이터를 ~/.cache로 다시 받습니다.
export HF_HOME=/database/hyunwoo/hf HF_HUB_CACHE=/database/hyunwoo/hf/hub \
       TRANSFORMERS_CACHE=/database/hyunwoo/hf/hub HF_DATASETS_CACHE=/database/hyunwoo/hf/datasets

# 설정 확정용 진단 (dense 출력 반환, 생성 궤적 무교란)
$ENV run_block_tail_diag.py --device cuda:2 --heavy-size 0.048 --selector hat \
    --tag cfg5_hat --subsets qa_1,niah_multikey_3 --samples-per-subset 2 \
    --max-new-tokens 3 --max-qrows 2 --output-dir ./results_tail_diag_v2

# Ablation (7개, density 5%, HAT)
for M in dense selector UTA UTA+jensen UTA+multibin UTA+multibin+jensen vAttention; do
  $ENV run_multibin_hat_bench.py --device cuda:2 --densities 0.05 --selector hat \
      --methods "$M" --bin-sizes 32 --eps 0.25 --delta 0.25 --output-dir ./results_v2
done

$ENV analyze_cost.py --density 0.05 --bin-size 32     # Phase 3 회계
$ENV make_report.py --report results/REPORT_V2.md     # 이 파일 갱신
```

단위 테스트 `pytest tests/test_uta_multibin.py -q` (16개)가 고정하는 불변식:
`Σ n_b == N_tail`, `bin_size=1` ⇒ exact attention, bin ≥ K ⇒ 기본 UTA,
그리고 bin을 나누는 데 쓴 값이 bin logit에 절대 새어 들어가지 않을 것.

---

## 6. 인용 금지 (V1에서 이월 + V2 추가)

| 결과 | 사유 |
|---|---|
| V1 `ours-b*-score(*)` 86.83 / 86.95 / 87.05 | score bin은 tail 전체 로짓이 필요 — **도달 불가 상한** |
| `vAttention-HAT@3pct` 54.58, `-s0.25` 55.49 | `eps=delta=0.4`로 실행(타 실행은 0.25), adaptive budget 약 18배 차이 |
| 모든 `@1pct` HAT 실행 | HAT이 1%에서 retrieval 불가(nk_3 ≈ 1) — 셀렉터 붕괴를 측정한 것 |
| `UTA_Jensen@alpha=1`, `UTA_LN` | j1 형태 — §1.2에서 형태 자체가 기각됨 |
| `CovUTA*` | 분자 보정 — §1.4에서 기여 ≈ 0 |
| 50샘플 실행 (`results_uta_low_density`) | subset당 SE ≈ 7%p |

---

## 7. 실행 인벤토리

<!-- AUTO:inventory START -->
| 결과 디렉터리 | 실행 수 | 완료 | 완료 subset 합 | 최종 기록 |
|---|---|---|---|---|
| `results_v2` | 7 | 7 | 49 | 08-17 15:24 |
| `results_multibin_hat` | 22 | 17 | 141 | 08-17 06:51 |
| `results_density_sweep` | 9 | 9 | 63 | 08-17 06:52 |
| `results_covuta` | 12 | 10 | 74 | 08-17 06:51 |
| `results_full_comparison` | 9 | 7 | 57 | 08-17 06:51 |
| `results_uta_low_density` | 6 | 6 | 42 | 08-17 06:52 |
| `results_damped_jensen_full` | 2 | 2 | 14 | 08-17 06:51 |
| `results_jensen_diagnostic` | 3 | 0 | 12 | 08-17 06:51 |
| `results_covuta_v2` | 2 | 0 | 10 | 08-17 06:51 |
<!-- AUTO:inventory END -->

---

## 8. 다음

1. **선택 토큰 차감 생략의 편향 측정** — 이게 MAC 예산 초과를 푸는 유일한 지렛대(§4).
2. **wall-clock 실측** — 해석적 회계를 실측으로 대체.
3. **bin 크기 스윕** — 정확도(작을수록 유리) vs 비용(클수록 유리)의 교차점 확정.
