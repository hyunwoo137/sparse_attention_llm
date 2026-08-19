# REPORT V3 — Latency: UTA ablation ladder vs vAttention

장비: **NVIDIA H100 NVL** (GPU 1) · torch 2.13.0+cu130 · Llama-3.1-8B 형상 (H=32, H_kv=8, D=128, bf16) · HAT selector · 밀도 ρ=5% 일치

모든 config는 `run_multibin_hat_bench.build_config`로 만들었다. `results_v2/`의 정확도 수치를 생성한 것과 **동일한 빌더**이므로, 이 리포트의 latency 열과 정확도 열은 같은 대상을 가리킨다.

## 0. 숫자를 인용하기 전에 반드시 읽을 것

**Tier 1이 재는 것은 프로토타입이지 배포 커널이 아니다.** 여기 나오는 모든 방법이 — 우리 것도, vAttention도 — full `(B,H,Q,K)` score matrix를 그대로 만든다 (`mb/T2_scores_fullQK`, `uta/T2_scores_fullQK`, `vatt/S1_expwts_fullQK`). 배포용 커널이라면 어느 쪽도 이런 일을 하지 않는다. 따라서:

- 같은 형상에서의 **방법 간 비율**은 의미가 있다. O(QK) 항이 모두에게 공통이기 때문이다.
- **절대 밀리초는 배포 성능 추정치가 아니다.**
- 프로토타입 전용 기계장치가 비용을 지배하는 항목(§3 참고)은 알고리즘이 아니라 코드에 대해 말해주는 것이다.

decode(Q=1)가 올바른 측정 영역이다. 어댑터는 context prefill을 `enable_sparse_mode()` **바깥에서** 수행하므로, 실제 실행에서 sparse attention 호출은 전부 decode step이다 ([huggingface.py:174](../sparse_attention_hub/adapters/huggingface.py#L174)).

## 1. 메인 비교 — K=32768, B=1, Q=1, ρ=5%

| method | mean ms | p50 | p95 | vs vAtt | 실측 ρ | peak MB | RULER avg |
|:---|---:|---:|---:|---:|---:|---:|---:|
| dense | 0.780 | 0.779 | 0.781 | 0.20× | 1.0000 | 671 | 87.04 |
| HAT | 2.937 | 2.931 | 2.988 | 0.76× | 0.0499 | 1396 | 71.55 |
| UTA | 4.827 | 4.827 | 4.865 | 1.25× | 0.0499 | 2010 | 81.25 |
| UTA+multibin(b32) | 6.590 | 6.590 | 6.618 | 1.71× | 0.0499 | 2206 | 81.44 |
| UTA+jensen | 21.332 | 21.330 | 21.395 | 5.53× | 0.0499 | 2155 | — |
| UTA+multibin+jensen(b32) | 6.595 | 6.591 | 6.628 | 1.71× | 0.0499 | 2206 | 81.43 |
| vAttention(HAT) e.25d.25 | 3.856 | 3.830 | 3.942 | 1.00× | 0.0490 | 1396 | 68.70 |
| vAttention(HAT) e.4d.4 | 3.824 | 3.820 | 3.847 | 0.99× | 0.0490 | 1396 | — |
| UTA+jensen (direct impl) | 5.345 | 5.359 | 5.396 | 1.39× | 0.0499 | 2543 | — |
| UTA+multibin(b32,fixed) | 4.194 | 4.163 | 4.287 | 1.09× | 0.0499 | 2080 | — |
| UTA+multibin+jensen(b32,fixed) | 4.177 | 4.167 | 4.277 | 1.08× | 0.0499 | 2080 | — |

`RULER avg`는 `results_v2/multibin_hat_summary.csv`에서 인용했다 (200 samples × 7 subsets). `—`는 latency는 쟀지만 정확도 실험은 아직 돌리지 않은 변형이라는 뜻이다.

**등가 비용 전제가 성립한다.** 실측 밀도가 모든 sparse 방법에서 0.0490–0.0499 범위에 있다. 즉 전부 2% 이내로 같은 개수의 KV 엔트리를 읽는다. vAttention이 명목치보다 살짝 *아래*에 있는 이유는 복원추출(with replacement)로 샘플링하기 때문이며, 따라서 이 비교는 오히려 vAttention에 유리한 쪽으로 기울어 있다.

## 2. 단계별 분해

호출당 CUDA 시간, 계측 활성 상태(`UTA_STAGE_TIMING=1`). 비율은 상위 단계 합계 대비이며, 들여쓴 행은 바로 위 행에 **포함된** 하위 단계라 합계에 다시 더하지 않는다.

### 2.1 vAttention — 오차 평가 → 추가 budget → 재샘플링 포함

"오차 평가 → 추가 budget 할당 → 재샘플링" 단은 **S3 → S4 → S5**에 해당한다. 이건 따로 덧붙인 게 아니라 `AdaptiveSamplingMasker.add_mask` 안에 들어 있어서 **구조적으로** 측정 구간에 포함된다 ([adaptive_sampling.py:366-410](../sparse_attention_hub/sparse_attention/research_attention/maskers/sampling/implementations/adaptive_sampling.py#L366-L410)).

계측 없는 총합 **3.856 ms** · 계측 포함 4.340 ms · 단계 합계 4.248 ms

| 단계 | ms/call | 비중 | 무엇인가 |
|:---|---:|---:|:---|
| `core/masked_attention_output` | 2.063 | 48.6% | sparse softmax + 출력 |
| `sel/AdaptiveSamplingMasker` | 1.195 | 28.1% | vAttention 샘플링 단 전체 |
| &nbsp;&nbsp;↳ `S1_expwts_fullQK` | 0.462 | 10.9% | full QK + exp (프로토타입에만 존재) |
| &nbsp;&nbsp;↳ `S2_static_denominator` | 0.032 | 0.8% | 이미 선택된 토큰들의 분모 |
| &nbsp;&nbsp;↳ `S3_base_sample_std` | 0.037 | 0.9% | base 샘플 → std 추정 = **오차 평가** |
| &nbsp;&nbsp;↳ `S4_error_eval_and_budget` | 0.139 | 3.3% | 오차 한계 → 행별 **추가 budget 산정** |
| &nbsp;&nbsp;↳ `S5_resample_extra_budget` | 0.315 | 7.4% | 새 budget으로 **재샘플링** |
| &nbsp;&nbsp;↳ `S6_merge_mask` | 0.087 | 2.0% | 샘플링 마스크를 선택 마스크에 병합 |
| `sel/HashAttentionTopKMasker` | 0.889 | 20.9% | HAT top-k 선택 (전 방법 공통) |
| `sel/LocalMasker` | 0.087 | 2.0% | local window (전 방법 공통) |
| `sel/SinkMasker` | 0.014 | 0.3% | sink 토큰 (전 방법 공통) |

### 2.2 UTA (tail 전체를 하나의 평균 proxy로)

계측 없는 총합 **4.827 ms** · 계측 포함 5.001 ms · 단계 합계 4.924 ms

| 단계 | ms/call | 비중 | 무엇인가 |
|:---|---:|---:|:---|
| `uta/T2_scores_fullQK` | 1.361 | 27.6% | full QK (프로토타입에만 존재) |
| `uta/T3_tail_kv_mean` | 1.175 | 23.9% | tail 구간의 K, V 평균 풀링 |
| `sel/HashAttentionTopKMasker` | 0.924 | 18.8% | HAT top-k 선택 (전 방법 공통) |
| `uta/T1_tail_mask` | 0.677 | 13.7% | tail 마스크 + tail 개수 |
| `uta/T6_merge_softmax` | 0.654 | 13.3% | 전역 softmax 병합 + 출력 |
| `sel/LocalMasker` | 0.094 | 1.9% | local window (전 방법 공통) |
| `uta/T5_proxy_logit` | 0.024 | 0.5% | q·k_mean + log N |
| `sel/SinkMasker` | 0.015 | 0.3% | sink 토큰 (전 방법 공통) |

### 2.3 UTA + Jensen, 직접 구현

`advanced.py`의 `UTAJensenAttention`: proxy 하나 + 편차 단. Jensen 비용을 깨끗하게 분리해서 보여주는 것이 이 행이다.

계측 없는 총합 **5.345 ms** · 계측 포함 5.545 ms · 단계 합계 5.452 ms

| 단계 | ms/call | 비중 | 무엇인가 |
|:---|---:|---:|:---|
| `utaj/T1_tail_mask` | 1.521 | 27.9% | tail 마스크 + tail 개수 |
| `uta/T2_scores_fullQK` | 1.364 | 25.0% | full QK (프로토타입에만 존재) |
| `sel/HashAttentionTopKMasker` | 0.945 | 17.3% | HAT top-k 선택 (전 방법 공통) |
| `uta/T6_merge_softmax` | 0.655 | 12.0% | 전역 softmax 병합 + 출력 |
| `utaj/T4_jensen_var` | 0.499 | 9.1% | **Jensen 편차 단 (합계)** |
| &nbsp;&nbsp;↳ `T4b_var_k_sq_mean` | 0.479 | 8.8% |   tail 구간 E[k²] → Var(k_d) |
| &nbsp;&nbsp;↳ `T4c_var_q2_dot` | 0.016 | 0.3% |   σ² = s²·Σ_d q_d²·Var(k_d) |
| `utaj/T3_tail_kv_mean` | 0.339 | 6.2% | tail 구간의 K, V 평균 풀링 |
| `sel/LocalMasker` | 0.100 | 1.8% | local window (전 방법 공통) |
| `sel/SinkMasker` | 0.016 | 0.3% | sink 토큰 (전 방법 공통) |
| `utaj/T5_proxy_logit` | 0.015 | 0.3% | z̄ + α·σ²/2 + log N |

### 2.4 UTA + multi-bin + Jensen (b=32, ladder config)

ladder의 `equalcount` 분할 — 정확도 수치를 생성한 바로 그 config다.

계측 없는 총합 **6.595 ms** · 계측 포함 6.790 ms · 단계 합계 6.696 ms

| 단계 | ms/call | 비중 | 무엇인가 |
|:---|---:|---:|:---|
| `mb/T4_jensen_var` | 2.299 | 34.3% | **Jensen 편차 단 (합계)** — 전환 이전 이름 |
| &nbsp;&nbsp;↳ `T4a_var_k_mean` | 0.955 | 14.3% |   bin별 E[k] |
| &nbsp;&nbsp;↳ `T4b_var_k_sq_mean` | 1.255 | 18.7% |   bin별 E[k²] |
| &nbsp;&nbsp;↳ `T4c_var_q2_dot` | 0.067 | 1.0% |   σ_b² = s²·Σ_d q_d²·Var_b(k_d) |
| `mb/T2_scores_fullQK` | 1.823 | 27.2% | full QK (프로토타입에만 존재) |
| `mb/T3_bin_stats` | 1.082 | 16.2% | bin별 n_b, v̄_b (K/V 리덕션) |
| `sel/HashAttentionTopKMasker` | 0.964 | 14.4% | HAT top-k 선택 (전 방법 공통) |
| `mb/T6_merge_softmax` | 0.265 | 4.0% | {heavy} ∪ {bins} 전역 softmax + 출력 |
| `sel/LocalMasker` | 0.100 | 1.5% | local window (전 방법 공통) |
| `mb/T2b_bin_ids` | 0.091 | 1.4% | **bin 분할** (토큰별 bin id) |
| `mb/T5_kappa_bin_logit` | 0.030 | 0.5% | κ = log(1+σ²/2), bin logit 계산 |
| `mb/T1_tail_mask` | 0.025 | 0.4% | 선택 결과로부터 tail 마스크 구성 |
| `sel/SinkMasker` | 0.016 | 0.2% | sink 토큰 (전 방법 공통) |

### 2.5 UTA + multi-bin + Jensen (b=32, `fixed` bins)

수식은 동일하고, tail을 인접 tail 토큰의 런이 아니라 **위치 블록**으로 나눈다. 경계가 query와 무관하므로 bin 통계를 KV 페이지와 함께 캐시할 수 있다 — 즉 이쪽이 배포 가능한 분할이다.

계측 없는 총합 **4.177 ms** · 계측 포함 4.299 ms · 단계 합계 4.212 ms

| 단계 | ms/call | 비중 | 무엇인가 |
|:---|---:|---:|:---|
| `mb/T2_scores_fullQK` | 1.826 | 43.4% | full QK (프로토타입에만 존재) |
| `sel/HashAttentionTopKMasker` | 0.889 | 21.1% | HAT top-k 선택 (전 방법 공통) |
| `mb/T4_jensen_var` | 0.794 | 18.8% | **Jensen 편차 단 (합계)** — 전환 이전 이름 |
| &nbsp;&nbsp;↳ `T4a_var_k_mean` | 0.207 | 4.9% |   bin별 E[k] |
| &nbsp;&nbsp;↳ `T4b_var_k_sq_mean` | 0.510 | 12.1% |   bin별 E[k²] |
| &nbsp;&nbsp;↳ `T4c_var_q2_dot` | 0.067 | 1.6% |   σ_b² = s²·Σ_d q_d²·Var_b(k_d) |
| `mb/T3_bin_stats` | 0.279 | 6.6% | bin별 n_b, v̄_b (K/V 리덕션) |
| `mb/T6_merge_softmax` | 0.266 | 6.3% | {heavy} ∪ {bins} 전역 softmax + 출력 |
| `sel/LocalMasker` | 0.089 | 2.1% | local window (전 방법 공통) |
| `mb/T5_kappa_bin_logit` | 0.029 | 0.7% | κ = log(1+σ²/2), bin logit 계산 |
| `mb/T1_tail_mask` | 0.025 | 0.6% | 선택 결과로부터 tail 마스크 구성 |
| `sel/SinkMasker` | 0.014 | 0.3% | sink 토큰 (전 방법 공통) |

### 2.6 ladder가 표현하는 방식의 UTA + Jensen (거대 bin 1개)

계측 없는 총합 **21.332 ms** · 계측 포함 21.498 ms · 단계 합계 21.404 ms

| 단계 | ms/call | 비중 | 무엇인가 |
|:---|---:|---:|:---|
| `mb/T4_jensen_var` | 10.410 | 48.6% | **Jensen 편차 단 (합계)** — 전환 이전 이름 |
| &nbsp;&nbsp;↳ `T4a_var_k_mean` | 5.036 | 23.5% |   bin별 E[k] |
| &nbsp;&nbsp;↳ `T4b_var_k_sq_mean` | 5.336 | 24.9% |   bin별 E[k²] |
| &nbsp;&nbsp;↳ `T4c_var_q2_dot` | 0.021 | 0.1% |   σ_b² = s²·Σ_d q_d²·Var_b(k_d) |
| `mb/T3_bin_stats` | 7.722 | 36.1% | bin별 n_b, v̄_b (K/V 리덕션) |
| `mb/T2_scores_fullQK` | 1.822 | 8.5% | full QK (프로토타입에만 존재) |
| `sel/HashAttentionTopKMasker` | 0.943 | 4.4% | HAT top-k 선택 (전 방법 공통) |
| `mb/T6_merge_softmax` | 0.247 | 1.2% | {heavy} ∪ {bins} 전역 softmax + 출력 |
| `sel/LocalMasker` | 0.101 | 0.5% | local window (전 방법 공통) |
| `mb/T2b_bin_ids` | 0.091 | 0.4% | **bin 분할** (토큰별 bin id) |
| `mb/T5_kappa_bin_logit` | 0.028 | 0.1% | κ = log(1+σ²/2), bin logit 계산 |
| `mb/T1_tail_mask` | 0.025 | 0.1% | 선택 결과로부터 tail 마스크 구성 |
| `sel/SinkMasker` | 0.016 | 0.1% | sink 토큰 (전 방법 공통) |

## 3. Jensen 편차 단의 실제 비용

| 측정 | 없을 때 ms | 있을 때 ms | Δ ms | Δ % |
|:---|---:|---:|---:|---:|
| proxy 1개 위의 Jensen (직접 구현) | 4.827 | 5.345 | +0.518 | +10.7% |
| proxy 1개 위의 Jensen (ladder: 거대 bin 1개) | 4.827 | 21.332 | +16.505 | +342.0% |
| 1024개 bin 위의 Jensen (equalcount) | 6.590 | 6.595 | +0.005 | +0.1% |
| 1024개 bin 위의 Jensen (fixed) | 4.194 | 4.177 | -0.017 | -0.4% |

이 표를 같이 읽으면:

1. **직접 구현하면 편차 단은 싸다.** `UTA+jensen (direct impl)`은 UTA 위에 차원별 `E[k²]` 리덕션 하나와 `Σ_d q_d²·Var(k_d)` 내적 하나를 더할 뿐이고, 그만큼의 작고 한정된 증분만 든다.
2. **multi-bin 위에서는 사실상 공짜다.** `var_mode="diag"`가 bin 평균이 이미 쓰는 것과 동일한 형태의 bin별 리덕션을 재활용하기 때문이다. `fixed` 분할에서는 한계 비용이 측정 노이즈 수준이다.
3. **ladder의 `UTA+jensen` 행은 측정 함정이다.** 이 행은 `bin_mode="equalcount", bin_size=1e9`, 즉 tail 전체를 덮는 bin 하나로 구성돼 있다 ([run_multibin_hat_bench.py:129-131](../run_multibin_hat_bench.py#L129-L131)). 그 결과 모든 tail 토큰이 bin id 0으로 scatter_add되어 리덕션 전체가 단일 주소 atomics로 직렬화된다. 수학적으로는 *가장 싼* 변형인데 표에서 가장 느린 이유가 이것이다. **이 행을 Jensen의 비용으로 인용하면 안 된다.** 1번 행이 정직한 숫자다.

§2.4에서 하나 더 보인다: `mb/T4a_var_k_mean`이 계산하는 bin별 `E[k]`는 `mb/T3_bin_stats`가 `z̄_b`를 구하려고 이미 손에 들고 있던 값이다. 이걸 넘겨주기만 하면 편차 단의 세 리덕션 중 하나를 공짜로 없앨 수 있다.

## 4. 연산량으로 잡히지 않는 vAttention 고유 비용

**매 layer, 매 step마다 device→host 동기화가 걸린다.** adaptive budget이 데이터에 의존하므로 샘플링 마스크의 크기를 device가 알려주기 전까지 알 수 없다. 그래서 [mask_attention_utils.py:145-150](../sparse_attention_hub/sparse_attention/utils/mask_attention_utils.py#L145-L150)의 `int(budgets_flat.sum().item())`이 강제 동기화를 일으킨다. K=32768에서 **0.057 ms/call**로 측정됐다 — 연산이 아니라 파이프라인 정지다. 단순히 32개 layer로 곱하면 토큰당 ~1.83 ms지만, 실제 모델에서 그중 얼마가 남는지는 CPU가 얼마나 앞서 달리고 있었는지에 달려 있고 단일 layer 하네스로는 알 수 없다. 확실한 것은 구조적인 사실 쪽이다: UTA 계열에는 host 동기화가 **하나도 없다**. 스텝당 작업량이 미리 정해지기 때문이다.

**vAttention의 추가 budget이 어디로 가는가.** 샘플링 단 전체가 1.195 ms/call이고, 그중 오차 평가 → budget → 재샘플링 사슬(S3+S4+S5)이 0.491 ms다.

**한쪽은 확률적이고 한쪽은 결정적이다 — 다만 그 결과는 이 벤치마크로 아직 정량화되지 않았다.** vAttention은 budget이 데이터에 반응하므로 실측 밀도가 반복마다 달라진다. 반면 모든 UTA 변형은 반복 간 완전히 동일하다:

| method | ρ 평균 | ρ 최대 | 반복 간 ρ | p50 ms | p95 ms | max ms | std ms |
|:---|---:|---:|---:|---:|---:|---:|---:|
| HAT | 0.04993 | 0.04993 | 결정적 | 2.931 | 2.988 | 3.070 | 0.0293 |
| UTA | 0.04993 | 0.04993 | 결정적 | 4.827 | 4.865 | 4.879 | 0.0252 |
| UTA+multibin(b32) | 0.04993 | 0.04993 | 결정적 | 6.590 | 6.618 | 6.657 | 0.0224 |
| UTA+jensen | 0.04993 | 0.04993 | 결정적 | 21.330 | 21.395 | 21.431 | 0.0317 |
| UTA+multibin+jensen(b32) | 0.04993 | 0.04993 | 결정적 | 6.591 | 6.628 | 6.695 | 0.0278 |
| vAttention(HAT) e.25d.25 | 0.04899 | 0.04902 | 변동 (+0.00003) | 3.830 | 3.942 | 4.554 | 0.1318 |
| vAttention(HAT) e.4d.4 | 0.04897 | 0.04900 | 변동 (+0.00003) | 3.820 | 3.847 | 3.934 | 0.0226 |
| UTA+jensen (direct impl) | 0.04993 | 0.04993 | 결정적 | 5.359 | 5.396 | 5.413 | 0.0388 |
| UTA+multibin(b32,fixed) | 0.04993 | 0.04993 | 결정적 | 4.163 | 4.287 | 4.679 | 0.1004 |
| UTA+multibin+jensen(b32,fixed) | 0.04993 | 0.04993 | 결정적 | 4.167 | 4.277 | 4.285 | 0.0349 |

여기서 조심할 것이 둘 있다:

- 밀도 열은 **실제로 두 방법을 구분한다.** vAttention은 확률적이고 UTA 계열은 정확히 결정적이다. 이건 진짜 구조적 차이다 — 서빙 시스템은 UTA 스텝의 작업량은 미리 산정할 수 있지만 vAttention 스텝은 산정할 수 없다.
- **하지만 latency 열은 두 방법을 구분하지 못한다.** vAttention의 std는 `UTA+multibin(b32,fixed)`의 std와 비슷한 수준이고, 편차는 전부 몇 % 이내다. 이 마이크로벤치는 모든 반복에 고정된 합성 텐서 하나를 먹이므로 vAttention의 budget이 거의 움직이지 않고, 따라서 데이터 의존성이 전혀 발현되지 않는다. tail latency 논증을 정량화하려면 실제 RULER 입력으로 여러 decode step에 걸친 스텝별 측정이 필요하다. 위 표로는 **입증되지 않으며, 위 표를 근거로 주장해서는 안 된다.**

아래 모든 스윕은 **두 가지 tail 분할을 함께** 싣는다: `equalcount`(인접 tail 토큰의 런 — `results_v2` 정확도 수치를 만든 분할)와 `fixed`(위치 블록 — 수식은 같고 경계가 query와 무관한, 배포 가능한 분할). 둘의 스케일링이 크게 다르고, 그 차이가 이번 실험의 가장 실용적인 결과다.

## 5. 시퀀스 길이 스케일링 (mean ms, B=1, Q=1)

모든 프로토타입이 O(K) full-QK 항을 지고 있어서 모든 열이 선형으로 자란다. 여기서 읽어야 할 것은 기울기가 아니라 **격차**다.

| seq_len | HAT | UTA | UTA+multibin+jensen(b32) | UTA+multibin+jensen(b32,fixed) | vAttention(HAT) e.25d.25 |
|:---|---:|---:|---:|---:|---:|
| 4096 | 1.266 | 1.572 | 2.548 | 2.068 | 2.120 |
| 8192 | 1.390 | 1.910 | 2.589 | 2.075 | 2.270 |
| 16384 | 1.907 | 2.869 | 3.879 | 2.593 | 2.754 |
| 32768 | 2.941 | 4.842 | 6.594 | 4.179 | 3.828 |
| 65536 | 5.271 | 8.841 | 12.139 | 7.396 | 6.536 |

**교차점이 존재한다.** `fixed` 분할은 K=16384까지는 vAttention보다 오히려 *빠르고* (K=4096: 0.98×, K=8192: 0.91×, K=16384: 0.94×, K=32768: 1.09×, K=65536: 1.13×), 긴 쪽 끝에서만 진다. 두 곡선 모두 동일한 프로토타입 전용 O(K) full-QK 항이 지배하므로, 이 교차점이 알려주는 것은 두 tail 추정기의 **상수 계수** 차이다. 그리고 그 부분이야말로 배포 커널에서도 남는 부분이다.

## 6. 배치 스케일링 (mean ms, K=32768, Q=1)

우리 쪽의 구조적 논거는, bin 통계가 query와 무관하므로 한 번 읽으면 배치 전체와 GQA 그룹 내 모든 query head에 재사용된다는 것이다. 반면 vAttention의 random gather는 query별이라 공유할 수 없다. **그런데 이 프로토타입은 아직 그 이점을 구현하지 않았다.** 양쪽 다 query마다 전부 다시 계산한다. 따라서 이 표는 논거의 증거가 아니라, 배포 경로가 넘어서야 할 기준선으로 읽어야 한다.

| batch | UTA+multibin+jensen(b32) | UTA+multibin+jensen(b32,fixed) | vAttention(HAT) e.25d.25 |
|:---|---:|---:|---:|
| 1 | 6.588 | 4.177 | 3.835 |
| 2 | 12.597 | 7.613 | 6.743 |
| 4 | 24.180 | 14.367 | 12.346 |
| 8 | 48.766 | 28.203 | 23.556 |

## 7. Bin 크기 다이얼 (mean ms, K=32768, B=1, Q=1)

bin당 토큰 수. bin이 작을수록 proxy가 많아지고 정확해지며 연산도 늘어난다. bin_size=1이면 exact attention이다. 이 표에서 쓸모 있는 부분은 **열이 거의 평평하다는 것**이다. 프로토타입에서 비용을 결정하는 것은 bin 개수가 아니라 O(K) 패스이므로, 이 구간에서는 정확도를 거의 공짜로 살 수 있다.

| bin_size | UTA+multibin+jensen(equalcount) | UTA+multibin+jensen(fixed) |
|:---|---:|---:|
| 8 | 7.009 | 4.751 |
| 16 | 6.731 | 4.377 |
| 32 | 6.602 | 4.200 |
| 64 | 6.547 | 4.058 |
| 128 | 6.557 | 4.009 |

## 8. Query 개수 스케일링 (mean ms, K=32768, B=1)

Q>1은 요청당 한 번 발생한다. 어댑터가 autoregressive decode를 시작하기 전에 질문 토큰들을 sparse forward 한 번으로 처리하는 시점이다.

| num_queries | UTA+multibin+jensen(b32) | UTA+multibin+jensen(b32,fixed) | vAttention(HAT) e.25d.25 |
|:---|---:|---:|---:|
| 1 | 6.611 | 4.180 | 3.843 |
| 8 | 27.411 | 6.891 | 4.824 |
| 32 | 99.291 | 13.040 | 7.603 |

### 8.1 스케일하지 않는 것은 `equalcount`의 scatter 경로다

`equalcount` bin은 인접한 *tail* 토큰의 런이므로 bin 소속이 선택 결과에 의존하고, 따라서 reshape으로 표현할 수 없다. 그래서 [`_aggregate_by_bin_id`](../sparse_attention_hub/sparse_attention/uta_attention/multibin.py#L226-L253)는 `(B, h_chunk, Q, K, D)` 버퍼를 실제로 만들어 scatter_add를 수행하고, 청킹 휴리스틱 `hstep = max(1, 8 // Q)`는 Q > 8이 되는 순간 1로 붕괴한다. 반면 `fixed` bin은 위치 블록이라 동일한 통계가 `reshape` + `sum`으로 끝난다 — scatter도, 거대 버퍼도 없고, 경계가 query와 무관하다. 그리고 바로 그 성질이 통계를 KV 페이지와 함께 캐시 가능하게 만드는 성질이다.

두 분할은 **같은 추정기**를 구현한다. 정확도 차이가 무엇으로 밝혀지든, 위 표들의 비용 차이는 충분히 크므로 `equalcount`가 RULER에서 값어치를 증명하지 못하는 한 `fixed`가 기본값이 되어야 한다 — 그리고 그 비교는 아직 돌리지 않았다.

## 9. 이 실험이 입증한 것과 입증하지 않은 것

**입증한 것:**

- **latency와 정확도가 둘 다 있는 config**: `UTA+multibin+jensen(b32)` (equalcount)는 **6.595 ms** — vAttention의 **3.856 ms** 대비 1.71× — 이고, vAttention 쪽 숫자에는 오차 평가 → budget → 재샘플링 단이 완전히 포함돼 있다. RULER는 **81.43** 대 vAttention의 **68.70**. 즉 이 ablation은 프로토타입 기준 1.71× latency로 RULER +12.7점을 산다.
- **가장 빠른 config**: 동일한 추정기를 `fixed` 위치 블록 위에서 돌리면 **4.177 ms** (1.08× vAttention). 단 이 변형의 RULER 정확도는 **아직 측정하지 않았으므로** 정확도 주장과 짝지을 수 없다 — 아래 다음 단계 2번이 그 실험이다. 4.18 ms를 81.43 옆에 나란히 인용하지 말 것.
- Jensen 편차 단은 직접 구현하면 작고 한정된 증분이고(UTA 대비 +0.5 ms, +11%), multi-bin 위에서는 거의 공짜다.
- vAttention은 UTA 계열에 전혀 없는 layer당 device→host 동기화를 지불하며, 실측 밀도가 확률적인 반면 UTA 계열은 정확히 결정적이다.
- 모든 방법이 실제로 동일한 밀도에 있다(§1). 즉 이 비교 자체가 유효하다.

**입증하지 않은 것 (그리고 주장하지 않는 것):**

- 배포 시 비용. 양쪽 다 여전히 full score matrix를 계산한다. 구조적 비대칭 — 우리 bin 통계는 query와 무관하고 KV 페이지와 함께 캐시 가능한 반면 vAttention의 gather는 그렇지 않다 — 은 [analyze_cost.py](../analyze_cost.py)에 모델링돼 있지만 **아직 측정되지 않았다**. 그게 Tier 2다.
- `UTA+jensen` ladder 행에서 나오는 어떤 주장도. 이유는 §3의 atomics 문제.
- tail latency / 예측 가능성 논증. 밀도 열은 이를 뒷받침하지만, 측정된 latency 편차는 여기서 두 방법을 구분하지 못한다(§4).
- end-to-end tokens/s. 이건 attention layer 하나를 격리해서 잰 것이고, layer 간 오버헤드나 32× layer 배수는 포함돼 있지 않다.

**권장하는 다음 단계 (우선순위 순):**

1. `fixed` 분할의 정확도를 측정한다. 배포 가능한 분할이면서 여기서 더 빠른 프로토타입 경로이기도 하다. RULER에서 `equalcount`와 대등하다면 이쪽이 헤드라인 config가 되어야 한다.
2. `UTA+jensen` ablation 단을 `bin_size=1e9`에서 떼어내 `UTAJensenAttention` (또는 `bin_mode="fixed"`)으로 교체한다. 그래야 그 단이 atomics 핫스팟이 아니라 보정 자체를 측정하게 된다. `results_v2`의 해당 정확도 행도 아직 미완성이다.
3. `mb/T3_bin_stats`의 `E[k]`를 편차 단으로 넘겨서 중복 계산을 없앤다.
4. 그 다음 Tier 2: bin 통계를 KV 페이지 메타데이터로 캐시하고 per-query 비용만 측정.
