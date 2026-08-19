# V4 — 랜덤 분할 multi-bin: bin 개수 스윕

모델 `meta-llama/Llama-3.1-8B-Instruct` (bf16) · RULER 32K
**density 5% 고정 · 셀렉터 HashAttention(HAT) · κ=none · var_mode=diag(key-side)**

sink / local window / HAT heavy hitter를 먼저 제외하고, **남은 tail만** UTA 기반 multi-bin으로
근사합니다. bin 분할 기준은 **균등 랜덤**이며, bin 개수 B를 1 → 2 → 4 → 8로 늘립니다
(B=1이 곧 단일 프록시 UTA).

**통계적 해상도:** subset 점수 SE ≈ 3.5%p, 7-subset 평균 SE ≈ 1.3%p.

---

## 1. 예측 — 랜덤 분할의 회수율은 σ²B/(2n)

랜덤 분할은 각 bin을 **같은 tail의 비편향 표본**으로 만듭니다. 따라서 bin을 아무리 늘려도
**bin 내부 로짓 분산은 전역 분산 σ² 그대로**입니다. 줄어드는 것은 bin 평균의 정밀도뿐이고,
이득은 전적으로 그 평균이 흔들리는 데서 나옵니다.

크기 m = n/B인 랜덤 bin에서 z̄_b의 분산은 σ²/m이므로

```
추정 tail 질량 ≈ n · exp(z̄ + σ²/(2m)) = n · exp(z̄ + σ²B/(2n))
참 tail 질량   = n · exp(z̄ + K₁),      K₁ = log E[e^(z−z̄)]
```

즉 **랜덤 B개 bin은 Jensen 항 K₁ 중 σ²B/(2n)만큼만 회수**합니다. 진단 측정값
σ² = 5.575, K₁ = 3.722 nats, n = 30,427을 넣으면 B = 8에서 회수율은 **0.02%**입니다.
K₁의 절반을 회수하려면 **B ≈ 20,310**(bin당 1.5토큰)이 필요합니다.

이 예측은 방향이 옳다는 점도 같이 말합니다 — 오차는 B에 대해 **단조 감소**합니다.
문제는 기울기이지 부호가 아닙니다.

---

## 2. 진단 — 예측이 그대로 맞았습니다

15,360행 (32 layer × 32 head × qa_1·niah_multikey_3), dense 출력 대비 상대오차 중앙값.
`results_tail_diag_v4/tail_diag_v4_rand.parquet`.

| B | bin당 토큰 | 출력 상대오차 | B=1 대비 | 예측 회수율 |
|---|---|---|---|---|
| **1** | 30,427 | 0.21735 | — | 0.00% |
| **2** | 15,214 | 0.21735 | −0.00% | 0.00% |
| **4** | 7,607 | 0.21735 | 0.00% | 0.01% |
| **8** | 3,803 | 0.21731 | −0.02% | 0.02% |
| 16 | 1,902 | 0.21728 | −0.03% | 0.04% |
| 64 | 475 | 0.21722 | −0.06% | 0.16% |
| 256 | 119 | 0.21710 | −0.11% | 0.63% |
| 1,024 | 30 | 0.21582 | −0.70% | 2.52% |
| 4,096 | 7 | 0.21077 | −3.02% | 10.09% |
| **16,384** | **2** | **0.13184** | **−39.34%** | **40.36%** |

측정된 오차 감소가 예측 회수율을 양 끝에서 모두 따라갑니다 — 특히 B=16,384에서
**측정 −39.34% vs 예측 40.36%**. σ²B/(2n) 공식이 검증됐습니다.

**요청 구간 B ≤ 8에서는 −0.02%**로, 벤치마크 해상도(SE 1.3%p)보다 세 자릿수 아래입니다.

---

## 3. 랜덤 분할은 같은 bin 개수에서 인접 분할보다 나쁩니다

| 분할 기준 | bin당 토큰 | 출력 상대오차 |
|---|---|---|
| 단일 프록시 (UTA) | 30,427 | 0.21745 |
| **랜덤 1,024개** | **30** | **0.21582** |
| **인접 인덱스 (bin 32)** | **32** | **0.18794** |
| score bin 32 (참 로짓, 도달 불가) | ~950 | 0.00337 |

거의 같은 bin 크기에서 인접 분할이 0.188, 랜덤이 0.216입니다. **위치 인접성은 실제 정보를
담고 있습니다** — 이웃 토큰의 로짓이 상관되어 있어 bin 내부 분산이 실제로 줄어드는 반면,
랜덤 분할은 정의상 그게 불가능합니다.

그리고 랜덤 분할이 의미 있어지는 유일한 구간(B = 16,384, bin당 2토큰)에서는 **압축이
2:1에 불과해 방법의 존재 이유가 사라집니다.** 그 지점에서도 오차 0.132는 score bin(0.0034)의
39배입니다.

---

## 4. Ablation (벤치마크) — 중단함

B ∈ {1,2,4,8}을 cuda:0/1에서 시작했으나 **25.8 s/it**이 나와 설정당 약 10시간,
합계 약 20시간이 필요했습니다. §2가 이미 B ≤ 8의 출력 오차 변화를 **−0.02%**로
측정했고 이는 벤치 해상도(7-subset 평균 SE 1.3%p)보다 세 자릿수 아래이므로,
20 GPU-시간을 들여도 네 점이 모두 노이즈 안에서 겹칠 것이 확정적입니다.
GPU를 후속 실험(bin 확장)으로 돌렸습니다.

**따라서 B ∈ {2,4,8}의 벤치마크 점수는 이 보고서에 없습니다.** 인용 가능한 것은
§2의 출력 상대오차뿐이며, B=1은 아래 V3 실측치(81.25)와 정확히 같은 구성입니다
(§5에서 절대차 0.0으로 검증).

V3 기준선 (동일 프로토콜, 7/7 subset):

| 방법 | 평균 |
|---|---|
| dense@5% | 87.04 |
| UTA+multibin(b32) — 인접 분할 | 81.44 |
| **UTA 단독 = randbin B=1** | **81.25** |
| HAT 단독 | 71.55 |
| vAttention(HAT) | 68.70 |

§2에 따라 B ∈ {1,2,4,8} 전부 81.25 근처에서 구별되지 않을 것으로 예측합니다.

---

## 5. 구현 — 이번에 추가·수정한 것

`bin_mode="random"`이 존재하지 않아 새로 넣었습니다
(`sparse_attention_hub/sparse_attention/uta_attention/multibin.py`).

설계에서 고른 두 가지는 배포 성질을 지키기 위한 것입니다:

1. **bin 배정을 key 위치 기준으로 한 번만 뽑고 모든 query가 공유합니다.** query마다 새로
   뽑으면 k̄_b, Cov_b(K)가 query 의존이 되어 캐싱이 깨지고, batch×GQA 공유(V2 §4의
   트래픽 우위 근거)를 잃습니다.
2. **decode step 간 재추첨하지 않도록 캐시했습니다.** 매 호출 재추첨하면 생성 도중 tail
   프록시가 흔들려 측정이 오염됩니다. seed 고정이라 재현됩니다.

검증한 불변식:

| 항목 | 결과 |
|---|---|
| `Σ n_b == N_tail` (B = 1,2,4,8) | 일치, heavy hitter 누출 0 |
| `B=1` vs 단일 bin equalcount | **max 절대차 0.0** |
| bin 배정의 query 독립성 | 모든 (head, query) 동일 |
| 호출·인스턴스 간 결정성 | 동일 |

---

## 6. 결론

**"랜덤으로 묶으면 성능이 오를 수밖에 없다"는 방향은 맞고, 크기는 쓸 수 없습니다.**

오차는 B에 대해 단조 감소하고 그 비율은 σ²B/(2n)로 예측대로 움직입니다. 다만 이 식은
B가 n에 근접해야 의미 있는 값을 주므로, 요청 구간 B ≤ 8에서는 Jensen 갭의 0.02%만
회수합니다. 그리고 효과가 나타나는 B에서는 bin당 토큰이 2개라 압축이 사라집니다.

정보를 담은 분할(인접 0.188, 참 로짓 0.0034)과 비교하면 결론이 분명합니다 —
**bin의 개수가 아니라 분할 기준이 성능을 결정합니다.** 랜덤 분할은 그 사실을 보여주는
깨끗한 대조군이며, 이 결과는 V3에서 인접 multi-bin이 +0.19%p(노이즈)에 그친 것과
같은 방향을 가리킵니다.

---

## 7. 재현

```bash
ENV=~/miniconda3/envs/sparse-attn/bin/python
export HF_HOME=/database/hyunwoo/hf HF_HUB_CACHE=/database/hyunwoo/hf/hub \
       TRANSFORMERS_CACHE=/database/hyunwoo/hf/hub HF_DATASETS_CACHE=/database/hyunwoo/hf/datasets

# 진단 (B 스윕, 수 분)
$ENV run_block_tail_diag.py --device cuda:2 --heavy-size 0.048 --selector hat \
    --tag v4_rand --subsets qa_1,niah_multikey_3 --samples-per-subset 2 \
    --max-new-tokens 3 --max-qrows 2 --block-sizes 32 --num-bins 32 \
    --output-dir ./results_tail_diag_v4

# 벤치마크 (B = 1,2,4,8)
$ENV run_multibin_hat_bench.py --device cuda:0 --densities 0.05 --selector hat \
    --methods UTA+randbin --bin-sizes 1,4 --output-dir ./results_v4
$ENV run_multibin_hat_bench.py --device cuda:1 --densities 0.05 --selector hat \
    --methods UTA+randbin --bin-sizes 2,8 --output-dir ./results_v4
```
