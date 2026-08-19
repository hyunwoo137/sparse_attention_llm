# V5 — bin 확장: tail 통계를 보정이 아니라 **조준**에 쓰기

모델 `meta-llama/Llama-3.1-8B-Instruct` (bf16) · RULER 32K, subset당 200샘플
**density 5% 고정 · 셀렉터 HashAttention(HAT)** — 모든 방법이 동일 셀렉터·동일 밀도

**통계적 해상도:** subset 점수 SE ≈ 3.5%p, 7-subset 평균 SE ≈ 1.3%p.

---

## 1. 한 줄 요약

tail의 bin 통계 σ_b를 **질량 보정 계수**로 쓰면 해롭거나 무의미했고(V2/V3),
같은 통계를 **어느 bin을 정확히 계산할지 고르는 랭킹 신호**로 쓰니
**81.25 → 85.35 (+4.10%p)**, dense와의 격차를 5.79 → 1.69%p로 줄였습니다.

---

## 2. 최종 Ablation (5%, HAT, 7/7 subset)

| 방법 | 추가 토큰 | 평균 | qa_1 | qa_2 | vt | fwe | nk_2 | nk_3 | nmv |
|---|---|---|---|---|---|---|---|---|---|
| dense@5% (full attention) | — | **87.04** | 79.0 | 50.0 | 95.2 | 93.33 | 95.0 | 100.0 | 96.75 |
| **UTA+expand m16 c3** | 512 | **85.50** | 78.5 | 49.0 | 93.0 | 90.00 | 94.5 | 98.0 | 95.50 |
| **UTA+expand m8 c3** | **256** | **85.35** | 78.0 | 49.0 | 92.8 | 89.50 | 94.5 | 98.0 | 95.62 |
| **UTA+expand m4 c3** | 128 | **84.95** | 78.0 | 49.0 | 91.8 | 88.83 | 93.0 | 97.5 | 96.50 |
| **UTA+expandrand m8** *(대조군)* | 256 | **81.81** | 76.0 | 49.5 | 89.7 | 85.50 | 88.0 | 88.5 | 95.50 |
| UTA+multibin(b32) | 0 | 81.44 | 78.0 | 48.0 | 90.5 | 85.83 | 86.5 | 85.5 | 95.75 |
| UTA+multibin+jensen(b32) | 0 | 81.43 | 76.0 | 51.0 | 90.2 | 86.17 | 85.0 | 86.0 | 95.62 |
| UTA 단독 | 0 | 81.25 | 74.5 | 48.0 | 90.7 | 85.67 | 85.5 | 88.5 | 95.88 |
| UTA+jensen | 0 | 78.33 | 73.5 | 48.0 | 90.7 | 84.50 | 78.0 | 78.5 | 95.12 |
| HAT 단독 | 0 | 71.55 | 68.0 | 44.5 | 81.9 | 87.33 | 72.5 | 50.5 | 96.12 |
| vAttention(HAT) | — | 68.70 | 66.0 | 46.5 | 85.1 | 91.33 | 64.5 | 34.5 | 93.00 |

참고선 (셀렉터가 완벽할 때): `UTA(oracle)@5pct` **86.04**, `vAttention(oracle)@5pct` 86.72.
**HAT + 확장(85.35)이 오라클 셀렉터 UTA(86.04)에 0.7%p까지 접근합니다.**

---

## 3. 이득의 85%는 예산이 아니라 랭킹에서 옵니다

`expandrand`는 **완전히 동일한 예산(256토큰)**을 무작위로 고른 bin에 씁니다.

| | 평균 | UTA 대비 |
|---|---|---|
| UTA 단독 | 81.25 | — |
| **UTA+expandrand m8** (예산만) | 81.81 | **+0.56** |
| **UTA+expand m8 c3** (예산+랭킹) | 85.35 | **+4.10** |

쌍별 차이 (bound − random):

| qa_1 | qa_2 | vt | fwe | nk_2 | nk_3 | nmv | 평균 |
|---|---|---|---|---|---|---|---|
| +2.00 | −0.50 | +3.10 | +4.00 | +6.50 | **+9.50** | +0.12 | **+3.53** |

SE 1.337, **t = 2.64 (df=6), p = 0.039**. 7개 중 6개에서 같은 방향이고, 유일한 예외
qa_2는 dense 50.0에 모든 방법이 붙어 있는 천장 subset입니다.

---

## 4. 방법

```
1. sink + local + HAT top-k 로 heavy hitter 선택          (기존과 동일)
2. 나머지(tail)를 인접 인덱스 32토큰 런으로 분할           (bin_mode="equalcount")
3. bin별 key-side 통계로 랭킹:  û_b = s·q·k̄_b + 3·σ_b
     σ_b² = s²·Σ_d q_d²·Var_b(k_d)                       (대각, query 무관, 캐싱 가능)
4. 상위 m=8개 bin을 프록시 집합에서 제거하고 그 토큰들을 exact 계산
5. 나머지 bin은 평균 프록시로, heavy·확장 토큰과 하나의 softmax로 합성
```

**순환이 없습니다.** k̄_b와 Cov_b(K)는 query와 무관해 캐싱되고, 랭킹은 O(d)/bin
내적입니다. 참 로짓은 확장된 bin의 토큰에 대해서만 계산합니다.

`kappa_mode="none"` — jensen 보정은 쓰지 않습니다(§6).

---

## 5. 왜 되는가 — E0/E1/E2 진단

**E0 (동점 구조).** HAT 점수는 32비트 ±1 시그니처의 내적이라 값이 33개뿐입니다.
경계 레벨 τ의 중앙값이 **0**(= 무작위 수준 일치)이고, 예산의 **32%**가 1,150개짜리
동점 그룹에서 인덱스 순서로 임의 충당됩니다. 그러나 놓친 tail 질량 중 동점 그룹 안에
있는 비율은 **12%**뿐이고, 가장 무거운 놓친 토큰의 **62.5%는 τ보다 아래**에 있습니다
→ **동점 처리는 원인이 아니고, HAT은 애초에 순위를 못 매깁니다.**

**E1/E2 (확장).** 15,360행, dense 출력 대비 상대오차 중앙값 (기준 UTA = 0.21745):

| m | 추가 토큰 | E1 oracle (천장) | **E2 bound c=3** | 랜덤 대조 |
|---|---|---|---|---|
| 1 | 32 | 0.0920 | 0.1116 | 0.1879 |
| 4 | 128 | 0.0818 | 0.0919 | 0.1875 |
| **8** | **256** | **0.0758** | **0.0854** | 0.1870 |
| 16 | 512 | 0.0677 | 0.0776 | 0.1847 |
| 64 | 2048 | 0.0450 | 0.0542 | 0.1749 |

- **bin 하나(32토큰)만 정확히 계산해도 오차가 절반 이하**(−57.7%)로 떨어집니다.
  tail의 손상은 확산된 질량이 아니라 소수의 거대 로짓 토큰입니다.
- **E2가 치팅 상한의 88%를 회수합니다** (m=8: 0.0854 vs 0.0758). 참 로짓 없이.
- **σ_b 항이 결정적입니다**: c=0 → −30.7%, c=3 → −48.7% (m=1 기준).

---

## 6. 같은 통계, 정반대 결과

| σ_b² 사용법 | 결과 |
|---|---|
| **질량 보정** `ℓ_b += log(1+σ_b²/2)` (jensen) | 단일 bin **−2.92%p**, bin32 **−0.01%p** |
| **랭킹 신호** `û_b = z̄_b + 3σ_b` (expand) | **+4.10%p** |

"이 bin에 뭔가 튄 게 있다"는 정보는 처음부터 유효했습니다. 틀린 것은 그것을
**보정 계수로 환산한 것**입니다 — outlier가 든 bin의 질량을 고치려 하지 말고,
그 bin을 꺼내서 정확히 계산하면 됩니다. σ_b²가 클수록 보정은 발산하지만
(V2에서 j1이 붕괴하고 j2도 단일 bin에서 −2.92%p), 랭킹은 정확히 그 bin을 지목합니다.

---

## 7. 비용

decode query·head당, top-k 위에 **추가되는** 양. K=32768, d=128, ρ=5%, bin 32,
batch=32, GQA=4.

| | MAC | vAtt 대비 | bytes/query | vAtt 대비 |
|---|---|---|---|---|
| vAttention 샘플링 (819샘플) | 209.7K | 1.00× | 419.3K **(랜덤)** | 1.00× |
| **UTA+expand m8** | 841.9K | 4.02× | **131.1K (연속 블록 8개)** | **0.31×** |
| UTA+expand m16 | 907.4K | 4.33× | 262.1K | 0.63× |

- **트래픽이 vAttention의 0.31배**이고, 819개 랜덤 접근이 아니라 **연속 32토큰 블록
  8개**라 메모리 친화적입니다. 디코딩이 메모리 바운드라는 점에서 결정적입니다.
- **MAC은 4.02배 초과**입니다. 다만 확장 자체는 65.5K로 싸고, 지배항은 bin 통계 373K와
  선택 토큰 차감 403K입니다 — 둘 다 확장과 무관한 기존 항목입니다.
- **m=8이 운영점**입니다. m16은 예산 2배에 +0.15%p(85.50 vs 85.35)로 노이즈 안입니다.

---

## 8. 한계

- **wall-clock 미측정.** 구현은 여전히 merge와 tail 마스크를 위해 full QK를 한 번
  계산하는 프로토타입입니다. §7은 해석적 회계입니다.
- **MAC 예산 초과(4.02×)가 남아 있습니다.** bin 크기를 키우면 통계 항 373K가 줄지만
  확장 품질에 미치는 영향은 미측정입니다.
- **density 5% 한 점**에서만 측정했습니다. 3%/10%로의 일반화는 미확인입니다.
- **c=3은 진단 격자에서 고른 값**이며 벤치마크에서 스윕하지 않았습니다.

---

## 9. 재현

```bash
ENV=~/miniconda3/envs/sparse-attn/bin/python
export HF_HOME=/database/hyunwoo/hf HF_HUB_CACHE=/database/hyunwoo/hf/hub \
       TRANSFORMERS_CACHE=/database/hyunwoo/hf/hub HF_DATASETS_CACHE=/database/hyunwoo/hf/datasets

# 진단 (E0 동점구조 + E1/E2 확장 격자)
$ENV run_block_tail_diag.py --device cuda:0 --heavy-size 0.048 --selector hat \
    --tag e12_hat --subsets qa_1,niah_multikey_3 --samples-per-subset 2 \
    --max-new-tokens 3 --max-qrows 2 --block-sizes 32 --num-bins 32 \
    --output-dir ./results_tail_diag_e12

# 벤치마크 (GPU 4개 병렬)
$ENV run_multibin_hat_bench.py --device cuda:0 --densities 0.05 --selector hat \
    --methods UTA+expand --bin-sizes 4  --expand-c 3.0 --output-dir ./results_v5
$ENV run_multibin_hat_bench.py --device cuda:1 --densities 0.05 --selector hat \
    --methods UTA+expand --bin-sizes 8  --expand-c 3.0 --output-dir ./results_v5
$ENV run_multibin_hat_bench.py --device cuda:2 --densities 0.05 --selector hat \
    --methods UTA+expand --bin-sizes 16 --expand-c 3.0 --output-dir ./results_v5
$ENV run_multibin_hat_bench.py --device cuda:3 --densities 0.05 --selector hat \
    --methods UTA+expandrand --bin-sizes 8 --output-dir ./results_v5
```

`expand_m=0`은 확장 없는 multibin과 비트 단위로 동일하므로 기존 결과는 영향받지 않습니다.

---

## 10. 다음

1. **density 일반화** — 3%/10%에서 `UTA / UTA+expand(m8) / expandrand(m8)` 3개.
   부정 결과(jensen·multibin)와 긍정 결과(expand) 양쪽 모두 5% 한 점에 걸려 있습니다.
2. **c 스윕을 벤치마크에서** — c ∈ {1, 2, 3, 5}. 진단은 c=3에서 포화였으나 점수 기준
   최적점은 다를 수 있습니다.
3. **bin 크기 × m 교차** — MAC 지배항인 bin 통계 373K를 줄이는 유일한 축.
4. **wall-clock 실측** — 해석적 회계를 실측으로 대체.
