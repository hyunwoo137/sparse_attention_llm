# V6 — 확장(expansion) 방법 상세 · density 일반화 · wall-clock

모델 `meta-llama/Llama-3.1-8B-Instruct` (bf16) · RULER 32K, subset당 200샘플 · 셀렉터 HAT

V5에서 확정한 방법을 **구현 수준에서 서술**하고, 그 결과가 density 5%에만 걸려 있다는
한계를 푸는 **일반화 실험**을 정의하며, **실측 wall-clock**과 그 아래에 깔린
**메모리 계층 동작**을 정리합니다.

**통계적 해상도:** subset 점수 SE ≈ 3.5%p, 7-subset 평균 SE ≈ 1.3%p.

---

## 1. 방법 — `UTA + expand`

### 1.1 한 decode step에서 일어나는 일

```
입력:  q (한 토큰), KV 캐시 K,V (길이 N), 캐시된 bin 통계
──────────────────────────────────────────────────────────────────────
T1  선택      sink ∪ local ∪ HAT top-k          -> 인덱스 S,  |S| = ρN
T2  bin 로짓  z̄_b = s · q·k̄_b                   -> bin당 O(d),  B개 bin
T3  bin 분산  σ_b² = s² Σ_d q_d² Var_b(k_d)      -> bin당 O(d)
T4  랭킹      û_b = z̄_b + c·σ_b,  c = 3          -> top-m bin 선택,  m = 8
T5  확장 읽기 top-m bin의 토큰 인덱스 E           -> |E| = m · bin_size = 256
T6  정확 로짓 z_i = s · q·k_i  for i ∈ S ∪ E     -> 1,828개 (전체의 5.6%)
T7  합성      softmax( {z_i}_{i∈S∪E} ∪ {ℓ_b}_{b∉top-m} )
              ℓ_b = z̄_b + log n_b
              출력 = Σ_{i∈S∪E} p_i v_i + Σ_{b∉top-m} p_b v̄_b
```

**bin 통계는 매 step 만드는 것이 아니라 읽습니다.** k̄_b, Var_b(k), v̄_b는 K와 V만의
함수라 query와 무관합니다. prefill에서 한 번 만들고, 디코딩 중에는 새 토큰이 붙을 때
마지막 bin만 증분 갱신합니다.

### 1.2 왜 순환이 없는가

로짓으로 bin을 나누려면 로짓이 필요하고, 그게 우리가 피하려는 full QK입니다. 분리하면:

| 필요한 것 | 순환? | 근거 |
|---|---|---|
| bin **멤버십** | **안 걸림** | 인접 인덱스 32개 런 — 위치만으로 결정 |
| bin 평균 로짓 z̄_b | 안 걸림 | 캐시된 k̄_b와 O(d) 내적 |
| bin 로짓 분산 σ_b² | 안 걸림 | Var_b(k)는 query 무관, 캐싱 가능 |
| **어느 bin을 펼칠지** | 안 걸림 | z̄_b, σ_b만으로 랭킹 |
| 펼친 bin 안의 **참 로짓** | — | 그 256개만 실제 계산 |

### 1.3 왜 되는가 — 보정이 아니라 조준

V2/V3에서 같은 σ_b²를 **질량 보정**에 썼습니다: ℓ_b += log(1 + σ_b²/2). 결과는
단일 bin −2.92%p, bin 32 −0.01%p. σ_b²가 클수록 보정이 발산해서, outlier가 있는
bin일수록 더 틀립니다.

V5는 같은 통계를 **랭킹 신호**로 씁니다: +4.10%p. 차이의 이유는 진단에 있습니다 —
tail의 손상은 확산된 질량이 아니라 **소수의 거대 로짓 토큰**입니다.

| 진단 지표 (15,360행, 5% HAT) | 값 |
|---|---|
| tail이 가진 attention 질량 (중앙값) | 19.5% |
| tail 최대 로짓 − tail 평균 (`headroom`) | **13.28 nats** (≈ e¹³ = 59만 배) |
| 놓친 heavy 토큰이 존재하는 행 | 100.0% |

bin 32개 안에 13 nats 튄 토큰 하나가 섞이면 **bin 질량 ≈ 그 토큰 하나**입니다.
평균 벡터 v̄_b로 대표하는 순간 그 value가 31개에 희석되고, 이건 보정항으로 못 고칩니다.
**bin을 꺼내서 정확히 계산하면** 희석 자체가 사라집니다.

그래서 σ_b는 "이 bin에 뭔가 튀었다"는 **탐지기**로는 유효하고, "얼마나 튀었는지"를
**정량화**하는 데는 무효였던 것입니다.

### 1.4 c와 m

| m | 추가 토큰 | 진단 오차 | 벤치 평균 |
|---|---|---|---|
| 0 | 0 | 0.2175 | 81.25 |
| 4 | 128 | 0.0919 | 84.95 |
| **8** | **256** | **0.0854** | **85.35** |
| 16 | 512 | 0.0776 | 85.50 |

**m = 8이 운영점**입니다. m16은 예산 2배에 +0.15%p로 노이즈 안입니다.
c는 진단에서 c=0 → 3으로 단조 개선(−30.7% → −48.7%, m=1 기준)이고 c=3에서 포화입니다.
**c는 벤치마크에서 스윕하지 않았습니다** — §2의 남은 과제입니다.

### 1.5 코드

`sparse_attention_hub/sparse_attention/uta_attention/multibin.py`:
`expand_m`, `expand_c`, `expand_rank` 세 필드. `expand_m=0`이면 확장 없는 multibin과
**비트 단위로 동일**하므로 기존 결과는 영향받지 않습니다.

```python
UTAMultiBinConfig(masker_configs=..., bin_mode="equalcount", bin_size=32,
                  kappa_mode="none", var_mode="diag",
                  expand_m=8, expand_c=3.0, expand_rank="bound")
```

`expand_rank="random"`은 **같은 예산을 랭킹 없이 쓰는 대조군**입니다. V5에서 이 대조군이
81.81(UTA 81.25 대비 +0.56)에 그쳐, 이득 +4.10 중 **87%가 랭킹에서** 온다는 것이
쌍별 t = 2.64, p = 0.039로 확인됐습니다.

---

## 2. density 일반화 실험 — 무엇을, 왜

### 2.1 문제

지금까지의 **모든** 결론이 density 5% 한 점에 걸려 있습니다. 긍정(expand +4.10%p)과
부정(jensen −2.92%p, multibin +0.19%p, 랜덤 분할 0.02%) 양쪽 모두입니다.

이게 위험한 이유는 tail의 성질이 density에 강하게 의존하기 때문입니다. 밀도가 낮으면
tail이 커지고 질량도 커져서 확장할 대상이 많아지지만 m개로는 부족할 수 있고, 밀도가
높으면 tail 질량 자체가 작아져 확장의 여지가 사라집니다. 즉 **5%가 우연히 단 지점**일
가능성을 배제하지 못합니다.

### 2.2 설계

density **3%**와 **10%**에서 V5의 핵심 3개를 그대로 반복합니다. 셀렉터·bin 크기·c·m은
5%와 동일하게 고정하고 **density만** 바꿉니다.

| 방법 | 역할 |
|---|---|
| `UTA` | 기준선 — 확장 없는 tail 프록시 |
| `UTA+expand(m8, c3)` | 방법 |
| `UTA+expandrand(m8)` | **대조군** — 같은 예산, 랭킹 없음 |

3개 × 2 density = 6런, GPU 2/3에 density별로 배치(각 3런).

### 2.3 무엇을 판정하는가

| 관측 | 해석 |
|---|---|
| 3%와 10% 모두에서 expand > rand > UTA | 결론이 density-robust. 논문 주장 성립 |
| 10%에서 격차 소멸 | 예상된 결과 — tail 질량이 작아 확장할 게 없음. 방법의 **적용 범위**를 정의 |
| **3%에서 격차 소멸** | 심각. 5%가 우연한 단 지점이라는 뜻 |
| 랜덤 대조군이 어느 density에서든 expand를 따라잡음 | "랭킹이 일한다"는 주장 붕괴 |

세 번째가 진짜 위험입니다. 3%에서는 HAT이 이미 무너지고(HAT@3pct = 63.48) tail 질량이
훨씬 크므로, m=8(256토큰)로는 부족해서 확장이 안 먹힐 수 있습니다. 그러면 **m을 density에
비례시켜야 한다**는 설계 변경이 필요합니다.

### 2.4 상태

<!-- AUTO:v6 START -->
_실행 중: cuda:2 = density 3%, cuda:3 = density 10%. 각 3개 방법 × 7 subset._
<!-- AUTO:v6 END -->

기존 3% 참고선 (동일 프로토콜, `results_multibin_hat`): `HAT@3pct` 63.48,
`UTA-HAT@3pct` 74.35, `MultiBin32-HAT@3pct` 75.65, `vAttention-HAT@3pct` 54.58.

---

## 3. full QK는 필수가 아닙니다

질문: 현재 구현이 full QK를 쓰는데, 이게 알고리즘적으로 필요한가?

**아닙니다.** `_multibin_output`에서 `raw`(전체 QK 행렬)를 소비하는 곳은 정확히 세 군데이고
셋 다 대체 가능합니다.

| 위치 | 코드 | 왜 필요 없나 |
|---|---|---|
| `valid` | `raw > NEG` | causal/padding 유효성. **attention_mask에서 직접 유도** |
| `row_max` | `raw.amax(-1)` | softmax 수치 안정용 shift. **max(선택 로짓, 확장 로짓, ℓ_b)**로 충분 |
| 분자/분모 | `exp(raw − row_max) * exact` | `exact = S ∪ E`. **1,828개 위치에서만** 로짓 필요 |

`_bin_ids_score`만 tail 전체 로짓을 요구하는데 그건 `bin_mode="score"`(도달 불가 상한)
전용이고 우리 경로(`equalcount`)에는 없습니다.

따라서 **배포 구현은 로짓의 5.6%만 계산합니다.** 아래 wall-clock은 그 경로
(`expand_gather`)를 실제로 구현해서 잰 것이며, full QK 단계를 빼고 차감한 값이 아닙니다 —
gather의 인덱싱·비연속 접근 비용이 차감으로는 안 잡히기 때문입니다.

---

## 4. Wall-clock 실측

`bench_expand_latency.py`, H100 NVL, K=32768, Hq/Hkv = 32/8, d=128, ρ=5%, m=8, bf16.
CUDA event, warmup 25 / 측정 150회, 중앙값 ms. **유휴 GPU 단독 실행.**

| 변형 | B=1 | B=8 | B=32 | B=32 vs dense |
|---|---|---|---|---|
| dense (full attention) | **0.121** | **0.414** | **1.594** | 1.00× |
| expand_fullqk (V5 현재 구현) | 0.274 | 0.549 | 1.904 | 1.19× |
| topk_only (HAT, tail 폐기) | 0.138 | 0.738 | 2.620 | 1.64× |
| **vAttention** (topk ρ/2 + 샘플링) | 0.631 | — | **2.815** | **1.77×** |
| uta (topk + 프록시 1개) | 0.341 | 0.873 | 2.840 | 1.78× |
| **expand_gather (V5 배포 형태)** | 0.623 | 1.066 | **3.411** | **2.14×** |

### 4.1 32K·H100(VRAM 상주)에서는 dense가 가장 빠릅니다 — vAttention 포함

**sparse 계열 전부가 dense보다 느립니다.** vAttention도 1.77×로 예외가 아닙니다.
유효 대역폭이 이유입니다 (B=32, payload ÷ 시간):

| | payload | 시간 | 유효 대역폭 |
|---|---|---|---|
| dense | 4,295 MB | 1.594 ms | **2.69 TB/s** (피크의 69%) |
| topk_only | 825 MB | 2.620 ms | **0.32 TB/s** (피크의 8%) |

dense는 KV를 완전히 연속으로 스트리밍해 대역폭의 69%를 씁니다. sparse는 5%만 읽지만
흩어진 접근이라 8%에 그칩니다. **5% ÷ 8% > 1이므로 적게 읽는 것이 순 손해입니다.**
이건 우리 방법 고유의 문제가 아니라 sparse attention 일반의 문제입니다.

### 4.2 이 측정의 한계

**PyTorch 연산 조합은 fused 커널을 대표할 수 없습니다.** `index_select`가 모은 KV를
DRAM에 실체화한 뒤 다시 읽습니다 — B=32에서 825 MB 쓰고 825 MB 다시 읽는 왕복이
추가됩니다. 진짜 커널은 흩어진 K/V를 레지스터·shared memory로 직접 읽어 소비하므로
이 왕복이 없습니다. `expand_fullqk`(1.19×)가 `expand_gather`(2.14×)보다 빠른 역전이
그 증거입니다 — full QK는 cuBLAS GEMM 한 번이라 실체화 왕복이 없습니다.

따라서 **위 표는 "PyTorch 수준에서 표현했을 때"의 수치이지 방법의 하한이 아닙니다.**

---

## 5. Offload 영역 — 여기서도 졌습니다

`bench_offload_latency.py`. KV 캐시를 pinned host DRAM에 두고 PCIe로 가져옵니다.
**host-side gather 비용을 부과**했고(흩어진 행을 모으는 것도 실제 비용이므로),
GPU 연산은 제외했습니다(버스가 수십 배 지배적이라 섞으면 비교가 흐려짐).

| case | bytes MB | B=1 ms | B=8 ms | B=8 vs dense |
|---|---|---|---|---|
| dense (KV 캐시 전체) | 1073.7 | 2.812 | **19.340** | 1.00× |
| topk_only (선택 KV) | 206.2 | 3.481 | 47.524 | 2.46× |
| vAttention (topk ρ/2 + 샘플) | 206.2 | 3.409 | 47.360 | 2.45× |
| ours m=8 (선택+확장+통계) | 287.6 | 3.945 | 54.998 | 2.84× |

**V6 초안의 "offload에서 33배 유리" 예측은 틀렸습니다.** 그 계산은 PCIe 전송 시간만
세고 **흩어진 행을 누가 모으는가**를 빠뜨렸습니다. 실측하면 그 gather가 병목입니다:
dense는 1073.7 MB를 19.34 ms에 옮겨 **55.5 GB/s**(PCIe Gen5 실측 상한 근처)를 내는데,
sparse는 206.2 MB를 47.5 ms에 옮겨 **4.3 GB/s**에 그칩니다. 바이트는 5배 적은데
시간은 2.5배 더 듭니다.

이 측정에도 천장이 있습니다 — CPU `index_select`가 단일 경로로 모읍니다. 실제 오프로딩
엔진은 GPU가 직접 zero-copy/UVA로 흩어진 행을 읽거나 DMA descriptor list를 씁니다.
**그건 PyTorch로 표현할 수 없어 측정하지 못했습니다.**

---

## 5b. 비용 회계 정정 — 우리가 vAttention보다 **더** 읽습니다

이전 보고서(V2 §4, V5 §7)의 `analyze_cost.py` 기반 수치에 두 가지 오류가 있었습니다.

| 오류 | 실제 |
|---|---|
| `share = batch × gqa = 128`으로 bin 통계를 상각 | bin 통계는 **시퀀스별**이라 배치 공유 불가. 공유는 **GQA 그룹 g=4**뿐 |
| "top-k 위에 추가되는 양"만 비교 | 두 방법의 **top-k 크기가 다름** — 같은 밀도에서 vAttention은 절반만 selector에 씀 |

정정된 회계 (query head·layer당, ρ=5%, K=32768, KV bf16, bin 통계 int8):

| | 토큰 | KB |
|---|---|---|
| ours 선택 | 1,636 | 837.6 |
| ours 확장 (m=8) | 256 | 131.1 |
| ours bin 통계 ÷ g=4 | — | 93.3 (원본 373.2) |
| **ours 합계** | **1,892** | **1,062.0** |
| vAtt selector (ρ/2) | 817 | 418.3 |
| vAtt 샘플 (ρ/2) | 819 | 419.3 |
| **vAtt 합계** | **1,636** | **837.6** |

**ours / vAttention = 1.27배 바이트, 1.16배 토큰.** "0.31배로 더 싸다"는 이전 주장은
철회합니다. 정확한 서술은 이것입니다 — **같은 밀도 예산에서 우리는 27% 더 읽고
16.65%p 더 정확합니다** (85.35 vs 68.70).

접근 *패턴*의 이점은 남아 있습니다: 확장 토큰은 연속 32토큰 블록 8개이고 bin 통계는
query 무관(g=4 공유)인 반면, vAttention의 819개 샘플은 query마다 다른 개별 랜덤
접근입니다. 다만 §4·§5가 보여주듯 **그 이점은 아직 wall-clock으로 실현되지 않았습니다.**

---

## 5c. VRAM / DRAM / GPU-CPU 데이터 흐름

| 데이터 | 위치 | 크기 (Llama-3.1-8B, 32K, 시퀀스 1개) |
|---|---|---|
| 모델 가중치 | **VRAM (HBM)** | 16.1 GB (bf16) |
| **KV 캐시** | **VRAM (HBM)** | **4.295 GB** = 32 layer × 8 kv-head × 32768 × 128 × 2(K,V) × 2B |
| bin 통계 | **VRAM (HBM)** | **95.5 MB** (int8) = KV의 **2.2%** |
| 토큰 ID, 샘플링 결과 | 호스트 DRAM | 수 KB |

H100 NVL: HBM3 94 GB @ **약 3.9 TB/s**, **L2 50 MB**, PCIe Gen5 ×16 **약 64 GB/s**
(실측 55.5 GB/s).

```
[호스트 DRAM]  토큰 ID  ── PCIe(수 바이트) ──▶  [VRAM]
                                                  │
[VRAM] 가중치 16.1 GB ───────────────────────────▶ SM
[VRAM] KV 캐시                                     │
   ├ dense : 4.295 GB 전부 연속 스트리밍 ──────────▶ SM   (2.69 TB/s 달성)
   └ ours  : 선택 1,636 + 확장 256 토큰, 흩어짐 ───▶ SM   (0.32 TB/s 달성)
[VRAM] bin 통계 95.5 MB ── 층당 3 MB, L2 상주 ────▶ SM   (g=4 query head 공유)
                                                  │
[VRAM] 샘플링된 토큰 1개 ── PCIe(수 바이트) ──▶ [호스트 DRAM]
```

**PCIe는 디코딩 루프에 사실상 등장하지 않습니다** — 스텝당 토큰 ID 몇 바이트뿐입니다.
모델과 KV가 VRAM에 상주하는 한 병목은 전적으로 **HBM ↔ SM**이고, 거기서는 §4.1대로
**접근 패턴이 바이트 수보다 중요합니다.** KV가 VRAM을 벗어나면 병목이 PCIe로 옮겨가지만,
§5가 보여주듯 이번엔 **흩어진 행을 모으는 비용**이 새 병목이 됩니다.

---

## 6. 남은 것

1. **density 일반화 완료** (§2, 진행 중) — 모든 결론이 5% 한 점에 걸려 있음.
2. **fused 커널** — §4.2의 실체화 왕복 제거 없이는 트래픽 우위가 실현되지 않음.
3. **c 스윕을 벤치마크에서** — c=3은 진단 격자에서만 고른 값.
4. **bin 크기 × m 교차** — 접근 단위를 키우는 유일한 축이자 MAC 지배항(bin 통계)을
   줄이는 축.
5. **KV 오프로딩 실측** (§5.4) — 이 방법이 이기는 영역의 직접 측정.

---

## 7. 재현

```bash
ENV=~/miniconda3/envs/sparse-attn/bin/python
export HF_HOME=/database/hyunwoo/hf HF_HUB_CACHE=/database/hyunwoo/hf/hub \
       TRANSFORMERS_CACHE=/database/hyunwoo/hf/hub HF_DATASETS_CACHE=/database/hyunwoo/hf/datasets \
       HF_HUB_OFFLINE=1

# density 일반화
$ENV run_multibin_hat_bench.py --device cuda:2 --densities 0.03 --selector hat \
    --methods UTA,UTA+expand,UTA+expandrand --bin-sizes 8 --expand-c 3.0 \
    --output-dir ./results_v6
$ENV run_multibin_hat_bench.py --device cuda:3 --densities 0.10 --selector hat \
    --methods UTA,UTA+expand,UTA+expandrand --bin-sizes 8 --expand-c 3.0 \
    --output-dir ./results_v6

# wall-clock (유휴 GPU에서 단독 실행할 것 — 공유 GPU에서는 무의미)
for B in 1 8 32; do
  $ENV bench_expand_latency.py --device cuda:1 --batch $B --json results/latency_b$B.json
done
```
