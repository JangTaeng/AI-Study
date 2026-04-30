# 📘 Attention Is All You Need (Transformer) 완전 분석

> Vaswani et al., NeurIPS 2017
> 논문 링크: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

비전공자를 위한 트랜스포머 논문 분석 문서입니다. 끝까지 따라오면 면접에서 트랜스포머를 술술 설명할 수 있게 됩니다.

---

## 📑 목차

1. [Abstract (3줄 요약)](#1-abstract-3줄-요약)
2. [Introduction + Conclusion 요약](#2-introduction--conclusion-요약)
3. [핵심 구조 설명 ⭐](#3-핵심-구조-설명-)
4. [구조 시각화](#4-구조-시각화)
5. [논문 ↔ 코드 연결 (PyTorch)](#5-논문--코드-연결-pytorch)
6. [데이터 흐름 추적 (Shape 변화)](#6-데이터-흐름-추적-shape-변화)
7. [면접 예상 질문 (5개)](#7-면접-예상-질문-5개)

---

## 1. Abstract (3줄 요약)

| 구분 | 내용 |
|---|---|
| **문제** | 기존 번역 모델(RNN/LSTM)은 단어를 하나씩 순서대로 처리해야 해서 학습이 느리고 병렬화가 불가능했다. |
| **방법** | 순환(recurrence)과 합성곱(convolution)을 완전히 제거하고, 오직 **어텐션(Attention)**만으로 작동하는 새로운 구조 "Transformer"를 제안했다. |
| **결과** | WMT 2014 영어→독일어 번역에서 BLEU 28.4로 SOTA를 달성했고, 학습 시간도 기존 모델 대비 훨씬 빨라졌다 (8 GPU로 12시간 ~ 3.5일). |

---

## 2. Introduction + Conclusion 요약

### 🤔 이 논문이 왜 나왔는지

2017년 당시 번역, 언어 모델링 같은 **시퀀스(순서가 있는 데이터)** 작업의 최강자는 **RNN/LSTM/GRU** 였습니다. 인코더-디코더 구조에 어텐션을 살짝 얹어 쓰는 게 대세였죠.

### ❌ 기존 방식(RNN)의 문제점

1. **순차 처리의 본질적 한계**
   - RNN은 t번째 단어를 처리하려면 t-1번째 결과(hidden state $h_{t-1}$)가 반드시 먼저 나와야 합니다.
   - 즉, **단어를 하나씩 차례대로** 계산해야 하므로 GPU의 병렬 연산 능력을 살릴 수 없습니다.

2. **긴 문장에서 정보 손실**
   - 문장이 길어질수록 앞쪽 단어 정보가 뒤로 갈수록 흐려집니다 (장기 의존성 문제).

3. **거리에 비례한 연산량**
   - 멀리 떨어진 두 단어의 관계를 학습하려면 그 거리만큼 연산을 거쳐야 합니다.

### ✅ 이 논문의 개선

- **재귀(RNN) 완전 제거** → 모든 단어를 **동시에(병렬로)** 처리
- **Self-Attention** 도입 → 모든 단어 쌍의 관계를 **한 번에** 계산 (거리 무관)
- **결과**: 더 빠른 학습 + 더 좋은 번역 품질

> 💡 **핵심 한 줄**: "RNN아 안녕, 어텐션 하나면 충분해!"

---

## 3. 핵심 구조 설명 ⭐

> (핵심) 표시: 이 논문에서 새롭게 등장한 핵심 개념

### 🔑 (핵심) Self-Attention (자기 어텐션)

**1) 이게 뭔지 (비유)**

교실에서 한 학생("making"이라는 단어)이 친구들 전체를 둘러보고 "내가 누구랑 가장 관련 있지?"를 스스로 판단하는 것입니다. "making"은 멀리 떨어진 "more difficult"를 보면서 "아, 나는 저 단어들이랑 짝이구나!"를 깨닫습니다.

즉, **한 문장 안에서 단어들끼리 서로를 참고**하는 메커니즘입니다.

**2) 입력값**
- Query (Q), Key (K), Value (V) — 세 개 모두 같은 입력에서 만들어짐 (그래서 "Self")
- Shape: `(batch, seq_len, d_model)` → 각각을 선형변환해서 `(batch, seq_len, d_k)`

**3) 출력값**
- 입력과 같은 shape: `(batch, seq_len, d_model)`
- 각 단어가 "다른 단어들의 정보를 가중평균으로 흡수한 새로운 벡터"

**4) 하이퍼파라미터**
- `d_model = 512` (단어 벡터 차원)
- `d_k = d_v = 64` (Q, K, V 차원)

---

### 🔑 (핵심) Scaled Dot-Product Attention

**1) 이게 뭔지 (비유)**

도서관에서 책을 찾는 과정과 같습니다.
- **Query** = "내가 찾는 주제" (예: "사과")
- **Key** = "책마다 붙은 주제 라벨"
- **Value** = "책의 실제 내용"

Query와 모든 Key의 **유사도(점수)**를 구하고 → 가장 비슷한 책의 Value를 많이 가져오는 방식입니다.

**2) 수식**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- $QK^T$: Query와 Key의 내적(유사도 점수)
- $\sqrt{d_k}$로 나눔: 점수가 너무 커지면 softmax가 극단적으로 쏠려서 **gradient가 0에 가까워지는 문제**를 방지 (이게 "Scaled"의 의미!)
- softmax: 점수를 0~1 확률로 변환 (다 더하면 1)
- 마지막에 V를 곱함: 확률로 가중평균

**3) 입력값**
- Q: `(batch, seq_len, d_k)`
- K: `(batch, seq_len, d_k)`
- V: `(batch, seq_len, d_v)`

**4) 출력값**
- `(batch, seq_len, d_v)`

**5) 하이퍼파라미터**
- $\sqrt{d_k}$ (스케일링 팩터)

---

### 🔑 (핵심) Multi-Head Attention

**1) 이게 뭔지 (비유)**

한 사람이 책을 읽는 게 아니라, **8명의 전문가**가 각자 다른 관점에서 책을 읽고 의견을 모으는 것입니다.
- 1번 헤드: 문법 관계 분석
- 2번 헤드: 주어-동사 관계 분석
- 3번 헤드: 멀리 떨어진 단어 관계 분석
- ...

각자의 관점을 모아서 더 풍부한 표현을 만듭니다.

**2) 작동 방식**

- 입력 Q, K, V를 **h개(=8개)**의 다른 선형변환으로 쪼개기
- 각 헤드에서 독립적으로 Scaled Dot-Product Attention 수행
- 결과들을 **이어붙이기(Concat)** → 다시 선형변환

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

**3) 입력값**
- Q, K, V: `(batch, seq_len, d_model=512)`

**4) 출력값**
- `(batch, seq_len, d_model=512)`

**5) 하이퍼파라미터**
- `h = 8` (헤드 개수)
- `d_k = d_v = d_model / h = 64`

---

### 🔑 (핵심) Positional Encoding (위치 인코딩)

**1) 이게 뭔지 (비유)**

RNN은 단어를 순서대로 받으니 자동으로 "몇 번째 단어"인지 압니다. 하지만 트랜스포머는 모든 단어를 **동시에** 처리하기 때문에 순서 정보가 사라집니다.

그래서 **각 단어에 "좌석 번호표"를 붙여줘야** 합니다. 그게 Positional Encoding입니다!

**2) 어떻게 만드는지**

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$

$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

- 짝수 차원은 sin, 홀수 차원은 cos
- 각 위치마다 고유한 패턴이 만들어짐 → 모델이 위치를 인식 가능
- 학습되지 않는 고정값 (Sinusoidal)

**3) 입력값**
- 단어 임베딩: `(batch, seq_len, d_model)`

**4) 출력값**
- 위치 정보가 더해진 벡터: `(batch, seq_len, d_model)` (덧셈이라 shape 그대로)

**5) 하이퍼파라미터**
- 없음 (수식 고정)

---

### Encoder (인코더) — 입력 문장 이해 담당

**1) 비유**

영어 문장을 읽고 **의미를 압축한 메모**를 만드는 통역사.

**2) 구조 (N=6번 반복)**

- Multi-Head Self-Attention
- Add & Norm (잔차 연결 + LayerNorm)
- Feed-Forward Network
- Add & Norm

**3) 입력값**: `(batch, src_seq_len, d_model)`

**4) 출력값**: `(batch, src_seq_len, d_model)` — 디코더로 전달

**5) 하이퍼파라미터**: `N = 6`

---

### Decoder (디코더) — 번역문 생성 담당

**1) 비유**

인코더의 메모를 보면서 한 단어씩 독일어로 받아쓰는 작가. 단, **미래 단어는 절대 보면 안 됨** (커닝 금지!)

**2) 구조 (N=6번 반복)**

- **Masked** Multi-Head Self-Attention ⚠️ (마스킹: 미래 단어 가림)
- Add & Norm
- Multi-Head **Encoder-Decoder Attention** (인코더 출력을 K, V로 사용)
- Add & Norm
- Feed-Forward Network
- Add & Norm

**3) 입력값**:
- 디코더 입력: `(batch, tgt_seq_len, d_model)`
- 인코더 출력: `(batch, src_seq_len, d_model)`

**4) 출력값**: `(batch, tgt_seq_len, vocab_size)` — 다음 단어 확률

**5) 하이퍼파라미터**: `N = 6`

---

### Position-wise Feed-Forward Network (FFN)

**1) 비유**

어텐션이 모은 정보를 **각 단어마다 따로따로** 한 번 더 가공하는 작업.

**2) 수식**

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

Linear → ReLU → Linear

**3) 입력/출력**: `(batch, seq_len, 512)` → `(batch, seq_len, 2048)` → `(batch, seq_len, 512)`

**4) 하이퍼파라미터**
- `d_ff = 2048` (중간 차원)

---

### Add & Norm (잔차 연결 + 레이어 정규화)

**1) 비유**

"원본을 잃지 말자!" — 변형된 결과에 **원본을 더해서**(Residual) 정보 손실을 막고, 분포를 안정화(LayerNorm)합니다.

**2) 수식**: `LayerNorm(x + Sublayer(x))`

---

## 4. 구조 시각화

```
                    [입력 문장: "I am a student"]
                              │
                              ▼
                    ┌─────────────────────┐
                    │  Input Embedding    │  (batch, src_len, 512)
                    └─────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │ + Positional Encoding│  순서 정보 주입
                    └─────────────────────┘
                              │
                              ▼
       ┌──────────────────────────────────────────┐
       │         ENCODER (×6 반복)                 │
       │  ┌────────────────────────────────────┐  │
       │  │  Multi-Head Self-Attention          │  │
       │  │  └→ Add & Norm                      │  │
       │  │  Feed-Forward (512→2048→512)       │  │
       │  │  └→ Add & Norm                      │  │
       │  └────────────────────────────────────┘  │
       └──────────────────────────────────────────┘
                              │
                  인코더 출력 (memory)
                  (batch, src_len, 512)
                              │
                              │  ┌────────────────────────┐
                              │  │ [디코더 입력: "<sos> Ich"] │
                              │  └────────────────────────┘
                              │              │
                              │              ▼
                              │   Output Embedding + PE
                              │              │
                              ▼              ▼
       ┌──────────────────────────────────────────┐
       │         DECODER (×6 반복)                 │
       │  ┌────────────────────────────────────┐  │
       │  │  Masked Multi-Head Self-Attention   │  │ ← 미래 가림
       │  │  └→ Add & Norm                      │  │
       │  │  Encoder-Decoder Attention          │  │ ← 인코더 출력 참조
       │  │  └→ Add & Norm                      │  │   (Q는 디코더, K/V는 인코더)
       │  │  Feed-Forward                       │  │
       │  │  └→ Add & Norm                      │  │
       │  └────────────────────────────────────┘  │
       └──────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  Linear (→vocab_size)│
                    └─────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │      Softmax         │
                    └─────────────────────┘
                              │
                              ▼
                  [출력 확률: 다음 단어]
```

### 데이터 형태 변화 한눈에

| 단계 | Shape |
|---|---|
| 입력 토큰 | `(batch, src_len)` |
| 임베딩 후 | `(batch, src_len, 512)` |
| 인코더 통과 후 | `(batch, src_len, 512)` |
| 디코더 통과 후 | `(batch, tgt_len, 512)` |
| Linear 후 | `(batch, tgt_len, vocab_size)` |
| Softmax 후 | `(batch, tgt_len, vocab_size)` (확률) |

---

## 5. 논문 ↔ 코드 연결 (PyTorch)

### 논문 ↔ 코드 매핑 표

| 논문 설명 | PyTorch 코드 |
|---|---|
| "queries, keys, values를 dk, dk, dv로 선형 사영" | `self.W_q = nn.Linear(d_model, d_model)` |
| "QK^T / √dk" | `scores = Q @ K.transpose(-2,-1) / math.sqrt(d_k)` |
| "softmax로 가중치 계산" | `attn = F.softmax(scores, dim=-1)` |
| "마스킹 (디코더에서 -∞)" | `scores.masked_fill(mask==0, -1e9)` |
| "h=8개 헤드를 병렬" | `view(batch, -1, h, d_k).transpose(1,2)` |
| "ReLU 활성화의 FFN" | `nn.Sequential(nn.Linear(512,2048), nn.ReLU(), nn.Linear(2048,512))` |
| "잔차 연결 + LayerNorm" | `x = self.norm(x + sublayer(x))` |
| "sinusoidal positional encoding" | `pe[:,0::2]=sin(...); pe[:,1::2]=cos(...)` |
| "N=6 identical layers" | `nn.ModuleList([EncoderLayer() for _ in range(6)])` |

### 전체 모델 구현 예시

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 1) Scaled Dot-Product Attention
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # 논문 식 (1)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)   # 마스킹
    attn = F.softmax(scores, dim=-1)
    return attn @ V


# 2) Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, h=8):
        super().__init__()
        self.h = h
        self.d_k = d_model // h     # 64
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        B = Q.size(0)
        # (B, seq, d_model) -> (B, h, seq, d_k)  [헤드 분할]
        Q = self.W_q(Q).view(B, -1, self.h, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(B, -1, self.h, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(B, -1, self.h, self.d_k).transpose(1, 2)
        out = scaled_dot_product_attention(Q, K, V, mask)
        # 다시 합치기
        out = out.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.W_o(out)


# 3) Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)   # 짝수 차원
        pe[:, 1::2] = torch.cos(pos * div)   # 홀수 차원
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# 4) Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


# 5) Encoder Layer (한 층)
class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, h=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, h)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 잔차 연결 + LayerNorm
        x = self.norm1(x + self.dropout(self.attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


# 6) Decoder Layer (한 층)
class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, h=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h)     # masked
        self.cross_attn = MultiHeadAttention(d_model, h)    # enc-dec
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        # Q는 디코더, K/V는 인코더 출력
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


# 7) 전체 Transformer
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, N=6, h=8, d_ff=2048):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)
        self.pe = PositionalEncoding(d_model)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, h, d_ff) for _ in range(N)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, h, d_ff) for _ in range(N)])
        self.out = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 인코더
        e = self.pe(self.src_embed(src) * math.sqrt(512))
        for layer in self.encoder:
            e = layer(e, src_mask)
        # 디코더
        d = self.pe(self.tgt_embed(tgt) * math.sqrt(512))
        for layer in self.decoder:
            d = layer(d, e, src_mask, tgt_mask)
        return self.out(d)
```

---

## 6. 데이터 흐름 추적 (Shape 변화)

> 가정: `batch=64, src_len=10, tgt_len=12, d_model=512, h=8, vocab=37000`

### 인코더 흐름

```
입력 토큰 ID:           (64, 10)
  ↓ [Embedding]                       단어를 512차원 벡터로 변환
                        (64, 10, 512)
  ↓ [+ Positional Encoding]           위치 정보 더하기 (shape 그대로)
                        (64, 10, 512)
  ↓ [Multi-Head Attention]            각 단어가 다른 단어 정보를 모음
                        (64, 10, 512)
  ↓ [Add & Norm]                      잔차 연결 + 정규화
                        (64, 10, 512)
  ↓ [Feed-Forward (512→2048→512)]    위치별 비선형 변환
                        (64, 10, 512)
  ↓ [Add & Norm]
                        (64, 10, 512)
  ↓ [×6 반복]
인코더 출력:            (64, 10, 512)  ← 디코더로 전달
```

### Multi-Head Attention 내부 (한 헤드 분할 과정)

```
입력:                   (64, 10, 512)
  ↓ [Linear W_q]                      Q, K, V 각각 만들기
  ↓ [view + transpose]                헤드 분할: 512 = 8 × 64
                        (64, 8, 10, 64)   ← (batch, heads, seq, d_k)
  ↓ [Q @ K^T]                         어텐션 점수 계산
                        (64, 8, 10, 10)   ← seq×seq 점수 행렬
  ↓ [/ √64, softmax]                  스케일링 + 확률화
                        (64, 8, 10, 10)
  ↓ [@ V]                             값에 가중평균
                        (64, 8, 10, 64)
  ↓ [transpose + view]                헤드 합치기
                        (64, 10, 512)
  ↓ [Linear W_o]                      최종 사영
                        (64, 10, 512)
```

### 디코더 흐름

```
디코더 입력 토큰:        (64, 12)
  ↓ [Embedding + PE]
                        (64, 12, 512)
  ↓ [Masked Self-Attn]                미래 단어 가리고 자기끼리 어텐션
                        (64, 12, 512)
  ↓ [Encoder-Decoder Attn]            Q=디코더, K/V=인코더 출력
                        (64, 12, 512)     입력 문장에서 정보 가져오기
  ↓ [Feed-Forward]
                        (64, 12, 512)
  ↓ [×6 반복]
                        (64, 12, 512)
  ↓ [Linear → vocab]                  단어 확률로 변환
                        (64, 12, 37000)
  ↓ [Softmax]
                        (64, 12, 37000)   ← 다음 단어 예측 확률
```


---

## 🎯 정리

| 구성 요소 | 역할 | 한 줄 요약 |
|---|---|---|
| **Self-Attention** | 단어 간 관계 학습 | 한 문장 안에서 서로 참고 |
| **Multi-Head** | 다양한 관점 | 8명의 전문가가 각자 분석 |
| **Scaled** | 안정적 학습 | √d_k로 나눠서 gradient 보존 |
| **Positional Encoding** | 순서 정보 | 좌석 번호표 |
| **Encoder** | 입력 이해 | 의미 압축 메모 작성 |
| **Decoder** | 출력 생성 | 메모 보며 한 단어씩 받아쓰기 |
| **Masking** | 커닝 방지 | 미래 단어 가리기 |
| **Add & Norm** | 안정화 | 원본 보존 + 분포 정규화 |

---

## 📚 참고문헌

```bibtex
@inproceedings{vaswani2017attention,
  title={Attention is All you Need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, Łukasz and Polosukhin, Illia},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
}
```

---

> 작성: Transformer 논문 분석 (비전공자용)
> 라이선스: MIT
