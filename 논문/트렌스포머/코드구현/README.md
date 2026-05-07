---
title: Transformer
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.29.0
python_version: "3.10"
app_file: app.py
pinned: false
---

# Transformer — 논문 완전 재현

**논문**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)  
**저자**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (NIPS 2017)  
**데모**: [🤗 Hugging Face Spaces — JangTaeng/Transformer](https://huggingface.co/spaces/JangTaeng/Transformer)

> 시퀀스 모델의 고질적인 한계였던 **순차 계산(RNN)** 과 **거리에 비례한 연산(CNN)** 을  
> 오직 attention 메커니즘만으로 대체하여, 병렬화·장거리 의존성·학습 효율을 동시에 잡은  
> Transformer 논문을 처음부터 끝까지 재현한 프로젝트입니다.  
> 논문의 Figure 1(전체 구조), Figure 2(Scaled Dot-Product / Multi-Head Attention), 식 (1)·(2), 학습 설정까지 모두 반영했습니다.

---

## 데모 바로가기

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20HuggingFace-Spaces-blue)](https://huggingface.co/spaces/JangTaeng/Transformer)

숫자열을 입력하면 **직접 구현·학습한 Transformer**가 시퀀스를 뒤집어 출력하고,  
디코더의 **cross-attention 가중치를 히트맵으로 시각화**합니다.  
🔢 1 2 3 4 5 → 5 4 3 2 1 · 🔢 9 8 7 → 7 8 9 처럼 학습이 잘 된 모델은 **반대각선(anti-diagonal) 패턴**을 또렷이 그립니다.

---

## 프로젝트 구조

```
├── app.py              # Gradio 데모 (UI + 부팅 시 자동 학습 + 추론 + 시각화)
├── transformer.py      # Transformer 모델 본체 (Attention, MHA, PE, Encoder/Decoder Layer)
├── requirements.txt    # 패키지 목록
└── README.md           # 이 파일
```

---

## 논문 구현 포인트

### 1. 전체 아키텍처 (논문 §3.1 + Figure 1)

base 모델 기준 하이퍼파라미터와 구조:

| 구성 요소 | 논문 base | 본 데모 | 논문 위치 |
|----------|----------|--------|----------|
| 인코더/디코더 층수 N | 6 | 2 | §3.1 |
| 모델 차원 d_model | 512 | 64 | §3.1 |
| FFN 내부 차원 d_ff | 2048 | 128 | §3.3 |
| 헤드 수 h | 8 | 4 | §3.2.2 |
| 헤드별 차원 d_k = d_v | 64 | 16 | §3.2.2 |
| 어휘 크기 | 37K (BPE) | 13 (toy) | §5.1 |
| 파라미터 | 약 6,500만 개 | 약 8만 개 | — |

**전체 흐름**: 입력 → Embedding × √d_model → +PE → Encoder ×N → Decoder ×N → Linear → Softmax

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab,
                 d_model=512, N=6, h=8, d_ff=2048, dropout=0.1, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)
        self.pe = PositionalEncoding(d_model, max_len)
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, h, d_ff, dropout) for _ in range(N)
        ])
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, h, d_ff, dropout) for _ in range(N)
        ])
        self.out = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 인코더: 임베딩 × √d_model 후 PE 더하고 N층 통과
        e = self.pe(self.src_embed(src) * math.sqrt(self.d_model))
        for layer in self.encoder:
            e = layer(e, src_mask)
        # 디코더: 인코더 출력 e를 cross-attention의 K/V로 받음
        d = self.pe(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder:
            d = layer(d, e, src_mask, tgt_mask)
        return self.out(d)
```

---

### 2. Scaled Dot-Product Attention — 핵심 아이디어 (논문 §3.2.1, Figure 2 왼쪽)

Transformer의 모든 것의 출발점인 attention 공식:

```
Attention(Q, K, V) = softmax( QKᵀ / √d_k ) V        (논문 식 1)
```

- Q와 K의 **내적**으로 "얼마나 닮았는지" 점수를 계산
- **softmax**로 확률 분포로 변환 (가중치 합 = 1)
- 그 가중치로 **V를 weighted sum**

**√d_k로 나누는 이유** (각주 4): q·k의 분산이 d_k가 되어 값이 커지면 softmax가 한쪽으로 포화 → 그래디언트가 0에 가까워짐 → 스케일링으로 방지.

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # 식 (1)의 분자/분모
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)    # 마스킹된 위치는 -∞ 근사
    attn = F.softmax(scores, dim=-1)
    return attn @ V                                     # V에 가중합
```

---

### 3. Multi-Head Attention (논문 §3.2.2, Figure 2 오른쪽)

단일 attention 대신 **8개 헤드(데모는 4개)** 로 병렬 계산. 서로 다른 표현 부분공간에서 다른 위치 관계를 동시에 학습합니다.

```
MultiHead(Q,K,V) = Concat(head₁, ..., head_h) Wᴼ
where head_i = Attention(QWᵢQ, KWᵢK, VWᵢV)
```

핵심은 **(B, seq, d_model) → (B, h, seq, d_k)** 로 reshape해서 헤드 차원을 앞으로 빼는 것. 그래야 batched matmul로 헤드별 attention을 한 번에 계산할 수 있습니다.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, h=8):
        super().__init__()
        self.h, self.d_k = h, d_model // h
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        B = Q.size(0)
        # (B, seq, d_model) → (B, h, seq, d_k)  헤드 분할
        Q = self.W_q(Q).view(B, -1, self.h, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(B, -1, self.h, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(B, -1, self.h, self.d_k).transpose(1, 2)
        out = scaled_dot_product_attention(Q, K, V, mask)
        # 헤드 합치기
        out = out.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.W_o(out)
```

> **논문 ablation (Table 3 행 A)**: 헤드 1개면 BLEU 0.9 하락, 32개로 늘려도 오히려 떨어짐 → h=8이 sweet spot

---

### 4. Attention의 3가지 사용 방식 (논문 §3.2.3)

Transformer 안에서 multi-head attention은 **목적이 다른 3곳**에 쓰입니다:

| 위치 | Q | K, V | 마스크 | 본 데모에서의 역할 |
|------|---|------|--------|------------------|
| **Encoder self-attention** | 인코더 입력 | 인코더 입력 | 패딩 마스크 | 입력 숫자들끼리 관계 파악 |
| **Decoder masked self-attention** | 디코더 입력 | 디코더 입력 | 패딩 + **causal** | 미래 출력 못 보게 차단 |
| **Encoder-Decoder cross-attention** | **디코더** 출력 | **인코더** 출력 | 인코더 패딩 마스크 | "지금 출력할 위치 = 입력의 어디?" → 시각화 대상 |

**결론**: 본 데모의 시각화는 세 번째 cross-attention. 뒤집기 태스크가 잘 학습되면 **반대각선 패턴**이 나타납니다.

```python
# DecoderLayer 안의 cross-attention — Q는 디코더, K/V는 인코더
x = self.norm2(x + self.dropout(
    self.cross_attn(x, enc_out, enc_out, src_mask)
))
```

---

### 5. Positional Encoding (논문 §3.5)

Transformer는 재귀가 없어서 위치 정보를 따로 주입해야 합니다. sin/cos를 쓰는 이유: 임의의 offset k에 대해 PE_{pos+k}가 PE_{pos}의 **선형변환으로 표현 가능** → 모델이 상대 위치를 쉽게 학습.

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)   # 짝수 차원
        pe[:, 1::2] = torch.cos(pos * div)   # 홀수 차원
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

> **논문 ablation (Table 3 행 E)**: 학습된 위치 임베딩과 sin/cos는 거의 동일한 성능. 그러나 sin/cos는 **학습 시 보지 못한 더 긴 시퀀스에도 외삽 가능** → 논문은 sin/cos 채택

---

### 6. Position-wise Feed-Forward Network (논문 §3.3, 식 2)

```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

각 위치에 **독립적으로 동일하게** 적용되는 2층 MLP. 차원은 512 → 2048 → 512로 **4배 늘렸다 줄임**.

```python
class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
```

> 논문 표현: "**커널 크기 1짜리 합성곱 2개**로 봐도 같다" — 위치 독립성 강조

---

### 7. 학습 설정 (논문 §5)

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=5e-4,                # 본 데모는 warmup 없이 고정 lr (toy task)
    betas=(0.9, 0.98),      # 논문 §5.3: β₂가 일반적인 0.999가 아닌 0.98
    eps=1e-9,
)
loss_fn = nn.CrossEntropyLoss(
    ignore_index=PAD_IDX,   # 패딩 토큰은 손실에서 제외
    label_smoothing=0.1,    # 논문 §5.4
)
# 배치 크기: 128, 학습 step: 2000, gradient clipping = 1.0
# Dropout: P=0.1 (논문 §5.4 Residual Dropout)
```

논문 원래 LR 스케줄 (식 3): `d_model^(-0.5) · min(step^(-0.5), step · warmup^(-1.5))` — 워밍업 4000 step 후 역제곱근 감소. 본 데모는 모델이 작아 warmup 없이도 안정적으로 수렴합니다.

---

### 8. Residual Connection + Layer Normalization (논문 §3.1)

각 sub-layer 출력은 다음과 같이 처리됩니다:

```
output = LayerNorm(x + Sublayer(x))
```

- **잔차 연결**: 그래디언트 소실 방지, 깊은 층 학습 가능 (ResNet과 동일 철학)
- **LayerNorm**: 배치가 아닌 **feature 차원**으로 정규화 → 시퀀스 길이가 달라져도 안정적

본 구현은 **Post-LN** 방식(원논문). 최근에는 학습 안정성 때문에 **Pre-LN** (`x + Sublayer(LayerNorm(x))`)을 더 많이 씁니다.

```python
class EncoderLayer(nn.Module):
    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout(self.attn(x, x, x, mask)))   # Post-LN
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x
```

---

### 9. Toy Task 설계 (본 구현 추가 사항)

논문은 WMT14 EN-DE/EN-FR 번역(~4.5M~36M 문장 쌍)으로 검증했지만, 무료 Space에서 그건 불가능합니다. 그래서 **숫자 시퀀스 뒤집기**라는 toy task로 대체했습니다.

```python
# 학습 데이터 — 매 step 무작위 생성, 외부 데이터 불필요
def make_batch(batch_size=128, min_len=3, max_len=10):
    src_list, tgt_list = [], []
    for _ in range(batch_size):
        L = np.random.randint(min_len, max_len + 1)
        digits = np.random.randint(0, 10, size=L).tolist()
        src_list.append(digits_to_ids(digits))         # [BOS, 1, 2, 3, EOS]
        tgt_list.append(digits_to_ids(digits[::-1]))   # [BOS, 3, 2, 1, EOS]
    return pad_and_stack(src_list), pad_and_stack(tgt_list)
```

**왜 뒤집기인가?**

- 어휘 13개(0~9 + PAD/BOS/EOS)로 모델이 작아도 됨
- 출력 i번째 = 입력의 반대편 위치 → **장거리 의존성 강제**
- cross-attention 시각화가 **가장 극적**(반대각선 패턴)
- 무한 데이터 (런타임 생성)

**효과**: ~80K 파라미터 모델이 부팅 시 **30초 학습으로 정확도 99%+** 달성.

---

## 구현 중 발생한 주요 오류와 해결

### 오류 1: Space YAML 메타데이터 누락

```
configuration error
Missing SDK in configuration
```

**원인**: Hugging Face Space는 `README.md` 최상단에 `sdk: gradio` 같은 YAML 메타데이터가 필수입니다.

```yaml
# 해결: README.md 최상단에 YAML 블록 추가
---
title: Transformer
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.29.0
python_version: "3.10"
app_file: app.py
pinned: false
---
```

---

### 오류 2: Python 3.13의 `audioop` 제거

```
ModuleNotFoundError: No module named 'audioop'
```

**원인**: Python 3.13에서 표준 라이브러리 `audioop`이 제거됨.  
Gradio 내부 `pydub`가 이걸 필요로 해서 빌드 실패.

**해결**: Space의 Python 버전을 3.10으로 명시.

```yaml
python_version: "3.10"   # README.md YAML에 추가
```

---

### 오류 3: matplotlib backend 문제 (헤드리스 환경)

```
RuntimeError: main thread is not in main loop
qt.qpa.xcb: could not connect to display
```

**원인**: Space는 GUI 없는 헤드리스 환경인데, matplotlib 기본 backend는 GUI를 시도함.

**해결**: import 직후 backend를 `Agg`로 강제 지정.

```python
import matplotlib
matplotlib.use("Agg")          # ← 반드시 pyplot import 전에!
import matplotlib.pyplot as plt
```

---

### 오류 4: 임베딩 스케일링 누락 → 학습 불안정

```
loss: 2.5 → 2.4 → 2.4 → 2.4 → ...   # 거의 안 떨어짐
```

**원인**: Positional Encoding이 임베딩보다 크면 위치 정보가 의미를 압도함. 논문 §3.4에 명시된 **임베딩 × √d_model** 스케일링을 빠뜨리면 발생.

```python
# 오류
e = self.pe(self.src_embed(src))                     # PE가 압도

# 해결
e = self.pe(self.src_embed(src) * math.sqrt(self.d_model))
```

---

### 오류 5: 마스크 broadcasting shape 불일치

```
RuntimeError: The size of tensor a (8) must match the size of tensor b (1)
at non-singleton dimension 1
```

**원인**: `scaled_dot_product_attention` 내부 scores는 `(B, h, seq_q, seq_k)` 모양이지만, 마스크를 `(B, seq)` 그대로 넘기면 broadcast 실패.

```python
# 오류
src_mask = (src != PAD)                              # (B, S) — 차원 부족

# 해결: 헤드/쿼리 축에 broadcast되도록 차원 추가
src_mask = (src != PAD).unsqueeze(1).unsqueeze(2)    # (B, 1, 1, S)

# causal mask와 결합 시
tgt_pad = (tgt != PAD).unsqueeze(1).unsqueeze(2)     # (B, 1, 1, T)
causal = torch.tril(torch.ones(T, T)).bool()         # (T, T)
tgt_mask = tgt_pad & causal.unsqueeze(0).unsqueeze(0)  # (B, 1, T, T)
```

---

## 논문 성능 결과

WMT 2014 번역 벤치마크에서 Transformer가 달성한 성능 (논문 Table 2):

| 모델 | EN-DE BLEU | EN-FR BLEU | 학습 비용 (FLOPs) | 파라미터 |
|------|-----------|-----------|-------------------|---------|
| GNMT + RL | 24.6 | 39.92 | 2.3 × 10¹⁹ | — |
| ConvS2S | 25.16 | 40.46 | 9.6 × 10¹⁸ | — |
| MoE | 26.03 | 40.56 | 2.0 × 10¹⁹ | — |
| GNMT + RL Ensemble | 26.30 | 41.16 | 1.8 × 10²⁰ | — |
| ConvS2S Ensemble | 26.36 | 41.29 | 7.7 × 10¹⁹ | — |
| **Transformer (base)** | **27.3** | **38.1** | **3.3 × 10¹⁸** | **65M** |
| **Transformer (big)** | **28.4** | **41.8** | **2.3 × 10¹⁹** | **213M** |

**WMT 2014 EN-DE에서 이전 SOTA(앙상블 포함) 대비 +2.0 BLEU**, 학습 비용은 1/4 이하 🏆  
EN-FR에서는 **8× P100 GPU로 단 3.5일** 만에 단일 모델 신기록 41.8 BLEU 달성.

---

## 논문 §4 — Why Self-Attention

Transformer 논문의 핵심 통찰. RNN/CNN 대비 self-attention의 우월성:

| 레이어 종류 | 층당 복잡도 | 순차 연산 | 최대 경로 길이 |
|------------|-----------|-----------|---------------|
| **Self-Attention** | O(n² · d) | **O(1)** ⬇️ | **O(1)** ⬇️ |
| Recurrent | O(n · d²) | O(n) ⬆️ | O(n) ⬆️ |
| Convolutional | O(k · n · d²) | O(1) | O(log_k n) |

핵심은 **최대 경로 길이가 O(1)** 이라는 점입니다. 시퀀스 안 어떤 두 토큰이든 단 한 번의 연산으로 직접 연결되므로, **장거리 의존성 학습**이 RNN(O(n))이나 CNN(O(log n))보다 본질적으로 유리합니다.

본 데모의 뒤집기 태스크가 이 우월성을 직접 보여줍니다 — 길이 10의 입력 첫 토큰이 출력 마지막에서 어떻게 정확히 매칭되는지 cross-attention 히트맵으로 확인할 수 있어요.

---

## 로컬 실행

```bash
# 1) 의존성 설치
pip install torch gradio matplotlib numpy

# 2) 데모 실행 (첫 실행 시 자동 학습, ~30초)
python app.py

# → http://127.0.0.1:7860 에서 열림
```

학습된 모델은 `model.pt`로 캐싱되어 다음 실행부터는 즉시 시작됩니다.

---

## 데모 화면 & 테스트 팁

**예측 결과 예시:**
```
입력     : 1 2 3 4 5 6 7
예측 출력 : 7 6 5 4 3 2 1
정답     : 7 6 5 4 3 2 1
일치 여부 : ✅ 정답!
```

**테스트 추천 입력**:
- 길이 3~10의 숫자열 (학습 분포 내)
- `1 2 3 4 5`, `9 8 7 6 5`, `0 1 2 3 4 5 6 7 8 9` 같은 명확한 패턴
- 시각화 탭에서 **반대각선 패턴**이 또렷할수록 학습이 잘 된 것

**한계**:
- 학습 길이(3~10) 밖은 정확도 급락
- 영어/한글 등 일반 자연어는 처리 불가 (toy 어휘 13개만 학습)

---

## 참고 논문

```bibtex
@inproceedings{vaswani2017attention,
  title     = {Attention Is All You Need},
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki
               and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N
               and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle = {Advances in Neural Information Processing Systems},
  pages     = {5998--6008},
  year      = {2017}
}
```

## 관련 자료

- 📄 논문: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- 🤗 데모: [Hugging Face Spaces](https://huggingface.co/spaces/JangTaeng/Transformer)
- 📝 The Annotated Transformer: [http://nlp.seas.harvard.edu/annotated-transformer/](http://nlp.seas.harvard.edu/annotated-transformer/)
- 🎥 The Illustrated Transformer (Jay Alammar): [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
