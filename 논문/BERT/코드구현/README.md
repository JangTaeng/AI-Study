---
title: BERT 데모 - 사전학습 & 파인튜닝 태스크
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
python_version: "3.11"
app_file: app.py
pinned: false
license: apache-2.0
tags:
  - bert
  - nlp
  - transformers
  - masked-language-modeling
  - question-answering
  - text-classification
  - named-entity-recognition
short_description: BERT 논문의 사전학습과 파인튜닝 태스크 실습 데모
---

# BERT — 논문 완전 재현

**논문**: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)  
**저자**: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova (NAACL 2019)  
**데모**: [🤗 Hugging Face Spaces — JangTaeng/BERT](https://huggingface.co/spaces/JangTaeng/BERT)

> 기존 언어 모델은 단방향(left-to-right)이라 문맥의 절반만 활용했습니다.  
> BERT는 **Masked Language Model(MLM)** 과 **Next Sentence Prediction(NSP)** 이라는  
> 두 가지 사전학습 과제로 **깊은 양방향(bidirectional) 표현**을 학습하고,  
> 단 하나의 출력 레이어만 추가해 11개 NLP 태스크에서 SOTA를 달성한 논문을 재현한 프로젝트입니다.  
> 논문의 Section 3(사전학습), Section 4(파인튜닝), Figure 2(입력 표현),  
> Figure 4(태스크별 파인튜닝 구조)를 모두 반영했습니다.

---

## 데모 바로가기

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20HuggingFace-Spaces-blue)](https://huggingface.co/spaces/JangTaeng/BERT)

문장이나 문단을 입력하면 BERT의 **여섯 가지 핵심 기능**을 직접 체험해볼 수 있습니다.  
🎭 마스크 단어 예측 · 🔗 다음 문장 판별 · 🎯 문장 쌍 함의 분류 (MNLI) ·  
😀 감성 분석 (SST-2) · ❓ 질의응답 (SQuAD v1.1) · 🏷️ 개체명 인식 (CoNLL-2003 NER)

---

## 프로젝트 구조

```
├── app.py                  # Gradio 데모 (6개 태스크 탭 UI)
├── requirements.txt        # 패키지 목록 + 호환성 핀
├── train_glue.py           # GLUE 태스크 파인튜닝 스크립트 (논문 §4.1)
├── train_squad.py          # SQuAD v1.1 파인튜닝 스크립트 (논문 §4.2)
├── inference_examples.py   # CLI에서 바로 돌려보는 최소 추론 예제
└── README.md               # 이 파일
```

---

## 논문 구현 포인트

### 1. 전체 아키텍처 (논문 3절 + Figure 1)

BERT는 **Transformer Encoder만 쌓아 올린 구조**입니다 (Vaswani et al. 2017의 인코더 부분).  
GPT(디코더)와 달리 양방향 self-attention을 사용해 모든 토큰이 좌우 문맥을 모두 봅니다.

| 모델 | 레이어 수 (L) | 히든 크기 (H) | 어텐션 헤드 (A) | 파라미터 |
|------|:------------:|:------------:|:--------------:|:--------:|
| BERT-Base  | 12 | 768  | 12 | 110M |
| BERT-Large | 24 | 1024 | 16 | 340M |

```python
# BERT의 핵심 구조 — Transformer Encoder block만 L번 쌓음
class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([
            BertLayer(config) for _ in range(config.num_hidden_layers)  # L=12 or 24
        ])

    def forward(self, hidden_states, attention_mask):
        for layer in self.layer:
            # 각 레이어 = Multi-Head Self-Attention + FFN + Residual + LayerNorm
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states
```

**전체 흐름**:
```
입력 토큰 → Embedding (token + segment + position) → Transformer Encoder × L → 출력 표현
                                                              ↓
                                  태스크별 헤드 (MLM / NSP / 분류 / QA / NER)
```

---

### 2. 입력 표현 (논문 Figure 2)

BERT의 입력은 **세 가지 임베딩의 합**으로 구성됩니다:

```
입력 임베딩 = Token Embedding + Segment Embedding + Position Embedding
```

| 임베딩 | 역할 | 어휘 크기 |
|--------|------|----------|
| Token | WordPiece 단위 단어 표현 | 30,000 |
| Segment | 문장 A인지 B인지 구분 | 2 |
| Position | 순서 정보 (학습됨, 절대 위치) | 512 |

특수 토큰:
- `[CLS]`: 모든 시퀀스의 첫 토큰 — 분류 태스크에서 문장 전체 표현으로 사용
- `[SEP]`: 두 문장(A, B)을 구분하는 분리자
- `[MASK]`: MLM 학습 시 가리는 토큰

```python
# 토크나이저가 자동으로 [CLS], [SEP], segment id를 만들어 줍니다
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer(
    "The man went to the store.",      # sentence A → segment id = 0
    "He bought a gallon of milk.",     # sentence B → segment id = 1
    return_tensors="pt",
)
# inputs.input_ids:    [CLS] the man went to the store [SEP] he bought ... [SEP]
# inputs.token_type_ids: [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, ..., 1]
```

---

### 3. Masked Language Model — 핵심 아이디어 (논문 3.1절, Task #1)

전통적인 LM은 좌→우 또는 우→좌 단방향이라 깊은 양방향 표현을 못 만듭니다.  
BERT는 입력 토큰의 **15%를 무작위로 마스킹**한 뒤 그 자리에 올 단어를 예측하게 해서  
양방향 문맥을 강제로 활용하게 만듭니다.

**마스킹 전략** (논문 3.1, Appendix A.1):
- 15% 토큰 위치를 선택
  - 그 중 **80%**: `[MASK]` 토큰으로 교체
  - 그 중 **10%**: 다른 무작위 단어로 교체
  - 그 중 **10%**: 원래 단어 그대로 유지

> 80/10/10 전략의 목적: 사전학습에는 `[MASK]`가 등장하지만 파인튜닝에는 없는  
> **사전학습-파인튜닝 불일치(mismatch)** 를 줄이기 위함.

```python
# MLM 추론 예제 (본 프로젝트 inference_examples.py 발췌)
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

text = "The capital of France is [MASK]."
inputs = tokenizer(text, return_tensors="pt")
mask_idx = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

with torch.no_grad():
    logits = model(**inputs).logits

# 마스크 위치에 대한 상위 5개 예측
top_5 = torch.topk(logits[0, mask_idx], 5, dim=-1)
for token_id, score in zip(top_5.indices[0], top_5.values[0]):
    print(f"{tokenizer.decode([token_id]):<15} 로짓={score.item():.3f}")
# paris   로짓=12.3
# lyon    로짓= 8.7
# ...
```

---

### 4. Next Sentence Prediction (논문 3.1절, Task #2)

QA와 NLI 같은 태스크는 **두 문장의 관계**를 이해해야 합니다.  
NSP는 문장 B가 실제로 문장 A의 다음 문장인지 이진 분류합니다.

- 50%: 실제 다음 문장 → 라벨 `IsNext`
- 50%: 코퍼스에서 무작위로 뽑은 문장 → 라벨 `NotNext`

분류는 `[CLS]` 토큰의 최종 히든 벡터 `C`로 수행:

```python
# NSP 예측 — [CLS] 위치의 출력을 이진 분류 헤드로 통과
from transformers import AutoModelForNextSentencePrediction

model = AutoModelForNextSentencePrediction.from_pretrained("bert-base-uncased")
inputs = tokenizer(
    "The man went to the store.",
    "He bought a gallon of milk.",
    return_tensors="pt",
)
logits = model(**inputs).logits   # shape: (1, 2)
probs = torch.softmax(logits, dim=-1)[0]
# probs[0] = P(IsNext), probs[1] = P(NotNext)
```

> 📊 사전학습 후 NSP 정확도는 **97~98%** 에 달합니다 (논문 각주 5).

---

### 5. 사전학습 데이터 및 학습 설정 (논문 3.1 + Appendix A.2)

| 항목 | 값 |
|------|---|
| 사전학습 코퍼스 | BooksCorpus (800M words) + English Wikipedia (2,500M words) |
| 총 단어 수 | 약 3.3B |
| 배치 크기 | 256 시퀀스 × 512 토큰 = 128,000 토큰/배치 |
| Optimizer | Adam (β₁=0.9, β₂=0.999) |
| 학습률 | 1e-4, warmup 10,000 steps, linear decay |
| Weight decay | 0.01 |
| Dropout | 0.1 (모든 레이어) |
| 활성화 함수 | GELU (ReLU 아님) |
| 학습 step | 1,000,000 (약 40 epoch) |
| 학습 시간 | BERT-Base 4 TPU × 4일, BERT-Large 16 TPU × 4일 |

```python
# 본 프로젝트에서는 파인튜닝 시 동일 설정을 그대로 사용 (Appendix A.3)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5,            # 파인튜닝은 {5e-5, 3e-5, 2e-5} 중 선택
    betas=(0.9, 0.999),
    weight_decay=0.01,
)
```

---

### 6. 파인튜닝 — 네 가지 태스크 카테고리 (논문 Figure 4)

BERT의 강점은 **사전학습된 모델 + 출력 레이어 하나만 추가**하면 거의 모든  
NLP 태스크를 풀 수 있다는 점입니다. 파인튜닝은 모든 파라미터를 end-to-end로 업데이트합니다.

| 태스크 유형 | 입력 형태 | 출력 위치 | 본 데모 모델 |
|------------|----------|----------|------------|
| (a) 문장 쌍 분류 | `[CLS] A [SEP] B [SEP]` | `[CLS]` → 분류 헤드 | MNLI |
| (b) 단일 문장 분류 | `[CLS] A [SEP]` | `[CLS]` → 분류 헤드 | SST-2 |
| (c) 질의응답 | `[CLS] Q [SEP] P [SEP]` | 모든 P 토큰 → start/end | SQuAD v1.1 |
| (d) 단일 문장 태깅 | `[CLS] A [SEP]` | 각 토큰 → 라벨 | CoNLL-2003 NER |

---

### 7. 질의응답(SQuAD) 파인튜닝 (논문 4.2절, Figure 4c)

SQuAD는 지문에서 답변의 **시작/끝 토큰 위치**를 예측하는 추출형 QA입니다.

```
P_i = exp(S · T_i) / Σ_j exp(S · T_j)     (시작 토큰)
P_i = exp(E · T_i) / Σ_j exp(E · T_j)     (끝 토큰)
```

- `S`, `E`: 새로 학습되는 시작/끝 벡터 (논문 4.2 추가 파라미터의 전부!)
- `T_i`: i번째 토큰의 최종 BERT 히든 표현
- 점수가 가장 높은 (i, j) span (단, j ≥ i)을 답변으로 선택

```python
# train_squad.py 발췌 — 답변 span을 학습 라벨로 변환
def preprocess_train(examples):
    # 토크나이저가 (question, context)를 [CLS] Q [SEP] P [SEP] 형식으로 패킹
    tokenized = tokenizer(
        examples["question"], examples["context"],
        max_length=384, truncation="only_second",  # 질문은 절대 자르지 않음
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )
    # 각 example의 정답 character span을 token span으로 매핑
    for i, offsets in enumerate(tokenized["offset_mapping"]):
        start_char = examples["answers"][...]["answer_start"][0]
        end_char = start_char + len(answer_text)
        # offsets 정보로 토큰 인덱스 찾기 → start_positions, end_positions
        ...
```

---

### 8. 학습 결과 — 11개 NLP 태스크 SOTA (논문 4절)

#### GLUE 벤치마크 (논문 Table 1)

| 시스템 | MNLI-m/mm | QQP | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE | **평균** |
|--------|:--------:|:---:|:----:|:-----:|:----:|:-----:|:----:|:---:|:--------:|
| Pre-OpenAI SOTA | 80.6/80.1 | 66.1 | 82.3 | 93.2 | 35.0 | 81.0 | 86.0 | 61.7 | 74.0 |
| OpenAI GPT      | 82.1/81.4 | 70.3 | 87.4 | 91.3 | 45.4 | 80.0 | 82.3 | 56.0 | 75.1 |
| **BERT-Base**   | 84.6/83.4 | 71.2 | 90.5 | 93.5 | 52.1 | 85.8 | 88.9 | 66.4 | **79.6** |
| **BERT-Large**  | **86.7/85.9** | **72.1** | **92.7** | **94.9** | **60.5** | **86.5** | **89.3** | **70.1** | **82.1** |

GLUE 평균에서 **+7.0%p**, MNLI에서 **+4.6%p** 향상 — 같은 시점 대비 압도적.

#### SQuAD v1.1 (논문 Table 2)

| 시스템 | Dev EM | Dev F1 | Test EM | Test F1 |
|--------|:------:|:------:|:-------:|:-------:|
| 인간 (Human)        |  —   |  —   | 82.3 | 91.2 |
| #1 Ensemble nlnet  |  —   |  —   | 86.0 | 91.7 |
| BiDAF + ELMo       |  —   | 85.6 |  —   | 85.8 |
| **BERT-Large 단일** | 84.1 | 90.9 |  —   |  —   |
| **BERT-Large 앙상블 + TriviaQA** | **86.2** | **92.2** | **87.4** | **93.2** |

> 인간 성능(F1 91.2)을 **+2.0 F1** 초과 달성한 첫 모델 중 하나.

#### SQuAD v2.0 / SWAG

| 데이터셋 | 이전 SOTA | BERT-Large | 향상폭 |
|---------|:--------:|:----------:|:------:|
| SQuAD v2.0 (Test F1) | 78.0 | **83.1** | +5.1 |
| SWAG (Test Acc)      | 78.0 | **86.3** | +8.3 |

---

### 9. 사전학습 태스크의 효과 (논문 5.1절 — Ablation)

논문은 MLM과 NSP가 정말로 필요한지 검증합니다:

| 사전학습 설정 | MNLI | QNLI | MRPC | SST-2 | SQuAD F1 |
|--------------|:----:|:----:|:----:|:-----:|:--------:|
| **BERT-Base (MLM + NSP)** | **84.4** | **88.4** | **86.7** | **92.7** | **88.5** |
| No NSP                    | 83.9 | 84.9 | 86.5 | 92.6 | 87.9 |
| LTR & No NSP (= GPT 방식)  | 82.1 | 84.3 | 77.5 | 92.1 | 77.8 |
| + BiLSTM 추가              | 82.1 | 84.1 | 75.7 | 91.6 | 84.9 |

**결론**: MLM이 깊은 양방향 표현을 만드는 핵심이고, NSP는 문장 쌍 태스크에 보탬이 됩니다.

---

## 구현 중 발생한 주요 오류와 해결

> 본 Space를 배포하면서 실제로 마주친 4가지 오류와 해결 과정입니다.  
> Gradio 4.44는 2024년 10월 릴리스라, 2026년 환경의 새 의존성과 부딪히는 케이스가 많았습니다.

### 오류 1: `short_description` 길이 제한 초과

```
YAML Metadata Error: "short_description" length must be less than or equal to 60 characters long
```

**원인**: Hugging Face Space의 `short_description`은 **60자 이하**여야 함.

```yaml
# 오류 (61자)
short_description: Interactive demo of BERT's pre-training and fine-tuning tasks

# 해결 (29자)
short_description: BERT 논문의 사전학습과 파인튜닝 태스크 실습 데모
```

---

### 오류 2: Python 3.13의 `audioop` 모듈 제거

```
ModuleNotFoundError: No module named 'audioop'
  File "pydub/utils.py", line 14
```

**원인**: Python 3.13에서 표준 라이브러리 `audioop`이 제거됨 (PEP 594).  
Gradio 4.44의 내부 의존성인 `pydub`가 이걸 import하려다 실패.  
Hugging Face Space는 기본적으로 Python 3.13을 사용함.

**해결**: README YAML에서 Python 버전을 3.11로 고정 + 백포트 패키지를 안전망으로 추가.

```yaml
# README.md YAML
python_version: "3.11"
```

```text
# requirements.txt
audioop-lts; python_version >= "3.13"
```

---

### 오류 3: `HfFolder` ImportError

```
ImportError: cannot import name 'HfFolder' from 'huggingface_hub'
  File "gradio/oauth.py", line 13
```

**원인**: `huggingface_hub` **v1.0.0** (2025년 11월 릴리스)에서 deprecated였던  
`HfFolder` 클래스가 제거됐는데, Gradio 4.44는 아직 옛 API를 import 중.

**해결**: 호환 가능한 버전으로 다운그레이드 — Facebook의 EdgeTAM Space가 검증한 0.34.3.

```text
# requirements.txt
huggingface_hub==0.34.3
```

---

### 오류 4: Starlette `TemplateResponse` 시그니처 변경

```
TypeError: unhashable type: 'dict'
  File "jinja2/utils.py", line 515
  in templates.TemplateResponse(...)
```

**원인**: Starlette **v1.0.0** (2026년 3월 릴리스)이 deprecated였던  
`TemplateResponse(name, context)` 시그니처를 제거하고 `TemplateResponse(request, name, ...)`  
으로 강제 변경. Gradio 4.44는 옛 시그니처로 호출하므로, dict가 첫 인자로 넘어가  
Jinja2 캐시에서 해시 키로 사용되려다 폭발.  
앱이 기동은 되지만 **모든 HTTP 요청마다** 이 에러로 죽음.

**해결**: starlette와 fastapi 둘 다 1.0 미만으로 핀.

```text
# requirements.txt
starlette<1.0
fastapi<0.116           # 새 FastAPI는 starlette>=1.0을 요구하므로 함께 핀
```

---

### 최종 requirements.txt 핀 요약

| 핀 | 이유 | 깨진 버전 |
|---|------|----------|
| `python_version: "3.11"` | `audioop` stdlib 제거 회피 | Python 3.13+ |
| `huggingface_hub==0.34.3` | `HfFolder` 클래스 제거 | hf_hub ≥ 1.0 |
| `starlette<1.0` | `TemplateResponse` 시그니처 변경 | starlette ≥ 1.0 |
| `fastapi<0.116` | 최신 fastapi가 starlette≥1.0 요구 | fastapi ≥ 0.116 |
| `audioop-lts; python_version >= "3.13"` | 3.13 fallback 안전망 | — |

> 💡 **장기적 해결책**: Gradio 5.x로 업그레이드하면 위 핀 대부분이 불필요해집니다.  
> 다만 컴포넌트 API가 일부 바뀌어서 코드 수정이 필요할 수 있습니다.

---

## 로컬 실행

```bash
# 1) 의존성 설치
pip install -r requirements.txt

# 2) 데모 실행 (HuggingFace Hub의 사전학습 모델 자동 다운로드)
python app.py
# → http://127.0.0.1:7860 에서 접속

# 3) 직접 파인튜닝 (선택)
# GLUE — SST-2 감성 분류
python train_glue.py \
    --task_name sst2 \
    --model_name_or_path bert-base-uncased \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32

# SQuAD v1.1
python train_squad.py \
    --model_name_or_path bert-base-uncased \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32

# 4) CLI 추론 예제 (전체 태스크 한 번에 확인)
python inference_examples.py
```

---

## 데모 화면 & 테스트 팁

**Masked LM 출력 예시**:
```
입력: The capital of France is [MASK].

1. paris   — 확률 0.9234
2. lyon    — 확률 0.0123
3. nice    — 확률 0.0087
...
```

**테스트 추천 문장**:
- **MLM**: 한 문장에 `[MASK]` 토큰을 정확히 **하나**만 포함시킬 것
- **NSP**: 같은 주제의 두 문장 vs 전혀 다른 주제의 두 문장을 비교해 볼 것
- **SQuAD**: 지문은 길수록 흥미로움 (단, 384 토큰 이하). 질문은 짧고 구체적으로
- **NER**: 영어 고유명사를 포함한 문장이 가장 잘 검출됨

> 📝 본 데모는 영어 BERT(`bert-base-uncased`)를 사용합니다 (원 논문이 영어 코퍼스로 학습).  
> 한국어로 같은 실험을 해보려면 `app.py`의 모델 ID를 `klue/bert-base` 등으로 교체하세요.

---

## 참고 논문

```bibtex
@inproceedings{devlin-etal-2019-bert,
    title     = "{BERT}: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    author    = "Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina",
    booktitle = "Proceedings of NAACL-HLT",
    year      = "2019",
    pages     = "4171--4186",
    url       = "https://aclanthology.org/N19-1423/"
}
```

## 관련 자료

- 📄 논문: [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- 🤗 데모: [Hugging Face Spaces — JangTaeng/BERT](https://huggingface.co/spaces/JangTaeng/BERT)
- 💻 원 저자 코드: [google-research/bert](https://github.com/google-research/bert)
- 🤖 사용 모델: [bert-base-uncased](https://huggingface.co/bert-base-uncased)
- 📚 데이터셋: [GLUE](https://gluebenchmark.com/) · [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) · [CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/)
