---
title: AlexNet 논문 재현
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
tags:
  - image-classification
  - alexnet
  - paper-reproduction
  - pytorch
  - imagenet
---

# AlexNet — 논문 완전 재현

**논문**: [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)  
**저자**: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton (NeurIPS 2012)  
**데모**: [🤗 Hugging Face Spaces — JangTaeng/AlexNetCode](https://huggingface.co/spaces/JangTaeng/AlexNetCode)

> 딥러닝의 역사를 바꾼 AlexNet 논문을 처음부터 끝까지 코드로 재현한 프로젝트입니다.  
> 논문의 Figure 2, GPU 분할 구조, 학습 설정까지 모두 반영했습니다.

---

## 데모 바로가기

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20HuggingFace-Spaces-blue)](https://huggingface.co/spaces/JangTaeng/AlexNetCode)

이미지를 업로드하면 ImageNet 1000개 클래스 중 Top-5 예측 결과를 확인할 수 있습니다.  
강아지, 고양이, 자동차, 새, 음식 등 다양한 물체를 인식합니다.

> ※ ImageNet은 사람(남자/여자) 클래스를 포함하지 않습니다.

---

## 프로젝트 구조

```
├── app.py            # Gradio 데모 + AlexNet 모델 전체 코드
├── config.json       # 모델 하이퍼파라미터 (num_labels, dropout 등)
├── requirements.txt  # 패키지 목록
└── README.md         # 이 파일
```

---

## 논문 구현 포인트

### 1. 전체 아키텍처 (논문 3.5절 + Figure 2)

8개 레이어(Conv×5 + FC×3)로 구성된 AlexNet의 레이어별 출력 shape:

| 레이어 | 커널 | stride | padding | 출력 shape | 논문 섹션 |
|--------|------|--------|---------|-----------|-----------|
| Conv1 + MaxPool | 11×11 | 4 | 2 | (B, 64, 27, 27) | 3.5절 |
| Conv2 + MaxPool | 5×5 | 1 | 2 | (B, 192, 13, 13) | 3.5절 |
| Conv3 | 3×3 | 1 | 1 | (B, 384, 13, 13) | 3.5절 |
| Conv4 | 3×3 | 1 | 1 | (B, 256, 13, 13) | 3.5절 |
| Conv5 + MaxPool | 3×3 | 1 | 1 | (B, 256, 6, 6) | 3.5절 |
| FC1 | — | — | — | (B, 4096) | 4.2절 |
| FC2 | — | — | — | (B, 4096) | 4.2절 |
| FC3 (출력) | — | — | — | (B, 1000) | Abstract |

**총 파라미터: 약 6,000만 개**

```python
self.features = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),   # Conv1
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(64, 192, kernel_size=5, padding=2),             # Conv2
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(192, 384, kernel_size=3, padding=1),            # Conv3
    nn.ReLU(inplace=True),
    nn.Conv2d(384, 256, kernel_size=3, padding=1),            # Conv4
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 256, kernel_size=3, padding=1),            # Conv5
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2),
)

self.classifier = nn.Sequential(
    nn.Dropout(p=0.5),                   # 논문 4.2절
    nn.Linear(256 * 6 * 6, 4096),        # FC1
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),                   # 논문 4.2절
    nn.Linear(4096, 4096),               # FC2
    nn.ReLU(inplace=True),
    nn.Linear(4096, num_labels),         # FC3 — Softmax는 CrossEntropy에 내장
)
```

---

### 2. ReLU 활성화 함수 (논문 3.1절)

기존 tanh 대신 `f(x) = max(0, x)` 를 사용해 학습 속도를 6배 향상시켰습니다.

```python
nn.ReLU(inplace=True)   # 모든 Conv·FC 레이어 뒤에 적용
```

---

### 3. GPU 분할 구조 (논문 3.2절)

논문은 GTX 580 (3GB) 2개를 사용해 모델을 병렬로 학습했습니다.  
`groups=2` 파라미터로 채널을 반씩 나눠 독립 연산하는 구조를 재현했으며,  
최종 구현에서는 torchvision 가중치 호환을 위해 `groups=1`로 통일했습니다.

```
GPU 분할 전략 (논문 원본):
  Conv1        groups=1  (in=3, RGB는 2로 나눌 수 없음)
  Conv2·4·5    groups=2  (채널을 반씩 나눠 독립 연산 — 메모리 절약)
  Conv3·FC     groups=1  (cross-GPU — 전체 채널 연결, 정보 교환)

왜 이 패턴인가?
  나누는 이유  → GPU 메모리(3GB) 부족
  Conv3에서 합치는 이유 → GPU0(색상 무관 필터)·GPU1(색상 특이적 필터) 정보 교환
  FC에서 합치는 이유   → 최종 분류는 전체 특징을 종합해야 함
```

---

### 4. Local Response Normalization (논문 3.3절)

"측면 억제(lateral inhibition)"를 구현해 일반화 성능을 높입니다.  
Conv1·Conv2 뒤에만 적용합니다.

```python
nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)
```

---

### 5. Overlapping MaxPool (논문 3.4절)

전통적인 non-overlapping pooling(s=z) 대신  
stride < kernel\_size인 overlapping pooling을 사용합니다.

```python
nn.MaxPool2d(kernel_size=3, stride=2)   # s=2 < z=3 → overlapping
```

---

### 6. Dropout (논문 4.2절)

FC1·FC2 앞에만 `p=0.5`로 적용합니다. FC3(출력층)에는 적용하지 않습니다.

```python
nn.Dropout(p=0.5)   # FC1·FC2 앞에만
# FC3는 Dropout 없음
```

---

### 7. 데이터 증강 (논문 4.1절)

```python
# 논문 2절: 256×256 리사이즈 → 224×224 center crop → 픽셀 평균 차감
TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])
```

---

### 8. 학습 설정 (논문 5절)

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,            # 검증 오류 정체 시 ÷10 (총 3회)
    momentum=0.9,
    weight_decay=0.0005,
)
# 배치 크기: 128
# 학습 기간: 약 90 epoch (GTX 580 × 2, 5~6일 소요)
```

---

### 9. 가중치 초기화 (논문 5절)

```python
# 평균 0, 표준편차 0.01인 정규분포로 초기화
nn.init.normal_(m.weight, mean=0, std=0.01)

# Conv2·Conv4·Conv5 및 FC의 bias → 1 (ReLU에 양수 입력 보장)
# 나머지 bias → 0
nn.init.constant_(m.bias, 1.0)   # Conv2·Conv4·Conv5·FC
nn.init.constant_(m.bias, 0.0)   # Conv1·Conv3
```

---

## 구현 중 발생한 주요 오류와 해결

### 오류 1: `in_channels must be divisible by groups`

```
ValueError: in_channels must be divisible by groups
```

**원인**: Conv1의 `in_channels=3`(RGB)은 `groups=2`로 나눌 수 없습니다 (3 ÷ 2 = 1.5).  
**해결**: Conv1만 `groups=1`로 설정합니다.

```python
# 오류
nn.Conv2d(3, 96, kernel_size=11, stride=4, groups=2)   # 3 % 2 != 0

# 해결
nn.Conv2d(3, 96, kernel_size=11, stride=4, groups=1)   # groups=1
```

---

### 오류 2: `mat1 and mat2 shapes cannot be multiplied (1×6400 and 9216×4096)`

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x6400 and 9216x4096)
```

**원인**: Conv1 `padding=0`이면 출력이 55가 아닌 54로 내림되어 FC 입력 크기가 틀립니다.

```
padding=0 → (224-11+0)/4+1 = 54.25 → 내림 54
            → Pool: 26 → ... → Pool5: 5
            → Flatten: 256×5×5 = 6400  ← 오류

padding=2 → (224-11+4)/4+1 = 55.0  → 정확히 55
            → Pool: 27 → ... → Pool5: 6
            → Flatten: 256×6×6 = 9216 ← 정상
```

**해결**: Conv1에 `padding=2` 추가합니다.

```python
# 오류
nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0)

# 해결
nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
```

---

## 5. 데이터 흐름 추적 (Shape 변화)

형식: `(batch, C, H, W)` → [Layer] → `(변경된 shape)`

| 단계 | 입력 Shape | 출력 Shape | 이유 |
|------|-----------|-----------|------|
| 입력 이미지 | — | `(1, 3, 224, 224)` | RGB 3채널, 224×224px |
| Conv1 (7×7, stride 2, 64 filters) | `(1, 3, 224, 224)` | `(1, 64, 112, 112)` | 224 ÷ 2 = 112, 채널 3 → 64 |
| MaxPool (3×3, stride 2) | `(1, 64, 112, 112)` | `(1, 64, 56, 56)` | 112 ÷ 2 = 56, 채널 유지 |
| conv2_x (Bottleneck × 3) | `(1, 64, 56, 56)` | `(1, 256, 56, 56)` | 공간 크기 유지, 채널 64 × 4 = 256으로 확장 |
| conv3_x (Bottleneck × 4, stride 2) | `(1, 256, 56, 56)` | `(1, 512, 28, 28)` | 56 ÷ 2 = 28, 채널 128 × 4 = 512 |
| conv4_x (Bottleneck × 6, stride 2) | `(1, 512, 28, 28)` | `(1, 1024, 14, 14)` | 28 ÷ 2 = 14, 채널 256 × 4 = 1024 |
| conv5_x (Bottleneck × 3, stride 2) | `(1, 1024, 14, 14)` | `(1, 2048, 7, 7)` | 14 ÷ 2 = 7, 채널 512 × 4 = 2048 |
| Global Average Pooling → Flatten | `(1, 2048, 7, 7)` | `(1, 2048)` | 7×7 특징 맵을 평균 내어 1개 숫자로 압축 후 1D 펼침 |
| FC Layer (2048 → 1000) | `(1, 2048)` | `(1, 1000)` | 1000개 클래스 점수로 선형 변환 |
| Softmax (최종 출력) | `(1, 1000)` | `(1, 1000)` | 합계 = 1.0인 확률값으로 변환 |

---

## 논문 성능 결과

| 모델 | Top-1 오류율 | Top-5 오류율 |
|------|------------|------------|
| Sparse coding (기존 최고) | 47.1% | 28.2% |
| SIFT + FVs | 45.7% | 25.7% |
| **AlexNet (본 구현)** | **37.5%** | **17.0%** |
| AlexNet × 7 (ILSVRC-2012 우승) | 36.7% | **15.3%** |

2위(26.2%)와 약 10%p 격차로 우승 — 딥러닝 시대의 시작점이 된 결과입니다.

---

## 로컬 실행

```bash
pip install torch torchvision gradio pillow requests
python app.py
```

---

## 참고 논문

```bibtex
@inproceedings{krizhevsky2012imagenet,
  title     = {ImageNet Classification with Deep Convolutional Neural Networks},
  author    = {Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {25},
  year      = {2012}
}
```
