# 📘 AlexNet 논문 분석

> **논문 제목:** ImageNet Classification with Deep Convolutional Neural Networks  
> **저자:** Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton (2012)

---

## 📋 목차

1. [Abstract (3줄 요약)](#1-abstract-3줄-요약)
2. [Introduction + Conclusion 요약](#2-introduction--conclusion-요약)
3. [핵심 구조 설명](#3-핵심-구조-설명)
4. [전체 모델 PyTorch 구현](#4-전체-모델-pytorch-구현)
5. [데이터 흐름 추적 (Shape 변화)](#5-데이터-흐름-추적-shape-변화)
6. [논문 ↔ 코드 연결](#6-논문--코드-연결-pytorch)

---

## 1. Abstract (3줄 요약)


**문제** | 수백만 장의 고해상도 이미지를 1000개 카테고리로 정확하게 분류하는 것이 기존 방식으로는 한계가 있었다.  

**방법** | 5개의 합성곱 레이어(Convolutional Layer)와 3개의 완전연결 레이어(Fully-Connected Layer)로 구성된 깊은 신경망을, GPU 2개와 ReLU·Dropout 등 새로운 기법을 활용해 학습시켰다.  

**결과** |  ImageNet 대회(ILSVRC-2012)에서 **top-5 오류율 15.3%** 를 달성해 2위(26.2%)를 압도적으로 따돌리며 딥러닝 시대의 서막을 열었다.

---

## 2. Introduction + Conclusion 요약

### 왜 이 논문이 나왔는가?

과거에는 이미지 인식에 **"사람이 직접 특징을 뽑아주는"** 방식(SIFT, FV 등)을 썼습니다.  
예를 들어 "귀가 뾰족하면 고양이"처럼 규칙을 사람이 직접 설계했죠.  
하지만 현실 세계의 이미지는 너무 다양해서 이 방식은 한계에 부딪혔습니다.

### 기존 방식의 문제점

- 데이터셋이 너무 작았다 (수만 장 수준)
- 사람이 직접 특징을 설계하니 복잡한 패턴 인식에 한계
- CNN은 이론적으로 강력하지만, 계산량이 너무 많아 대규모로 쓰기 어려웠다

### AlexNet의 개선점

- ✅ **GPU 2개**로 대규모 병렬 학습을 실현
- ✅ **ReLU** 활성화 함수로 학습 속도를 획기적으로 향상
- ✅ **Dropout**으로 과적합(Overfitting) 방지
- ✅ **데이터 증강(Data Augmentation)**으로 학습 데이터를 인위적으로 늘림

> 💡 **논문의 핵심 주장:** "깊이(Depth)가 정말 중요하다"  
> 중간 레이어 하나만 제거해도 성능이 2%씩 떨어졌습니다.

---

## 3. 핵심 구조 설명

### ① Convolutional Layer — 합성곱 레이어

> 🔍 **비유:** 돋보기로 사진을 꼼꼼히 훑어보는 것.  
> 작은 창문(필터)을 이미지 위에서 슬라이딩하며 패턴(엣지, 색깔, 질감)을 찾아냅니다.

| 구분 | 내용 |
|------|------|
| 입력값 | 이미지 데이터 `(batch, 채널 수, 높이, 너비)` 형태의 숫자 배열 |
| 출력값 | 특징 맵(Feature Map) — 패턴이 어디에 있는지를 담은 숫자 배열 |
| 하이퍼파라미터 | 커널 크기(11×11, 5×5, 3×3), 필터 개수(96, 256, 384...), 스트라이드, 패딩 |

---

### ② ReLU Nonlinearity — 렐루 활성화 함수 ⭐ 핵심

> 🚦 **비유:** 신호등.  
> 음수 값은 무조건 0(빨간불)으로 막고, 양수 값은 그대로 통과(초록불)시킵니다.

$$f(x) = \max(0, x)$$

기존에 쓰던 `tanh`나 `sigmoid`는 값이 커지면 기울기가 거의 0이 되어  
(**기울기 소실, Vanishing Gradient**) 학습이 느려집니다.  
ReLU는 이 문제가 없어서 **6배 빠른 학습**을 가능케 했습니다.

| 구분 | 내용 |
|------|------|
| 입력값 | 합성곱 연산 후 나온 숫자 |
| 출력값 | 음수는 0, 양수는 그대로 |
| 하이퍼파라미터 | 없음 (수식이 고정되어 있음) |

---

### ③ Max Pooling — 맥스 풀링 ⭐ 핵심

> 🖼️ **비유:** 사진을 축소할 때 각 구역에서 가장 눈에 띄는 픽셀만 남기는 것.  
> 중요한 정보만 남기고 나머지는 버립니다.

AlexNet은 일반적인 풀링과 달리 겹치는 영역을 포함하는 **Overlapping Pooling**을 사용했습니다.  
(창 크기 `z=3`, 이동 간격 `s=2`로 겹치게 설정)

| 구분 | 내용 |
|------|------|
| 입력값 | ReLU를 통과한 특징 맵 |
| 출력값 | 크기가 줄어든 특징 맵 |
| 하이퍼파라미터 | 풀링 창 크기(`z=3`), 스트라이드(`s=2`) |

---

### ④ Local Response Normalization (LRN) — 국소 반응 정규화 ⭐ 핵심

> 🏫 **비유:** 학교에서 한 학생이 너무 튀면 주변 학생들이 상대적으로 눌리는 것.  
> 이웃한 필터들 사이에서 경쟁을 시켜 너무 강한 반응을 억제합니다.

실제 뇌 신경세포(뉴런)의 **"측면 억제(lateral inhibition)"** 현상을 모방한 기법입니다.

| 구분 | 내용 |
|------|------|
| 입력값 | ReLU를 통과한 특징 맵 |
| 출력값 | 정규화된 특징 맵 (같은 크기) |
| 하이퍼파라미터 | `k=2`, `n=5`, `α=10⁻⁴`, `β=0.75` |

---

### ⑤ Fully-Connected Layer — 완전연결 레이어

> ⚖️ **비유:** 앞에서 찾아낸 모든 특징들을 모아 최종 판단을 내리는 판사.  
> "귀가 뾰족하고 수염이 있으니 고양이!"

| 구분 | 내용 |
|------|------|
| 입력값 | 앞 레이어에서 나온 벡터 (4096차원) |
| 출력값 | 다음 레이어의 뉴런값 (4096차원 → 최종 1000차원) |
| 하이퍼파라미터 | 뉴런 수 (4096, 4096, 1000) |

---

### ⑥ Dropout ⭐ 핵심

> ⚽ **비유:** 팀 훈련에서 매번 랜덤하게 일부 선수를 쉬게 하는 것.  
> 특정 선수에게만 의존하지 않고 모든 선수가 실력을 키우게 됩니다.

학습 중에 뉴런을 확률 50%로 무작위로 꺼버립니다.  
덕분에 모델이 특정 패턴에만 의존하는 **과적합(Overfitting)을 방지**합니다.

| 구분 | 내용 |
|------|------|
| 입력값 | FC 레이어의 출력값 |
| 출력값 | 일부 뉴런이 0이 된 출력값 |
| 하이퍼파라미터 | 드롭아웃 확률 (`p=0.5`) |

---

### ⑦ Softmax — 소프트맥스

> 📊 **비유:** 점수 환산기.  
> "고양이일 확률 87%, 개일 확률 10%, 토끼일 확률 3%"처럼  
> 모든 클래스의 확률 합이 1이 되도록 변환합니다.

| 구분 | 내용 |
|------|------|
| 입력값 | 1000개의 점수 숫자 |
| 출력값 | 1000개 클래스 각각의 확률 (합계 = 1.0) |
| 하이퍼파라미터 | 없음 |

---

## 4. 전체 모델 PyTorch 구현

```python
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        
        # ===== 합성곱 레이어 5개 =====
        self.features = nn.Sequential(
            # Conv1: 224x224x3 → 55x55x96
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),  # LRN
            nn.MaxPool2d(kernel_size=3, stride=2),  # → 27x27x96

            # Conv2: 27x27x96 → 27x27x256
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),  # LRN
            nn.MaxPool2d(kernel_size=3, stride=2),  # → 13x13x256

            # Conv3: 13x13x256 → 13x13x384 (풀링 없음)
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv4: 13x13x384 → 13x13x384 (풀링 없음)
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv5: 13x13x384 → 13x13x256 → 풀링 후 6x6x256
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # → 6x6x256
        )
        
        # ===== 완전연결 레이어 3개 =====
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),                  # FC1에 Dropout 적용
            nn.Linear(6 * 6 * 256, 4096),       # 9216 → 4096
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),                  # FC2에 Dropout 적용
            nn.Linear(4096, 4096),              # 4096 → 4096
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),       # 4096 → 1000
        )
    
    def forward(self, x):
        x = self.features(x)           # 합성곱 처리
        x = x.view(x.size(0), -1)     # Flatten: (batch, 256, 6, 6) → (batch, 9216)
        x = self.classifier(x)        # FC 레이어 처리
        return x

# 모델 생성 및 테스트
model = AlexNet(num_classes=1000)
dummy_input = torch.randn(1, 3, 224, 224)  # 배치 1개, RGB, 224x224
output = model(dummy_input)
print(output.shape)  # → torch.Size([1, 1000])
```

---

## 5. 데이터 흐름 추적 (Shape 변화)

| 단계 | 레이어 | 출력 Shape |
|------|--------|-----------|
| 입력 | Input | `(1, 3, 224, 224)` |
| 1 | Conv1 (11×11, stride=4) | `(1, 96, 55, 55)` |
| 2 | ReLU + LRN | `(1, 96, 55, 55)` |
| 3 | MaxPool (3×3, stride=2) | `(1, 96, 27, 27)` |
| 4 | Conv2 (5×5, pad=2) | `(1, 256, 27, 27)` |
| 5 | ReLU + LRN | `(1, 256, 27, 27)` |
| 6 | MaxPool (3×3, stride=2) | `(1, 256, 13, 13)` |
| 7 | Conv3 (3×3, pad=1) | `(1, 384, 13, 13)` |
| 8 | Conv4 (3×3, pad=1) | `(1, 384, 13, 13)` |
| 9 | Conv5 (3×3, pad=1) | `(1, 256, 13, 13)` |
| 10 | MaxPool (3×3, stride=2) | `(1, 256, 6, 6)` |
| 11 | Flatten | `(1, 9216)` |
| 12 | FC1 + Dropout | `(1, 4096)` |
| 13 | FC2 + Dropout | `(1, 4096)` |
| 14 | FC3 (출력) | `(1, 1000)` |

---

## 6. 논문 ↔ 코드 연결 (PyTorch)

| 논문 내용 | PyTorch 코드 |
|-----------|-------------|
| "96 kernels of size 11×11×3, stride 4" | `nn.Conv2d(3, 96, kernel_size=11, stride=4)` |
| "ReLU nonlinearity" | `nn.ReLU(inplace=True)` |
| "Local Response Normalization" | `nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)` |
| "Overlapping max-pooling, z=3, s=2" | `nn.MaxPool2d(kernel_size=3, stride=2)` |
| "Dropout with probability 0.5" | `nn.Dropout(p=0.5)` |
| "three fully-connected layers, 4096 neurons" | `nn.Linear(9216, 4096)`, `nn.Linear(4096, 4096)` |
| "1000-way softmax" | `nn.Linear(4096, 1000)` + `nn.Softmax(dim=1)` |  

---
## 📚 참고 자료

- [원본 논문 (NeurIPS 2012)](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
- [PyTorch 공식 AlexNet 구현](https://pytorch.org/vision/stable/models/alexnet.html)
