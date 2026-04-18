# ResNet Keynote — QA

## Section 0. Keynote Intro

```text
앞 발표에서는 ResNet이 왜 필요했는지 봤습니다.
핵심은 아주 단순합니다. 깊은 네트워크는 복잡한 함수를 못 배워서가 아니라, 가장 단순한 identity조차 안정적으로 구현하지 못했습니다. 그래서 ResNet은 H(x)를 직접 학습시키는 대신 H(x)=x+F(x)로 바꿔서, identity를 학습 목표가 아니라 구조적 기본값으로 만들어 버렸습니다.

그런데 저는 오늘 여기서 한 단계 더 내려가 보려고 합니다.
이 x + F(x)는 논문 안에서는 학습 구조이지만, 실제 시스템 안에서는 그냥 수식이 아닙니다. 이 식은 코드가 되고, quantization contract가 되고, GPU에서는 fusion boundary가 되고, NPU에서는 SRAM residency와 dataflow scheduling 문제가 됩니다. 즉 residual add는 단순한 덧셈이 아니라, 두 실행 경로가 다시 만나는 merge point입니다.

왜 이게 중요할까요.
몇 ms, 몇 초 빠르게 만드는 건 벤치마크 숫자를 예쁘게 만드는 일이 아닙니다. 추론 사업에서는 속도가 곧 비용이기 때문입니다. 같은 요청을 더 빨리 끝내면 필요한 칩 수가 줄고, 전력과 냉각 비용이 줄고, 결국 운영비가 내려갑니다. 그래서 latency reduction은 기술 취향이 아니라 돈 이야기입니다.

특히 지금은 그 비용 압박이 커지면서, 단순히 더 큰 모델만이 아니라 그 모델을 더 싸고 안정적으로 돌리는 가속기와 최적화 기술에도 돈이 몰리고 있습니다. 여기서 NPU가 중요해집니다. NPU는 아무 연산이나 유연하게 처리하는 장치라기보다, 작은 온칩 SRAM 안에서 정해진 데이터 흐름을 매우 효율적으로 반복하도록 만든 장치입니다. 그래서 지원 연산, 타일링, SRAM, DMA 순서가 더 빡빡합니다. 이건 단점이 아니라, 효율을 위해 자유도를 줄인 결과입니다.

그래서 오늘 제 발표는 ResNet을 다시 설명하는 자리가 아닙니다.
ResNet을 통해 좋은 모델 구조가 좋은 실행 구조로 자동 변환되지는 않는다는 점, 그리고 더 나은 코드란 결국 backend가 더 자연스럽게 실행할 수 있는 코드라는 점을 같이 고민해보는 자리입니다.

오늘 저는 세 가지를 보여드리겠습니다.
첫째, 논문에서 말하는 identity path를 코드에서는 어떻게 읽어야 하는가.
둘째, residual block 하나가 quantization, fusion, memory movement와 어떻게 연결되는가.
셋째, 같은 ResNet block이라도 왜 어떤 구현은 더 잘 실행되고, 어떤 구현은 더 많은 비용을 부르는가.

코드는 길게 보지 않겠습니다. 대신 아주 짧은 예제로, 같아 보이는 residual block도 어떤 구현은 merge point가 선명하고, 어떤 구현은 reformat, spill, extra kernel launch를 유발할 수 있다는 점만 보겠습니다.

제 목표는 ResNet paper를 읽고 끝내는 것이 아니라, 여러분이 다음부터는 코드를 볼 때 이런 질문을 한 번 더 하게 만드는 것입니다.
이 residual add는 진짜 싼가?
이 block은 quantization 후에도 규칙이 안 깨지는가?
이 구조는 compiler가 크게 fuse할 수 있는가?
이건 GPU에 맞는가, 아니면 NPU에 더 맞는가?

한 줄로 끝내면 이렇습니다.
오늘 발표는 ResNet 설명이 아니라, ResNet을 통해 더 나은 코드가 무엇인지 다시 생각해보는 발표입니다.
```

---

## Section 1. 문제 게임 1 — `+`는 왜 갑자기 비싸질까?

```text
문제 상황.
같은 ResNet block인데 float에서는 잘 돌던 residual add가, quantization으로 넘어가면 갑자기 별도 취급됩니다.
겉보기에는 그냥 + 한 줄인데, 왜 quantized ResNet은 이 지점을 더 조심스럽게 다룰까요?

찾아야 할 것.
이 + 가 정말 단순 덧셈인지, 아니면 두 경로가 다시 만나는 특별한 지점인지 찾으면 됩니다.

정답.
float에서는 둘 다 같은 float 체계 안에 있으니 그냥 더해도 됩니다.
하지만 quantization에서는 main path와 skip path가 같은 규칙 안에서 해석되고, 같은 execution flow 안에서 합쳐져야 합니다.
즉 residual add는 단순 산술이 아니라 merge point입니다.

그래서 핵심은 + 자체가 틀렸다는 게 아닙니다.
여기가 합류 지점이라는 정보가 중요하다는 뜻입니다.
이 정보가 약해지면 backend는 add, relu, format conversion을 따로따로 보게 될 수 있고,
사용자 입장에서는 왜 같은 ResNet인데 생각보다 안 빠르지? 라는 결과로 돌아옵니다.

가져갈 질문.
이 residual add는 그냥 덧셈인가, 아니면 merge contract인가?
```

```python
# 예제로 볼 포인트만 남긴 코드
# float 관점: 그냥 더하기처럼 보임
out = out + identity
out = relu(out)

# quantized 관점: 합류 지점을 명시
out = add_relu(out, identity)
```

```text
이 섹션의 한 줄 결론.
float에서는 더하기지만, quantized에서는 merge contract입니다.
```

---

## Section 2. 문제 게임 2 — 왜 main path를 먼저 정리해야 할까?

```text
문제 상황.
residual add가 중요하다는 건 알겠습니다.
그런데 실행기 입장에서 그 합류 지점이 잘 보이려면, 왜 그 앞의 Conv-BN-ReLU fusion이 먼저 중요할까요?

찾아야 할 것.
block 안에서 어떤 부분은 한 경로 안의 연속 연산이고, 어떤 부분은 두 경로가 다시 만나는 경계인지 구분하면 됩니다.

정답.
Conv-BN-ReLU fusion은 main path 안을 길게 묶는 일입니다.
반면 residual add는 그 정리된 길이 skip path와 다시 만나는 곳입니다.
즉 fusion과 residual add는 같은 일이 아니라, 앞뒤 순서로 연결된 일입니다.

그래서 main path가 먼저 깔끔해야 merge도 깔끔해집니다.
좋은 fusion은 residual merge를 더 쉽게 만듭니다.

여기서 v2가 중요해집니다.
v1은 Add 뒤에 ReLU가 있어서 merge 뒤 경계가 한 번 더 섞입니다.
반면 v2는 Add에서 블록이 끝나기 때문에 merge boundary가 더 선명합니다.
즉 clean identity path는 학습뿐 아니라 실행기 입장에서도 더 예측 가능한 구조입니다.

가져갈 질문.
이 block은 main path가 충분히 규칙적인가?
skip path는 너무 많이 변형되지 않았는가?
merge 이후 구조는 짧고 선명한가?
```

```python
# v1 느낌: merge 뒤에 한 번 더 비선형
out = conv_bn_relu(x)
out = conv_bn(out)
out = out + identity
out = relu(out)

# v2 느낌: merge에서 블록 종료
out = preact_conv(x)
out = preact_conv(out)
out = out + identity
```

```text
이 섹션의 한 줄 결론.
fusion은 길을 정리하는 일이고, residual add는 그 길이 다시 만나는 지점입니다.
```

---

## Section 3. 문제 게임 3 — 왜 GPU에서는 괜찮아 보이는데, NPU에서는 갑자기 빡빡해질까?

```text
문제 상황.
어떤 ResNet block은 GPU에서는 큰 문제 없어 보이는데, NPU로 가면 갑자기 지원 연산, 타일링, SRAM, DMA 순서 이야기가 나옵니다.
왜 같은 residual block이 장치가 바뀌면 이렇게 예민해질까요?

찾아야 할 것.
GPU가 보는 비용과 NPU가 보는 비용이 무엇인지 분리해서 보면 됩니다.

정답.
GPU에서 residual의 핵심 비용은 FLOPs보다, main path와 skip path를 같은 형식과 같은 layout으로 얼마나 싸게 맞추느냐에 가깝습니다.
즉 GPU에서는 residual이 quantization과 memory-format 문제로 보입니다.

그런데 여기서 중요한 구분이 있습니다.
표준 ResNet은 GPU compiler가 다시 최적화할 수 있는 일반 연산 그래프입니다.
반면 eager quantized ResNet은 CPU quantized backend 경로에 가깝습니다.
그래서 양자화했으니 GPU에서 더 빨라지겠지, 라는 기대는 항상 성립하지 않습니다.
문제는 수학이 아니라 백엔드 생태계입니다.

NPU는 더 빡빡합니다.
NPU는 아무 연산이나 일단 돌려보고 최적화하는 장치가 아니라,
작은 온칩 SRAM 안에서 정해진 dataflow를 반복할 때 효율이 극대화되는 장치입니다.
그래서 지원 연산, 타일링, SRAM residency, DMA 순서가 전부 더 민감합니다.

결국 NPU에서 residual merge는 단순 add가 아니라,
두 경로와 metadata를 SRAM 안에 같이 올려 두고 제때 합칠 수 있느냐의 문제가 됩니다.
조금만 꼬여도 spill, reload, extra DMA가 생기고, 그 순간 residual은 더 이상 가벼운 마지막 연산이 아닙니다.

가져갈 질문.
이 모델은 GPU compiler 경로에 올라타는가?
아니면 CPU quantized backend 경로인가?
이 residual merge는 SRAM 안에서 끝날 수 있는가?
```

```python
# GPU에서의 질문
# 같은 형식인가? 같은 layout인가?
can_merge_cheaply = same_dtype and same_layout and no_extra_reformat

# NPU에서의 질문
# 둘과 metadata를 SRAM 안에 같이 놓을 수 있는가?
can_finish_onchip = fits_input and fits_skip and fits_output and fits_scale_metadata
```

```text
이 섹션의 한 줄 결론.
GPU에서 residual은 quantization과 memory-format 문제이고,
NPU에서는 그 문제가 더 직접적으로 SRAM residency와 dataflow scheduling 문제로 나타납니다.
```

---

## Section 4. 결론 — 이제 어떤 질문으로 코드를 볼 것인가

```text
오늘 발표의 메시지는 단순합니다.
ResNet은 identity를 배우기 쉽게 만든 구조이지만,
실제 시스템에서는 그 identity path와 residual add가 quantization, fusion, memory movement, GPU/NPU scheduling 문제로 바뀝니다.

그래서 좋은 모델은 정확도만 높은 모델이 아닙니다.
하드웨어가 좋아하는 실행 형태로 표현된 모델이어야 합니다.

또 좋은 코드가 자동으로 좋은 실행이 되지도 않습니다.
겉보기에는 같은 residual block처럼 보여도,
어떤 구현은 backend가 자연스럽게 처리하고,
어떤 구현은 reformat, spill, extra kernel launch를 부를 수 있습니다.

그래서 앞으로는 질문이 달라져야 합니다.
이 residual add는 진짜 싼가?
quantization 후에도 규칙이 안 깨지는가?
compiler가 이 구조를 크게 fuse할 수 있는가?
이건 GPU에 맞는가, NPU에 맞는가?

마지막 한 줄은 이렇습니다.
오늘 발표의 목적은 ResNet을 다시 설명하는 것이 아니라,
ResNet을 통해 더 나은 코드가 무엇인지 다시 생각해보는 것이었습니다.
```
