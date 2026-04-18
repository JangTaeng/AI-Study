# 최적화 관점에서 다시 읽는 ResNet

## 하드웨어가 좋아하는 코드 구조

## 들어가며: 수식에서 실행 비용으로

ResNet은 깊은 신경망이 identity조차 안정적으로 학습하지 못하는 문제를 `H(x) = x + F(x)`라는 구조로 풀었습니다.

하지만 이 수식은 실제 시스템에 들어가면 단순한 덧셈으로 끝나지 않습니다. 코드가 되고, 양자화 규칙이 되고, GPU에서는 fusion boundary가 되고, NPU에서는 SRAM과 DMA 스케줄 문제가 됩니다. 즉 residual add는 수학적으로는 `+`이지만, 실행 관점에서는 두 경로가 다시 만나는 merge point입니다.

이걸 왜 알아야 할까요. 추론 시스템에서는 속도가 곧 비용이기 때문입니다. 요청을 더 빨리 끝내면 필요한 칩 수가 줄고, 전력과 냉각 비용이 줄고, 운영 효율이 올라갑니다. 그래서 모델 구조를 이해한다는 것은 정확도만 보는 일이 아니라, 실제 비용 구조를 같이 보는 일입니다.

이 문서는 ResNet 블록을 세 가지 관점에서 짧게 다시 읽습니다.

1. 양자화에서는 왜 residual add를 따로 보아야 하는가
2. 왜 fusion과 clean identity path가 같이 중요해지는가
3. 왜 GPU와 NPU에서 같은 블록이 전혀 다른 비용으로 보이는가

---

## 1. 양자화에서는 왜 residual add를 따로 보나

float 환경에서는 residual add가 그냥 `out += identity`처럼 보입니다. 두 값이 모두 float이기 때문에, 겉으로 보면 단순한 덧셈입니다.

하지만 quantization 환경에서는 다릅니다. main path와 skip path가 다시 만나는 순간, 두 값이 같은 스케일과 같은 해석 규칙 안에서 합쳐져야 합니다. 그래서 이 지점은 단순 덧셈이 아니라 “두 경로를 같은 규칙으로 다시 맞추는 합류 지점”이 됩니다.

그래서 quantized 구현에서는 이 부분을 단순 `+` 대신 별도의 helper나 merge 표현으로 드러내는 경우가 많습니다. 이렇게 해야 시스템이 “여기가 residual merge point다”라는 사실을 더 분명히 이해할 수 있습니다.

이걸 명확히 드러내지 않으면 어떤 문제가 생길까요.

- backend가 이 지점을 그냥 일반 add로 볼 수 있음
- add 앞뒤에 format conversion이 따로 생길 수 있음
- add와 relu가 따로 놀면서 실행 비용이 커질 수 있음

핵심은 `+`가 틀렸다는 뜻이 아닙니다. 핵심은 **merge point라는 의미가 중요하다**는 점입니다.

한 줄 요약:

> float에서는 더하기처럼 보이지만, quantization에서는 residual add가 merge contract가 됩니다.

---

## 2. 그게 Conv-BN-ReLU fusion과 무슨 관련이 있나

여기서 fusion과 residual add는 같은 일이 아닙니다.

- Conv-BN-ReLU fusion은 main path 안의 연속 연산을 길게 묶는 일입니다.
- residual add는 그 main path가 skip path와 다시 만나는 지점입니다.

즉 순서는 이렇습니다. 먼저 main path가 깔끔하게 정리되어 있어야, 마지막 merge도 깔끔하게 처리할 수 있습니다. 그래서 좋은 fusion은 residual merge를 더 쉽게 만듭니다.

이 관점에서 ResNet v1과 v2의 차이도 다시 볼 수 있습니다.

- v1은 `Add -> ReLU`라서 merge 뒤에 연산이 하나 더 있습니다.
- v2는 `Add`에서 블록이 끝나므로 merge boundary가 더 선명합니다.

이 차이는 학습 관점에서만 중요한 것이 아닙니다. 실행기 입장에서도 중요합니다. v2처럼 merge 지점이 더 깨끗하게 남아 있으면, 어디까지가 transform path이고 어디서 merge가 일어나는지가 더 예측 가능해집니다.

그래서 모델 구조를 볼 때는 이런 기준이 중요합니다.

- main path를 불필요하게 복잡하게 만들지 말 것
- skip path를 가능한 한 깨끗하게 유지할 것
- merge 이후 구조를 짧게 유지할 것

한 줄 요약:

> fusion은 main path를 정리하는 일이고, clean identity path는 merge boundary를 더 선명하게 만들어 다음 최적화를 쉽게 합니다.

---

## 3. GPU에서는 왜 quantization과 같이 봐야 하고, NPU에서는 왜 더 빡빡한가

GPU에서 residual의 비용은 단순 add 자체보다, main path와 skip path를 같은 형식과 같은 layout으로 얼마나 싸게 다시 맞출 수 있느냐에 더 가깝습니다.

좋은 경우에는 add가 거의 마지막 epilogue처럼 처리됩니다. 하지만 형식이나 layout이 어긋나면 reformat, requantize, extra kernel이 붙으면서 residual이 비싸집니다. 그래서 GPU에서는 residual을 quantization과 함께 봐야 합니다.

여기서 중요한 구분이 하나 있습니다. 표준 ResNet과 eager quantized ResNet은 이름은 비슷하지만 실제로는 같은 실행 경로가 아닙니다.

- 표준 ResNet은 일반 PyTorch 연산 그래프라서 GPU compiler가 다시 최적화할 수 있습니다.
- eager quantized ResNet은 CPU quantized backend 경로에 더 가깝습니다.

즉 “양자화했으니 GPU에서 더 빨라지겠지”라는 기대는 여기서는 성립하지 않을 수 있습니다. 문제는 수학이 아니라 backend 생태계입니다.

그리고 NPU는 더 빡빡합니다. NPU는 범용성이 높은 장치라기보다, 작은 온칩 SRAM 안에서 정해진 데이터 흐름을 반복할 때 효율이 극대화되는 장치입니다. 그래서 지원 연산, 타일링, SRAM residency, DMA 순서가 모두 더 민감합니다.

NPU에서 residual merge는 단순 add가 아니라, 다음 질문으로 바뀝니다.

- main path와 skip path를 동시에 SRAM 안에 둘 수 있는가
- quantization metadata까지 같이 올려둘 수 있는가
- add 시점에 데이터가 제때 도착하는가

조금만 어긋나도 spill, reload, extra DMA가 생기고 비용이 급격히 커집니다.

한 줄 요약:

> GPU에서는 residual이 quantization과 memory-format 문제로 보이고, NPU에서는 그 문제가 더 직접적으로 SRAM residency와 dataflow scheduling 문제로 보입니다.

---

## 결론

이 문서의 핵심은 단순합니다.

ResNet은 identity를 배우기 쉽게 만든 구조이지만, 실제 시스템에서는 그 identity path와 residual add가 quantization, fusion, memory movement, GPU/NPU scheduling 문제로 바뀝니다.

그래서 좋은 모델은 정확도만 높은 모델이 아닙니다. 하드웨어가 좋아하는 실행 형태로 표현된 모델이어야 합니다.

또 좋은 코드가 자동으로 좋은 실행이 되지도 않습니다. 겉보기에는 같은 residual block처럼 보여도, 어떤 구현은 backend가 자연스럽게 처리하고, 어떤 구현은 reformat, spill, extra kernel launch를 부를 수 있습니다.

앞으로 모델이나 코드를 볼 때는 이런 질문이 중요합니다.

1. 이 residual add는 진짜 싼가
2. quantization 후에도 규칙이 안 깨지는가
3. compiler가 이 구조를 크게 fuse할 수 있는가
4. 이건 GPU에 맞는가, NPU에 맞는가

마지막 한 줄은 이렇습니다.

> ResNet을 다시 읽는다는 것은 논문을 복습하는 것이 아니라, 하드웨어가 자연스럽게 소화할 수 있는 더 나은 코드 구조를 배우는 일입니다.
