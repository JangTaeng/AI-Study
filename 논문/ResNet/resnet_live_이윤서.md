# Section 0. Intro

## 문제 상황

ResNet은 보통 이렇게 배운다.

- 깊은 네트워크는 학습이 어렵다.
- 그래서 `H(x) = x + F(x)` 구조를 넣었다.
- skip connection 덕분에 학습이 쉬워졌다.

여기까지는 맞다.

그런데 오늘 질문은 여기서 끝나지 않는다.

**이 수식이 실제 코드와 하드웨어 안에서는 무엇이 되는가?**

---

## 같이 보기

논문에서는 `x + F(x)`가 예쁜 수식처럼 보인다.
하지만 실제 시스템에서는 이 한 줄이 아래 문제로 바뀐다.

- quantization 규칙
- fusion 경계
- memory movement
- GPU layout 정렬
- NPU SRAM residency
- DMA 순서

즉 residual add는 단순한 더하기가 아니라, **두 경로가 다시 만나는 merge point**다.

---

## 오늘 볼 3가지

1. 왜 quantization에서는 residual add를 따로 보나
2. 그게 왜 Conv-BN-ReLU fusion과 연결되나
3. 왜 GPU에서는 그냥 보이던 문제가 NPU에서는 갑자기 빡빡해지나

---

## 핵심 질문

- 이 residual add는 진짜 싼가?
- 이 block은 quantization 후에도 규칙이 안 깨지는가?
- compiler가 이 구조를 크게 fuse할 수 있는가?
- 이건 GPU에 맞는가, 아니면 NPU에 더 맞는가?

---

# Section 1. 문제 게임 1 — `+`는 왜 갑자기 비싸질까?

## 문제 상황

float ResNet에서는 residual add가 그냥 `+`처럼 보인다.

그런데 quantized ResNet에서는 이 지점을 굳이 따로 표시한다.

**왜 같은 덧셈인데, 한쪽은 그냥 `+`이고 다른 쪽은 별도 취급일까?**

---

## 먼저 눈으로 보기

### float 쪽 느낌

```python
out = out + identity
out = relu(out)
```

### quantized 쪽 느낌

```python
out = add_relu(out, identity)
```

겉보기에는 둘 다 비슷하다.
하지만 시스템이 읽는 의미는 다르다.

---

## 청중이 찾아야 할 포인트

여기서 찾아야 하는 것은 이것이다.

**이 `+`가 진짜 그냥 덧셈인가, 아니면 두 경로가 다시 만나는 특별한 지점인가?**

---

## 정답

float에서는 둘 다 같은 float 체계 안에 있으니 그냥 더해도 된다.

하지만 quantization에서는 다르다.
main path와 skip path가 마지막에 만날 때,

- 같은 scale 안에서 읽혀야 하고
- 같은 규칙으로 해석돼야 하고
- 같은 execution flow 안에서 합쳐져야 한다

즉 이 지점은 단순한 산술이 아니라 **merge contract**다.

그래서 quantization 환경에서는 “여기가 합류 지점이다”라는 정보를 시스템에 더 분명히 주는 쪽이 유리하다.

---

## 왜 이게 실제 문제로 이어지나

이 정보를 시스템이 잘 못 보면,

- add를 따로 보고
- relu를 따로 보고
- format conversion을 따로 보고
- 중간에 한 번 더 옮기고
- kernel이 하나 더 생길 수 있다

사용자 입장에서는 결국 이렇게 느낀다.

**“왜 같은 ResNet인데 생각보다 안 빠르지?”**

---

## 이 섹션에서 가져갈 질문

- 이 residual add는 그냥 덧셈인가?
- 아니면 merge point인가?
- 시스템이 그 사실을 분명하게 볼 수 있는가?

---

## 한 줄 정리

**float에서는 더하기지만, quantized에서는 merge contract다.**

---

# Section 2. 문제 게임 2 — 왜 main path를 먼저 정리해야 할까?

## 문제 상황

이제 residual add가 중요하다는 건 알았다.

그런데 왜 사람들은 또 Conv-BN-ReLU fusion 이야기를 할까?
왜 merge 이야기와 fusion 이야기가 같이 나오나?

---

## 먼저 눈으로 보기

### 익숙한 v1 느낌

```python
out = conv1(x)
out = bn1(out)
out = relu(out)

out = conv2(out)
out = bn2(out)

out = out + identity
out = relu(out)
```

여기서 block 안에는 두 종류의 구간이 있다.

1. 한 경로 안에서 연속적으로 이어지는 부분
2. 두 경로가 다시 만나는 부분

---

## 청중이 찾아야 할 포인트

어디까지가 **main path 안의 정리 구간**이고,
어디부터가 **skip path와 다시 만나는 merge boundary**인지 찾으면 된다.

---

## 정답

Conv-BN-ReLU fusion은 main path 안을 길게 묶는 일이다.

반면 residual add는 그 정리된 길이 skip path와 다시 만나는 지점이다.

즉 둘은 같은 일이 아니다.
하지만 순서대로 연결된 일이다.

- 먼저 main path가 깔끔해야 하고
- 그 다음 merge가 깔끔해질 수 있다

그래서 좋은 fusion은 좋은 merge를 돕는다.

---

## v1과 v2를 왜 같이 보나

### v1 느낌

```python
out = ...
out = out + identity
out = relu(out)
```

### v2 느낌

```python
out = ...
out = out + identity
# 여기서 block 종료
```

v1은 add 뒤에 ReLU가 있어서 merge 직후 경계가 한 번 더 섞인다.

v2는 add에서 블록이 끝난다.
즉 시스템 입장에서는 “여기서 합쳐졌다”는 경계가 더 선명하다.

그래서 v2의 clean identity path는 학습만 돕는 것이 아니라,
실행기 입장에서도 더 예측 가능한 구조가 된다.

---

## 이 섹션에서 가져갈 질문

- main path가 충분히 규칙적인가?
- skip path는 너무 많이 변형되지 않았는가?
- merge 이후 구조는 짧고 선명한가?

---

## 한 줄 정리

**fusion은 길을 정리하는 일이고, residual add는 그 길이 다시 만나는 지점이다.**

---

# Section 3. 문제 게임 3 — 왜 GPU에서는 괜찮아 보이는데, NPU에서는 갑자기 빡빡해질까?

## 문제 상황

어떤 ResNet block은 GPU에서는 별문제 없어 보인다.

그런데 NPU로 가면 갑자기 이런 말이 튀어나온다.

- 지원 연산
- 타일링
- SRAM residency
- DMA 순서

왜 같은 residual block이 장치가 바뀌면 이렇게 예민해질까?

---

## 먼저 눈으로 보기

### GPU가 보는 질문

```python
can_merge_cheaply = same_dtype and same_layout and no_extra_reformat
```

### NPU가 보는 질문

```python
can_finish_onchip = fits_input and fits_skip and fits_output and fits_scale_metadata
```

같은 block인데 질문 자체가 다르다.

---

## 청중이 찾아야 할 포인트

GPU가 보는 비용과 NPU가 보는 비용이 무엇인지 분리해서 보면 된다.

---

## 정답 1 — GPU에서는 무엇이 비싼가

GPU에서 residual의 핵심 비용은 FLOPs 그 자체보다,

- main path 결과와 skip path 결과를
- 같은 형식으로 맞추고
- 같은 layout으로 맞추고
- add를 싸게 끝내는 비용

에 더 가깝다.

즉 GPU에서는 residual이 **quantization + memory-format 문제**로 보인다.

좋은 경우에는 add가 거의 epilogue처럼 처리된다.
하지만 형식이 안 맞으면 reformat, requantize, extra kernel이 생긴다.

---

## 정답 2 — 여기서 꼭 구분해야 할 것

표준 ResNet과 eager quantized ResNet은 이름은 비슷하지만 실행 경로가 다르다.

- 표준 ResNet: GPU compiler가 다시 최적화할 수 있는 일반 연산 그래프
- eager quantized ResNet: CPU quantized backend 경로에 더 가까운 그래프

즉 “양자화했으니 GPU에서 더 빨라지겠지”라는 기대는 항상 맞지 않는다.
문제는 수학이 아니라 **백엔드 생태계**다.

---

## 정답 3 — NPU에서는 왜 더 빡빡한가

NPU는 아무 연산이나 유연하게 돌리는 장치가 아니다.
작은 온칩 SRAM 안에서 정해진 dataflow를 반복할 때 효율이 가장 좋다.

그래서 NPU에서는 아래가 더 민감하다.

- 지원 연산이 맞는가
- tile이 SRAM 안에 같이 들어가는가
- main path와 skip path를 동시에 잡을 수 있는가
- quantization metadata까지 같이 둘 수 있는가
- DMA가 제때 도착하는가

조금만 꼬여도 spill, reload, extra DMA가 생긴다.
그 순간 residual은 더 이상 가벼운 마지막 연산이 아니다.

---

## 이 섹션에서 가져갈 질문

- 이 모델은 GPU compiler 경로에 올라타는가?
- 아니면 CPU quantized backend 경로인가?
- 이 residual merge는 SRAM 안에서 끝날 수 있는가?
- format, metadata, DMA 순서까지 같이 맞는가?

---

## 한 줄 정리

**GPU에서 residual은 quantization과 memory-format 문제이고, NPU에서는 그 문제가 더 직접적으로 SRAM residency와 dataflow scheduling 문제로 나타난다.**

---

# Section 4. 결론 — 이제 어떤 질문으로 코드를 볼 것인가

## 다시 문제로 돌아가기

오늘 처음 질문은 이것이었다.

**같은 ResNet block처럼 보여도, 왜 어떤 구현은 잘 돌고 어떤 구현은 더 많은 비용을 부를까?**

이제 답을 낼 수 있다.

---

## 정답

ResNet은 identity를 배우기 쉽게 만든 구조다.
하지만 실제 시스템에서는 그 identity path와 residual add가 아래 문제로 바뀐다.

- quantization 규칙
- fusion 경계
- memory movement
- GPU/NPU scheduling

그래서 좋은 모델은 정확도만 높은 모델이 아니다.
**하드웨어가 좋아하는 실행 형태로 표현된 모델**이어야 한다.

또 좋은 코드가 자동으로 좋은 실행이 되지도 않는다.
겉보기에는 같은 residual block처럼 보여도,

- 어떤 구현은 backend가 자연스럽게 처리하고
- 어떤 구현은 reformat, spill, extra kernel launch를 부른다

즉 “논문적으로 맞는 코드”와 “실행 관점에서 더 나은 코드”는 다를 수 있다.

---

## 앞으로 바뀌어야 할 질문

이제는 코드나 모델을 볼 때 이렇게 물어야 한다.

- 이 residual add는 진짜 싼가?
- quantization 후에도 규칙이 안 깨지는가?
- compiler가 이 구조를 크게 fuse할 수 있는가?
- 이건 GPU에 맞는가, NPU에 맞는가?

---

## 마지막 한 줄

**오늘 발표의 목적은 ResNet을 다시 설명하는 것이 아니라, ResNet을 통해 더 나은 코드가 무엇인지 다시 생각해보는 것이었습니다.**
