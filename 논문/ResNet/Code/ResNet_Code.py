import torch                                       # 파이토치 딥러닝 프레임워크를 불러옴
import torch.nn as nn                              # neural network 모듈을 nn으로 명명


# ① 병목 블록 (Bottleneck Block) - ResNet-50/101/152용        # 좁게 줄여서 계산량을 줄이고 더 깊게 학습함

class BottleneckBlock(nn.Module):                  # nn.Module을 상속받아야 신경망인 것을 알 수 있다.
    expansion = 4  # 출력 채널 = 입력 채널 × 4

    def __init__(self, in_channels, mid_channels, stride=1):        # 초기화 함수(들어오는 블럭 채널 수, 병목으로 줄어든 중간 채널 수, 스트라이드 = 1 -> 이미지 크기를 얼마나 줄일지 (1=그대로, 2=절반). 기본값 1.
        super().__init__()                         # 부모 클래스를 먼저 실행

        # 논문의 F(x) 부분 (잔차 학습)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)   # 합성곱 층, 1x1 필터 
        self.bn1   = nn.BatchNorm2d(mid_channels)                    # 배치 정규화 -> 데이터를 "평균 0, 분산 1"로 맞춰줌. 학습을 안정화시키고 속도를 빠르게 해줌

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3,              # 합성곱 층, 3×3 필터
                               stride=stride, padding=1, bias=False)                   # 패딩 -> 가장자리에 0을 한 줄 덧대서 이미지 크기 유지
        self.bn2   = nn.BatchNorm2d(mid_channels)                    # 배치 정규화

        self.conv3 = nn.Conv2d(mid_channels, mid_channels * self.expansion,            # 합성곱 층, 1x1 필터 / 출력 채널 = mid_channels * 4 → 채널을 4배로 복원 (병목 확장 단계)
                               kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(mid_channels * self.expansion)   # 배치 정규화 (채널을 동일하게 늘림)

        self.relu  = nn.ReLU(inplace=True)                           # ReLU 활성화 함수-> 음수는 0으로, 양수는 그대로.

        # 숏컷 연결 - 채널 수나 크기가 다를 때 1×1 conv로 맞춤
        self.shortcut = nn.Sequential()                              # 레이어들을 순서대로 묶는 빈 컨테이너 생
        if stride != 1 or in_channels != mid_channels * self.expansion:  # stride가 1이 아니거나(=이미지 크기가 변함), 채널 수가 맞지 않으면
            self.shortcut = nn.Sequential(                               #  1×1 합성곱으로 크기/채널을 맞춰줌
                nn.Conv2d(in_channels, mid_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(mid_channels * self.expansion)
            )

    def forward(self, x):
        identity = x  # 원래 입력값 저장 (숏컷용)

        out = self.relu(self.bn1(self.conv1(x)))   # 1×1 conv
        out = self.relu(self.bn2(self.conv2(out))) # 3×3 conv
        out = self.bn3(self.conv3(out))            # 1×1 conv (ReLU 전)

        # 핵심! F(x) + x  -> 이게 없으면 그냥CNN
        out += self.shortcut(identity)            # 원본 x를 크기 맞춤
        out = self.relu(out)                      # 거기에 더함 (F(x) + x)
        return out


# ② 전체 ResNet-50 모델
class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        # conv1
        self.conv1   = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # conv2_x ~ conv5_x
        self.layer1 = self._make_layer(64,   64,  blocks=3, stride=1)
        self.layer2 = self._make_layer(256,  128, blocks=4, stride=2)
        self.layer3 = self._make_layer(512,  256, blocks=6, stride=2)
        self.layer4 = self._make_layer(1024, 512, blocks=3, stride=2)

        # 분류기
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(2048, num_classes)

    def _make_layer(self, in_channels, mid_channels, blocks, stride):
        layers = [BottleneckBlock(in_channels, mid_channels, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BottleneckBlock(mid_channels * 4, mid_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))  # conv1
        x = self.layer1(x)  # conv2_x
        x = self.layer2(x)  # conv3_x
        x = self.layer3(x)  # conv4_x
        x = self.layer4(x)  # conv5_x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# 사용 예시
model = ResNet50(num_classes=1000)
dummy = torch.randn(1, 3, 224, 224)  # 이미지 1장
output = model(dummy)
print(output.shape)  # → torch.Size([1, 1000])
