import torch                                                           # PyTorch 라이브러리(딥러닝 계산에 필요한 핵심 도구들)
import torch.nn as nn                                                  # 신경망(Neural Network)을 만드는 도구들 nn으로 명명

class AlexNet(nn.Module):                                              # AlexNet이라는 클래스 생성, nn.Module을 상속받아서 PyTorch 모델의 기본 기능을 모두 물려받음
    def __init__(self, num_classes=1000):                              # 초기화 함수, num_classes = 분류할 카테고리 수
        super(AlexNet, self).__init__()                                # 부모 클래스인 nn.Module의 초기화도 함께 실행. 이걸 안 하면 모델이 제대로 작동하지 않음
        
        # ===== 합성곱 레이어 5개 =====
        self.features = nn.Sequential(                                 # 여러 레이어를 순서대로 묶어주는 컨테이너를 만듬
            
            # Conv1: 224x224x3 → 55x55x96
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),     # 이미지 RGB 컬러 채널을 받아와서 96개 특징맵을 만드는 합성곱 연산 / 필터 사이즈 11x11, 4칸씩 이동 / 224×224 → 55×55
            nn.ReLU(inplace=True),                                     # ReLU 함수 사용 ( 음수는 0, 양수는 그대로 적용)
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),  # LRN  # 주변 채널들과 비교해서 값을 정규화. 강한 반응은 더 강하게, 약한 반응은 억제하는 역할 (현재는 잘 사용 X)
            nn.MaxPool2d(kernel_size=3, stride=2),                     # 3x3 영역 중 가장 큰 값만 뽑아서 크기 줄이기 / 55x55x96→ 27x27x96

            # Conv2: 27x27x96 → 27x27x256
            nn.Conv2d(96, 256, kernel_size=5, padding=2),              # 96 채널을 받아서 256으로 확장하는 합성곱 / 필터 사이즈는 5x5 패딩은 2 / 결과는 27x27
            nn.ReLU(inplace=True),                                     # ReLU 함수 사용 ( 음수는 0, 양수는 그대로 적용)
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),  # LRN
            nn.MaxPool2d(kernel_size=3, stride=2),                     # 3x3 영역 중 가장 큰 값만 뽑아서 크기 줄이기 / 27x27x96 → 13x13x256

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
