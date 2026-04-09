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
