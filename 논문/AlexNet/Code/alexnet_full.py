"""
AlexNet — 논문 완전 재현 (GPU 분할 구조 포함)
논문: "ImageNet Classification with Deep Convolutional Neural Networks"
      Krizhevsky, Sutskever, Hinton (NeurIPS 2012)

────────────────────────────────────────────────────────────────
논문 Figure 2 GPU 분할 규칙 (3.2절 + 3.5절)
────────────────────────────────────────────────────────────────
  Conv1  : 입력(3ch) → GPU0(48ch) / GPU1(48ch)  — 독립 연산
  Conv2  : GPU0(48ch→128ch) / GPU1(48ch→128ch)  — 독립 연산
  Conv3  : GPU0+GPU1 합쳐서(256ch→192ch each)   — cross-GPU ★
  Conv4  : GPU0(192ch→192ch) / GPU1(192ch→192ch) — 독립 연산
  Conv5  : GPU0(192ch→128ch) / GPU1(192ch→128ch) — 독립 연산
  FC1~3  : GPU0+GPU1 합쳐서 연산               — cross-GPU ★

  [합산 채널]  Conv1:96  Conv2:256  Conv3:384  Conv4:384  Conv5:256
────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
from torch import Tensor


# ──────────────────────────────────────────────────────────────
# 1. GPU 분할 합성곱 블록
# ──────────────────────────────────────────────────────────────

class ParallelConvBlock(nn.Module):
    """
    논문 3.2절 — GPU 2개에 나눠 독립 연산하는 합성곱 블록.

    입력 텐서를 채널 축으로 반씩 잘라
    GPU0·GPU1에 각각 별도의 Conv를 적용한 뒤 다시 합칩니다.

    실제 GPU가 1개인 환경에서는 두 연산이 같은 GPU에서 순차 실행되지만,
    논문의 '분리된 가중치' 구조는 그대로 재현됩니다.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,      # 두 GPU 합산 출력 채널 수
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 2,        # 논문: GPU 2개 → groups=2
        use_lrn: bool = False,
        use_pool: bool = False,
    ):
        super().__init__()
        # groups=2 로 설정하면 PyTorch가 채널을 자동으로 반씩 나눠
        # 각 그룹이 독립적인 커널을 갖습니다 — 논문의 GPU 분할과 동일한 효과
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,       # ← 핵심: 채널 분리
        )
        self.relu = nn.ReLU(inplace=True)

        # 논문 3.3절: LRN은 Conv1·Conv2 뒤에만 적용
        self.lrn = (
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)
            if use_lrn else None
        )
        # 논문 3.4절: overlapping max-pool (kernel=3, stride=2)
        self.pool = (
            nn.MaxPool2d(kernel_size=3, stride=2)
            if use_pool else None
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.conv(x))
        if self.lrn:
            x = self.lrn(x)
        if self.pool:
            x = self.pool(x)
        return x


# ──────────────────────────────────────────────────────────────
# 2. Cross-GPU 합성곱 블록 (Conv3)
# ──────────────────────────────────────────────────────────────

class CrossConvBlock(nn.Module):
    """
    논문 3.5절 — "The kernels of the third convolutional layer are
    connected to all kernel maps in the second layer."

    두 GPU의 출력 전체(256ch)를 받아 연산합니다 (groups=1).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=1,    # cross-GPU: 모든 채널을 하나로 연산
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.conv(x))


# ──────────────────────────────────────────────────────────────
# 3. AlexNet 메인 모델
# ──────────────────────────────────────────────────────────────

class AlexNet(nn.Module):
    """
    논문 Figure 2 구조를 GPU 분할까지 포함해 완전 재현한 AlexNet.

    레이어별 채널 흐름 (논문 3.5절 기준):
      입력          (B,   3, 224, 224)
      Conv1 + Pool  (B,  96,  27,  27)   GPU0: 48ch / GPU1: 48ch
      Conv2 + Pool  (B, 256,  13,  13)   GPU0:128ch / GPU1:128ch
      Conv3         (B, 384,  13,  13)   cross-GPU (groups=1)
      Conv4         (B, 384,  13,  13)   GPU0:192ch / GPU1:192ch
      Conv5 + Pool  (B, 256,   6,   6)   GPU0:128ch / GPU1:128ch
      FC1           (B, 4096)
      FC2           (B, 4096)
      FC3           (B, num_classes)
    """

    def __init__(self, num_classes: int = 1000, dropout: float = 0.5):
        super().__init__()

        # ── 합성곱 레이어 ─────────────────────────────────────────────────

        # Conv1: 224×224×3 → 55×55×96 → pool → 27×27×96
        # 논문: "96 kernels of size 11×11×3, stride 4" + LRN + MaxPool
        # GPU0: 48ch, GPU1: 48ch (groups=2)
        self.conv1 = ParallelConvBlock(
            in_channels=3, out_channels=96,
            kernel_size=11, stride=4, padding=0,
            groups=2, use_lrn=True, use_pool=True,
        )

        # Conv2: 27×27×96 → 27×27×256 → pool → 13×13×256
        # 논문: "256 kernels of size 5×5×48" (GPU당 48ch 입력 → 128ch 출력)
        # GPU0: 128ch, GPU1: 128ch (groups=2)
        self.conv2 = ParallelConvBlock(
            in_channels=96, out_channels=256,
            kernel_size=5, stride=1, padding=2,
            groups=2, use_lrn=True, use_pool=True,
        )

        # Conv3: 13×13×256 → 13×13×384  ★ cross-GPU
        # 논문: "connected to all kernel maps in the second layer"
        self.conv3 = CrossConvBlock(
            in_channels=256, out_channels=384,
            kernel_size=3, padding=1,
        )

        # Conv4: 13×13×384 → 13×13×384
        # 논문: "384 kernels of size 3×3×192" (GPU당 192ch)
        # GPU0: 192ch, GPU1: 192ch (groups=2)
        self.conv4 = ParallelConvBlock(
            in_channels=384, out_channels=384,
            kernel_size=3, stride=1, padding=1,
            groups=2, use_lrn=False, use_pool=False,
        )

        # Conv5: 13×13×384 → 13×13×256 → pool → 6×6×256
        # 논문: "256 kernels of size 3×3×192" + MaxPool
        # GPU0: 128ch, GPU1: 128ch (groups=2)
        self.conv5 = ParallelConvBlock(
            in_channels=384, out_channels=256,
            kernel_size=3, stride=1, padding=1,
            groups=2, use_lrn=False, use_pool=True,
        )

        # ── 완전연결 레이어 ───────────────────────────────────────────────
        # 논문 4.2절: FC1·FC2에 Dropout(p=0.5) 적용
        # FC3는 Dropout 없음

        self.classifier = nn.Sequential(
            # FC1: 9216 → 4096
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            # FC2: 4096 → 4096
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            # FC3: 4096 → num_classes (Softmax는 CrossEntropyLoss에 내장)
            nn.Linear(4096, num_classes),
        )

        self._initialize_weights()

    # ── 순전파 ────────────────────────────────────────────────────────────

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, 3, 224, 224)
        Returns:
            logits: (B, num_classes)
        """
        x = self.conv1(x)           # (B,  96, 27, 27)
        x = self.conv2(x)           # (B, 256, 13, 13)
        x = self.conv3(x)           # (B, 384, 13, 13)  cross-GPU
        x = self.conv4(x)           # (B, 384, 13, 13)
        x = self.conv5(x)           # (B, 256,  6,  6)
        x = x.view(x.size(0), -1)  # (B, 9216)
        x = self.classifier(x)      # (B, num_classes)
        return x

    # ── 가중치 초기화 ─────────────────────────────────────────────────────

    def _initialize_weights(self):
        """
        논문 5절 가중치 초기화.

        - 모든 가중치: N(0, 0.01) 정규분포
        - Conv2·Conv4·Conv5 및 FC의 bias → 1  (ReLU에 양수 입력 보장)
        - 나머지 bias → 0
        """
        # bias=1 을 적용할 레이어
        bias_one_layers = (self.conv2.conv, self.conv4.conv,
                           self.conv5.conv)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                bias_val = 1.0 if m in bias_one_layers else 0.0
                nn.init.constant_(m.bias, bias_val)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 1.0)


# ──────────────────────────────────────────────────────────────
# 4. 학습 설정
# ──────────────────────────────────────────────────────────────

def build_optimizer(model: nn.Module) -> torch.optim.SGD:
    """
    논문 5절 — SGD 설정 그대로 재현.

    lr 스케줄:
      - 초기 lr = 0.01
      - 검증 오류가 개선되지 않으면 lr ÷ 10
      - 총 3회 감소 후 종료 (약 90 epoch)
    """
    return torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0005,  # 논문: "small amount of weight decay (0.0005)"
    )


def build_lr_scheduler(optimizer: torch.optim.SGD):
    """
    논문 5절 — 검증 오류 정체 시 lr ÷ 10.
    patience=5로 설정 (5 epoch 개선 없으면 감소).
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,     # lr × 0.1 = lr ÷ 10
        patience=5,
    )


# ──────────────────────────────────────────────────────────────
# 5. 학습 루프 (미니멀 예시)
# ──────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    1 epoch 학습. 평균 손실을 반환합니다.

    논문 5절:
      - 배치 크기 128
      - CrossEntropyLoss = log-softmax + NLL (논문의 softmax 출력과 동일)
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)          # (B, 1000)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    device: torch.device,
) -> dict:
    """
    Top-1 / Top-5 오류율 계산.

    논문 2절: "top-5 error rate is the fraction of test images
    for which the correct label is not among the five labels
    considered most probable by the model."
    """
    model.eval()
    correct_top1 = correct_top5 = total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)

        # Top-5 예측
        _, top5_preds = logits.topk(5, dim=1)
        labels_expanded = labels.view(-1, 1).expand_as(top5_preds)

        correct_top1 += (top5_preds[:, 0] == labels).sum().item()
        correct_top5 += top5_preds.eq(labels_expanded).any(dim=1).sum().item()
        total += labels.size(0)

    return {
        "top1_error": 1 - correct_top1 / total,
        "top5_error": 1 - correct_top5 / total,
    }


# ──────────────────────────────────────────────────────────────
# 6. 동작 확인
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    model = AlexNet(num_classes=1000).to(device)
    optimizer = build_optimizer(model)
    scheduler = build_lr_scheduler(optimizer)

    # ── 구조 출력 ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("AlexNet — 논문 GPU 분할 구조 포함 완전 재현")
    print("=" * 60)

    dummy = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model(dummy)

    print(f"입력  shape : {dummy.shape}")       # (2, 3, 224, 224)
    print(f"출력  shape : {out.shape}")          # (2, 1000)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"총 파라미터: {total_params:,}개")   # 논문: 약 6,000만 개

    print("\n── 레이어별 출력 shape 확인 ──")
    x = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        x = model.conv1(x); print(f"Conv1 출력: {tuple(x.shape)}")
        x = model.conv2(x); print(f"Conv2 출력: {tuple(x.shape)}")
        x = model.conv3(x); print(f"Conv3 출력: {tuple(x.shape)}  ← cross-GPU")
        x = model.conv4(x); print(f"Conv4 출력: {tuple(x.shape)}")
        x = model.conv5(x); print(f"Conv5 출력: {tuple(x.shape)}")
        x = x.view(x.size(0), -1)
        print(f"Flatten    : {tuple(x.shape)}")
        x = model.classifier(x)
        print(f"출력       : {tuple(x.shape)}")
    print("=" * 60)
