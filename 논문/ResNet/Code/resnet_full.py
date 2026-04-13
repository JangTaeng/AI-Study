"""
Deep Residual Learning for Image Recognition (ResNet) - PyTorch 구현
논문: He et al., arXiv:1512.03385 (ILSVRC 2015 1위)

구현 내용:
  - BasicBlock       : ResNet-18, ResNet-34용 2-layer 블록
  - Bottleneck       : ResNet-50, ResNet-101, ResNet-152용 3-layer 블록
  - ResNet           : 범용 ResNet 클래스
  - 팩토리 함수      : resnet18 / resnet34 / resnet50 / resnet101 / resnet152
  - 학습 루프        : ImageNet 논문 설정 그대로 (SGD, lr=0.1, wd=1e-4, BN)
  - CIFAR-10 버전    : 논문 Section 4.2 구조 (6n+2 layer)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


# ──────────────────────────────────────────────
# 1. 잔차 블록 (Building Blocks)
# ──────────────────────────────────────────────

class BasicBlock(nn.Module):
    """
    ResNet-18 / ResNet-34용 기본 블록 (논문 Figure 2 왼쪽)

    구조:
        x ──┐
            │  3x3 conv → BN → ReLU
            │  3x3 conv → BN
        x ──┘ (shortcut)
        y = F(x) + x  →  ReLU

    expansion: 입력/출력 채널 비율 (BasicBlock은 1배)
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        # 첫 번째 합성곱: stride로 공간 크기 조절 가능
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 두 번째 합성곱: 항상 stride=1
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # 차원이 달라질 때 shortcut을 맞춰주는 projection (논문 식 2)
        # stride != 1  → 공간 크기가 달라짐
        # 채널 수가 달라짐 → 1x1 conv로 맞춤
        self.downsample = downsample

    def forward(self, x):
        identity = x  # shortcut 저장

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # shortcut 연결: 차원이 다를 때 projection 적용
        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity  # F(x) + x
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """
    ResNet-50 / ResNet-101 / ResNet-152용 보틀넥 블록 (논문 Figure 5 오른쪽)

    구조:
        x ──┐
            │  1x1 conv → BN → ReLU   (채널 축소: out_channels)
            │  3x3 conv → BN → ReLU   (주요 연산: out_channels)
            │  1x1 conv → BN           (채널 복원: out_channels * 4)
        x ──┘ (shortcut)
        y = F(x) + x  →  ReLU

    expansion=4: 출력 채널 = out_channels * 4
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        # 1×1: 채널 축소
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)

        # 3×3: 핵심 합성곱 (stride 적용)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1×1: 채널 복원 (expansion=4 배)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion,
            kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


# ──────────────────────────────────────────────
# 2. ResNet 본체 (ImageNet 버전)
# ──────────────────────────────────────────────

class ResNet(nn.Module):
    """
    논문 Table 1의 ImageNet용 ResNet 전체 구조

    Args:
        block       : BasicBlock 또는 Bottleneck
        layers      : [conv2_x 반복 수, conv3_x, conv4_x, conv5_x]
                      예) ResNet-50 → [3, 4, 6, 3]
        num_classes : 분류 클래스 수 (ImageNet=1000)

    전체 흐름:
        입력(224x224)
        → conv1: 7x7, 64, stride=2   → 112x112
        → maxpool: 3x3, stride=2      → 56x56
        → conv2_x (layer1)            → 56x56
        → conv3_x (layer2, stride=2)  → 28x28
        → conv4_x (layer3, stride=2)  → 14x14
        → conv5_x (layer4, stride=2)  → 7x7
        → avgpool (global)            → 1x1
        → fc (1000-d)
    """

    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64  # conv2_x 시작 채널 수

        # ── conv1: 7x7, 64, stride=2 ──
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ── conv2_x ~ conv5_x ──
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # ── Global Average Pooling + FC ──
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # ── 가중치 초기화 (논문 참조: He et al. ICCV 2015) ──
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        하나의 conv_x 단계를 구성하는 함수

        첫 번째 블록: stride 및 채널 변경 → downsample(projection) 필요할 수 있음
        이후 블록들: stride=1, 채널 동일
        """
        downsample = None

        # shortcut과 메인 경로의 채널/크기가 다를 때 projection 생성
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # 첫 블록 (stride, downsample 적용)
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        # 나머지 블록 (stride=1, downsample 없음)
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # stem
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        # residual stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ──────────────────────────────────────────────
# 3. 팩토리 함수 (논문 Table 1 구조)
# ──────────────────────────────────────────────

def resnet18(num_classes=1000):
    """ResNet-18: BasicBlock × [2, 2, 2, 2]  |  1.8G FLOPs"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet34(num_classes=1000):
    """ResNet-34: BasicBlock × [3, 4, 6, 3]  |  3.6G FLOPs"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def resnet50(num_classes=1000):
    """ResNet-50: Bottleneck × [3, 4, 6, 3]  |  3.8G FLOPs"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def resnet101(num_classes=1000):
    """ResNet-101: Bottleneck × [3, 4, 23, 3]  |  7.6G FLOPs"""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)

def resnet152(num_classes=1000):
    """ResNet-152: Bottleneck × [3, 8, 36, 3]  |  11.3G FLOPs"""
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


# ──────────────────────────────────────────────
# 4. CIFAR-10 버전 ResNet (논문 Section 4.2)
# ──────────────────────────────────────────────

class CIFARBasicBlock(nn.Module):
    """
    CIFAR-10용 잔차 블록 (논문 Section 4.2)

    - 3x3 conv 두 개
    - shortcut: 항상 identity (option A, zero-padding)
    - BN + ReLU 구조 동일
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Option A: zero-padding shortcut (파라미터 추가 없음)
        if self.stride != 1 or out.shape[1] != identity.shape[1]:
            # 공간 크기 맞추기: stride=2 → average pooling
            identity = F.avg_pool2d(identity, kernel_size=1, stride=self.stride)
            # 채널 맞추기: zero-padding
            pad = out.shape[1] - identity.shape[1]
            identity = F.pad(identity, (0, 0, 0, 0, 0, pad))

        out = out + identity
        out = self.relu(out)
        return out


class CIFARResNet(nn.Module):
    """
    CIFAR-10용 ResNet (논문 Section 4.2)

    총 레이어 수 = 6n + 2  (n: 각 stage의 블록 수)
      n=3  → 20층  (ResNet-20)
      n=5  → 32층  (ResNet-32)
      n=7  → 44층  (ResNet-44)
      n=9  → 56층  (ResNet-56)
      n=18 → 110층 (ResNet-110)

    구조:
        입력(32x32)
        → 3x3 conv, 16          → 32x32
        → stage1 (16 filters)   → 32x32
        → stage2 (32 filters)   → 16x16
        → stage3 (64 filters)   → 8x8
        → global avgpool        → 1x1
        → fc (10-d)

    논문 Table (CIFAR-10):
        output size | layers  | filters
        32×32       | 1 + 2n  | 16
        16×16       | 2n      | 32
        8×8         | 2n      | 64
    """
    def __init__(self, n, num_classes=10):
        super().__init__()
        # 초기 conv
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # 3개의 stage
        self.stage1 = self._make_stage(16, 16, n, stride=1)
        self.stage2 = self._make_stage(16, 32, n, stride=2)
        self.stage3 = self._make_stage(32, 64, n, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        # 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_stage(self, in_ch, out_ch, n, stride):
        layers = [CIFARBasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, n):
            layers.append(CIFARBasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def cifar_resnet20():  return CIFARResNet(n=3)   # 20층
def cifar_resnet32():  return CIFARResNet(n=5)   # 32층
def cifar_resnet44():  return CIFARResNet(n=7)   # 44층
def cifar_resnet56():  return CIFARResNet(n=9)   # 56층
def cifar_resnet110(): return CIFARResNet(n=18)  # 110층


# ──────────────────────────────────────────────
# 5. 학습 루프 (논문 Section 3.4 설정)
# ──────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    """한 에폭 학습"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, 100. * correct / total


def evaluate(model, loader, criterion, device):
    """검증/테스트 평가"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)

    return total_loss / total, 100. * correct / total


def get_cifar10_loaders(batch_size=128, data_root='./data'):
    """
    CIFAR-10 데이터로더 생성 (논문 Section 4.2 augmentation)

    학습: 4픽셀 패딩 후 32x32 랜덤 크롭, 수평 플립
    테스트: 원본 32x32 그대로
    """
    # 논문과 동일한 전처리
    train_transform = transforms.Compose([
        transforms.Pad(4),                       # 각 면에 4픽셀 패딩
        transforms.RandomCrop(32),               # 32x32 랜덤 크롭
        transforms.RandomHorizontalFlip(),       # 수평 플립
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        ),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        ),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    return train_loader, test_loader


def train_cifar10(model_name='resnet20', epochs=164, batch_size=128):
    """
    CIFAR-10 학습 (논문 Section 4.2 설정 그대로)

    - SGD: momentum=0.9, weight_decay=1e-4
    - lr: 0.1 시작 → 32k, 48k iteration에서 10으로 나눔
    - 총 64k iteration (약 164 에폭)
    - ResNet-110은 처음 400 iteration을 lr=0.01로 웜업
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 모델 선택
    model_map = {
        'resnet20':  cifar_resnet20,
        'resnet32':  cifar_resnet32,
        'resnet44':  cifar_resnet44,
        'resnet56':  cifar_resnet56,
        'resnet110': cifar_resnet110,
    }
    model = model_map[model_name]().to(device)
    print(f"Model: {model_name} | Params: {sum(p.numel() for p in model.parameters()):,}")

    train_loader, test_loader = get_cifar10_loaders(batch_size)
    criterion = nn.CrossEntropyLoss()

    # 논문: SGD, momentum=0.9, weight_decay=1e-4, dropout 사용 안 함
    # ResNet-110 웜업: 처음에는 lr=0.01
    init_lr = 0.01 if model_name == 'resnet110' else 0.1
    optimizer = optim.SGD(
        model.parameters(),
        lr=init_lr,
        momentum=0.9,
        weight_decay=1e-4
    )

    # lr 스케줄: 32k, 48k iteration → 에폭 기준으로 근사
    # 64k iter / (50000/128 배치) ≈ 164 에폭
    # 32k → 약 82 에폭, 48k → 약 122 에폭
    scheduler = MultiStepLR(optimizer, milestones=[82, 122], gamma=0.1)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        # ResNet-110 웜업: 400 iteration ≈ 1 에폭 후 lr을 0.1로
        if model_name == 'resnet110' and epoch == 2:
            for g in optimizer.param_groups:
                g['lr'] = 0.1

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss,  test_acc  = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f'{model_name}_best.pth')

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch [{epoch:3d}/{epochs}] "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}% | "
                f"Best: {best_acc:.2f}%"
            )

    print(f"\n학습 완료 | 최고 테스트 정확도: {best_acc:.2f}%")
    return model


# ──────────────────────────────────────────────
# 6. 유틸리티 함수
# ──────────────────────────────────────────────

def count_parameters(model):
    """모델 파라미터 수 출력"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"전체 파라미터: {total:,}")
    print(f"학습 가능 파라미터: {trainable:,}")
    return total


def model_summary(model, input_size=(1, 3, 224, 224)):
    """간단한 모델 구조 및 출력 크기 확인"""
    device = next(model.parameters()).device
    x = torch.randn(*input_size).to(device)
    print(f"입력 크기: {list(x.shape)}")

    # 각 주요 레이어의 출력 크기 확인
    hooks = []
    outputs = {}

    def make_hook(name):
        def hook(module, inp, out):
            outputs[name] = list(out.shape)
        return hook

    if hasattr(model, 'conv1'):  # ImageNet ResNet
        hooks.append(model.conv1.register_forward_hook(make_hook('conv1')))
        hooks.append(model.maxpool.register_forward_hook(make_hook('maxpool')))
        hooks.append(model.layer1.register_forward_hook(make_hook('layer1')))
        hooks.append(model.layer2.register_forward_hook(make_hook('layer2')))
        hooks.append(model.layer3.register_forward_hook(make_hook('layer3')))
        hooks.append(model.layer4.register_forward_hook(make_hook('layer4')))
        hooks.append(model.avgpool.register_forward_hook(make_hook('avgpool')))

    model.eval()
    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    for name, shape in outputs.items():
        print(f"  {name:12s}: {shape}")


# ──────────────────────────────────────────────
# 7. 메인 실행
# ──────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("ResNet PyTorch 구현 (논문: He et al., 2015)")
    print("=" * 60)

    # ── ImageNet ResNet 구조 확인 ──
    print("\n[1] ImageNet ResNet 모델 파라미터 수")
    print("-" * 40)
    for name, fn in [
        ('ResNet-18',  resnet18),
        ('ResNet-34',  resnet34),
        ('ResNet-50',  resnet50),
        ('ResNet-101', resnet101),
        ('ResNet-152', resnet152),
    ]:
        model = fn(num_classes=1000)
        params = sum(p.numel() for p in model.parameters())
        print(f"  {name:12s}: {params:>12,} 파라미터")

    # ── ResNet-50 구조 상세 확인 ──
    print("\n[2] ResNet-50 레이어별 출력 크기 (224x224 입력)")
    print("-" * 40)
    model = resnet50(num_classes=1000)
    model_summary(model, input_size=(1, 3, 224, 224))

    # ── CIFAR-10 ResNet 파라미터 수 ──
    print("\n[3] CIFAR-10 ResNet 모델 파라미터 수")
    print("-" * 40)
    for name, fn in [
        ('ResNet-20',  cifar_resnet20),
        ('ResNet-32',  cifar_resnet32),
        ('ResNet-44',  cifar_resnet44),
        ('ResNet-56',  cifar_resnet56),
        ('ResNet-110', cifar_resnet110),
    ]:
        model = fn()
        params = sum(p.numel() for p in model.parameters())
        print(f"  {name:12s}: {params:>10,} 파라미터")

    # ── 잔차 연결 동작 확인 ──
    print("\n[4] 잔차 연결 동작 확인")
    print("-" * 40)
    model = cifar_resnet20()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    print(f"  입력 크기:  {list(x.shape)}")
    print(f"  출력 크기:  {list(out.shape)}  (배치=2, 클래스=10)")

    # ── CIFAR-10 학습 실행 예시 ──
    print("\n[5] CIFAR-10 학습 (ResNet-20, 5 에폭 데모)")
    print("-" * 40)
    print("  실제 학습을 시작하려면 epochs=164로 변경하세요.")
    print("  아래 코드를 활성화하면 학습이 시작됩니다:\n")
    print("  train_cifar10(model_name='resnet20', epochs=164)")

    # 실제 학습 실행 (주석 해제 시 시작)
    # train_cifar10(model_name='resnet20', epochs=164)
