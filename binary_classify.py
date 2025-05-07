# not-tire / tire 에 대한 이진 분류
# MobileNetV2 활용 (앱 개발을 고려)
# Image Size: 224x224
# 타이어 객체가 이미지에 들어있는 지에 대한 여부를 판단

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ✅ 1. 하이퍼파라미터
BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001
DATA_DIR = './classify_dataset'

# ✅ 2. 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ✅ 3. 모델 로딩 (MobileNetV2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, 2)  # Binary classification
model = model.to(device)

# ✅ 4. Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ✅ 5. 학습 루프 + 진행률 + 최적 모델 저장
train_losses = []
test_accuracies = []
best_acc = 0.0  # 최적 성능 추적용

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    total_batches = len(train_loader)
    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # ✅ 진행 퍼센트 출력
        progress = (batch_idx + 1) / total_batches * 100
        print(f"\rProgress: {progress:.2f}%", end='')

    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # ✅ 6. 평가
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    test_accuracies.append(acc)

    print(f"\nEpoch Result => Loss: {train_loss:.4f} | Test Acc: {acc:.4f}")

    # ✅ 최적 모델 저장
    if acc > best_acc:
        best_acc = acc
        os.makedirs('model_weights', exist_ok=True)
        torch.save(model.state_dict(), 'model_weights/best_tire_classifier.pth')
        print(f"📌 Best model saved at Epoch {epoch+1} with Accuracy: {acc:.4f}")

# ✅ 7. 결과 시각화
plt.plot(train_losses, label='Train Loss')
plt.plot(test_accuracies, label='Test Accuracy')
plt.legend()
plt.title("Training Progress")
plt.show()

print(f"🎉 Training complete. Best Test Accuracy: {best_acc:.4f}")
