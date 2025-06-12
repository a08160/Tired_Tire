import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# 데이터 전처리 및 로더
# ========================
def get_dataloaders(train_dir, test_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, transform, train_dataset.classes

# ========================
# 모델 정의 (MobileNetV2)
# ========================
def build_model(num_classes=2):
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model.to(device)

# ========================
# 학습 루프
# ========================
def train(model, train_loader, epochs=5, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Accuracy={acc:.4f}")

# ========================
# 평가 및 TorchScript 저장
# ========================
def evaluate_and_export(model, test_loader, class_names, save_path="mobilenet_scripted.pt"):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\n[분류 리포트]")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # TorchScript 변환 및 저장
    example_input = torch.randn(1, 3, 224, 224).to(device)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save(save_path)
    print(f"TorchScript 모델 저장 완료: {save_path}")
    return save_path

def run_mobile_pipeline(train_dir, test_dir, save_script_path="mobilenet_classification.pt"):
    train_loader, test_loader, transform, class_names = get_dataloaders(train_dir, test_dir)
    model = build_model(num_classes=len(class_names))
    train(model, train_loader, epochs=5)
    model_script_path = evaluate_and_export(model, test_loader, class_names, save_path=save_script_path)
    return model_script_path, transform, class_names

train_dir = "classify_data/train"
test_dir = "classify_data/test"
model_script_path, transform, class_names = run_mobile_pipeline(train_dir, test_dir)
