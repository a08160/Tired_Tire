import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image


# 1. 하이퍼파라미터 및 설정
BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001
IMAGE_SIZE = (224, 224)
DATA_DIR = './classify_data'
MODEL_PATH = './model_weights/best_tire_classifier.pth'
TORCHSCRIPT_PATH = './model_weights/tire_classifier_script.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 2. 전처리 및 데이터셋 로딩
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# 3. 모델 정의
def get_model(num_classes=2):
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model.to(device)


# 4. 학습 함수
def train_model(model, train_loader, test_loader, epochs=EPOCHS, lr=LR):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses, test_accuracies = [], []
    best_acc = 0.0
    class_to_idx = train_dataset.class_to_idx

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        print(f"\nEpoch [{epoch+1}/{epochs}]")

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

            progress = (batch_idx + 1) / len(train_loader) * 100
            print(f"\rProgress: {progress:.2f}%", end='')

        avg_loss = running_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)

        # 평가
        acc = evaluate_model(model, test_loader, print_result=False)
        test_accuracies.append(acc)
        print(f"\nEpoch Result => Loss: {avg_loss:.4f} | Test Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            os.makedirs('model_weights', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_to_idx': class_to_idx
            }, MODEL_PATH)
            print(f"📌 Best model saved at Epoch {epoch+1} with Accuracy: {acc:.4f}")

    # 학습 곡선 시각화
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.legend()
    plt.title("Training Progress")
    plt.show()
    print(f"🎉 Training complete. Best Test Accuracy: {best_acc:.4f}")


# 5. 평가 함수
def evaluate_model(model, data_loader, print_result=True):
    model.eval()
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = correct / total
    if print_result:
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)
        disp.plot()
        plt.title("Confusion Matrix")
        plt.show()
        print(f"✅ Evaluation Accuracy: {acc:.4f}")
    return acc


# 6. 예측 함수 (단일 이미지용)
def predict_image(image_path, model, transform, class_names):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]


# 7. TorchScript 변환 함수 (앱용)
def export_to_torchscript(model):
    model.eval()
    example_input = torch.randn(1, 3, *IMAGE_SIZE).to(device)
    traced_model = torch.jit.trace(model, example_input)
    torch.jit.save(traced_model, TORCHSCRIPT_PATH)
    print(f"📦 TorchScript model saved to {TORCHSCRIPT_PATH}")


# 8. 실행 예시
if __name__ == "__main__":
    model = get_model()
    train_model(model, train_loader, test_loader)

    # 최적 모델 로딩
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    class_names = list(checkpoint['class_to_idx'].keys())
    
    # 예측 테스트
    test_image_path = 'test.jpg'  # 테스트 이미지 경로 지정
    if os.path.exists(test_image_path):
        result = predict_image(test_image_path, model, transform, class_names)
        print(f"🔍 Prediction for '{test_image_path}': {result}")

    # TorchScript 저장 (앱에서 사용 가능)
    export_to_torchscript(model)
