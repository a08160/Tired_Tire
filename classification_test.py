import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
from classification import get_model, predict_image

# 1. 하이퍼파라미터 및 설정
BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001
IMAGE_SIZE = (224, 224)
DATA_DIR = './classify_data'
MODEL_PATH = './model_weights/best_tire_classifier.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 2. 전처리 및 데이터셋 로딩
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# 8. 실행 예시
if __name__ == "__main__":
    # 최적 모델 로딩
    model = get_model()

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    class_names = list(checkpoint['class_to_idx'].keys())
    
    # 예측 테스트
    test_image_path = './classify_data/test/tire/defective (8).jpg'  # 테스트 이미지 경로 지정
    if os.path.exists(test_image_path):
        result = predict_image(test_image_path, model, transform, class_names)
        print(f"🔍 Prediction for '{test_image_path}': {result}")
