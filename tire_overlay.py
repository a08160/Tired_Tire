from torch.utils.data import Dataset, DataLoader
import os
import cv2
import torch
import numpy as np
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

# 1. 전처리 정의 (원본 해상도 유지)
transform = Compose([
    Resize(240, 240),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# 2. Dataset 클래스
class TireSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.filenames = [f for f in os.listdir(mask_dir) if f.endswith("_mask.png")]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        mask_name = self.filenames[idx]
        image_name = mask_name.replace("_mask.png", ".jpg")  # 확장자에 맞게 수정 필요

        img_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mask = (mask > 0).astype(np.float32)  # 0 또는 1

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask.unsqueeze(0)  # mask shape: (1, H, W)

# 3. 경로 설정
image_dir = "/content/drive/MyDrive/flat_data/tire_images"
mask_dir = "/content/drive/MyDrive/flat_data/tire_masks"

# 4. Dataset, DataLoader 생성
train_dataset = TireSegmentationDataset(image_dir, mask_dir, transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

import segmentation_models_pytorch as smp
import torch.nn as nn
import torch

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# U-Net 모델 정의
model = smp.Unet(
    encoder_name="resnet34",         # backbone
    encoder_weights="imagenet",      # pretrained
    in_channels=3,                   # RGB
    classes=1                        # 출력 채널 = 1 (binary mask)
).to(device)

num_epochs = 20  # 원하는 에폭 수 설정

for epoch in range(num_epochs):
    model.train()  # 학습 모드
    epoch_loss = 0.0

    for images, masks in train_loader:
        images = images.to(device)                      # shape: (B, 3, H, W)
        masks = masks.to(device)                        # shape: (B, 1, H, W)

        optimizer.zero_grad()

        outputs = model(images)                         # shape: (B, 1, H, W), logits
        loss = criterion(outputs, masks)                # BCEWithLogitsLoss

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}] 🔹 Loss: {epoch_loss / len(train_loader):.4f}")
    
    

import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

# 1. 전처리 정의
transform = Compose([
    Resize(240, 240),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# 2. 테스트 폴더 설정
test_folder = "/content/drive/MyDrive/flat_data/random_test"
file_list = sorted([f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

# 3. 모델 예측 루프
model.eval()
for fname in file_list:
    img_path = os.path.join(test_folder, fname)
    image = cv2.imread(img_path)
    if image is None:
        print(f"[스킵] 이미지 로드 실패: {fname}")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (240, 240))

    transformed = transform(image=image_resized)
    input_tensor = transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output)[0, 0].cpu().numpy()  # (H, W)
        pred_mask = (prob > 0.5).astype(np.uint8) * 255

    # 시각화
    plt.figure(figsize=(12, 3))
    plt.suptitle(fname, fontsize=10)

    plt.subplot(1, 3, 1)
    plt.imshow(image_resized)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(prob, cmap="gray")
    plt.title("Probability Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, cmap="gray")
    plt.title("Binary Mask")
    plt.axis("off")

    plt.show()
