import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

# 1. 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 모델 구조 (기존과 동일)
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
).to(device)

# 3. 저장된 fine-tuned 모델 weight 로드
model.load_state_dict(torch.load(
    "/content/drive/MyDrive/flat_data/wheel_model_finetuned_v2.pth",
    map_location=device
))
model.eval()

# 4. transform 정의 (학습과 동일해야 함)
transform = Compose([
    Resize(240, 240),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# 5. 후처리 함수: 가장 큰 컴포넌트만 유지
def keep_largest_connected_component(mask):
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.uint8)

# 6. 테스트 이미지 예측 및 시각화
test_folder = "/content/drive/MyDrive/flat_data/random_test"
file_list = sorted([f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

for fname in file_list:
    img_path = os.path.join(test_folder, fname)
    image = cv2.imread(img_path)
    if image is None:
        print(f"[스킵] 이미지 로드 실패: {fname}")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (240, 240))

    # 전처리
    transformed = transform(image=image_resized)
    input_tensor = transformed["image"].unsqueeze(0).to(device)

    # 예측
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output)[0, 0].cpu().numpy()
        pred_mask = (prob > 0.5).astype(np.uint8)

    # 후처리: 가장 큰 연결된 휠만 유지
    cleaned_mask = keep_largest_connected_component(pred_mask)

    # 오버레이
    overlay = image_resized.copy()
    overlay[cleaned_mask == 1] = [0, 0, 255]  # 파란색

    # 시각화
    plt.figure(figsize=(10, 4))
    plt.suptitle(fname, fontsize=10)

    plt.subplot(1, 2, 1)
    plt.imshow(image_resized)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title("Wheel Prediction Overlay (Post-processed)")
    plt.axis("off")

    plt.show()