import os, cv2, math, torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

# 1. 디바이스 및 transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = Compose([
    Resize(240, 240),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# 2. 모델 로드 함수
def load_model(path):
    model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

# 3. 가장 큰 컴포넌트만 유지
def keep_largest(mask):
    mask = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8)

# 4. 원활 넓이 계산
def circular_segment_area(r, h):
    if h <= 0 or h >= 2 * r:
        return 0
    try:
        # theta = math.acos((r - h) / r)
        # return r**2 * theta - (r - h) * math.sqrt(2 * r * h - h**2)
        theta = 2 * math.acos((r - h) / r)
        return (1/2)*r**2 * (theta - math.sin(theta))
    except:
        return 0

# 5. 모델 경로
wheel_model = load_model("Tired_Tire/flat_tire_detector/wheel_model_finetuned_v2.pth") ###### 경로 바꿔서 사용
tire_model  = load_model("Tired_Tire/flat_tire_detector/tire_model_finetuned.pth") ###### 경로 바꿔서 사용

# 6. 테스트 이미지 (1장)
img_path = "Tired_Tire/flat_tire_detector/random_test/random5.jpg"  ##### ← 파일경로 바꿔서 사용
img = cv2.imread(img_path)
assert img is not None, "이미지 파일을 찾을 수 없습니다."

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (240, 240))
input_tensor = transform(image=img_resized)["image"].unsqueeze(0).to(device)

# 7. 마스크 예측
with torch.no_grad():
    wheel_mask = torch.sigmoid(wheel_model(input_tensor))[0, 0].cpu().numpy()
    tire_mask = torch.sigmoid(tire_model(input_tensor))[0, 0].cpu().numpy()
wheel_mask = keep_largest((wheel_mask > 0.5).astype(np.uint8))
tire_mask = keep_largest((tire_mask > 0.5).astype(np.uint8))

# 8. 기준점 계산
ys, xs = np.where(wheel_mask == 1)
cx, cy = int(xs.mean()), int(ys.mean())  # 초록점

yst, xst = np.where(tire_mask == 1)
far_idx = np.argmax((xst - cx)**2 + (yst - cy)**2)
fx, fy = int(xst[far_idx]), int(yst[far_idx])  # 보라점

yw = np.where(wheel_mask[:, cx] == 1)[0]
yt = np.where(tire_mask[:, cx] == 1)[0]
assert len(yw) > 0 and len(yt) > 0, "하단 점 계산 실패"
bottom_y_wheel = int(np.max(yw))  # 주황
bottom_y_tire  = int(np.max(yt))  # 파랑

# 9. 거리 및 면적 계산
r  = math.sqrt((fx - cx)**2 + (fy - cy)**2)
d1 = bottom_y_wheel - cy
d2 = bottom_y_tire - cy
h1 = r - d1
h2 = r - d2

A_ideal   = circular_segment_area(r, h1)
A_missing = circular_segment_area(r, h2)
A_actual  = A_ideal - A_missing
air_pct   = (A_actual / A_ideal * 100) if A_ideal > 0 else 0

# 10. 시각화
overlay = img_resized.copy()
cv2.circle(overlay, (cx, cy), 4, (0, 255, 0), -1)         # 초록 (중심)
cv2.circle(overlay, (fx, fy), 4, (255, 0, 255), -1)       # 보라 (가장 먼 점)
cv2.circle(overlay, (cx, bottom_y_wheel), 4, (0, 165, 255), -1)  # 주황 (휠 하단)
cv2.circle(overlay, (cx, bottom_y_tire), 4, (255, 0, 0), -1)     # 파랑 (타이어 하단)
cv2.putText(overlay, f"Air: {air_pct:.2f}%", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

plt.figure(figsize=(5, 5))
plt.imshow(overlay)
plt.title(f"공기압: {air_pct:.2f}%")
plt.axis("off")
plt.show()
