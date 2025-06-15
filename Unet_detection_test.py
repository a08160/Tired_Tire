import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

predict_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TorchScript 모델 로딩
model = torch.jit.load("tire_crack_model_scripted.pt").to(device)
model.eval()

def predict_and_draw_cracks(model, image_path, transform, device):
    orig_img = Image.open(image_path).convert("RGB")
    orig_img_np = np.array(orig_img)

    # Transform 적용
    augmented = transform(image=orig_img_np)
    input_tensor = augmented['image'].unsqueeze(0).to(device)

    # 예측
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.sigmoid(output)[0, 0].cpu().numpy()
        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255

    # OpenCV용 변환
    resized_img = cv2.resize(orig_img_np, (256, 256))
    orig_img_cv = cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR)

    # 윤곽선 검출
    contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 윤곽선 그리기 (선택사항)
    cv2.drawContours(orig_img_cv, contours, -1, (0, 0, 255), 1)  # 초록색 윤곽선, 굵기 1

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < 50:
            continue
        cv2.rectangle(orig_img_cv, (x, y), (x + w, y + h), (0, 0, 255), 1)  # 빨간 테두리, 굵기 1
        cv2.putText(orig_img_cv, "crack", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)  # 글씨 작게, 얇게

    # 결과 출력
    result_img = cv2.cvtColor(orig_img_cv, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 6))
    plt.imshow(result_img)
    plt.axis("off")
    plt.title("Detected Cracks with Contours & Bounding Boxes")
    plt.show()


# 원하는 테스트 이미지 경로 입력
image_path = "defect_data/defective_train/Defective (40).jpg"
predict_and_draw_cracks(model, image_path, predict_transform, device)
