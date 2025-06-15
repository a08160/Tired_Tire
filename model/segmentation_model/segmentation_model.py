import os
import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp

# ===== 사용자 설정 =====
PIXEL_TO_MM = 0.1  # 1 픽셀 = 0.1mm
model_path = "best_model.pth"
image_path = "defect_data/defective_train/Defective (2).jpg"
min_area = 0  # 분석에 사용할 최소 크랙 면적 (pixel 단위)

# ===== 유틸 함수 =====
def sharpen_image(img_pil):
    img_cv = np.array(img_pil)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(img_cv, -1, kernel)
    return Image.fromarray(sharpened)

def slide_crop(image, crop_size=(227, 227), stride=50):
    w, h = image.size
    crops, coords = [], []
    for y in range(0, h - crop_size[1] + 1, stride):
        for x in range(0, w - crop_size[0] + 1, stride):
            box = (x, y, x + crop_size[0], y + crop_size[1])
            crops.append(image.crop(box))
            coords.append((x, y))
    return crops, coords

def stitch_mask(crop_preds, coords, image_size, crop_size=(227, 227)):
    full_mask = np.zeros(image_size, dtype=np.float32)
    count_map = np.zeros(image_size, dtype=np.float32)
    for crop_pred, (x, y) in zip(crop_preds, coords):
        crop_np = np.array(crop_pred.resize(crop_size))
        full_mask[y:y+crop_size[1], x:x+crop_size[0]] += crop_np
        count_map[y:y+crop_size[1], x:x+crop_size[0]] += 1.0
    stitched = (full_mask / np.maximum(count_map, 1)).astype(np.uint8)
    return stitched

# ===== 마스크 생성 및 필터링 =====
def generate_crop_pseudo_mask(model, device, image_path, crop_size=(227,227), stride=50, min_area=20):
    image = Image.open(image_path).convert("RGB")
    crops, coords = slide_crop(image, crop_size, stride)

    preds = []
    for crop in crops:
        crop = sharpen_image(crop)
        input_tensor = TF.to_tensor(TF.resize(crop, (256, 256)))
        input_tensor = TF.normalize(input_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        input_tensor = input_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = torch.sigmoid(model(input_tensor)).squeeze().cpu().numpy()
            pred_mask = (pred > 0.5).astype(np.uint8) * 255
            preds.append(Image.fromarray(pred_mask))

    stitched_mask = stitch_mask(preds, coords, image.size[::-1], crop_size)

    # 크랙 필터링
    binary_mask = (stitched_mask > 127).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)
    filtered_mask = np.zeros_like(binary_mask)
    crack_count = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered_mask[labels == i] = 1
            crack_count += 1

    crack_area_ratio = np.sum(filtered_mask) / (binary_mask.shape[0] * binary_mask.shape[1])
    return filtered_mask, crack_count, crack_area_ratio

# ===== 크랙 분석 (면적 mm²) =====
def analyze_cracks(mask, min_area=20):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    crack_areas_mm2 = []
    valid_labels = np.zeros_like(labels)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            crack_areas_mm2.append(area * PIXEL_TO_MM ** 2)
            valid_labels[labels == i] = len(crack_areas_mm2)
    return crack_areas_mm2, valid_labels

# ===== 위험도 시각화 및 점수 계산 =====
def visualize_crack_classes_transparent(image_path, crack_areas, labels, min_area=20):
    image = np.array(Image.open(image_path).convert("RGB"))
    overlay = np.zeros_like(image, dtype=np.uint8)

    filtered = [(i, area) for i, area in enumerate(crack_areas) if area >= min_area]
    if not filtered:
        return image, overlay, 0.0, "양호"

    # 면적 구간, 색상, 가중치 정의
    area_pixel_ranges = [(0, 35000), (35000, 70000), (70000, float("inf"))]
    colors = [(255, 255, 0), (255, 100, 0), (255, 0, 0)]
    weights = [0.15, 0.35, 0.5]
    area_sums_by_level = [0.0] * 3

    for idx, area_mm2 in filtered:
        area_pixels = area_mm2 / (PIXEL_TO_MM ** 2)
        for bin_idx, (low, high) in enumerate(area_pixel_ranges):
            if low <= area_pixels < high:
                overlay[labels == idx + 1] = colors[bin_idx]
                area_sums_by_level[bin_idx] += area_mm2
                break

    # 위험도 점수 계산
    total_area = sum(area_sums_by_level)
    if total_area == 0:
        risk_score = 0.0
    else:
        weighted_sum = sum(area * w for area, w in zip(area_sums_by_level, weights))
        risk_score = (weighted_sum / (total_area * max(weights))) * 100
        risk_score = 100-round(risk_score, 2)

    # 등급 산정
    if risk_score >= 70:
        grade = "양호"
    elif risk_score >= 35:
        grade = "주의"
    else:
        grade = "위험"

    # 최종 이미지 오버레이
    blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

    return blended, overlay, risk_score, grade

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    model = smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 마스크 및 분석 수행
    filtered_mask, crack_count, area_ratio = generate_crop_pseudo_mask(
        model, device, image_path,
        crop_size=(227, 227), stride=50, min_area=min_area
    )

    crack_areas_mm2, valid_labels = analyze_cracks(filtered_mask, min_area)

    blended_image, overlay_mask, risk_score, grade = visualize_crack_classes_transparent(
        image_path=image_path,
        crack_areas=crack_areas_mm2,
        labels=valid_labels,
        min_area=min_area
    )

    print(f"크랙 수: {crack_count}")
    print(f"위험도 점수: {risk_score} / 100")
    print(f"위험도 등급: {grade}")

    # 예: 이미지 확인용
    import matplotlib.pyplot as plt
    plt.imshow(blended_image)
    plt.title(f"Risk: {risk_score} ({grade})")
    plt.axis("off")
    plt.show()
