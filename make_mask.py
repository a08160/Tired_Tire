import os
import cv2
import numpy as np

def create_crack_mask(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. CLAHE로 대비 향상 (clipLimit 증가)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))  
    gray = clahe.apply(gray)

    # 2. Gaussian Blur로 노이즈 제거 (커널 크기 조정)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  

    # 3. Adaptive Thresholding 적용
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 4. Canny Edge Detection 최적화
    edges = cv2.Canny(binary, threshold1=10, threshold2=50)  

    # 5. Morphological Closing으로 균열 연결 (iterations 증가)
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)  

    # 6. Morphological Dilation 추가하여 균열 강조
    kernel_dilate = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(closed, kernel_dilate, iterations=1)  

    # 7. 바이너리 마스크 생성
    mask = np.where(dilated > 0, 255, 0).astype(np.uint8)
    return mask

def generate_masks(image_dir, mask_dir):
    os.makedirs(mask_dir, exist_ok=True)
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, filename)
            mask = create_crack_mask(image_path)
            save_path = os.path.join(mask_dir, os.path.splitext(filename)[0] + ".png")  
            cv2.imwrite(save_path, mask)
            print(f"Saved mask to {save_path}")

def main():
    base_path = "defect_data"
    target_folders = "defective_train"

    image_dir = os.path.join(base_path, target_folders)
    mask_dir = os.path.join(base_path, "defective_train_mask")

    if os.path.exists(image_dir):
        print(f"Processing {image_dir} ...")
        generate_masks(image_dir, mask_dir)
    else:
        print(f"Skipped: {image_dir} does not exist")

if __name__ == "__main__":
    main()