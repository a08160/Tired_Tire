import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
import imgaug

# 1. 이미지 전처리: 엣지 검출 및 추가 개선
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 히스토그램 평활화
    img_eq = cv2.equalizeHist(img)
    blur = cv2.GaussianBlur(img_eq, (5, 5), 0)
    edges = cv2.Canny(blur, threshold1=50, threshold2=150)
    
    # Adaptive Thresholding
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 결과 시각화
    plt.subplot(1, 2, 1)
    plt.imshow(edges, cmap='gray')
    plt.title("Canny Edge")
    plt.subplot(1, 2, 2)
    plt.imshow(thresh, cmap='gray')
    plt.title("Thresholded Image")
    plt.show()

    return edges

# 2. Mask R-CNN 데이터셋 준비 (데이터 증강 포함)
class TireCrackDataset(Dataset):
    def load_dataset(self, dataset_dir, subset):
        self.add_class("tire", 0, "crack")  # 크랙
        self.add_class("tire", 1, "good")   # 정상 타이어

        # 'defective_train'과 'good_train' 데이터 로드
        for subset_name in ['defective_train', 'good_train']:
            image_dir = os.path.join(dataset_dir, subset_name)
            for image_id in os.listdir(image_dir):
                if image_id.endswith(".jpg") or image_id.endswith(".png"):
                    image_path = os.path.join(image_dir, image_id)
                    self.add_image("tire", image_id=image_id, path=image_path)

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        image_path = image_info['path']
        
        # 마스크 생성
        mask = preprocess_image(image_path)
        mask = np.expand_dims(mask, axis=-1)  # (height, width, 1)

        # 크랙 데이터와 정상 데이터를 구별
        if 'defective' in image_path:
            return mask, np.array([1])  # 크랙 클래스 ID
        else:
            return mask, np.array([0])  # 정상 타이어 클래스 ID

    def augment(self, image, mask):
        seq = imgaug.augmenters.Sequential([
            imgaug.augmenters.Fliplr(0.5),  # 좌우 반전
            imgaug.augmenters.Affine(rotate=(-45, 45)),  # 회전
            imgaug.augmenters.Scale((0.8, 1.2))  # 크기 조정
        ])
        return seq(image=image, segmentation_maps=mask)

# 3. Mask R-CNN 구성
class TireCrackConfig(Config):
    NAME = "tire_crack"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 3  # 1: 크랙, 0: 정상 타이어, 배경은 제외
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 50

# 4. Mask R-CNN 모델 학습
def train_model():
    config = TireCrackConfig()
    model = MaskRCNN(mode="training", config=config, model_dir='./logs')

    # 학습용 데이터셋 로드
    dataset_train = TireCrackDataset()
    dataset_train.load_dataset('./defect_data', 'defective_train')
    dataset_train.prepare()

    # 데이터 증강 적용
    augmented_image, augmented_mask = dataset_train.augment(dataset_train.load_image(0), dataset_train.load_mask(0)[0])

    # 모델 훈련
    model.train(dataset_train, dataset_train,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                layers='heads')

# 5. Mask R-CNN 추론 (테스트)
def detect_crack(image_path):
    config = TireCrackConfig()
    model = MaskRCNN(mode="inference", config=config, model_dir='./logs')
    model.load_weights('./logs/tire_crack20180507T1014/mask_rcnn_tire_crack_0010.h5', by_name=True)

    # 이미지 로드
    image = cv2.imread(image_path)
    results = model.detect([image])

    r = results[0]
    plt.imshow(image)
    for i in range(len(r['rois'])):
        # 바운딩 박스와 마스크 시각화
        y1, x1, y2, x2 = r['rois'][i]
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2))
        mask = r['masks'][:, :, i]
        plt.imshow(mask, alpha=0.5, cmap='jet')
    
    plt.title("Detected Crack Instances")
    plt.axis('off')
    plt.show()

    return r

if __name__ == "__main__":
    # 모델 학습: python --version
    # train_model()

    # 테스트 이미지로 추론:
    test_image = './defect_data/defective_test/Defective (2).jpg'
    result = detect_crack(test_image)
    print("Detection Results:", result)
    # 결과 시각화
    plt.imshow(result['masks'][:, :, 0], cmap='jet', alpha=0.5)
    plt.title("Detected Crack Mask")
    plt.axis('off')
    plt.show()
