import os
import numpy as np
import cv2
from tensorflow.keras.utils import Dataset  # tensorflow.keras로 변경

class TireCrackDataset(Dataset):
    def load_dataset(self, dataset_dir, subset):
        self.add_class("crack", 1, "defective")  # defective
        self.add_class("good", 2, "good")        # good

        # 이미지 디렉토리
        image_dir = os.path.join(dataset_dir, subset)
        mask_path = os.path.join(dataset_dir, "mask.png")  # 마스크 이미지 경로

        # 파일 리스트 (이미지)
        for file_name in os.listdir(image_dir):
            if file_name.endswith(".jpg") or file_name.endswith(".png"):
                image_path = os.path.join(image_dir, file_name)

                # 각 이미지를 추가
                self.add_image(
                    "crack",  # class name
                    image_id=file_name,  # 고유 이미지 ID
                    path=image_path,  # 이미지 경로
                    mask_path=mask_path  # 동일한 마스크 이미지 경로
                )

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        mask_path = image_info['mask_path']

        # 단일 마스크 이미지 로드
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_bin = np.where(mask_img > 127, 1, 0)  # 마스크 이진화 처리

        # 바운딩 박스 계산
        bbox = cv2.boundingRect(mask_bin)

        return mask_bin, np.array([1])  # 클래스 ID (여기서는 1개만 사용)

    def image_reference(self, image_id):
        return self.image_info[image_id]['path']


from mrcnn.model import MaskRCNN
from mrcnn.config import Config

# 모델 설정
class TireCrackConfig(Config):
    NAME = "tire_crack"
    NUM_CLASSES = 3  # 2 classes + background
    IMAGES_PER_GPU = 2
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 50
    LEARNING_RATE = 0.001
    MAX_GT_INSTANCES = 1  # 한 이미지에서 크랙은 하나
    DETECTION_MIN_CONFIDENCE = 0.9  # 최소 confidence 기준

config = TireCrackConfig()

# 모델 생성
model = MaskRCNN(mode="training", config=config, model_dir="./")

# 데이터셋 로드 (훈련, 검증 데이터)
dataset_train = TireCrackDataset()
dataset_train.load_dataset("defect_data", subset="train")  # train 폴더
dataset_train.prepare()

dataset_val = TireCrackDataset()
dataset_val.load_dataset("defect_data", subset="val")  # val 폴더
dataset_val.prepare()

# 모델 학습
model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=30, layers='all')

from pycocotools.cocoeval import COCOeval

def evaluate_model(model, dataset, subset="val"):
    result = []
    
    # 모델 예측 수행
    for image_id in dataset.image_ids:
        image = dataset.load_image(image_id)
        results = model.detect([image])
        
        for i, r in enumerate(results):
            bbox = r['rois']
            segm = r['masks']
            score = r['scores']
            labels = r['class_ids']
            
            for j in range(len(bbox)):
                result.append({
                    'image_id': image_id,
                    'category_id': labels[j],
                    'bbox': bbox[j],
                    'segmentation': segm[j],
                    'score': score[j]
                })
    
    # COCOeval을 통해 IoU 및 mAP 계산
    coco_gt = dataset.load_coco(subset=subset, useCats=True)
    coco_dt = coco_gt.loadRes(result)
    
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats

# 모델 평가
eval_stats = evaluate_model(model, dataset_val)
print(f"mAP: {eval_stats[0]}")  # mAP는 stats[0]에 저장됨

import matplotlib.pyplot as plt

# 학습된 모델 로드
model = MaskRCNN(mode="inference", config=config, model_dir="./")
model.load_weights("best_model.h5", by_name=True)

# 예측할 이미지 로드
image_id = 1  # 예시 이미지
image = dataset_val.load_image(image_id)

# 예측 실행
results = model.detect([image])

# 예측 결과 시각화
plt.figure(figsize=(10, 10))
plt.imshow(image)
ax = plt.gca()

for i, r in enumerate(results[0]['rois']):
    # 바운딩 박스를 그리기
    rect = plt.Rectangle((r[1], r[0]), r[3] - r[1], r[2] - r[0], fill=False, color='red', linewidth=3)
    ax.add_patch(rect)

plt.show()