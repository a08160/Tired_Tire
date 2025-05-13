# JSON 파일을 활용한 마스크 생성

import os
import json
import numpy as np
import cv2
from glob import glob
from PIL import Image

def vgg_json_to_mask(json_dir, image_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    json_files = glob(os.path.join(json_dir, "*.json"))

    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)

        for key, item in data.items():
            filename = item["filename"]
            image_path = os.path.join(image_dir, filename)

            if not os.path.exists(image_path):
                print(f"⚠ 이미지 없음: {image_path}")
                continue

            img = Image.open(image_path)
            w, h = img.size
            mask = np.zeros((h, w), dtype=np.uint8)

            regions = item.get("regions", [])
            if isinstance(regions, dict):
                regions = list(regions.values())

            for region in regions:
                shape = region["shape_attributes"]
                if shape["name"] == "polygon":
                    all_x = shape["all_points_x"]
                    all_y = shape["all_points_y"]
                    pts = np.array([[x, y] for x, y in zip(all_x, all_y)], dtype=np.int32)
                    cv2.fillPoly(mask, [pts], 255)

            save_path = os.path.join(save_dir, filename.replace('.jpg', '.png'))
            cv2.imwrite(save_path, mask)
            print(f"✅ 저장됨: {save_path}")

    print("🎉 VGG JSON → PNG 마스크 변환 완료")

if __name__ == "__main__":
    vgg_json_to_mask(
        json_dir="defect_data/defective_train_mask",     # .json 파일이 있는 폴더
        image_dir="defect_data/defective_train",         # 원본 이미지 폴더
        save_dir="mask_result"                                # 마스크 저장 폴더
    )
