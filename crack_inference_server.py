from fastapi import FastAPI, UploadFile, File
import uvicorn
import os
import shutil

from classification_test import (
    load_classification_model, get_transform, get_class_names, predict_image_scripted
)

from segmentation_model import (
    load_segmentation_model, run_segmentation_inference
)

import torch

# FastAPI 인스턴스 생성
app = FastAPI()

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 사전 로드 (서버 구동 시 1회)
classification_model = load_classification_model("mobilenet_classification.pt")
classification_transform = get_transform()
class_names = get_class_names()

segmentation_model = load_segmentation_model("best_model.pth", device)

# 기본 테스트 API
@app.get("/")
async def root():
    return {"message": "Tired Tire 균열 진단 서버 정상 작동 중"}

# 진단 API 엔드포인트
@app.post("/diagnosis/")
async def diagnose_tire(file: UploadFile = File(...)):
    # 1. 업로드된 파일 저장
    save_dir = "uploads"
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. classification 예측
    predicted_label = predict_image_scripted(
        image_path=file_path,
        model=classification_model,
        transform=classification_transform,
        class_names=class_names
    )

    result = {"classification": "no_tire" if predicted_label == 0 else "tire"}

    # 3. segmentation 조건부 실행
    if predicted_label == 1:
        blended_image, crack_count, risk_score, grade = run_segmentation_inference(
            model=segmentation_model,
            device=device,
            image_path=file_path,
            crop_size=(227, 227),
            stride=100,
            min_area=0
        )

        result.update({
            "crack_count": crack_count,
            "risk_score": risk_score,
            "grade": grade
        })
    else:
        result.update({
            "crack_count": None,
            "risk_score": None,
            "grade": None
        })

    return result

if __name__ == "__main__":
    uvicorn.run("crack_inference_server:app", host="0.0.0.0", port=8000, reload=True)
