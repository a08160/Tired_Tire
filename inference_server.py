# inference_server.py

import math, torch, cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import segmentation_models_pytorch as smp
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

# FastAPI 앱 정의
app = FastAPI()

# 디바이스 설정 및 transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = Compose([
    Resize(240, 240),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# 모델 로드
def load_model(path):
    model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

wheel_model = load_model("flat_tire_detector/wheel_model_finetuned_v2.pth")
tire_model  = load_model("flat_tire_detector/tire_model_finetuned.pth")

# 도형 계산 함수
def keep_largest(mask):
    mask = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8)

def circular_segment_area(r, h):
    if h <= 0 or h >= 2 * r:
        return 0
    try:
        theta = 2 * math.acos((r - h) / r)
        return (1/2)*r**2 * (theta - math.sin(theta))
    except:
        return 0

# 실제 inference 함수
def run_inference(image_np: np.ndarray):
    img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (240, 240))
    input_tensor = transform(image=img_resized)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        wheel_mask = torch.sigmoid(wheel_model(input_tensor))[0, 0].cpu().numpy()
        tire_mask = torch.sigmoid(tire_model(input_tensor))[0, 0].cpu().numpy()
    wheel_mask = keep_largest((wheel_mask > 0.5).astype(np.uint8))
    tire_mask = keep_largest((tire_mask > 0.5).astype(np.uint8))

    ys, xs = np.where(wheel_mask == 1)
    cx, cy = int(xs.mean()), int(ys.mean())

    yst, xst = np.where(tire_mask == 1)
    far_idx = np.argmax((xst - cx)**2 + (yst - cy)**2)
    fx, fy = int(xst[far_idx]), int(yst[far_idx])

    yw = np.where(wheel_mask[:, cx] == 1)[0]
    yt = np.where(tire_mask[:, cx] == 1)[0]
    if len(yw) == 0 or len(yt) == 0:
        return {"error": "하단 점 계산 실패"}

    bottom_y_wheel = int(np.max(yw))
    bottom_y_tire  = int(np.max(yt))

    r  = math.sqrt((fx - cx)**2 + (fy - cy)**2)
    d1 = bottom_y_wheel - cy
    d2 = bottom_y_tire - cy
    h1 = r - d1
    h2 = r - d2

    A_ideal   = circular_segment_area(r, h1)
    A_missing = circular_segment_area(r, h2)
    A_actual  = A_ideal - A_missing
    air_pct   = (A_actual / A_ideal * 100) if A_ideal > 0 else 0

    return {"air_pct": round(air_pct, 2)}

# FastAPI 엔드포인트
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        file_bytes = np.asarray(bytearray(contents), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        result = run_inference(image)

        if "error" in result:
            return JSONResponse(content={"success": False, "message": result["error"]}, status_code=400)

        return JSONResponse(content={
            "success": True,
            "air_pct": result["air_pct"],
            "message": f"타이어 공기압 상태: {result['air_pct']}%"
        })

    except Exception as e:
        return JSONResponse(
            content={"success": False, "message": f"서버 오류: {str(e)}"},
            status_code=500
        )