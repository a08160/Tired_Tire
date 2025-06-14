import torch
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp

# 간단히 모델만 다시 정의
device = torch.device("cpu")
model = smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# 샘플 이미지 하나 불러서 crop
image = Image.open("defect_data/good_train/good (331).jpg").convert("RGB")
crop = image.crop((0, 0, 227, 227))  # 일단 첫 crop만 잘라서 테스트
crop = TF.to_tensor(TF.resize(crop, (256, 256)))
crop = TF.normalize(crop, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
input_tensor = crop.unsqueeze(0).to(device)

# 모델 추론
with torch.no_grad():
    pred = torch.sigmoid(model(input_tensor)).squeeze().cpu().numpy()
print("추론 성공!")
