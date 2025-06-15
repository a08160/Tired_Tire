# ========================
# 필요한 모듈 임포트
# ========================
import torch
from torchvision import transforms
from PIL import Image

# 디바이스 설정 (GPU 사용 가능 시 GPU, 아니면 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# 단일 이미지 예측 함수
# ========================
def predict_image_scripted(image_path, model, transform, class_names, show_image=False):
    """
    TorchScript 모델을 사용한 단일 이미지 예측 함수

    Args:
        image_path (str): 예측할 이미지 경로
        model: 로드된 TorchScript 모델 객체
        transform: torchvision.transforms 객체
        class_names (list[str]): 클래스 이름 목록
        show_image (bool): 이미지와 예측 결과를 시각화할지 여부

    Returns:
        int: 예측된 클래스 인덱스 (0 또는 1)
    """
    # 이미지 불러오기 및 전처리
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

    predicted_label = predicted.item()

    # 이 부분은 사실상 필요 없음 → 이미 int로 나옴
    return predicted_label

# ========================
# 모델 로드 및 설정 함수
# ========================
def load_classification_model(model_script_path):
    model = torch.jit.load(model_script_path).to(device)
    return model

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

def get_class_names():
    return ['no_tire', 'tire']