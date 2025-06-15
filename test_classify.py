# ========================
# 필요한 모듈 임포트
# ========================
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 디바이스 설정 (GPU 사용 가능 시 GPU, 아니면 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# 단일 이미지 예측 함수
# ========================
def predict_image_scripted(image_path, model_script_path, transform, class_names, show_image=False):
    """
    TorchScript 모델을 사용한 단일 이미지 예측 함수

    Args:
        image_path (str): 예측할 이미지 경로
        model_script_path (str): TorchScript 모델 경로 (.pt)
        transform: torchvision.transforms 객체
        class_names (list[str]): 클래스 이름 목록
        show_image (bool): 이미지와 예측 결과를 시각화할지 여부

    Returns:
        int: 예측된 클래스 인덱스
    """
    # 이미지 불러오기 및 전처리
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # TorchScript 모델 로드
    model = torch.jit.load(model_script_path).to(device)
    model.eval()

    # 예측
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

    predicted_label = predicted.item()

    if predicted_label == "no_tire":
        predicted_label = 0
    elif predicted_label == "tire":
        predicted_label = 1
        
    return predicted_label

# ========================
# 예시 실행 코드
# ========================
if __name__ == "__main__":
    # 이미지 전처리 방식
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # 클래스 이름 목록 (예: 0 = no_tire, 1 = tire)
    class_names = ['no_tire', 'tire']

    # 모델 경로와 예측할 이미지
    model_script_path = "mobilenet_classification.pt"
    image_path = "test/rabbit.jpg"

    # 예측 실행
    result = predict_image_scripted(image_path, model_script_path, transform, class_names, show_image=True)
    print(result)