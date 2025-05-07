import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class TireClassifier:
    def __init__(self, model_path: str, device: str = None):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
    
    def _load_model(self, model_path: str):
        class CNNModel(nn.Module):
            def __init__(self):
                super(CNNModel, self).__init__()
                self.model = models.mobilenet_v2(pretrained=False)
                self.model.classifier[1] = nn.Linear(self.model.last_channel, 2)

            def forward(self, x):
                return self.model(x)

        model = CNNModel()
        model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
        model.to(self.device)
        model.eval()
        return model

    def predict(self, image_path: str) -> str:
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            _, predicted = torch.max(output, 1)

        if predicted.item() == 0:
            result = 0 # 타이어 아님
        else:
            result = 1 # 타이어

        return result

'''
사용 예시
if __name__ == "__main__":
    classifier = TireClassifier(model_path='model_weights/best_tire_classifier.pth')
    result = classifier.predict('images.jpg')
    print("예측 결과:", result)
'''