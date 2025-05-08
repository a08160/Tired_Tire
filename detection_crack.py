import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.transforms.functional import to_pil_image
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 데이터셋 클래스 정의
class TireCrackDataset(Dataset):
    def __init__(self, dataset_dir, subset, transform=None):
        self.dataset_dir = dataset_dir
        self.subset = subset
        self.transform = transform
        self.image_paths = []
        self.load_dataset()

    def load_dataset(self):
        image_dir = os.path.join(self.dataset_dir, self.subset)
        for image_id in os.listdir(image_dir):
            if image_id.endswith(".jpg") or image_id.endswith(".png"):
                image_path = os.path.join(image_dir, image_id)
                self.image_paths.append(image_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Error loading image: {image_path}")

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        transform = transforms.Compose([
            transforms.Resize((512, 512)),  # 크기 먼저 조정
            transforms.ToTensor()
        ])
        image_tensor = transform(image_pil)

        target = {
            "boxes": torch.tensor([[0, 0, 512, 512]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
            "masks": torch.zeros((1, 512, 512), dtype=torch.float32),
        }

        return image_tensor, target

# 2. 모델 설정
def get_model():
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = MaskRCNNPredictor(in_features, 256, 2)

    return model.to(device)

# 3. 모델 학습 함수
def train_model():
    model = get_model()
    model.train()

    transform = transforms.Compose([
        transforms.Resize((512, 512))
    ])
    dataset_train = TireCrackDataset("./defect_data", "defective_train", transform=transform)

    # collate_fn 수정 → 크기 맞춰서 배치 형태 유지
    def collate_fn(batch):
        images, targets = zip(*batch)
        images = torch.stack([img.to(device) for img in images])  # 크기 통일 후 스택 적용
        targets = [{k: (v.to(device).long() if k == "labels" else v.to(device).float()) for k, v in t.items()} for t in targets]
        return images, targets

    dataset_train_loader = DataLoader(dataset_train, batch_size=2, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        for images, targets in dataset_train_loader:
            print(f"Training - Image Tensor Shape: {images.shape}")  # 디버깅용 출력
            
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            optimizer.step()

        print(f"Epoch {epoch} Loss: {losses.item()}")

    torch.save(model.state_dict(), "tire_crack_model.pth")
    print("모델이 저장되었습니다.")

# 4. 추론 함수
def detect_crack(image_path):
    model = get_model()
    model.load_state_dict(torch.load("tire_crack_model.pth"))
    model.eval()

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Error loading image: {image_path}")

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    transform = transforms.Compose([
        transforms.Resize((512, 512)),  
        transforms.ToTensor()
    ])
    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    print(f"Final Input Tensor Shape: {image_tensor.shape}")  # 배치 차원 확인

    with torch.no_grad():
        prediction = model(image_tensor)

    plt.imshow(np.array(image_pil))
    for i in range(len(prediction[0]["masks"])):
        score = prediction[0]["scores"][i].item()
        if score > 0.5:
            mask = prediction[0]["masks"][i, 0].mul(255).byte().cpu().numpy()
            plt.imshow(mask, alpha=0.3, cmap="jet")

    plt.title("Detected Crack Instances")
    plt.axis("off")
    plt.show()

    return prediction

# 5. 실행 엔트리포인트
if __name__ == "__main__":
    print("Main script running...")
    train_model()

    test_image_path = "Defective (1).jpg"
    result = detect_crack(test_image_path)
    print("Detection Results:", result)