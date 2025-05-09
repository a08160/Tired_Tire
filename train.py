import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms

class TireCrackDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(os.listdir(image_dir))  # 이미지 파일 목록
        self.mask_files = sorted(os.listdir(mask_dir))  # 마스크 파일 목록

        # 이미지와 마스크의 개수를 맞추기 위해 30개까지만 사용
        self.image_files = [f for f in self.image_files if f.endswith(('.jpg', '.png'))]
        self.mask_files = [f for f in self.mask_files if f.endswith('.png')]

        # 마스크가 없는 이미지들은 제외
        self.mask_files = self.mask_files[:30]
        self.image_files = self.image_files[:30]

        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # 마스크를 이진화 (1은 crack, 0은 no crack)
        mask = np.array(mask)
        mask[mask > 0] = 1

        sample = {"image": image, "mask": mask}

        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample


def collate_fn(batch):
    # 여러 개의 샘플(batch)을 합치는 collate 함수
    images = []
    masks = []
    
    for sample in batch:
        images.append(sample['image'])
        masks.append(sample['mask'])
    
    # 배치 차원 추가하여 텐서로 변환
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    
    return {'image': images, 'mask': masks}

# 모델 초기화 및 학습
class TireCrackModel:
    def __init__(self, image_dir, mask_dir, model_save_dir, batch_size=2, num_classes=2, lr=0.005, num_epochs=10, device=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.model_save_dir = model_save_dir
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.get_model(num_classes)
        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.transform = transforms.Compose([transforms.ToTensor()])
        
        # Dataset 및 DataLoader 설정
        self.dataset = TireCrackDataset(self.image_dir, self.mask_dir, self.transform)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

    def get_model(self, num_classes):
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, 256, num_classes)
        
        return model

    def train_model(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            for i, data in enumerate(self.dataloader):
                images = [data['image'].to(self.device)]
                masks = [data['mask'].to(self.device)]

                # Forward pass
                loss_dict = self.model(images, masks)

                # 손실 값 계산
                losses = sum(loss for loss in loss_dict.values())
                
                # 역전파 및 최적화
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

                if i % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.num_epochs}], Iter [{i}/{len(self.dataloader)}], Loss: {losses.item()}")

    def save_model(self, model_name="best_mask_rcnn.pth"):
        model_save_path = os.path.join(self.model_save_dir, model_name)
        torch.save(self.model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

# 학습 코드 예시
if __name__ == "__main__":
    # 데이터셋 및 모델 경로 설정
    image_dir = 'defect_data/defective_train'
    mask_dir = 'defect_data/mask_result'  # 초기 30개의 마스크
    model_save_dir = 'model_weights'

    # 모델 인스턴스 생성
    model = TireCrackModel(image_dir, mask_dir, model_save_dir, batch_size=2, num_classes=2, lr=0.005, num_epochs=10)

    # 모델 학습
    model.train_model()

    # 학습된 모델 저장
    model.save_model()
