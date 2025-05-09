import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.optim import Adam
import torchvision.transforms.functional as TF
from tqdm import tqdm

# --------------------------------------------------
# Dataset 정의
# --------------------------------------------------
class TireCrackDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None, target_size=(256, 256), use_mask=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
        self.transform = transform
        self.target_size = target_size
        self.use_mask = use_mask and mask_dir is not None

        if self.use_mask:
            self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
            paired = list(zip(self.image_files, self.mask_files))[:30]
            self.image_files, self.mask_files = zip(*paired)
        else:
            self.mask_files = [None] * len(self.image_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB").resize(self.target_size)
        image_tensor = self.transform(image) if self.transform else TF.to_tensor(image)

        if self.use_mask and self.mask_files[idx]:
            mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
            mask = Image.open(mask_path).convert("L").resize(self.target_size)
            mask = np.array(mask)
            mask[mask > 0] = 1

            boxes = self.get_bounding_boxes(mask)
            return {
                "image": image_tensor,
                "mask": torch.tensor(mask, dtype=torch.uint8),
                "boxes": boxes,
                "labels": torch.ones(len(boxes), dtype=torch.int64)
            }
        else:
            return {
                "image": image_tensor,
                "image_name": self.image_files[idx]
            }

    def get_bounding_boxes(self, mask):
        boxes = []
        unique_labels = np.unique(mask)
        for label in unique_labels:
            if label == 0:
                continue
            mask_idx = np.where(mask == label)
            min_x, max_x = np.min(mask_idx[1]), np.max(mask_idx[1])
            min_y, max_y = np.min(mask_idx[0]), np.max(mask_idx[0])
            boxes.append([min_x, min_y, max_x, max_y])
        return torch.tensor(boxes, dtype=torch.float32)

def collate_fn(batch):
    has_mask = 'mask' in batch[0]
    if has_mask:
        return {
            'image': torch.stack([b['image'] for b in batch]),  # 이미지 배치를 텐서로 처리
            'mask': torch.stack([b['mask'] for b in batch]),
            'boxes': [b['boxes'] for b in batch],
            'labels': [b['labels'] for b in batch]
        }
    else:
        return {
            'image': torch.stack([b['image'] for b in batch]),
            'image_name': [b['image_name'] for b in batch]
        }

# --------------------------------------------------
# 모델 정의 및 유틸
# --------------------------------------------------
def get_model(num_classes=2):
    model = maskrcnn_resnet50_fpn(pretrained=True)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)

    in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feat_mask, 256, num_classes)
    return model

def save_pseudo_masks(model, dataloader, save_dir, device):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating pseudo masks"):
            images = [img.to(device) for img in batch["image"]]
            names = batch["image_name"]
            preds = model(images)

            for name, pred in zip(names, preds):
                if len(pred["masks"]) == 0:
                    continue
                mask = pred["masks"][0, 0].cpu().numpy()
                mask_img = (mask > 0.3).astype(np.uint8) * 255
                Image.fromarray(mask_img).save(os.path.join(save_dir, name.replace('.jpg', '.png')))

# --------------------------------------------------
# 학습 함수
# --------------------------------------------------
def train_model(model, dataloader, device, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_batches = len(dataloader)
        for i, data in enumerate(dataloader):
            # 이미지와 마스크를 디바이스로 이동
            images = data['image'].to(device)
            
            # 마스크, 박스, 레이블을 디바이스로 이동
            masks = data['mask'].to(device) if 'mask' in data else None
            boxes = [box.to(device) for box in data['boxes']] if 'boxes' in data else None
            labels = [label.to(device) for label in data['labels']] if 'labels' in data else None

            # 마스크 크기 조정 (필요시)
            if masks is not None:
                masks = masks.unsqueeze(1)  # (H, W) -> (1, H, W)

            # 박스가 비어 있지 않은 경우에만 targets 생성
            targets = []
            if boxes and labels:  # boxes와 labels가 모두 있을 경우에만
                for b, l, m in zip(boxes, labels, masks if masks is not None else [None]*len(boxes)):
                    if len(b) > 0:  # 박스가 비어 있지 않으면
                        targets.append({'boxes': b, 'labels': l, 'masks': m})
            
            # targets가 비어 있다면 skip (빈 박스만 있는 경우는 건너뛰기)
            if not targets:
                continue

            # images와 targets의 길이가 일치하는지 확인
            if len(images) != len(targets):
                print(f"Warning: Mismatch between images ({len(images)}) and targets ({len(targets)})")
                continue  # 이미지와 타겟의 길이가 맞지 않으면 해당 배치를 건너뜀
            
            # Forward pass
            loss_dict = model(images, targets)

            # 손실 값 계산
            losses = sum(loss for loss in loss_dict.values())

            # 역전파 및 최적화
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # 진행상황 출력 (매 10번째 배치마다 출력)
            if i % 10 == 0:
                epoch_progress = (epoch + 1) / num_epochs * 100
                iter_progress = (i + 1) / total_batches * 100
                print(f"Epoch [{epoch+1}/{num_epochs}] ({epoch_progress:.2f}%), Iter [{i}/{total_batches}] ({iter_progress:.2f}%), Loss: {losses.item()}")

# --------------------------------------------------
# 메인 파이프라인
# --------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([transforms.ToTensor()])
    
    # 1단계: 30개 마스크로 학습
    labeled_dataset = TireCrackDataset("defect_data/defective_train", "defect_data/mask_result", transform=transform, use_mask=True)
    labeled_loader = DataLoader(labeled_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    model = get_model()
    optimizer = Adam(model.parameters(), lr=1e-5)  # optimizer 정의
    print("Training with labeled data...")
    train_model(model, labeled_loader, device, optimizer, num_epochs=10)
    
    # 2단계: 마스크 없는 이미지에 pseudo-mask 생성
    unlabeled_dataset = TireCrackDataset("defect_data/defective_train", use_mask=False, transform=transform)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    print("Generating pseudo labels...")
    save_pseudo_masks(model, unlabeled_loader, "defect_data/pseudo_mask", device)
    
    # 3단계: pseudo-mask를 포함해 전체 데이터로 재학습
    print("Re-training with pseudo-labeled data...")
    pseudo_dataset = TireCrackDataset("defect_data/defective_train", "defect_data/pseudo_mask", transform=transform, use_mask=True)
    pseudo_loader = DataLoader(pseudo_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    model = get_model()
    optimizer = Adam(model.parameters(), lr=1e-5)  # optimizer 정의
    train_model(model, pseudo_loader, device, optimizer, num_epochs=10)

    torch.save(model.state_dict(), "model_weights/final_model.pth")
    print("Final model saved.")
