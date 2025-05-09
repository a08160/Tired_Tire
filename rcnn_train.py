import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN
from torch.optim import Adam
from tqdm import tqdm

# COCO 데이터셋을 PyTorch 형식으로 변환하는 Dataset 클래스
class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, coco_json, image_dir, transform=None):
        self.coco = COCO(coco_json)
        self.image_dir = image_dir
        self.transform = transform
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")
        annotations = self.coco.loadAnns(self.coco.getAnnIds(img_id))

        masks = []
        boxes = []
        labels = []
        for ann in annotations:
            masks.append(self.coco.annToMask(ann))
            boxes.append(ann['bbox'])
            labels.append(ann['category_id'])

        # Bounding box를 [xmin, ymin, width, height]로 변환
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.stack([torch.as_tensor(mask, dtype=torch.uint8) for mask in masks])

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = torch.tensor([img_id])
        target['area'] = torch.tensor([box[2] * box[3] for box in boxes])
        target['iscrowd'] = torch.zeros(len(boxes), dtype=torch.int64)

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

# Transform 정의 (Resize, ToTensor 등)
transform = transforms.Compose([
    transforms.ToTensor()
])

# COCO JSON 형식 경로
coco_json = 'result/annotations.json'  # COCO 형식으로 변환된 JSON
image_dir = 'defect_data/defective_train'  # 이미지 경로

# COCO 데이터셋 로딩
dataset = CocoDataset(coco_json=coco_json, image_dir=image_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

# Mask R-CNN 모델 준비
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = maskrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # 배경 + 타이어 균열
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features, num_classes)

# 모델을 학습 모드로 전환
model.to(device)
model.train()

# 옵티마이저
optimizer = Adam(model.parameters(), lr=1e-4)

# 학습 루프
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for images, targets in tqdm(data_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

    print(f"Epoch {epoch+1} Loss: {running_loss/len(data_loader)}")

print("✅ 학습 완료!")

# 모델 저장
torch.save(model.state_dict(), 'mask_rcnn_model.pth')
