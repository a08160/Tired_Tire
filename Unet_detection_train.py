import torch
import torch.nn as nn

from typing import Optional


def auto_pad(kernel_size: int, dilation: int) -> int:
    """Padding mode = `same`"""
    padding = (kernel_size - 1) // 2 * dilation
    return padding


class Conv(nn.Module):
    """Convolutional Block"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 1,
            stride: int = 1,
            padding: Optional[int] = None,
            groups: int = 1,
            dilation: int = 1,
            bias: bool = False,
            act: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=auto_pad(kernel_size, dilation) if padding is None else padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DoubleConv(nn.Module):
    """Double Convolutional Block"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            mid_channels: Optional[int] = None,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = False,
            act: bool = True,
    ) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = Conv(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            act=act,
        )
        self.conv2 = Conv(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            act=act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class Down(nn.Module):
    """Feature Downscale"""

    def __init__(self, in_channels: int, out_channels: int, scale_factor=2) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=scale_factor)
        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)

        return x


class Up(nn.Module):
    """Feature Upscale"""

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=in_channels // 2, kernel_size=2, stride=scale_factor
        )
        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        x_ = torch.cat([x2, x1], dim=1)
        return self.conv(x_)


class UNet(nn.Module):
    """UNet Segmentation Model"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_conv = DoubleConv(in_channels, out_channels=64)

        # Downscale ⬇️
        self.down1 = Down(in_channels=64, out_channels=128, scale_factor=2)  # P/2
        self.down2 = Down(in_channels=128, out_channels=256, scale_factor=2)  # P/4
        self.down3 = Down(in_channels=256, out_channels=512, scale_factor=2)  # P/8
        self.down4 = Down(in_channels=512, out_channels=1024, scale_factor=2)  # P/16

        # Upscale ⬆️
        self.up1 = Up(in_channels=1024, out_channels=512, scale_factor=2)
        self.up2 = Up(in_channels=512, out_channels=256, scale_factor=2)
        self.up3 = Up(in_channels=256, out_channels=128, scale_factor=2)
        self.up4 = Up(in_channels=128, out_channels=64, scale_factor=2)

        self.output_conv = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.input_conv(x)

        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x_ = self.up1(x4, x3)
        x_ = self.up2(x_, x2)
        x_ = self.up3(x_, x1)
        x_ = self.up4(x_, x0)

        x_ = self.output_conv(x_)

        return x_
    
model = UNet(in_channels=3, out_channels=1)  # RGB 입력, 이진 마스크 출력

import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# UNet 모델 클래스 

# 1️⃣ Custom Dataset
class TireCrackDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '.png'))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = (mask > 0).float()  # 0 or 1

        return image, mask

# 2️⃣ Transforms & Dataloader
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

train_dataset = TireCrackDataset("defect_data/defective_train", "defect_data/mask_result", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# 3️⃣ 모델 정의
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=1).to(device)

# 4️⃣ 손실 함수 & 옵티마이저
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 5️⃣ 학습 함수
def train(model, dataloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"[Epoch {epoch+1}/{epochs}] Loss: {epoch_loss / len(dataloader):.4f}")

# 6️⃣ 결과 시각화 함수
def visualize_sample(model, dataloader):
    model.eval()
    with torch.no_grad():
        images, masks = next(iter(dataloader))
        images = images.to(device)
        outputs = model(images)
        preds = torch.sigmoid(outputs) > 0.5

        # to CPU
        images = images.cpu()
        preds = preds.cpu()
        masks = masks.cpu()

        fig, ax = plt.subplots(len(images), 3, figsize=(10, 10))
        for i in range(len(images)):
            ax[i, 0].imshow(images[i].permute(1, 2, 0))
            ax[i, 0].set_title("Input")
            ax[i, 1].imshow(masks[i][0], cmap='gray')
            ax[i, 1].set_title("Ground Truth")
            ax[i, 2].imshow(preds[i][0], cmap='gray')
            ax[i, 2].set_title("Prediction")

        plt.tight_layout()
        plt.show()

# 7️⃣ 실행
train(model, train_loader, criterion, optimizer, epochs=20)
visualize_sample(model, train_loader)

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def predict_and_draw_cracks(model, image_path, transform, device):
    model.eval()

    # 🔹 이미지 불러오기 및 전처리
    orig_img = Image.open(image_path).convert("RGB")
    resized_img = orig_img.resize((256, 256))  # 모델 입력 크기
    input_tensor = transform(resized_img).unsqueeze(0).to(device)

    # 🔹 모델 예측
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.sigmoid(output)[0, 0].cpu().numpy()
        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255  # 이진 마스크 (0 또는 255)

    # 🔹 OpenCV로 변환
    orig_img_cv = np.array(resized_img)
    orig_img_cv = cv2.cvtColor(orig_img_cv, cv2.COLOR_RGB2BGR)

    # 🔹 윤곽선 검출
    contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < 50:  # 너무 작은 잡음은 무시
            continue
        # 박스 그리기
        cv2.rectangle(orig_img_cv, (x, y), (x+w, y+h), (0, 0, 255), 2)
        # 레이블 추가
        cv2.putText(orig_img_cv, "crack", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 🔹 시각화 (RGB로 다시 변환)
    result_img = cv2.cvtColor(orig_img_cv, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 6))
    plt.imshow(result_img)
    plt.axis("off")
    plt.title("Detected Cracks with Bounding Boxes")
    plt.show()

# 예시 이미지 경로
image_path = "defect_data/file/Defective (31).jpg"  # 테스트 이미지 경로
predict_and_draw_cracks(model, image_path, transform, device)

# 모델 저장
torch.save(model.state_dict(), "tire_crack_detection_model.pth")

# 모델 불러오기
model = UNet(in_channels=3, out_channels=1)
model.load_state_dict(torch.load("tire_crack_detection_model.pth"))
model.eval()