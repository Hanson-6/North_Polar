# train.py

import os, sys

# 1. 定位到 model 目录（包含 dataset/ 和 deeplab/）
BASE_MODEL = os.path.abspath(os.path.dirname(__file__))

# 2. 把 deeplab/DeepLabV3Plus-Pytorch 放入 sys.path
deep_model_path = os.path.join(BASE_MODEL, "deeplab", "DeepLabV3Plus-Pytorch")
if deep_model_path not in sys.path:
    sys.path.insert(0, deep_model_path)

# —— 这样下面的 import network._deeplab 就能找到 network 包  ——
# 从backbone子模块导入DeepLabV3
from network._deeplab import DeepLabV3
from dataset.dataset import ArcticSegmentationDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

num_epochs = 50

def main():
    # ——— 3. 构造 Transform、Dataset、DataLoader ————
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std =[0.229,0.224,0.225]),
    ])
    # 这里就能正确识别 ArcticSegmentationDataset 了
    train_dataset = ArcticSegmentationDataset(
        polygons_dir=r"D:\Download\North_Polar\python\model\dataset\polygons",   # 你的路径
        images_dir=r"D:\Download\North_Polar\python\model\dataset\sentinel_2",
        output_size=(256,256),
        transform=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # ——— 4. 模型、损失、优化器、训练循环 ——————
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = DeepLabV3(num_classes=2, backbone_pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)





    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images  = images.to(device)
            masks   = masks.to(device)
            outputs = model(images)
            loss    = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        print(f"Epoch {epoch}/{num_epochs}, Loss: {running_loss/len(train_dataset):.4f}")
        scheduler.step()

        if epoch % 10 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/deeplab_epoch{epoch}.pth")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()