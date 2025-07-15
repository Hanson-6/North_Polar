import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import KFold
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping
import json

# 添加 DeeplabV3Plus-Pytorch 模型路径
BASE = os.path.abspath(os.path.dirname(__file__))
deep = os.path.join(BASE, "deeplab", "DeepLabV3Plus-Pytorch")
if deep not in sys.path:
    sys.path.insert(0, deep)
from network._deeplab import DeepLabV3
from dataset.dataset import ArcticSegmentationDataset

def compute_metrics(preds, truths, num_classes=2):
    p = preds.flatten()
    t = truths.flatten()
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pi, ti in zip(p, t):
        if ti < num_classes:
            conf[ti, pi] += 1
    ious = []
    for cls in range(num_classes):
        tp = conf[cls, cls]
        fp = conf[:, cls].sum() - tp
        fn = conf[cls, :].sum() - tp
        ious.append(tp / (tp + fp + fn + 1e-6))
    pixel_acc = (p == t).sum() / len(p)
    return pixel_acc, ious

def infer_and_save(model, loader, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    device = next(model.parameters()).device
    all_pa, all_miou, count = 0, 0, 0
    for idx, (imgs, masks) in enumerate(loader):
        imgs = imgs.to(device)
        truths = masks.numpy()
        with torch.no_grad():
            preds = torch.argmax(model(imgs), dim=1).cpu().numpy()
        pa, ious = compute_metrics(preds, truths)
        all_pa += pa; all_miou += np.mean(ious); count += 1
        for b in range(preds.shape[0]):
            mask = preds[b].astype(np.uint8)
            img_path, _ = loader.dataset.samples[idx * loader.batch_size + b]
            with rasterio.open(img_path) as src:
                profile = src.profile.copy()
            profile.update({'count': 1, 'dtype': 'uint8'})
            tif_path = os.path.join(out_dir, f"mask_{idx}_{b}.tif")
            with rasterio.open(tif_path, 'w', **profile) as dst:
                dst.write(mask, 1)
            polys = []
            for geom, val in shapes(mask, mask == 1, transform=profile['transform']):
                if val == 1:
                    polys.append(shape(geom))
            if not polys:
                continue
            union = polys[0]
            for poly in polys[1:]:
                union = union.union(poly)
            geojson = {
                "type": "FeatureCollection",
                "features": [{"type": "Feature", "properties": {}, "geometry": mapping(union)}]
            }
            json_path = os.path.join(out_dir, f"boundary_{idx}_{b}.geojson")
            with open(json_path, 'w') as f:
                json.dump(geojson, f)
    print(f"Fold inference: PixelAcc={all_pa/count:.4f}, MeanIoU={all_miou/count:.4f}")

def train_and_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ArcticSegmentationDataset(
        polygons_dir=args.polygons_dir,
        images_dir=args.images_dir,
        output_size=tuple(args.output_size),
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    )
    kf = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    for fold, (tr, val) in enumerate(kf.split(dataset), start=1):
        print(f"=== Fold {fold}/{args.kfold} ===")
        train_ds = Subset(dataset, tr)
        val_ds = Subset(dataset, val)
        train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=args.nw)
        val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=args.nw)
        model = DeepLabV3(num_classes=args.nc, backbone_pretrained=True).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        best_val = float('inf')
        ckpt_dir = os.path.join(args.ckpt, f"fold{fold}")
        os.makedirs(ckpt_dir, exist_ok=True)
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            for imgs, masks in train_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * imgs.size(0)
            train_loss /= len(train_ds)
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    outputs = model(imgs)
                    val_loss += criterion(outputs, masks).item() * imgs.size(0)
            val_loss /= len(val_ds)
            scheduler.step()
            print(f"Epoch {epoch}/{args.epochs} - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best.pth'))
        # 测试当前 fold
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'best.pth'), map_location='cpu'))
        infer_and_save(model, val_loader, f"outputs_fold{fold}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 直接写入默认路径，不再必须从命令行传入
    parser.add_argument('--polygons_dir', type=str,
                        default=r'D:\Download\North_Polar\python\model\dataset\polygons',
                        help='GeoJSON 多边形目录')
    parser.add_argument('--images_dir', type=str,
                        default=r'D:\Download\North_Polar\python\model\dataset\sentinel_2',
                        help='Sentinel-2 GeoTIFF 目录')
    parser.add_argument('--kfold', type=int, default=10, help='K 折数')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-6)
    parser.add_argument('--step', type=int, default=5)
    parser.add_argument('--nc', type=int, default=2)
    parser.add_argument('--nw', type=int, default=4)
    parser.add_argument('--output_size', nargs=2, type=int, default=(256,256))
    parser.add_argument('--ckpt', type=str, default='checkpoints')
    parser.add_argument('--split_dir', type=str, default='splits')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train_and_test(args)
