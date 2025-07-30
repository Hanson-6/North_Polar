import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import KFold # K折交叉验证
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping
import json

# 添加 DeeplabV3Plus-Pytorch 模型路径
BASE = os.path.abspath(os.path.dirname(__file__))
deep = os.path.join(BASE, "deeplab", "DeepLabV3Plus-Pytorch")
if deep not in sys.path:
    sys.path.insert(0, deep)
from DeepLabV3Plus.python._deeplab import DeepLabV3 # 导入 DeepLabV3+ 模型
from dataset.dataset import ArcticSegmentationDataset # 导入自定义数据集

def compute_metrics(preds, truths, num_classes=2):
    """
    计算像素精度（Pixel Accuracy）和各类别 IoU。
    preds: 预测标签数组，truths: 真实标签数组
    num_classes: 类别总数
    返回: pixel_acc, [各类别 iou 列表]
    """
    p = preds.flatten() # 展平到一维
    t = truths.flatten()
    # 混淆矩阵统计，行是真实类，列是预测类
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pi, ti in zip(p, t):
        if ti < num_classes:
            conf[ti, pi] += 1
            
    ious = []
    for cls in range(num_classes):
        tp = conf[cls, cls]  # 真正类：预测正确的像素数
        fp = conf[:, cls].sum() - tp # 假正类 FP：预测成 cls，但真实不是 cls
        fn = conf[cls, :].sum() - tp # 假负类 FN：真实是 cls，但预测错了
        ious.append(tp / (tp + fp + fn + 1e-6)) # IoU = TP / (TP + FP + FN)
    pixel_acc = (p == t).sum() / len(p) #.sum() 是预测正确的像素个数/像素总数得到准确率
    return pixel_acc, ious   

def infer_and_save(model, loader, out_dir):
    """
    对验证/测试集做推理，并将预测掩膜和预测边界导出为 TIFF 和 GeoJSON。
    model: 已加载参数的网络，loader: 数据加载器，out_dir: 输出目录
    """
    os.makedirs(out_dir, exist_ok=True) # 创建输出文件夹
    device = next(model.parameters()).device # 模型所在设备
    all_pa, all_miou, count = 0, 0, 0   # 初始化总体像素精度、IoU 和样本计数

    # 遍历所有 batch
    for idx, (imgs, masks) in enumerate(loader):
        imgs = imgs.to(device) # 将图像送入对应的模型
        truths = masks.numpy() # 真实掩膜转换为NumPy格式用于后续指标计算

        with torch.no_grad():  # 推理阶段无需反向传播
            preds = torch.argmax(model(imgs), dim=1).cpu().numpy()
            # 模型输出 shape 为 [B, C, H, W]，使用 argmax 取每像素概率最大的类别作为预测结果

        # 先累加像素精度、IoU
        pa, ious = compute_metrics(preds, truths)
        all_pa += pa; all_miou += np.mean(ious); count += 1

        # 遍历 batch 中的每一张图像
        for b in range(preds.shape[0]):
            mask = preds[b].astype(np.uint8)   # 当前预测单张掩膜

            # —— 这里是修正点 —— 
            # 如果是 Subset，就取它的原始 dataset 和 indices
            if isinstance(loader.dataset, Subset):
                orig_ds = loader.dataset.dataset
                sample_idx = loader.dataset.indices[idx * loader.batch_size + b]
            else:
                orig_ds = loader.dataset
                sample_idx = idx * loader.batch_size + b

            img_path, _ = orig_ds.samples[sample_idx]  # 通过 sample_idx 找回原始图像路径
            # —— 修正点结束 —— 

            # 下面保持你原有的 rasterio + shapes 逻辑
            with rasterio.open(img_path) as src:
                profile = src.profile.copy()
            profile.update({'count': 1, 'dtype': 'uint8'})
            tif_path = os.path.join(out_dir, f"mask_{idx}_{b}.tif")
            with rasterio.open(tif_path, 'w', **profile) as dst:
                dst.write(mask, 1)

            polys = []
            for geom, val in shapes(mask, mask == 1, transform=profile['transform']):
                if val == 1:  # 只保留类别为 1（建成区）对应的多边形
                    polys.append(shape(geom)) # 掩膜转换为多边形
            if not polys:
                continue
            # 多个多边形合并
            union = polys[0]
            for poly in polys[1:]:
                union = union.union(poly)
            # 构造 GeoJSON 格式数据并写入
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
    kf = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed) #构建K折交叉验证
    # 遍历每一折
    for fold, (tr, val) in enumerate(kf.split(dataset), start=1):
        print(f"=== Fold {fold}/{args.kfold} ===")
        train_ds = Subset(dataset, tr) # 训练集
        val_ds = Subset(dataset, val)  # 测试集

        train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=args.nw)
        val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=args.nw)
        
        # 初始化模型
        model = DeepLabV3(num_classes=args.nc, backbone_pretrained=True).to(device)
        # 设置优化器、学习率调度器、损失函数
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        criterion = nn.CrossEntropyLoss()

        # 保存当前fold最优模型的初始验证损失
        best_val = float('inf')

        # 为当前fold创建模型保存目录
        ckpt_dir = os.path.join(args.ckpt, f"fold{fold}")
        os.makedirs(ckpt_dir, exist_ok=True)

        # 开始训练多个epoch
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0

            # 遍历训练集的每一个batch
            for imgs, masks in train_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * imgs.size(0)
            train_loss /= len(train_ds) # 平均训练损失

            # 模型验证阶段
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    outputs = model(imgs)
                    val_loss += criterion(outputs, masks).item() * imgs.size(0)
            val_loss /= len(val_ds)    # 平均验证损失
            scheduler.step(val_loss)   # 根据验证损失动态调整学习率

            # 打印当前epoch的训练与验证损失
            print(f"Epoch {epoch}/{args.epochs} - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")
            # 若当前验证损失最优，则保存模型
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best.pth'))
        # 当前 fold 训练结束后，加载最优模型权重并进行推理
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
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--bs', type=int, default=16) # batch size
    parser.add_argument('--lr', type=float, default=1e-4) # 学习率
    parser.add_argument('--wd', type=float, default=1e-4) # weight decay
    parser.add_argument('--step', type=int, default=5)    
    parser.add_argument('--nc', type=int, default=2)      # number of classes
    parser.add_argument('--nw', type=int, default=2)      # num workers加载数据的进程数
    parser.add_argument('--output_size', nargs=2, type=int, default=(256,256))
    parser.add_argument('--ckpt', type=str, default='checkpoints') # 保存每个fold最优模型的目录
    parser.add_argument('--split_dir', type=str, default='splits')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train_and_test(args)

    
