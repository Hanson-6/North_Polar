import os
import sys
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import numpy as np

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录 (model目录)
parent_dir = os.path.dirname(current_dir)
# 将父目录添加到Python路径
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir+'/code')

from filtered_dataset.dataset import ArcticSegmentationDataset
from _deeplab import DeepLabV3

def compute_metrics(preds, truths, num_classes=2):
    """计算像素精度和各类别IoU"""
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

def train_epoch(model, loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_miou = 0.0
    num_batches = 0
    
    for batch_idx, (imgs, masks) in enumerate(loader):
        imgs, masks = imgs.to(device), masks.to(device)
        
        # 前向传播
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 计算指标
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            truths = masks.cpu().numpy()
            acc, ious = compute_metrics(preds, truths)
            
        total_loss += loss.item()
        total_acc += acc
        total_miou += np.mean(ious)
        num_batches += 1
        
        if batch_idx % 10 == 0:
            print(f'  Batch [{batch_idx}/{len(loader)}] Loss: {loss.item():.4f}')
    
    return total_loss / num_batches, total_acc / num_batches, total_miou / num_batches

def validate(model, loader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_miou = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            truths = masks.cpu().numpy()
            acc, ious = compute_metrics(preds, truths)
            
            total_loss += loss.item()
            total_acc += acc
            total_miou += np.mean(ious)
            num_batches += 1
    
    return total_loss / num_batches, total_acc / num_batches, total_miou / num_batches

def main(args):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 数据集准备
    print("Loading dataset...")
    dataset = ArcticSegmentationDataset(
        polygons_dir=args.polygons_dir,
        images_dir=args.images_dir,
        output_size=tuple(args.output_size),
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    
    # 划分训练集和验证集 (80:20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Dataset split - Train: {train_size}, Val: {val_size}")
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    # 初始化模型
    print("Initializing model...")
    model = DeepLabV3(num_classes=args.num_classes, backbone_pretrained=True)
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 创建checkpoint目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 训练循环
    best_val_loss = float('inf')
    best_val_miou = 0.0
    
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 50)
        
        # 训练
        train_loss, train_acc, train_miou = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, mIoU: {train_miou:.4f}")
        
        # 验证
        val_loss, val_acc, val_miou = validate(
            model, val_loader, criterion, device
        )
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, mIoU: {val_miou:.4f}")
        
        # 调整学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = os.path.join(args.checkpoint_dir, f"best_model_{timestamp}.pth")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_miou': val_miou,
                'args': args
            }, checkpoint_path)
            
            print(f"Saved best model to {checkpoint_path}")
        
        # 定期保存checkpoint
        if epoch % args.save_interval == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch{epoch}_{timestamp}.pth")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_miou': val_miou,
                'args': args
            }, checkpoint_path)
            
            print(f"Saved checkpoint to {checkpoint_path}")
    
    print("\nTraining completed!")
    print(f"Best validation mIoU: {best_val_miou:.4f}")

if __name__ == '__main__':
    file_dirpath = os.getcwd()
    filtered_dataset_dirpath = '/share/home/u20169/North_Polar/python/model/filtered_dataset'
    result_dirpath = '/share/home/u20169/North_Polar/python/model/python/checkpoints'
    
    parser = argparse.ArgumentParser(description='Train DeepLabV3Plus model')
    
    # 数据相关参数
    parser.add_argument('--polygons_dir', type=str,
                        default=f'{filtered_dataset_dirpath}/filtered_polygons',
                        help='Path to GeoJSON polygons directory')
    parser.add_argument('--images_dir', type=str,
                        default=f'{filtered_dataset_dirpath}/filtered_sentinel_2',
                        help='Path to Sentinel-2 images directory')
    parser.add_argument('--output_size', nargs=2, type=int, default=[256, 256],
                        help='Output image size (default: 256 256)')
    
    # 模型相关参数
    parser.add_argument('--num_classes', '--nc', type=int, default=2,
                        help='Number of classes (default: 2)')
    
    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--batch_size', '--bs', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', '--wd', type=float, default=1e-5,
                        help='Weight decay (default: 1e-5)')
    parser.add_argument('--num_workers', '--nw', type=int, default=2,
                        help='Number of data loading workers (default: 2)')
    
    # 保存相关参数
    parser.add_argument('--checkpoint_dir', '--ckpt', type=str, 
                        default=f'result_dirpath',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # 创建必要的目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(result_dirpath, exist_ok=True)
    
    main(args)