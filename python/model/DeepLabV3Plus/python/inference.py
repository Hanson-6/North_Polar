import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import rasterio
from typing import Union, Tuple, Optional, Dict

from _deeplab import DeepLabV3

class DeepLabV3PlusInference:
    """
    DeepLabV3Plus 模型推理接口
    
    用于加载训练好的模型并进行图像分割推理
    """
    
    def __init__(self, checkpoint_path: str, device: Optional[str], num_classes: int = 2):
        """
        初始化推理接口
        
        Args:
            checkpoint_path: 模型checkpoint文件路径
            device: 运行设备 ('cuda', 'cpu' 或 None自动选择)
            num_classes: 分类数量
        """
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # 加载模型
        self.model = DeepLabV3(num_classes=num_classes, backbone_pretrained=False)
        self.load_checkpoint(checkpoint_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.num_classes = num_classes
        
    def load_checkpoint(self, checkpoint_path: str):
        """加载模型权重"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        
        # 加载模型权重
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            if 'val_miou' in checkpoint:
                print(f"Model validation mIoU: {checkpoint['val_miou']:.4f}")
        else:
            # 兼容只保存了state_dict的情况
            self.model.load_state_dict(checkpoint)
            print("Loaded model weights")
    
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image], 
                        target_size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
        """
        预处理输入图像
        
        Args:
            image: 输入图像 (文件路径、numpy数组或PIL Image)
            target_size: 目标尺寸 (height, width)
            
        Returns:
            预处理后的图像张量
        """
        # 处理不同类型的输入
        if isinstance(image, str):
            # 检查是否为GeoTIFF
            if image.lower().endswith(('.tif', '.tiff')):
                with rasterio.open(image) as src:
                    img_array = src.read([1, 2, 3])  # RGB波段
                    img_array = np.transpose(img_array, (1, 2, 0))
                    pil_image = Image.fromarray(img_array).convert("RGB")
            else:
                pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:  # 灰度图
                image = np.stack([image] * 3, axis=-1)
            pil_image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise ValueError("Unsupported image type")
        
        # 调整大小
        pil_image = pil_image.resize(target_size)
        
        # 应用预处理变换
        img_tensor = self.transform(pil_image)
        
        # 添加batch维度
        return img_tensor.unsqueeze(0)
    
    def predict(self, image: Union[str, np.ndarray, Image.Image], 
                target_size: Tuple[int, int] = (256, 256),
                return_probs: bool = False) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        对图像进行分割预测
        
        Args:
            image: 输入图像
            target_size: 预测时的图像尺寸
            return_probs: 是否返回概率图
            
        Returns:
            如果return_probs=False: 预测的分割掩膜 (H, W)
            如果return_probs=True: 包含'mask'和'probs'的字典
        """
        # 预处理图像
        img_tensor = self.preprocess_image(image, target_size)
        img_tensor = img_tensor.to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(img_tensor)  # [1, num_classes, H, W]
            probs = torch.softmax(outputs, dim=1)  # 转换为概率
            mask = torch.argmax(outputs, dim=1)  # [1, H, W]
        
        # 转换为numpy数组
        mask_np = mask[0].cpu().numpy().astype(np.uint8)
        
        if return_probs:
            probs_np = probs[0].cpu().numpy()  # [num_classes, H, W]
            return {
                'mask': mask_np,
                'probs': probs_np
            }
        else:
            return mask_np
    
    def predict_batch(self, images: list, target_size: Tuple[int, int] = (256, 256),
                     batch_size: int = 8) -> list:
        """
        批量预测
        
        Args:
            images: 图像列表
            target_size: 预测时的图像尺寸
            batch_size: 批处理大小
            
        Returns:
            预测掩膜列表
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            
            # 预处理批量图像
            batch_tensors = []
            for img in batch_images:
                img_tensor = self.preprocess_image(img, target_size)
                batch_tensors.append(img_tensor)
            
            # 拼接成批
            batch = torch.cat(batch_tensors, dim=0).to(self.device)
            
            # 批量推理
            with torch.no_grad():
                outputs = self.model(batch)
                masks = torch.argmax(outputs, dim=1)
            
            # 添加到结果
            for mask in masks:
                results.append(mask.cpu().numpy().astype(np.uint8))
        
        return results
    
    def compute_metrics(self, pred_mask: np.ndarray, true_mask: np.ndarray) -> Dict[str, float]:
        """
        计算预测指标
        
        Args:
            pred_mask: 预测掩膜
            true_mask: 真实掩膜
            
        Returns:
            包含pixel_accuracy和各类IoU的字典
        """
        p = pred_mask.flatten()
        t = true_mask.flatten()
        
        # 混淆矩阵
        conf = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        for pi, ti in zip(p, t):
            if ti < self.num_classes:
                conf[ti, pi] += 1
        
        # 计算IoU
        ious = {}
        for cls in range(self.num_classes):
            tp = conf[cls, cls]
            fp = conf[:, cls].sum() - tp
            fn = conf[cls, :].sum() - tp
            iou = tp / (tp + fp + fn + 1e-6)
            ious[f'class_{cls}_iou'] = float(iou)
        
        # 计算像素精度
        pixel_acc = float((p == t).sum() / len(p))
        
        # 计算平均IoU
        mean_iou = float(np.mean(list(ious.values())))
        
        return {
            'pixel_accuracy': pixel_acc,
            'mean_iou': mean_iou,
            **ious
        }
    
    def visualize_prediction(self, image: Union[str, np.ndarray, Image.Image],
                           target_size: Tuple[int, int] = (256, 256),
                           alpha: float = 0.5) -> Image.Image:
        """
        可视化预测结果
        
        Args:
            image: 输入图像
            target_size: 预测时的图像尺寸
            alpha: 掩膜透明度
            
        Returns:
            叠加了预测掩膜的图像
        """
        # 获取原始图像
        if isinstance(image, str):
            if image.lower().endswith(('.tif', '.tiff')):
                with rasterio.open(image) as src:
                    img_array = src.read([1, 2, 3])
                    img_array = np.transpose(img_array, (1, 2, 0))
                    original = Image.fromarray(img_array).convert("RGB")
            else:
                original = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            original = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        else:
            original = image.convert("RGB")
        
        original = original.resize(target_size)
        
        # 预测
        mask = self.predict(image, target_size)
        
        # 创建彩色掩膜
        color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        # 类别0（背景）为黑色，类别1（建成区）为红色
        color_mask[mask == 1] = [255, 0, 0]
        
        # 叠加
        color_mask_img = Image.fromarray(color_mask)
        result = Image.blend(original, color_mask_img, alpha)
        
        return result


# 使用示例
if __name__ == "__main__":
    # 初始化推理器
    inferencer = DeepLabV3PlusInference(
        checkpoint_path="../checkpoints/best_model_20240729_143022.pth",
        device="cuda",
        num_classes=2
    )
    
    # 单张图像预测
    image_path = "test_image.tif"
    mask = inferencer.predict(image_path)
    print(f"Predicted mask shape: {mask.shape}")
    print(f"Unique values in mask: {np.unique(mask)}")
    
    # 获取概率图
    result = inferencer.predict(image_path, return_probs=True)
    print(f"Probability map shape: {result['probs'].shape}")
    
    # 批量预测
    image_list = ["image1.tif", "image2.tif", "image3.tif"]
    masks = inferencer.predict_batch(image_list)
    print(f"Predicted {len(masks)} masks")
    
    # 可视化结果
    vis_result = inferencer.visualize_prediction(image_path)
    vis_result.save("prediction_visualization.png")
    print("Saved visualization to prediction_visualization.png")