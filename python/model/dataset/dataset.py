import os
import json
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import rasterio

class ArcticSegmentationDataset(Dataset):
    def __init__(self,
                 polygons_dir,   # e.g. .../dataset/polygons
                 images_dir,     # e.g. .../dataset/sentinel_2
                 output_size=(256,256),
                 transform=None):
        self.polygons_dir = polygons_dir
        self.images_dir   = images_dir
        self.output_size  = output_size
        self.transform    = transform or transforms.ToTensor()

        # 1) 递归搜集所有影像文件路径，并记录它们所属的“region”名
        self.samples = []
        valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        for root, _, files in os.walk(self.images_dir):
            for fn in files:
                if fn.lower().endswith(valid_exts):
                    img_path = os.path.join(root, fn)
                    rel = os.path.relpath(img_path, self.images_dir)
                    region = rel.split(os.sep)[0]
                    self.samples.append((img_path, region))
        if not self.samples:
            raise RuntimeError(f"No images found under {self.images_dir}")
        print(f"Found {len(self.samples)} images in regions:",
              sorted({r for _, r in self.samples}))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
      img_path, region = self.samples[idx]

      # 2) 读图并保留原始大小
      with rasterio.open(img_path) as src:
          img_array = src.read([1,2,3])  # assuming RGB bands
          transform = src.transform
          width, height = src.width, src.height
      img = Image.fromarray(np.transpose(img_array, (1,2,0))).convert("RGB")
      # 将图像缩放到指定输出尺寸
      img = img.resize(self.output_size)

      # 3) 载入多边形经纬度坐标
      poly_path = os.path.join(self.polygons_dir, f"{region}.json")
      with open(poly_path, 'r') as f:
          data = json.load(f)

    # 4) 解析 JSON，提取多边形坐标列表
      polygons = []
      if isinstance(data, list) and isinstance(data[0], list):
          polygons = data
      else:
          raise RuntimeError(f"无法识别的多边形结构：{poly_path}")

      # 5) 创建一个与原始影像大小相同的空白掩膜，用于绘制多边形区域
      mask = Image.new('L', (width, height), 0)
      drawer = ImageDraw.Draw(mask)
      for ring in polygons:
          try:
              pixel_coords = []
              # 将地理坐标 (lon, lat) 转为图像像素坐标 (col, row)
              for lon, lat in ring:
                  col, row = ~transform * (float(lon), float(lat))
                  # 四舍五入为整数像素
                  pixel_coords.append((int(round(col)), int(round(row))))
              # 在掩膜上绘制多边形并填充
              drawer.polygon(pixel_coords, outline=1, fill=1)
          except Exception as e:
              print(f"[ERROR] Polygon drawing failed at {region}, polygon={ring}, error={e}")
              raise

      # 6) 将掩膜缩放到与图像相同的输出尺寸，使用最近邻插值保留标签
      mask = mask.resize(self.output_size, resample=Image.NEAREST)

      # 7) 将图像和掩膜转换为 PyTorch 张量
      img_tensor  = self.transform(img)                            # [3,H,W]
      mask_np     = np.array(mask, dtype=np.int64)                 # [H,W]
      mask_tensor = torch.from_numpy(mask_np)                      # dtype long

      return img_tensor, mask_tensor

