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
        img = img.resize(self.output_size)

        # 3) 载入多边形经纬度坐标
        poly_path = os.path.join(self.polygons_dir, f"{region}.json")
        with open(poly_path, 'r') as f:
            geo_coords = json.load(f)  # 可能是多环结构

        # 4) 处理多环或单环结构，摊平所有点
        if geo_coords and isinstance(geo_coords[0][0], list):
            coords_list = [pt for ring in geo_coords for pt in ring]
        else:
            coords_list = geo_coords

        # 5) 经纬度转像素坐标 (col, row)
        pix_coords = []
        for pt_idx, coord in enumerate(coords_list):
            try:
                lon, lat = coord[:2]
                lon = float(lon)
                lat = float(lat)
                col_f, row_f = ~transform * (lon, lat)
            except Exception as e:
                print(f"[ERROR] sample={idx}, point={pt_idx}, coord={coord}, type={type(coord)}")
                raise
            pix_coords.append((col_f, row_f))

        # 6) 转成整数并画掩膜
        int_pts = [(int(round(x)), int(round(y))) for x, y in pix_coords]
        mask = Image.new('L', (width, height), 0)
        ImageDraw.Draw(mask).polygon(int_pts, outline=1, fill=1)
        mask = mask.resize(self.output_size, resample=Image.NEAREST)

        # 7) 转张量
        img_tensor  = self.transform(img)                            # [3,H,W]
        mask_np     = np.array(mask, dtype=np.int64)                 # [H,W]
        mask_tensor = torch.from_numpy(mask_np)                      # dtype long

        return img_tensor, mask_tensor