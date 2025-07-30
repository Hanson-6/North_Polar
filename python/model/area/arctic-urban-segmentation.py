# Arctic Urban Area Segmentation using Segment-Geospatial and Google Earth Engine
# 基于SAM和GEE的北极地区建城区语义分割

import ee
import geemap
import numpy as np
import pandas as pd
from samgeo import SamGeo, SamGeo2, tms_to_geotiff
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from datetime import datetime

# 初始化 Google Earth Engine
ee.Initialize()

class ArcticUrbanSegmentation:
    """北极地区建城区语义分割类"""
    
    def __init__(self, country_code='IS', year=2023):
        self.country_code = country_code  # IS for Iceland
        self.year = year
        self.output_dir = f"./outputs/{country_code}_{year}"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def get_sentinel2_composite(self, aoi, start_date, end_date):
        """
        获取 Sentinel-2 合成影像
        使用中值合成减少云覆盖影响
        """
        # 云掩膜函数
        def maskS2clouds(image):
            qa = image.select('QA60')
            cloudBitMask = 1 << 10
            cirrusBitMask = 1 << 11
            mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
            return image.updateMask(mask).divide(10000)
        
        # 获取 Sentinel-2 SR 数据集
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(aoi) 
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
            .map(maskS2clouds)
        
        # 计算中值合成
        composite = collection.median()
        
        # 选择关键波段
        bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']  # Blue, Green, Red, NIR, SWIR1, SWIR2
        
        return composite.select(bands)
    
    def calculate_indices(self, image):
        """
        计算有助于识别建城区的指数
        """
        # NDVI - 植被指数
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        
        # NDBI - 建筑用地指数
        ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')
        
        # MNDWI - 改进的归一化水体指数
        mndwi = image.normalizedDifference(['B3', 'B11']).rename('MNDWI')
        
        # UI - 城市指数
        ui = image.expression(
            '(SWIR2 - NIR) / (SWIR2 + NIR)', {
                'NIR': image.select('B8'),
                'SWIR2': image.select('B12')
            }).rename('UI')
        
        return image.addBands([ndvi, ndbi, mndwi, ui])
    
    def prepare_training_data(self, aoi_geojson, building_footprints_fc):
        """
        准备训练数据
        
        Args:
            aoi_geojson: 研究区域的 GeoJSON
            building_footprints_fc: 建筑物轮廓的 Feature Collection (来自 GEE)
        """
        # 转换 AOI 为 ee.Geometry
        aoi = ee.Geometry(aoi_geojson)
        
        # 获取 Sentinel-2 合成影像
        start_date = f'{self.year}-05-01'
        end_date = f'{self.year}-09-30'
        
        s2_composite = self.get_sentinel2_composite(aoi, start_date, end_date)
        s2_with_indices = self.calculate_indices(s2_composite)
        
        # 导出影像到 Google Drive
        export_params = {
            'image': s2_with_indices,
            'description': f's2_composite_{self.country_code}_{self.year}',
            'folder': 'arctic_urban_segmentation',
            'scale': 10,
            'region': aoi,
            'maxPixels': 1e13,
            'crs': 'EPSG:32627'  # UTM zone for Iceland
        }
        
        task = ee.batch.Export.image.toDrive(**export_params)
        task.start()
        
        print(f"导出任务已启动: {export_params['description']}")
        print("请在 Google Drive 中检查导出进度")
        
        return task
    
    def create_urban_masks(self, building_footprints_path, image_path, buffer_distance=50):
        """
        从建筑物轮廓创建建城区掩膜
        
        Args:
            building_footprints_path: 建筑物轮廓 shapefile 路径
            image_path: Sentinel-2 影像路径
            buffer_distance: 建筑物缓冲区距离（米）
        """
        # 读取建筑物轮廓
        buildings = gpd.read_file(building_footprints_path)
        
        # 读取影像元数据
        with rasterio.open(image_path) as src:
            img_crs = src.crs
            img_transform = src.transform
            img_shape = (src.height, src.width)
        
        # 确保建筑物数据与影像 CRS 匹配
        if buildings.crs != img_crs:
            buildings = buildings.to_crs(img_crs)
        
        # 创建建筑物缓冲区（形成建城区）
        urban_areas = buildings.buffer(buffer_distance).unary_union
        
        # 将建城区转换为栅格掩膜
        if isinstance(urban_areas, gpd.GeoSeries):
            geometries = urban_areas.geometry
        else:
            geometries = [urban_areas]
        
        # 创建掩膜
        mask = geometry_mask(
            geometries,
            transform=img_transform,
            out_shape=img_shape,
            invert=True  # True 表示几何内部
        )
        
        return mask.astype(np.uint8)
    
    def segment_with_sam(self, image_path, checkpoint='sam_vit_h_4b8939.pth'):
        """
        使用 SAM 进行自动分割
        """
        # 初始化 SamGeo
        sam = SamGeo(
            model_type='vit_h',
            checkpoint=checkpoint,
            automatic=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # 生成掩膜
        masks = sam.generate(
            image_path,
            output_dir=self.output_dir,
            foreground=True,
            unique=True
        )
        
        return masks
    
    def segment_with_sam2(self, image_path):
        """
        使用 SAM2 进行分割（更先进的版本）
        """
        sam2 = SamGeo2(
            model_id="sam2-hiera-large",
            automatic=True,
            apply_postprocessing=True,
            points_per_side=32,
            points_per_batch=64,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            box_nms_thresh=0.7,
            min_mask_region_area=100
        )
        
        masks = sam2.generate(image_path)
        return masks
    
    def train_urban_classifier(self, sam_masks, urban_ground_truth):
        """
        训练分类器识别哪些 SAM 掩膜对应建城区
        
        Args:
            sam_masks: SAM 生成的掩膜
            urban_ground_truth: 建城区真值掩膜
        """
        # 提取每个掩膜的特征
        features = []
        labels = []
        
        for mask_id, mask in enumerate(sam_masks):
            # 计算掩膜特征
            mask_area = np.sum(mask)
            mask_compactness = self._calculate_compactness(mask)
            
            # 计算与建城区的重叠度
            overlap = np.sum(mask & urban_ground_truth) / np.sum(mask)
            
            # 标记是否为建城区（重叠度 > 0.5）
            is_urban = 1 if overlap > 0.5 else 0
            
            features.append([mask_area, mask_compactness])
            labels.append(is_urban)
        
        # 训练简单分类器
        from sklearn.ensemble import RandomForestClassifier
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        accuracy = clf.score(X_test, y_test)
        print(f"分类器准确率: {accuracy:.2f}")
        
        return clf
    
    def _calculate_compactness(self, mask):
        """计算掩膜的紧凑度"""
        area = np.sum(mask)
        if area == 0:
            return 0
        
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return 0
        
        perimeter = cv2.arcLength(contours[0], True)
        compactness = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        return compactness
    
    def post_process_urban_areas(self, urban_mask, min_area=1000):
        """
        后处理建城区掩膜
        
        Args:
            urban_mask: 原始建城区掩膜
            min_area: 最小面积阈值（像素）
        """
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # 闭运算填充小孔
        closed = cv2.morphologyEx(urban_mask, cv2.MORPH_CLOSE, kernel)
        
        # 开运算去除小区域
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        
        # 去除小于阈值的区域
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            opened.astype(np.uint8), connectivity=8
        )
        
        # 创建最终掩膜
        final_mask = np.zeros_like(opened)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                final_mask[labels == i] = 1
        
        return final_mask
    
    def visualize_results(self, image_path, urban_mask, save_path=None):
        """
        可视化分割结果
        """
        # 读取原始影像
        with rasterio.open(image_path) as src:
            rgb = src.read([3, 2, 1])  # RGB bands
            rgb = np.moveaxis(rgb, 0, -1)
            
        # 归一化
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        
        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始影像
        axes[0].imshow(rgb)
        axes[0].set_title('Original Sentinel-2 Image')
        axes[0].axis('off')
        
        # 建城区掩膜
        axes[1].imshow(urban_mask, cmap='gray')
        axes[1].set_title('Urban Area Mask')
        axes[1].axis('off')
        
        # 叠加显示
        overlay = rgb.copy()
        overlay[urban_mask == 1] = [1, 0, 0]  # 红色标记建城区
        axes[2].imshow(overlay)
        axes[2].set_title('Urban Areas Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_pipeline(self, aoi_geojson, building_fc_path, s2_image_path):
        """
        运行完整的建城区分割流程
        """
        print("=== 北极地区建城区语义分割流程 ===")
        
        # 1. 准备数据
        print("\n1. 准备训练数据...")
        # export_task = self.prepare_training_data(aoi_geojson, building_fc_path)
        
        # 2. 创建建城区真值掩膜
        print("\n2. 创建建城区真值掩膜...")
        urban_gt = self.create_urban_masks(
            building_fc_path, 
            s2_image_path,
            buffer_distance=50
        )
        
        # 3. 使用 SAM 进行分割
        print("\n3. 使用 SAM 进行自动分割...")
        sam_masks = self.segment_with_sam(s2_image_path)
        
        # 4. 训练建城区分类器
        print("\n4. 训练建城区分类器...")
        classifier = self.train_urban_classifier(sam_masks, urban_gt)
        
        # 5. 生成最终建城区掩膜
        print("\n5. 生成最终建城区掩膜...")
        urban_prediction = np.zeros_like(urban_gt)
        
        for mask_id, mask in enumerate(sam_masks):
            features = [[np.sum(mask), self._calculate_compactness(mask)]]
            if classifier.predict(features)[0] == 1:
                urban_prediction |= mask
        
        # 6. 后处理
        print("\n6. 后处理优化结果...")
        urban_final = self.post_process_urban_areas(urban_prediction)
        
        # 7. 可视化结果
        print("\n7. 可视化结果...")
        vis_path = os.path.join(self.output_dir, 'urban_segmentation_results.png')
        self.visualize_results(s2_image_path, urban_final, vis_path)
        
        # 8. 保存结果
        print("\n8. 保存分割结果...")
        output_path = os.path.join(self.output_dir, 'urban_areas.tif')
        self.save_segmentation_result(urban_final, s2_image_path, output_path)
        
        print(f"\n分割完成！结果保存在: {self.output_dir}")
        
        return urban_final
    
    def save_segmentation_result(self, mask, reference_image_path, output_path):
        """
        保存分割结果为 GeoTIFF
        """
        with rasterio.open(reference_image_path) as src:
            profile = src.profile.copy()
            profile.update({
                'count': 1,
                'dtype': 'uint8',
                'compress': 'lzw'
            })
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(mask.astype(np.uint8), 1)


# 使用示例
if __name__ == "__main__":
    # 初始化分割器
    segmenter = ArcticUrbanSegmentation(country_code='IS', year=2023)
    
    # 定义冰岛研究区域
    iceland_aoi = {
        "type": "Polygon",
        "coordinates": [[
            [-24.5, 63.0],
            [-13.0, 63.0],
            [-13.0, 67.0],
            [-24.5, 67.0],
            [-24.5, 63.0]
        ]]
    }
    
    # 运行分割流程
    # 注意：需要先从 GEE 导出数据，然后提供本地路径
    # segmenter.run_pipeline(
    #     aoi_geojson=iceland_aoi,
    #     building_fc_path='path/to/iceland_buildings.shp',
    #     s2_image_path='path/to/s2_composite.tif'
    # )
    
    print("请确保已安装以下依赖:")
    print("pip install segment-geospatial geemap geopandas rasterio scikit-learn")
    print("pip install torch torchvision # for GPU support")
