# Google Colab + GEE 北极地区建城区语义分割
# 使用 segment-geospatial 和 Google Earth Engine

# ===== 第1部分：环境设置（在 Colab 中运行） =====

# !pip install segment-geospatial -q
# !pip install geemap -q
# !pip install geopandas -q
# !pip install earthengine-api -q

import os
import ee
import geemap
import numpy as np
import pandas as pd
from samgeo import SamGeo, SamGeo2, tms_to_geotiff
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime
import json
from google.colab import drive
from IPython.display import Image, display

# 挂载 Google Drive
# drive.mount('/content/drive')

# 认证 Google Earth Engine
# ee.Authenticate()
ee.Initialize()

# ===== 第2部分：主要功能类 =====

class GEEUrbanSegmentation:
    """基于 GEE 和 SAM 的建城区分割"""
    
    def __init__(self, project_name='arctic_urban'):
        self.project_name = project_name
        self.output_folder = f'/content/drive/MyDrive/{project_name}'
        os.makedirs(self.output_folder, exist_ok=True)
        
    def get_country_boundary(self, country_name='Iceland'):
        """获取国家边界"""
        countries = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017')
        country = countries.filter(ee.Filter.eq('country_na', country_name))
        return country.geometry()
    
    def get_s2_composite(self, aoi, year=2023, months=[5, 6, 7, 8, 9]):
        """
        获取 Sentinel-2 无云合成影像
        """
        # 云掩膜函数
        def maskS2clouds(image):
            qa = image.select('QA60')
            cloudBitMask = 1 << 10
            cirrusBitMask = 1 << 11
            mask = qa.bitwiseAnd(cloudBitMask).eq(0) \
                     .And(qa.bitwiseAnd(cirrusBitMask).eq(0))
            return image.updateMask(mask).divide(10000) \
                       .copyProperties(image, ["system:time_start"])
        
        # 创建日期范围
        start_date = f'{year}-{months[0]:02d}-01'
        end_date = f'{year}-{months[-1]:02d}-30'
        
        # 获取 Sentinel-2 集合
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(aoi) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
            .map(maskS2clouds)
        
        # 中值合成
        composite = collection.median()
        
        # 添加光谱指数
        composite = self.add_indices(composite)
        
        return composite.clip(aoi)
    
    def add_indices(self, image):
        """添加有助于识别建城区的光谱指数"""
        # NDVI
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        
        # NDBI - 建筑指数
        ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')
        
        # MNDWI - 水体指数
        mndwi = image.normalizedDifference(['B3', 'B11']).rename('MNDWI')
        
        # UI - 城市指数
        ui = ndbi.subtract(ndvi).rename('UI')
        
        # BU - 建成区指数
        bu = ndbi.subtract(ndvi).add(1).divide(2).rename('BU')
        
        return image.addBands([ndvi, ndbi, mndwi, ui, bu])
    
    def create_urban_training_data(self, aoi, building_fc):
        """
        从建筑物数据创建建城区训练样本
        """
        # 创建建筑物密度图
        building_density = building_fc.distance(50, 100).Not() \
                                     .reduceToImage([], ee.Reducer.mean()) \
                                     .reproject(crs='EPSG:4326', scale=10)
        
        # 定义建城区阈值（建筑物密度 > 0.3）
        urban_areas = building_density.gt(0.3)
        
        # 创建非建城区样本（NDVI > 0.5 的植被区域）
        s2_composite = self.get_s2_composite(aoi)
        non_urban = s2_composite.select('NDVI').gt(0.5)
        
        return urban_areas, non_urban
    
    def sample_training_points(self, image, urban_mask, non_urban_mask, n_samples=5000):
        """
        采样训练点
        """
        # 采样建城区点
        urban_points = image.sample(
            region=urban_mask.geometry(),
            scale=10,
            numPixels=n_samples,
            seed=42,
            geometries=True
        ).map(lambda f: f.set('class', 1))
        
        # 采样非建城区点
        non_urban_points = image.sample(
            region=non_urban_mask.geometry(),
            scale=10,
            numPixels=n_samples,
            seed=42,
            geometries=True
        ).map(lambda f: f.set('class', 0))
        
        # 合并训练数据
        training_data = urban_points.merge(non_urban_points)
        
        return training_data
    
    def train_classifier(self, training_data, bands):
        """
        训练随机森林分类器
        """
        classifier = ee.Classifier.smileRandomForest(
            numberOfTrees=100,
            variablesPerSplit=None,
            minLeafPopulation=1,
            bagFraction=0.5,
            seed=42
        ).train(
            features=training_data,
            classProperty='class',
            inputProperties=bands
        )
        
        return classifier
    
    def export_for_sam(self, image, aoi, description, scale=10):
        """
        导出影像供 SAM 处理
        """
        # 选择 RGB 波段用于 SAM
        rgb = image.select(['B4', 'B3', 'B2'])
        
        # 可视化参数
        vis_params = {
            'min': 0,
            'max': 0.3,
            'bands': ['B4', 'B3', 'B2']
        }
        
        # 导出到 Drive
        task = ee.batch.Export.image.toDrive(
            image=rgb.visualize(**vis_params),
            description=description,
            folder=self.project_name,
            scale=scale,
            region=aoi,
            fileFormat='GeoTIFF',
            maxPixels=1e13
        )
        
        task.start()
        print(f"导出任务已启动: {description}")
        print(f"请在 Google Drive/{self.project_name} 文件夹中查看")
        
        return task
    
    def segment_with_sam_text(self, image_path, text_prompt="buildings urban areas"):
        """
        使用文本提示的 SAM 分割
        """
        from samgeo import LangSAM
        
        # 初始化 LangSAM
        sam = LangSAM()
        
        # 使用文本提示进行分割
        sam.predict(
            image_path,
            text_prompt=text_prompt,
            box_threshold=0.24,
            text_threshold=0.24,
            output_dir=self.output_folder
        )
        
        return sam
    
    def create_urban_map(self, aoi, building_fc, year=2023):
        """
        创建建城区地图的完整流程
        """
        print("=== 开始建城区识别流程 ===\n")
        
        # 1. 获取 Sentinel-2 合成影像
        print("1. 获取 Sentinel-2 合成影像...")
        s2_composite = self.get_s2_composite(aoi, year)
        
        # 2. 创建训练数据
        print("2. 创建训练数据...")
        urban_mask, non_urban_mask = self.create_urban_training_data(aoi, building_fc)
        
        # 3. 采样训练点
        print("3. 采样训练点...")
        bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI', 'NDBI', 'UI', 'BU']
        training_data = self.sample_training_points(
            s2_composite.select(bands), 
            urban_mask, 
            non_urban_mask
        )
        
        # 4. 训练分类器
        print("4. 训练随机森林分类器...")
        classifier = self.train_classifier(training_data, bands)
        
        # 5. 分类
        print("5. 执行分类...")
        classified = s2_composite.select(bands).classify(classifier)
        
        # 6. 后处理
        print("6. 后处理优化...")
        # 使用形态学操作优化结果
        kernel = ee.Kernel.circle(radius=1)
        urban_final = classified.focal_mode(radius=1, kernelType='circle')
        
        return urban_final, s2_composite
    
    def visualize_results(self, urban_map, s2_composite, aoi):
        """
        在 GEE 中可视化结果
        """
        # 创建地图
        Map = geemap.Map()
        Map.centerObject(aoi, 10)
        
        # 添加 Sentinel-2 影像
        vis_params_s2 = {
            'min': 0,
            'max': 0.3,
            'bands': ['B4', 'B3', 'B2']
        }
        Map.addLayer(s2_composite, vis_params_s2, 'Sentinel-2')
        
        # 添加建城区分类结果
        urban_params = {
            'min': 0,
            'max': 1,
            'palette': ['green', 'red']
        }
        Map.addLayer(urban_map, urban_params, 'Urban Areas')
        
        # 添加图例
        legend_dict = {
            'Non-urban': 'green',
            'Urban': 'red'
        }
        Map.add_legend(legend_dict=legend_dict)
        
        return Map
    
    def run_complete_workflow(self, country_name='Iceland'):
        """
        运行完整的工作流程
        """
        # 1. 获取国家边界
        print(f"处理国家: {country_name}")
        aoi = self.get_country_boundary(country_name)
        
        # 2. 获取建筑物数据（假设已上传到 GEE）
        # 这里需要替换为您的实际建筑物 Feature Collection 路径
        building_fc = ee.FeatureCollection('users/your_username/iceland_buildings')
        
        # 3. 创建建城区地图
        urban_map, s2_composite = self.create_urban_map(aoi, building_fc)
        
        # 4. 可视化结果
        Map = self.visualize_results(urban_map, s2_composite, aoi)
        
        # 5. 导出结果
        self.export_results(urban_map, aoi, f'{country_name}_urban_areas')
        
        return Map
    
    def export_results(self, image, aoi, description):
        """
        导出最终结果
        """
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=description,
            folder=self.project_name,
            scale=10,
            region=aoi,
            fileFormat='GeoTIFF',
            maxPixels=1e13
        )
        
        task.start()
        print(f"\n导出任务已启动: {description}")


# ===== 第3部分：使用示例 =====

# 初始化
segmenter = GEEUrbanSegmentation(project_name='arctic_urban_segmentation')

# 示例1：处理冰岛特定区域
iceland_test_area = ee.Geometry.Rectangle([-22.0, 64.0, -21.5, 64.5])

# 获取 Sentinel-2 合成影像
s2_composite = segmenter.get_s2_composite(iceland_test_area, year=2023)

# 创建交互式地图
Map = geemap.Map()
Map.centerObject(iceland_test_area, 11)
Map.addLayer(s2_composite, {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 0.3}, 'Sentinel-2')
Map.addLayer(s2_composite.select('NDBI'), {'min': -0.5, 'max': 0.5, 'palette': ['blue', 'white', 'red']}, 'NDBI')
Map.addLayer(s2_composite.select('UI'), {'min': -1, 'max': 1, 'palette': ['green', 'yellow', 'red']}, 'Urban Index')

# 显示地图
# Map

# 示例2：使用 SAM 进行精细分割
# 首先导出影像
# export_task = segmenter.export_for_sam(s2_composite, iceland_test_area, 'iceland_test_rgb')

# 等待导出完成后，使用 SAM 处理
# image_path = '/content/drive/MyDrive/arctic_urban_segmentation/iceland_test_rgb.tif'
# sam_results = segmenter.segment_with_sam_text(image_path, "buildings roads urban infrastructure")

print("工作流程设置完成！")
print("\n下一步：")
print("1. 将您的建筑物轮廓数据上传到 GEE")
print("2. 运行 segmenter.run_complete_workflow('Iceland')")
print("3. 在 Google Drive 中查看导出的结果")
