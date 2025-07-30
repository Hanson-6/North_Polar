# python/gee_modules/GEE.py

import ee
import math
import numpy as np
from PIL import Image
import requests
import io

from gee_modules.config import *
from gee_modules.schema import Platform
from annotation.utils import Tool
from gee_modules.GMap import GMap

from IPython.display import display

class GEEBase:
    """
    GEE基础类，实现基本的功能服务
    """

    def __init__(self, platform_name):
        # 初始化GMap - 修改这里，传入self实例
        self.gmap = GMap(gee_instance=self)

        # 设置影像平台
        self.platform = Platform(platform_name)


    """
    以下函数为功能函数，服务于用户操作平台的功能
    """

    def connect(self, project_id):
        """
        连接到GEE引擎

        参数：
            project_id[str]: GEE项目ID
        """

        try:
            ee.Authenticate()
            ee.Initialize(project=project_id)
            print("✅ 成功连接到Google Earth Engine")

        except Exception as e:
            print(f"❌ GEE连接失败: {str(e)}")
            raise

        
    """
    这些函数是为了训练模型而做准备。
    """

    def makeGrids(self, rect_bound, num_rows, num_cols):
        """
        根据给定的矩形边界和行列数创建网格（方便探索地图需要放缩到什么比例，才能看到地面细节）

        参数：
            rect_bound[ee_polygon]: 矩形边界
            num_rows[int]: 行数
            num_cols[int]: 列数

        返回：
            ee_featureCollection: 包含网格单元的FeatureCollection
        """
        
        if rect_bound is None:
            raise ValueError("rect_bound must be provided")

        coords = ee_list(rect_bound.coordinates().get(0))
        get_val = lambda coord_idx, point_idx: ee_number(ee_list(coords.get(coord_idx)).get(point_idx))
        
        num_rows = ee_number(num_rows)
        num_cols = ee_number(num_cols)

        x_min = get_val(0, 0)
        y_min = get_val(0, 1)
        x_max = get_val(2, 0)
        y_max = get_val(2, 1)

        width = (x_max.subtract(x_min)).divide(num_cols)
        height = (y_max.subtract(y_min)).divide(num_rows)

        col_seq = ee_list.sequence(0, num_cols.subtract(1))
        row_seq = ee_list.sequence(0, num_rows.subtract(1))
        
        def create_row(row_idx):
            def create_col(col_idx):
                x1 = x_min.add(width.multiply(col_idx))
                y1 = y_min.add(height.multiply(row_idx))
                x2 = x1.add(width)
                y2 = y1.add(height)

                rect = ee_geometry.Rectangle([x1, y1, x2, y2])
                return ee_feature(rect, {'row': row_idx, 'col': col_idx})
            
            return ee_list(col_seq.map(create_col))
        
        all_cells = ee_list(row_seq.map(create_row))
        grid_fc = ee_featureCollection(ee_list(all_cells).flatten())

        return grid_fc
    
    def maskClouds(self, image):
        """
        使用QA60波段掩码云和卷云

        参数：
            image[ee_image]: 输入影像

        返回：
            ee_image: 掩码后的影像
        """

        qa = image.select('QA60')
        # 云的位掩码
        cloudBitMask = 1 << 10
        # 卷云的位掩码
        cirrusBitMask = 1 << 11
        # 两个标志都应该设置为零，表示晴朗的条件
        mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(
            qa.bitwiseAnd(cirrusBitMask).eq(0)
        )
        # 应用掩码并保留原始属性
        return image.updateMask(mask).select(
            image.bandNames()  # 保留所有波段
        ).copyProperties(image, ['system:time_start'])
    
    def addIndices(self, image):
        """
        添加对建筑物检测有用的指数
        """
        # NDVI - 植被指数（帮助区分植被和建筑物）
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        
        # NDBI - 归一化建筑指数
        ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')
        
        # NDWI - 水体指数（区分水体）
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
        
        # BUI - 建筑区指数
        bui = ndbi.subtract(ndvi).rename('BUI')
        
        # BSI - 裸土指数（帮助区分裸土和建筑物）
        bsi = image.expression(
            '((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2))',
            {
                'B11': image.select('B11'),
                'B4': image.select('B4'),
                'B8': image.select('B8'),
                'B2': image.select('B2')
            }
        ).rename('BSI')
        
        return image.addBands([ndvi, ndbi, ndwi, bui, bsi])
    
    def exportPic(self,
        coords,
        band_type='rgb',
        start_date='2022-01-01',
        end_date='2023-12-31',
        vis_mode='true_color',
        add_indices=False,
        composite_method='median'):
        """
        导出指定区域的影像

        参数：
            coords[ee_geometry]: 区域坐标（Geometry或FeatureCollection）
            band_type[str]: 波段类型（如'rgb', 'ndvi', 'ndbi', 'ndwi', 'bui', 'bsi'）
            start_date[str]: 开始日期（格式：YYYY-MM-DD）
            end_date[str]: 结束日期（格式：YYYY-MM-DD）
            vis_mode[str]: 可视化模式（如'true_color', 'false_color', 'urban_false_color'）
            add_indices[bool]: 是否添加指数
            composite_method[str]: 合成方法（如'median', 'mean', 'mosaic'）

        返回：
            dict: 包含影像、可视化参数、边界和其他信息的字典
        """

        # 确保coords是Geometry
        if isinstance(coords, (ee_feature, ee_featureCollection, ee_element)):
            coords = coords.geometry()
        
        # 获取影像集合
        plat_coll = self.platform.get_collection()
        sentinel_coll = ee_imgColl(plat_coll) \
                        .filterBounds(coords) \
                        .filterDate(start_date, end_date)
        
        # 云量过滤 - 使用元数据中的云量百分比
        filters = self.platform.get_filter()
        if filters: 
            sentinel_coll = sentinel_coll.filter(
                ee_filter.lt(filters['cloudCover'], filters['maxCloud'])
            )
        
        # 应用云掩码
        if self.platform.name == 'sentinel2':
            sentinel_coll = sentinel_coll.map(self.maskClouds)
        
        # 检查可用影像数量
        image_count = sentinel_coll.size()
        
        # 如果需要，添加指数
        if add_indices:
            sentinel_coll = sentinel_coll.map(self.addIndices)
        
        # 创建合成影像
        if composite_method == 'median':
            composite = sentinel_coll.median()
        elif composite_method == 'mean':
            composite = sentinel_coll.mean()
        elif composite_method == 'mosaic':
            composite = sentinel_coll.mosaic()
        else:
            composite = sentinel_coll.median()  # 默认
        
        # 裁剪到区域
        composite = composite.clip(coords)
        
        # 获取可视化波段
        if vis_mode in self.platform.vis_modes:
            vis_bands = self.platform.vis_modes[vis_mode]['bands']
        else:
            # 兼容旧版本的band_type参数
            vis_bands = self.platform.get_bands(band_type)
        
        # 准备可视化参数
        base_vis = self.platform.get_vis_params()
        vis_params = {
            'bands': vis_bands,
            'min': base_vis.get('min'),
            'max': base_vis.get('max'),
            'gamma': base_vis.get('gamma', 1.4)
        }
        
        # 特殊处理某些可视化模式
        if vis_mode == 'urban_false_color':
            # 城市假彩色通常需要不同的参数
            vis_params['min'] = 0
            vis_params['max'] = 2000
            vis_params['gamma'] = 1.5
        elif vis_mode == 'false_color':
            # 假彩色可能需要更高的最大值
            vis_params['max'] = 4000
        
        # 返回结果
        result = {
            'image': composite,
            'vis_params': vis_params,
            'bounds': coords,
            'platform': self.platform.name,
            'image_count': image_count,
            'vis_mode': vis_mode,
            'composite_method': composite_method
        }
        
        # 如果添加了指数，包含在结果中
        if add_indices:
            result['indices'] = ['NDVI', 'NDBI', 'NDWI', 'BUI', 'BSI']
        
        return result
    
    def exportSingleAreaForModel(self,
                                geometry,
                                output_size=(256, 256),
                                start_date='2022-01-01',
                                end_date='2023-12-31',
                                vis_mode='true_color'):
        """
        导出单个区域的影像供模型使用
        
        参数：
            geometry[ee_geometry]: 区域几何
            output_size[tuple]: 输出尺寸 (width, height)
            start_date[str]: 开始日期
            end_date[str]: 结束日期
            vis_mode[str]: 可视化模式
            
        返回：
            dict: 包含图像数据和元信息的字典
        """
        try:
            # 获取影像数据
            pic_data = self.exportPic(
                coords=geometry,
                start_date=start_date,
                end_date=end_date,
                vis_mode=vis_mode,
                composite_method='median'
            )
            
            # 直接使用已经配置好的image和vis_params
            image = pic_data['image']
            vis_params = pic_data['vis_params']
            
            # 生成图片URL
            url = image.visualize(**vis_params).getThumbURL({
                'dimensions': output_size,
                'format': 'png',
            })
            
            # 下载图像数据
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # 转换为PIL Image
            pil_image = Image.open(io.BytesIO(response.content))
            
            # 转换为numpy数组
            image_array = np.array(pil_image)
            
            # 获取边界信息
            bounds_info = geometry.bounds().getInfo()
            
            return {
                'image': pil_image,
                'image_array': image_array,
                'bounds': bounds_info,
                'geometry': geometry.getInfo(),
                'vis_mode': vis_mode,
                'size': output_size
            }
            
        except Exception as e:
            print(f"导出图像失败: {str(e)}")
            return None
    
    def importData(self, dir_path):
        """
        从指定目录导入国家边界数据（json数据）
        """
        
        countries_polygons = Tool.readBunchJSON(dir_path)
        
        dataset = {}
        for country in countries_polygons.keys():
            coords = countries_polygons[country]
            coords = ee_featureCollection(list(map(lambda coord : ee_feature(ee_poly(coord)), coords)))
            bounds = ee_featureCollection(coords.map(lambda coord : ee_feature(coord.geometry().bounds())))
            
            dataset[country] = {
                'coords': coords,
                'bounds': bounds
            }
        
        return dataset
    
    def expandBound(self, bound, ratio=0.3):
        """
        扩展bound的面积，使其增加指定比例（比如，0.3=30%）

        参数：
            bound[list]: 原始边界坐标列表，例如 [[lon1, lat1], [lon2, lat2], ..., [lon1, lat1]]
            ratio[float]: 扩展比例，默认0.3（30%）

        返回：
            ee_rect: 扩展后的边界矩形
        """

        # 一次循环，获得边界的最小和最大经纬度
        min_lon, max_lon = float('inf'), float('-inf')
        min_lat, max_lat = float('inf'), float('-inf')

        for coord in bound:
            if coord[0] < min_lon: min_lon = coord[0]
            if coord[0] > max_lon: max_lon = coord[0]
            if coord[1] < min_lat: min_lat = coord[1]
            if coord[1] > max_lat: max_lat = coord[1]

        # 获取中心点，和半宽高
        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2
        half_width = (max_lon - min_lon) / 2
        half_height = (max_lat - min_lat) / 2

        # 获取扩展比例
        scale = math.sqrt(1 + ratio)

        # 计算新的边界
        new_half_width = half_width * scale
        new_half_height = half_height * scale

        new_min_lon = center_lon - new_half_width
        new_max_lon = center_lon + new_half_width
        new_min_lat = center_lat - new_half_height
        new_max_lat = center_lat + new_half_height

        new_bound = [
            [new_min_lon, new_min_lat],
            [new_max_lon, new_min_lat],
            [new_max_lon, new_max_lat],
            [new_min_lon, new_max_lat],
            [new_min_lon, new_min_lat]  # 闭合
        ]

        # return new_bound
        return new_bound



class GEE(GEEBase):
    """
    继承GEEBase类，拓展功能。

    这是真正的操作类
    """

    
    def __init__(self, GEE_PROJECT_ID):
        # 连接GEE引擎
        self.connect(GEE_PROJECT_ID)
        
        # 继承基础类
        super().__init__(platform_name='sentinel2')

    
    def main(self):
        """
        主函数，执行GEE用户平台
        """
        # 启动地图
        self.gmap.start()

    """
    这些函数是为了训练模型而做准备。
    """

    def exportBunchPics(self,
                        coords_list,
                        band_type='rgb',
                        start_date='2022-01-01',
                        end_date='2023-12-31',
                        **kwargs):
        """
        批量导出图片，支持所有exportPic的参数
        """
        # 处理FeatureCollection
        if hasattr(coords_list, 'getInfo'):
            coords_list = coords_list.getInfo()['features']
        
        pics_list = []
        total = len(coords_list)
        
        print(f"开始处理 {total} 个区域...")
        
        for idx, coords in enumerate(coords_list):
            try:
                # 显示进度
                if (idx + 1) % 10 == 0:
                    print(f"进度: {idx + 1}/{total}")
                
                # 导出图片
                info = self.exportPic(
                    coords=ee_geometry(coords['geometry']),
                    band_type=band_type,
                    start_date=start_date,
                    end_date=end_date,
                    **kwargs  # 传递所有额外参数
                )
                
                # 添加索引信息
                info['index'] = idx
                pics_list.append(info)
                
            except Exception as e:
                print(f"⚠️ 区域 {idx} 处理失败: {str(e)}")
                continue
        
        print(f"✅ 完成！成功处理 {len(pics_list)}/{total} 个区域")
        return pics_list
    
    def expandBunchBound(self, bounds, ratio=0.3):
        """
        批量扩展边界
        """
        expanded_bounds = []
        for bound in bounds:
            expanded_bound = self.expandBound(bound, ratio)
            expanded_bounds.append(expanded_bound)
        
        return expanded_bounds