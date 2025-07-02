import ee

from gee_modules.config import *
from gee_modules.schema import FeatColl, Platform
from annotation.utils import Tool



class GEE:
    def __init__(self, project_id, platform_name):
        self.project_id = project_id
        self.connect()
        self.platform = Platform(platform_name)
        
        # 定义可视化模式
        self.vis_modes = {
            'true_color': {
                'bands': ['B4', 'B3', 'B2'],
                'name': '真彩色'
            },
            'false_color': {
                'bands': ['B8', 'B4', 'B3'],
                'name': '标准假彩色'
            },
            'urban_false_color': {
                'bands': ['B12', 'B11', 'B4'],
                'name': '城市假彩色'
            },
            'vegetation': {
                'bands': ['B8', 'B11', 'B2'],
                'name': '植被分析'
            },
            'swir': {
                'bands': ['B12', 'B8A', 'B4'],
                'name': '短波红外'
            }
        }
    

    def connect(self):
        """
        Connect to Google Earth Engine
        """
        try:
            ee.Authenticate()
            ee.Initialize(project=self.project_id)
            print("✅ 成功连接到Google Earth Engine")
        except Exception as e:
            print(f"❌ GEE连接失败: {str(e)}")
            raise


    def makeGrids(self, rect_bounds, num_rows, num_cols):
        """
        Create a grid of rectangles within the specified bounds.
        """
        if rect_bounds is None:
            raise ValueError("rect_bounds must be provided")

        coords = ee_list(rect_bounds.coordinates().get(0))
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
        grid_fc = ee_featColl(ee_list(all_cells).flatten())

        return grid_fc
    

    def maskClouds(self, image):
        """
        稳定的云掩码算法，仅使用QA60波段
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
        Export a picture from Google Earth Engine.
        
        Args:
            coords: 几何边界
            band_type: 波段类型（兼容旧版本）
            start_date: 开始日期
            end_date: 结束日期
            vis_mode: 可视化模式
            add_indices: 是否添加建筑物相关指数
            composite_method: 合成方法 ('median', 'mean', 'mosaic')
        """

        # 确保coords是Geometry
        if isinstance(coords, (ee_feature, ee_featColl, ee_element)):
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
        if vis_mode in self.vis_modes:
            vis_bands = self.vis_modes[vis_mode]['bands']
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
    

    def importData(self, dir_path):
        """
        Import JSON data from given directory path
        """
        countries_polygons = Tool.readBunchJSON(dir_path)
        
        dataset = {}
        for country in countries_polygons.keys():
            coords = countries_polygons[country]
            coords = ee_featColl(list(map(lambda coord : ee_feature(ee_poly(coord)), coords)))
            bounds = ee_featColl(coords.map(lambda coord : ee_feature(coord.geometry().bounds())))
            
            dataset[country] = {
                'coords': coords,
                'bounds': bounds
            }
        
        return dataset
    

    def analyzeImageQuality(self, bounds, start_date, end_date):
        """
        分析指定区域和时间范围内的影像质量
        """
        # 获取影像集合
        collection = ee.ImageCollection(self.platform.get_collection()) \
            .filterBounds(bounds) \
            .filterDate(start_date, end_date)
        
        # 基本统计
        total_count = collection.size()
        
        # 云量统计
        if self.platform.name == 'sentinel2':
            cloud_stats = collection.aggregate_stats('CLOUDY_PIXEL_PERCENTAGE')
            
            # 低云量影像数
            low_cloud = collection.filter(
                ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)
            ).size()
            
            print("📊 影像质量分析报告")
            print(f"时间范围: {start_date} 到 {end_date}")
            print(f"总影像数: {total_count.getInfo()}")
            print(f"低云量影像 (<20%): {low_cloud.getInfo()}")
            
            stats = cloud_stats.getInfo()
            if stats:
                print(f"平均云量: {stats.get('mean', 'N/A'):.1f}%")
                print(f"最小云量: {stats.get('min', 'N/A'):.1f}%")
                print(f"最大云量: {stats.get('max', 'N/A'):.1f}%")
        
        return collection