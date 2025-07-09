import os
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from datetime import datetime
import logging
from tqdm import tqdm
import requests


from PIL import Image
import io
import numpy as np
from osgeo import gdal, osr


class Downloader:
    """
    Downloader is specific for downloading images required.
    """

    def __init__(self,
                 gee_instance,
                 max_workers=5,
                 retry_times=3,
                 country=None
        ):
        self.gee = gee_instance
        self.max_workers = max_workers
        self.retry_times = retry_times
        self.download_lock = Lock()
        self.success_count = 0
        self.failed_count = 0
        self.country = country
        
        self.setup_saveDir()
        self.setup_logging()
    

    def setup_saveDir(self):
        self.base_dir = f"{os.getcwd()}/model/dataset"
        self.date = datetime.now().strftime('%Y-%m-%d')
        
        self.save_img_dir = f"{self.base_dir}/sentinel_2/{self.country}/img/{self.date}"
        self.save_geotiff_dir = f"{self.base_dir}/sentinel_2/{self.country}/geotiff/{self.date}"
        
        self.bound_dir = f"{self.base_dir}/bounds"
        

        if not os.path.exists(self.save_img_dir):
            os.makedirs(self.save_img_dir)

        if not os.path.exists(self.bound_dir):
            os.makedirs(self.bound_dir)

        if not os.path.exists(self.save_geotiff_dir):
            os.makedirs(self.save_geotiff_dir)
        

    def setup_logging(self):
        # setup logging        
        logging.basicConfig(
            level=logging.INFO, # level of general level
            format='%(asctime)s - %(levelname)s - %(message)s', # file formart
            handlers=[
                logging.FileHandler(f'{self.save_img_dir}/log.log'), # save log file
                logging.StreamHandler() # print on terminal
            ]
        )
        self.logger = logging.getLogger(__name__)


    def download_bounds_list(self, pics_list):
        total_count = len(pics_list)
        bounds = [None] * total_count

        msg = f"{self.date} 开始从pics_list下载"
        self.logger.info(msg)

        msg = f"总共需要下载 {total_count} 个bounds"
        self.logger.info(msg)

        # count
        avail = 0
        failed = 0

        for idx, pic in enumerate(pics_list):
            bound = pic.get('bounds', None)
            if bound:
                bounds[idx] = bound
                avail += 1
            else:
                failed += 1

        msg = f"获取bounds：成功: {avail}, 失败: {failed}"
        self.logger.info(msg)

        # 多线程下载
        def download_one_bound(idx, bound):
            tmp = bound.coordinates().getInfo()[0]
            bounds[idx] = tmp

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(
                    download_one_bound,
                    idx,
                    bound
                ): (idx, bound) for (idx, bound) in enumerate(bounds)
            }
            # 使用进度条
            with tqdm(total=avail, desc="下载进度") as pbar:
                for future in as_completed(future_to_task):
                    idx, bound = future_to_task[future]
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.error(f"任务异常: 第{idx}个bound, 错误: {str(e)}")
                    pbar.update(1)

        # bounds存储
        self.logger.info(f"\n下载完成！")

        with open(f"{self.bound_dir}/{self.country}.json", "w") as file:
            json.dump(bounds, file, indent=4)        


    def download_single_from_pic(self,
                               pic_data,
                               img_name,
                               output_path,
                               output_size=(256, 256)):
        """
        从pics_list中的单个pic对象下载图片
        
        Args:
            pic_data: 从exportPic返回的字典
            img_name: 图片名称
            output_path: 输出路径
            output_size: 输出尺寸
        """
        for attempt in range(self.retry_times):
            try:
                # 直接使用已经配置好的image和vis_params
                image = pic_data['image']
                vis_params = pic_data['vis_params']
                
                # 生成图片URL
                url = image.visualize(**vis_params).getThumbURL({
                    'dimensions': output_size,
                    'format': 'png',
                    # 'format': 'tiff',
                })
                
                # 下载
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # 保存
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                # 更新成功计数
                with self.download_lock:
                    self.success_count += 1
                
                # msg = f"成功下载: {img_name}"
                # self.logger.info(msg)
                return True
                
            except Exception as e:
                if attempt < self.retry_times - 1:
                    time.sleep(2 ** attempt)
                else:
                    with self.download_lock:
                        self.failed_count += 1
                    self.logger.error(f"最终失败: {img_name}, 错误: {str(e)}")
                    return False
    


    def download_from_pics_list(self, pics_list, output_size=(256, 256)):
        """
        从已经导出的pics_list下载图片
        
        Args:
            pics_list: 从exportBunchPics返回的列表
            output_size: 输出尺寸
        """
        msg = f"{self.date} 开始从pics_list下载"
        self.logger.info(msg)
        
        total_count = len(pics_list)
        self.logger.info(f"总共需要下载 {total_count} 张图片")
        
        # 获取vis_mode信息
        if pics_list and len(pics_list) > 0:
            vis_mode = pics_list[0].get('vis_mode', 'unknown')
            self.logger.info(f"使用可视化模式: {vis_mode}")
        
        # 准备下载任务
        download_tasks = []
        for pic in pics_list:
            idx = pic.get('index', 0)
            img_name = f"{self.country}_{idx}.png"
            # img_name = f"{self.country}_{idx}.tiff"
            output_path = f"{self.save_img_dir}/{img_name}"
            
            # 跳过已存在的文件
            if os.path.exists(output_path):
                self.logger.info(f"跳过已存在的文件: {img_name}")
                self.success_count += 1
                continue
            
            # 准备即将的下载任务
            download_tasks.append({
                'pic_data': pic,
                'img_name': img_name,
                'output_path': output_path,
                'output_size': output_size
            })
        
        # 使用线程池执行下载
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(
                    self.download_single_from_pic,
                    task['pic_data'],
                    task['img_name'],
                    task['output_path'],
                    task['output_size']
                ): task for task in download_tasks
            }
            
            # 使用进度条
            with tqdm(total=len(download_tasks), desc="下载进度") as pbar:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.error(f"任务异常: {task['output_path']}, 错误: {str(e)}")
                    pbar.update(1)
        
        # 打印最终统计
        self.logger.info(f"\n下载完成！成功: {self.success_count}, 失败: {self.failed_count}")
        
        return {
            'success': self.success_count,
            'failed': self.failed_count,
            'total': total_count
        }
    

    def png_to_geotiff(self, png_input, output_path, bounds, epsg=4326):
        try:
            # 1. 读取PNG图像
            if isinstance(png_input, str):
                img = Image.open(png_input)
            elif isinstance(png_input, bytes):
                img = Image.open(io.BytesIO(png_input))
            else:
                img = Image.open(png_input)
            
            # 转换为numpy数组
            img_array = np.array(img)
            
            # 2. 处理bounds
            if hasattr(bounds, 'getInfo'):
                # 如果是GEE对象，获取实际坐标
                bounds_info = bounds.bounds().getInfo()
                coords = bounds_info['coordinates'][0]
                west = coords[0][0]
                south = coords[0][1]
                east = coords[2][0]
                north = coords[2][1]
            else:
                # 如果是列表 [west, south, east, north]
                west, south, east, north = bounds
            
            # 3. 获取图像尺寸
            if len(img_array.shape) == 3:
                height, width, bands = img_array.shape
            else:
                height, width = img_array.shape
                bands = 1
            
            # 4. 创建GeoTIFF
            driver = gdal.GetDriverByName('GTiff')
            
            # 确定数据类型
            if img_array.dtype == np.uint8:
                gdal_dtype = gdal.GDT_Byte
            elif img_array.dtype == np.uint16:
                gdal_dtype = gdal.GDT_UInt16
            else:
                gdal_dtype = gdal.GDT_Float32
            
            # 创建数据集
            dataset = driver.Create(output_path, width, height, bands, gdal_dtype)
            
            # 5. 设置地理变换参数
            # GeoTransform = [左上角X, 像素宽度, 旋转, 左上角Y, 旋转, 像素高度]
            pixel_width = (east - west) / width
            pixel_height = (south - north) / height  # 注意是负值
            
            geotransform = [west, pixel_width, 0, north, 0, pixel_height]
            dataset.SetGeoTransform(geotransform)
            
            # 6. 设置投影
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(epsg)
            dataset.SetProjection(srs.ExportToWkt())
            
            # 7. 写入图像数据
            if bands == 1:
                dataset.GetRasterBand(1).WriteArray(img_array)
            else:
                for i in range(bands):
                    dataset.GetRasterBand(i + 1).WriteArray(img_array[:, :, i])
            
            # 8. 刷新缓存
            dataset.FlushCache()
            dataset = None
            
            return True
            
        except Exception as e:
            print(f"创建GeoTIFF失败: {str(e)}")
            return False
        

    def convert_pngs_to_geotiff(self):
        # 检查PNG文件夹是否存在
        if not os.path.exists(self.save_img_dir):
            self.logger.error(f"PNG图片目录不存在: {self.save_img_dir}")
            return {'success': 0, 'failed': 0, 'total': 0}
        
        # 检查bounds是否存在
        if not os.path.exists(f"{self.bound_dir}/{self.country}.json"):
            self.logger.error(f"bounds文件不存在: s{self.country}.json")
            return {'success': 0, 'failed': 0, 'total': 0}
        
        with open(f"{self.bound_dir}/{self.country}.json", 'r') as file:
            bounds = json.load(file)
        
        # 获取所有PNG文件
        img_names = [f[:-4] for f in os.listdir(self.save_img_dir) if f.endswith('.png')]
        if not img_names:
            self.logger.warning(f"PNG图片目录为空: {self.save_img_dir}")
            return {'success': 0, 'failed': 0, 'total': 0}
        
        tasks = []

        for img_name in img_names:
            idx = int(img_name.split("_")[1])
            img_path = f"{self.save_img_dir}/{img_name}.png"
            tiff_path = f"{self.save_geotiff_dir}/{img_name}.tiff"

            bound = bounds[idx]

            west = bound[0][0]
            south = bound[0][1]
            east = bound[2][0]
            north = bound[2][1]

            tasks.append({
                "idx": idx,
                "img_path": img_path,
                "output_path": tiff_path,
                "bound": [west, south, east, north],
                "epsg": 4326
            })
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
            executor.submit(
                self.png_to_geotiff,
                task["img_path"],
                task["output_path"],
                task["bound"],
                task["epsg"]
                ): task for task in tasks
        }
        
        # 使用进度条
        with tqdm(total=len(tasks), desc="转换进度") as pbar:
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"任务异常 {task['png_file']}: {str(e)}")
                pbar.update(1)