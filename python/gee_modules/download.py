import ee
import os
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from datetime import datetime
import logging
from tqdm import tqdm
import requests



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
        self.date = datetime.now().strftime('%Y-%m-%d')
        self.save_dir = f"{os.getcwd()}/model/dataset/sentinel_2/{self.country}/img_{self.date}"

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        

    def setup_logging(self):
        # setup logging        
        logging.basicConfig(
            level=logging.INFO, # level of general level
            format='%(asctime)s - %(levelname)s - %(message)s', # file formart
            handlers=[
                logging.FileHandler(f'{self.save_dir}/log.log'), # save log file
                logging.StreamHandler() # print on terminal
            ]
        )
        self.logger = logging.getLogger(__name__)

    
    def download_single_image(self,
                            bounds,
                            img_name,
                            output_path,
                            params):
        """
        download single image
        
        Args:
            bounds: ee.Geometry object
            img_name: 图片名称
            output_path: 输出路径
            params: image information
        """

        for attempt in range(self.retry_times):
            try:
                # export pics from gee - 传递所有参数包括vis_mode
                img_data = self.gee.exportPic(
                    coords=bounds,
                    band_type=params.get('band_type', 'rgb'),
                    start_date=params.get('start_date', '2022-01-01'),
                    end_date=params.get('end_date', '2023-12-31'),
                    vis_mode=params.get('vis_mode', 'true_color'),  # 使用vis_mode参数
                    add_indices=params.get('add_indices', False),
                    composite_method=params.get('composite_method', 'median')
                )
                
                # gain img url
                image = img_data['image']
                vis_params = img_data['vis_params']
                
                # generate img url
                url = image.visualize(**vis_params).getThumbURL({
                    'dimensions': params.get('output_size', (256, 256)),
                    'format': 'png'
                })
                
                # downlaod
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # save
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                # update success count
                with self.download_lock:
                    self.success_count += 1
                
                msg = f"成功下载: {img_name}"
                self.logger.info(msg)
                return True
                
            except Exception as e:
                if attempt < self.retry_times - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    with self.download_lock:
                        self.failed_count += 1

                    self.logger.error(f"最终失败: {img_name}, 错误: {str(e)}")
                    return False


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
                    'format': 'png'
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
                
                msg = f"成功下载: {img_name}"
                self.logger.info(msg)
                return True
                
            except Exception as e:
                if attempt < self.retry_times - 1:
                    time.sleep(2 ** attempt)
                else:
                    with self.download_lock:
                        self.failed_count += 1
                    self.logger.error(f"最终失败: {img_name}, 错误: {str(e)}")
                    return False


    def download_batch(self, bounds_fc, params=None):
        """
        download in the form of batch
        
        Args:
            bounds_fc: ee.FeatureCollection
            params: 导出参数
        """
        msg = f"{self.date} 开始下载"
        self.logger.info(msg)

        # 默认参数 - 添加vis_mode支持
        if params is None:
            params = {
                'output_size': (256, 256),
                'band_type': 'rgb',
                'start_date': '2022-01-01',
                'end_date': '2023-12-31',
                'vis_mode': 'true_color',  # 默认使用真彩色
                'add_indices': False,
                'composite_method': 'median'
            }
        
        # 获取所有 bounds
        bounds_list = bounds_fc.getInfo()['features']
        total_count = len(bounds_list)
        self.logger.info(f"总共需要下载 {total_count} 张图片")
        self.logger.info(f"使用可视化模式: {params.get('vis_mode', 'true_color')}")
        
        # 准备下载任务
        download_tasks = []
        for idx, feature in enumerate(bounds_list):
            geometry = ee.Geometry(feature['geometry'])

            img_name = f"{self.country}_{idx}.png"
            output_path = f"{self.save_dir}/{img_name}"
            
            # 跳过已存在的文件
            if os.path.exists(output_path):
                self.logger.info(f"跳过已存在的文件: {img_name}")
                self.success_count += 1
                continue
                
            download_tasks.append({
                'bounds': geometry,                
                'img_name': img_name,
                'output_path': output_path,
                'params': params,
            })
        
        # 使用线程池执行下载
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(
                    self.download_single_image,
                    task['bounds'],
                    task['img_name'],
                    task['output_path'],
                    task['params']
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
            output_path = f"{self.save_dir}/{img_name}"
            
            # 跳过已存在的文件
            if os.path.exists(output_path):
                self.logger.info(f"跳过已存在的文件: {img_name}")
                self.success_count += 1
                continue
            
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


    def save_metadata(self, pics_list, filename="metadata.json"):
        """
        保存pics_list的元数据信息
        
        Args:
            pics_list: 从exportBunchPics返回的列表
            filename: 元数据文件名
        """
        metadata_path = os.path.join(self.save_dir, filename)
        
        # 准备元数据
        metadata = {
            'date': self.date,
            'country': self.country,
            'total_images': len(pics_list),
            'images': []
        }
        
        # 添加每张图片的信息
        for pic in pics_list:
            img_info = {
                'index': pic.get('index'),
                'filename': f"{self.country}_{pic.get('index')}.png",
                'vis_mode': pic.get('vis_mode'),
                'composite_method': pic.get('composite_method'),
                'image_count': pic.get('image_count').getInfo() if hasattr(pic.get('image_count'), 'getInfo') else pic.get('image_count'),
                'platform': pic.get('platform')
            }
            
            # 如果包含indices信息
            if 'indices' in pic:
                img_info['indices'] = pic['indices']
            
            metadata['images'].append(img_info)
        
        # 保存元数据
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"元数据已保存到: {metadata_path}")