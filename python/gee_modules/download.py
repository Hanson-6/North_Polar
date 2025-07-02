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
            params: image information
        """

        for attempt in range(self.retry_times):
            try:
                # export pics from gee
                img_data = self.gee.exportPic(
                    coords=bounds,
                    # output_size=params.get('output_size', (256, 256)),
                    band_type=params.get('band_type', 'rgb'),
                    start_date=params.get('start_date', '2022-01-01'),
                    end_date=params.get('end_date', '2023-12-31')
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

                    self.logger.error(f"最终失败: {img_name}")
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

        # 默认参数
        if params is None:
            params = {
                'output_size': (256, 256),
                'band_type': 'rgb',
                'start_date': '2022-01-01',
                'end_date': '2023-12-31'
            }
        
        # 获取所有 bounds
        bounds_list = bounds_fc.getInfo()['features']
        total_count = len(bounds_list)
        self.logger.info(f"总共需要下载 {total_count} 张图片")
        
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