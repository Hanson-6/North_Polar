import json
from datetime import datetime
from ipywidgets import Button, Label, HBox, VBox, Output, HTML
import ee
import geemap


class PolygonAnnotationTool:
    """多边形标注工具（支持恢复历史记录）"""

    def __init__(self, feature_collection_id, region_name=None):
        self.fc_id = feature_collection_id
        self.fc = ee.FeatureCollection(feature_collection_id)

        # 如果没有提供region_name，尝试从fc_id中提取
        if region_name:
            self.region_name = region_name.lower().replace(' ', '_')
        else:
            # 尝试从fc_id中提取区域名（假设格式类似 'projects/xxx/iceland_features'）
            self.region_name = self._extract_region_name(feature_collection_id)

        self.annotations = []
        self.current_polygons = []
        self.current_layers = []  # 存储当前绘制的图层引用
        self.saved_polygons_count = 0  # 已保存的多边形总数
        self.saved_features_covered = 0  # 已保存多边形覆盖的特征总数
        self.map = None
        self.polygon_counter = 0  # 用于生成唯一的图层名称
        self.saved_polygon_counter = 0  # 用于生成保存的多边形图层名称
        self._setup_ui()

    def _extract_region_name(self, fc_id):
        """尝试从feature collection ID中提取区域名"""
        # 简单的提取逻辑，可以根据实际ID格式调整
        parts = fc_id.split('/')
        if parts:
            name = parts[-1].lower()
            # 移除常见后缀
            for suffix in ['_features', '_collection', '_fc', '_points']:
                if name.endswith(suffix):
                    name = name[:-len(suffix)]
            return name
        return 'unknown_region'

    def _setup_ui(self):
        """设置用户界面"""
        # 创建地图
        self.map = geemap.Map()
        self.map.centerObject(self.fc, zoom=7)

        # 添加要标注的点
        self.map.addLayer(
            self.fc,
            {'color': 'red', 'pointSize': 5},
            'Features to Cover'
        )

        # 添加绘图控制
        self._add_drawing_controls()

        # 创建控制按钮（添加tooltip说明）
        self.save_btn = Button(
            description='保存当前多边形',
            tooltip='将当前绘制的所有多边形保存为一个批次，保存后无法撤销',
            button_style='primary'
        )
        self.clear_btn = Button(
            description='清除所有多边形',
            tooltip='清除所有未保存的多边形，已保存的不受影响',
            button_style='warning'
        )
        self.export_btn = Button(
            description='导出标注',
            tooltip='将所有已保存的标注批次导出为JSON文件',
            button_style='success'
        )
        self.undo_btn = Button(
            description='撤销上一个',
            tooltip='撤销最后一个绘制的多边形（仅限未保存的）',
            button_style='info'
        )

        # 添加手动导入按钮
        self.manual_import_btn = Button(
            description='手动导入JSON',
            tooltip='显示导入历史记录的代码示例',
            button_style='info'
        )
        self.manual_import_btn.on_click(self._manual_import)

        # 信息显示
        self.info_label = Label(value='开始绘制多边形覆盖红色特征点...')
        self.stats_label = Label(value='')
        self.region_label = Label(value=f'当前区域: {self.region_name}')

        # 绑定事件
        self.save_btn.on_click(self._save_current)
        self.clear_btn.on_click(self._clear_all)
        self.export_btn.on_click(self._export_annotations)
        self.undo_btn.on_click(self._undo_last)

        # 输出区域（用于调试）
        self.output = Output()

        # 创建帮助说明
        help_html = """
        <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin: 10px 0;">
            <h4 style="margin-top: 0;">📖 功能说明：</h4>
            <ul style="list-style-type: none; padding-left: 0;">
                <li><b>• 保存当前多边形：</b>将当前绘制的所有多边形保存为一个批次，保存后变为绿色且无法撤销</li>
                <li><b>• 撤销上一个：</b>撤销最后一个绘制的多边形（仅对蓝色未保存的多边形有效）</li>
                <li><b>• 清除所有多边形：</b>清除所有蓝色未保存的多边形，绿色已保存的不受影响</li>
                <li><b>• 导出标注：</b>将所有已保存的标注批次导出为JSON文件，用于训练或备份</li>
                <li><b>• 手动导入JSON：</b>显示如何导入之前导出的标注历史记录</li>
            </ul>
            <h4>🎨 颜色说明：</h4>
            <p style="margin: 5px 0;">
                <span style="color: red;">●</span> 红色点 = 待覆盖特征 | 
                <span style="color: blue;">■</span> 蓝色多边形 = 未保存 | 
                <span style="color: green;">■</span> 绿色多边形 = 已保存
            </p>
        </div>
        """
        self.help_widget = HTML(value=help_html)

        # 布局
        controls = HBox([
            self.save_btn,
            self.undo_btn,
            self.clear_btn,
            self.export_btn,
            self.manual_import_btn
        ])

        info_box = VBox([self.region_label, self.info_label, self.stats_label])

        # 组合所有元素
        self.ui = VBox([
            self.map,
            controls,
            info_box,
            self.help_widget,
            self.output
        ])

        # 显示初始统计信息
        self._update_stats()

    def _add_drawing_controls(self):
        """添加绘图控制"""
        # 使用geemap的绘图工具
        draw_control = self.map.draw_control

        # 设置多边形绘制选项
        draw_control.polygon = {
            "shapeOptions": {
                "color": "#0000FF",
                "fillColor": "#0000FF",
                "fillOpacity": 0.2,
                "weight": 2
            }
        }

        # 禁用其他绘制工具（设置为空字典）
        draw_control.rectangle = {}
        draw_control.circle = {}
        draw_control.marker = {}
        draw_control.polyline = {}

        # 监听绘制事件
        def handle_draw(target, action, geo_json):
            if action == 'created' and geo_json['geometry']['type'] == 'Polygon':
                self._on_polygon_drawn(geo_json)

        draw_control.on_draw(handle_draw)

    def _on_polygon_drawn(self, geo_json):
        """处理绘制的多边形"""
        if geo_json['geometry']['type'] == 'Polygon':
            coords = geo_json['geometry']['coordinates'][0]

            # 创建多边形记录
            polygon = {
                'coordinates': coords,
                'geometry': geo_json['geometry'],
                'timestamp': datetime.now().isoformat(),
                'covered_features': self._count_covered_features(geo_json['geometry'])
            }

            # 生成唯一的图层名称
            layer_name = f'temp_polygon_{self.polygon_counter}'
            self.polygon_counter += 1

            # 将多边形添加到地图上
            poly_geom = ee.Geometry.Polygon(coords)
            self.map.addLayer(
                poly_geom,
                {'color': 'blue', 'fillOpacity': 0.2},
                layer_name
            )

            # 记录多边形和对应的图层名称
            polygon['layer_name'] = layer_name
            self.current_polygons.append(polygon)
            self.current_layers.append(layer_name)

            self._update_stats()

            # 清除绘制控件中的临时图形
            self.map.draw_control.clear()

    def _count_covered_features(self, geometry):
        """计算多边形覆盖的特征数量"""
        poly_geom = ee.Geometry(geometry)
        covered = self.fc.filterBounds(poly_geom)
        return covered.size().getInfo()

    def _save_current(self, b):
        """保存当前标注"""
        if not self.current_polygons:
            self.info_label.value = '⚠️ 没有绘制任何多边形'
            return

        # 更新已保存的统计信息
        self.saved_polygons_count += len(self.current_polygons)
        self.saved_features_covered += sum(p['covered_features'] for p in self.current_polygons)

        annotation = {
            'feature_collection': self.fc_id,
            'region_name': self.region_name,
            'polygons': self.current_polygons,
            'metadata': {
                'total_features': self.fc.size().getInfo(),
                'total_polygons': len(self.current_polygons),
                'timestamp': datetime.now().isoformat()
            }
        }

        self.annotations.append(annotation)
        self.info_label.value = f'✅ 已保存 {len(self.current_polygons)} 个多边形'

        # 将临时多边形转换为永久保存的多边形
        for i, polygon in enumerate(self.current_polygons):
            # 移除临时图层
            if polygon['layer_name'] in self.map.ee_layer_names:
                self.map.remove_layer(polygon['layer_name'])

            # 添加永久图层
            poly_geom = ee.Geometry.Polygon(polygon['coordinates'])
            layer_name = f'Saved_Polygon_{self.saved_polygon_counter}'
            self.saved_polygon_counter += 1
            self.map.addLayer(
                poly_geom,
                {'color': 'green', 'fillOpacity': 0.1},
                layer_name
            )

        # 清空当前状态
        self.current_polygons = []
        self.current_layers = []
        self._update_stats()

    def _clear_all(self, b):
        """清除所有未保存的多边形"""
        # 从地图上移除所有临时图层
        for layer_name in self.current_layers:
            if layer_name in self.map.ee_layer_names:
                self.map.remove_layer(layer_name)

        # 清空数据
        self.current_polygons = []
        self.current_layers = []

        # 清除绘制控件
        self.map.draw_control.clear()

        self.info_label.value = '🗑️ 已清除所有未保存的多边形'
        self._update_stats()

    def _undo_last(self, b):
        """撤销最后一个多边形"""
        if self.current_polygons:
            # 获取最后一个多边形
            last_polygon = self.current_polygons.pop()
            last_layer = self.current_layers.pop()

            # 从地图上移除对应的图层
            if last_layer in self.map.ee_layer_names:
                self.map.remove_layer(last_layer)

            self.info_label.value = '↩️ 已撤销最后一个多边形'
            self._update_stats()
        else:
            self.info_label.value = '⚠️ 没有可撤销的多边形'

    def _update_stats(self):
        """更新统计信息"""
        current_covered = sum(p['covered_features'] for p in self.current_polygons)
        total_covered = self.saved_features_covered + current_covered
        total_polygons = self.saved_polygons_count + len(self.current_polygons)
        total_features = self.fc.size().getInfo()
        coverage = (total_covered / total_features * 100) if total_features > 0 else 0

        self.stats_label.value = (
            f'总多边形数: {total_polygons} (已保存: {self.saved_polygons_count}, 当前: {len(self.current_polygons)}) | '
            f'覆盖特征数: {total_covered}/{total_features} ({coverage:.1f}%)'
        )

    def _export_annotations(self, b):
        """导出标注数据"""
        with self.output:
            print(f"开始导出... 当前有 {len(self.annotations)} 个标注批次")

        if not self.annotations:
            self.info_label.value = '⚠️ 没有标注数据可导出'
            return

        # 生成训练数据格式
        export_data = {
            'region_name': self.region_name,
            'feature_collection_id': self.fc_id,
            'export_timestamp': datetime.now().isoformat(),
            'total_saved_polygons': self.saved_polygons_count,
            'total_saved_features_covered': self.saved_features_covered,
            'annotations': [],
            'training_samples': []
        }

        for ann in self.annotations:
            # 保存原始标注数据
            export_data['annotations'].append(ann)
            # 为每个标注生成训练样本
            sample = self._create_training_sample(ann)
            export_data['training_samples'].append(sample)

        # 保存为JSON，文件名包含区域名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'polygon_annotations_{self.region_name}_{timestamp}.json'

        # 实际保存文件
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)

        self.info_label.value = f'✅ 已导出到 {filename} (包含 {len(self.annotations)} 个标注批次)'

        with self.output:
            print(f"导出完成: {filename}")

    def _create_training_sample(self, annotation):
        """创建训练样本"""
        # 获取区域边界
        fc = ee.FeatureCollection(annotation['feature_collection'])
        bounds = fc.geometry().bounds()
        region = bounds.getInfo()['coordinates'][0]

        # 创建训练样本
        sample = {
            'input': {
                'region': region,
                'feature_collection_id': annotation['feature_collection'],
                'feature_count': annotation['metadata']['total_features']
            },
            'target': {
                'polygons': [
                    {
                        'coordinates': poly['coordinates'],
                        'type': 'Polygon',
                        'covered_count': poly['covered_features']
                    }
                    for poly in annotation['polygons']
                ],
                'total_polygons': len(annotation['polygons'])
            },
            'metadata': annotation['metadata']
        }

        return sample

    def _manual_import(self, b):
        """手动导入JSON文件"""
        with self.output:
            self.output.clear_output()
            print("📥 手动导入JSON文件的步骤：")
            print("="*50)
            print("1. 首先确保你有之前导出的JSON文件")
            print("2. 在新的代码单元格中运行以下代码：")
            print("")
            print("```python")
            print("# 读取JSON文件")
            print("import json")
            print("with open('your_file.json', 'r') as f:")
            print("    data = json.load(f)")
            print("")
            print("# 导入数据到当前标注工具")
            print("tool.import_from_dict(data)")
            print("```")
            print("")
            print("3. 将 'your_file.json' 替换为实际的文件名")
            print("4. 导入成功后，之前保存的多边形会以绿色显示在地图上")
            print("="*50)

    def import_from_dict(self, import_data):
        """从字典导入数据（公开方法）"""
        try:
            # 验证区域名是否匹配
            if import_data.get('region_name') != self.region_name:
                self.info_label.value = (
                    f"❌ 导入失败：文件区域 '{import_data.get('region_name')}' "
                    f"与当前区域 '{self.region_name}' 不匹配。"
                    f"请确保导入的是同一区域的标注数据。"
                )
                return False

            # 验证feature collection ID是否匹配
            if import_data.get('feature_collection_id') != self.fc_id:
                self.info_label.value = "❌ 导入失败：Feature Collection ID不匹配。请确保导入的是同一数据集的标注。"
                return False

            # 恢复标注数据
            self._restore_annotations(import_data)
            self.info_label.value = f"✅ 成功导入历史记录：恢复了 {len(import_data['annotations'])} 个标注批次"
            return True

        except Exception as e:
            self.info_label.value = f"❌ 导入失败：{str(e)}。请检查文件格式是否正确。"
            return False

    def _restore_annotations(self, import_data):
        """恢复导入的标注数据"""
        with self.output:
            print(f"开始恢复 {len(import_data.get('annotations', []))} 个标注批次")

        # 恢复统计信息
        self.saved_polygons_count = import_data.get('total_saved_polygons', 0)
        self.saved_features_covered = import_data.get('total_saved_features_covered', 0)

        # 恢复标注数据
        self.annotations = import_data.get('annotations', [])

        # 在地图上显示所有已保存的多边形
        polygon_count = 0
        for ann_idx, annotation in enumerate(self.annotations):
            for poly_idx, polygon in enumerate(annotation['polygons']):
                # 添加多边形到地图
                poly_geom = ee.Geometry.Polygon(polygon['coordinates'])
                layer_name = f'Restored_Polygon_{self.saved_polygon_counter}'
                self.saved_polygon_counter += 1
                polygon_count += 1

                self.map.addLayer(
                    poly_geom,
                    {'color': 'green', 'fillOpacity': 0.1},
                    layer_name
                )

        with self.output:
            print(f"已恢复 {polygon_count} 个多边形到地图上")

        # 更新统计显示
        self._update_stats()

    def display(self):
        """显示标注工具"""
        return self.ui