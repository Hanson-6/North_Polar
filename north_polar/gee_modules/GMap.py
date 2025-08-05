"""
GMap - Interactive Map with AI Tools
====================================

A modular, extensible map interface built on geemap with AI-powered analysis tools.
Follows SOLID principles and clean architecture patterns.

Author: Liu Haisu
Version: 2.0.0
License: MIT
"""

import geemap
import ipywidgets as widgets
from IPython.display import display
import logging
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import ee
import os
import numpy as np
from PIL import Image
import io
import tempfile
from datetime import datetime
import ipyleaflet
from pathlib import Path
import json

from model.DeepLabV3Plus.code.model_interface import DeepLabV3PlusInterface

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger('GMap')


# ============================================================================
# Constants and Configuration
# ============================================================================

class MapConfig:
    """Centralized configuration for the map application."""
    
    # Zoom constraints
    MIN_ZOOM_FOR_TOOLS = 7
    DEFAULT_ZOOM = 5
    
    # UI Dimensions
    BUTTON_WIDTH = "135px"
    BUTTON_HEIGHT = "40px"
    BUTTON_BORDER_RADIUS = "20px"
    BUTTON_PADDING = "8px 18px"
    BUTTON_GAP = "12px"
    
    # Colors (Material Design palette)
    COLORS = {
        'primary': '#2196F3',
        'success': '#4CAF50',
        'info': '#17a2b8',
        'warning': '#ffc107',
        'danger': '#dc3545',
        'light': '#f8f9fa',
        'dark': '#343a40',
        'disabled': '#cccccc'
    }
    
    # Messages
    MESSAGES = {
        'tools_locked': '🔒 工具已锁定',
        'tools_available': '✓ 工具可用',
        'zoom_required': '需要缩放到{}级或以上才能使用',
        'click_to_use': '点击使用此功能',
        'draw_rectangle': '请在地图上绘制矩形区域',
        'processing': '正在处理，请稍候...',
        'complete': '处理完成！',
        'error': '处理失败：{}'
    }
    
    # Paths
    MASK_SAVE_PATH = "model/results/area_ai_masks"


class ButtonStyle(Enum):
    """Enumeration of available button styles."""
    PRIMARY = 'primary'
    SUCCESS = 'success'
    INFO = 'info'
    WARNING = 'warning'
    DANGER = 'danger'


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class ToolButton:
    """Data model for a tool button."""
    id: str
    text: str
    icon: str
    style: ButtonStyle
    tooltip: str
    handler: Optional[Callable] = None
    min_zoom: int = MapConfig.MIN_ZOOM_FOR_TOOLS


# ============================================================================
# Abstract Base Classes
# ============================================================================

class MapComponent(ABC):
    """Abstract base class for map components."""
    
    @abstractmethod
    def create(self) -> widgets.Widget:
        """Create and return the widget component."""
        pass
    
    @abstractmethod
    def update(self, **kwargs) -> None:
        """Update the component state."""
        pass


class MapTool(ABC):
    """Abstract base class for map tools."""
    
    @abstractmethod
    def activate(self) -> None:
        """Activate the tool."""
        pass
    
    @abstractmethod
    def deactivate(self) -> None:
        """Deactivate the tool."""
        pass


# ============================================================================
# UI Components
# ============================================================================

class ZoomIndicator(MapComponent):
    """Component for displaying zoom level and tool availability status."""
    
    def __init__(self):
        self.widget = widgets.HTML()
        self._current_zoom = MapConfig.DEFAULT_ZOOM
        
    def create(self) -> widgets.HTML:
        """Create the zoom indicator widget."""
        self.update(zoom_level=self._current_zoom)
        return self.widget
        
    def update(self, zoom_level: int) -> None:
        """Update the zoom indicator display."""
        self._current_zoom = zoom_level
        status = (MapConfig.MESSAGES['tools_available'] 
                 if zoom_level >= MapConfig.MIN_ZOOM_FOR_TOOLS 
                 else MapConfig.MESSAGES['tools_locked'])
        
        self.widget.value = f'''
        <div style="
            padding: 12px 16px;
            margin: 10px 0;
            border-left: 4px solid {MapConfig.COLORS['info']};
            background-color: {MapConfig.COLORS['light']};
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        ">
            <div>
                <strong>缩放级别：</strong> {zoom_level}
            </div>
            <div style="font-size: 12px; color: #666;">
                {status}
            </div>
        </div>
        '''


class ToolButtonWidget(MapComponent):
    """Component for tool buttons with consistent styling."""
    
    def __init__(self, button_config: ToolButton):
        self.config = button_config
        self.widget = None
        
    def create(self) -> widgets.Button:
        """Create a styled button widget."""
        self.widget = widgets.Button(
            description=self.config.text,
            button_style=self.config.style.value,
            icon=f'fa-{self.config.icon}',
            layout=widgets.Layout(
                width=MapConfig.BUTTON_WIDTH,
                height=MapConfig.BUTTON_HEIGHT,
                margin='0px',
                border_radius=MapConfig.BUTTON_BORDER_RADIUS,
                padding=MapConfig.BUTTON_PADDING,
                font_weight='500'
            ),
            disabled=True,
            tooltip=MapConfig.MESSAGES['zoom_required'].format(self.config.min_zoom)
        )
        
        if self.config.handler:
            self.widget.on_click(self.config.handler)
            
        return self.widget
        
    def update(self, enabled: bool = False) -> None:
        """Update button state."""
        if self.widget:
            self.widget.disabled = not enabled
            self.widget.tooltip = (MapConfig.MESSAGES['click_to_use'] if enabled 
                                 else MapConfig.MESSAGES['zoom_required'].format(self.config.min_zoom))


class ButtonContainer(MapComponent):
    """Container for managing multiple tool buttons."""
    
    def __init__(self):
        self.buttons: Dict[str, ToolButtonWidget] = {}
        self.widget = None
        
    def add_button(self, button: ToolButtonWidget) -> None:
        """Add a button to the container."""
        self.buttons[button.config.id] = button
        self._refresh_container()
        
    def remove_button(self, button_id: str) -> None:
        """Remove a button from the container."""
        if button_id in self.buttons:
            del self.buttons[button_id]
            self._refresh_container()
            
    def create(self) -> widgets.HBox:
        """Create the button container widget."""
        self.widget = widgets.HBox(
            layout=widgets.Layout(
                margin='10px 0',
                justify_content='flex-start',
                gap=MapConfig.BUTTON_GAP,
                padding='5px'
            )
        )
        self._refresh_container()
        return self.widget
        
    def update(self, enabled: bool = False) -> None:
        """Update all buttons' state."""
        for button in self.buttons.values():
            button.update(enabled=enabled)
            
    def _refresh_container(self) -> None:
        """Refresh the container's children."""
        if self.widget:
            self.widget.children = [btn.create() for btn in self.buttons.values()]


class StatusPanel(MapComponent):
    """Component for displaying status messages."""
    
    def __init__(self):
        self.widget = widgets.HTML()
        self.hide()
        
    def create(self) -> widgets.HTML:
        """Create the status panel widget."""
        return self.widget
        
    def update(self, message: str, status_type: str = 'info') -> None:
        """Update the status panel with a message."""
        color = MapConfig.COLORS.get(status_type, MapConfig.COLORS['info'])
        self.widget.value = f'''
        <div style="
            padding: 10px 15px;
            margin: 10px 0;
            background-color: {color}22;
            border: 1px solid {color};
            border-radius: 4px;
            color: #333;
        ">
            {message}
        </div>
        '''
        self.show()
        
    def show(self) -> None:
        """Show the status panel."""
        self.widget.layout.display = 'block'
        
    def hide(self) -> None:
        """Hide the status panel."""
        self.widget.layout.display = 'none'


class ResultPanel(MapComponent):
    """Component for displaying results with save/cancel options."""
    
    def __init__(self, on_save: Callable = None, on_cancel: Callable = None):
        self.on_save = on_save
        self.on_cancel = on_cancel
        self.widget = None
        self.save_button = None
        self.cancel_button = None
        
    def create(self) -> widgets.VBox:
        """Create the result panel widget."""
        self.save_button = widgets.Button(
            description='保存结果',
            button_style='success',
            icon='fa-save'
        )
        
        self.cancel_button = widgets.Button(
            description='取消',
            button_style='danger',
            icon='fa-times'
        )
        
        if self.on_save:
            self.save_button.on_click(lambda _: self.on_save())
        if self.on_cancel:
            self.cancel_button.on_click(lambda _: self.on_cancel())
            
        button_box = widgets.HBox([
            self.save_button,
            self.cancel_button
        ], layout=widgets.Layout(gap='10px'))
        
        self.widget = widgets.VBox([
            widgets.HTML('<h4>处理完成</h4>'),
            button_box
        ], layout=widgets.Layout(
            padding='10px',
            border='1px solid #ddd',
            border_radius='5px',
            margin='10px 0'
        ))
        
        self.hide()
        return self.widget
        
    def update(self, **kwargs) -> None:
        """Update the result panel."""
        pass
        
    def show(self) -> None:
        """Show the result panel."""
        if self.widget:
            self.widget.layout.display = 'block'
            
    def hide(self) -> None:
        """Hide the result panel."""
        if self.widget:
            self.widget.layout.display = 'none'


# ============================================================================
# Map Tools Implementation
# ============================================================================

class AreaAITool(MapTool):
    """AI-powered area analysis tool."""
    
    def __init__(self, map_instance: 'GMap'):
        self.map = map_instance
        self.is_active = False
        self.is_drawing = False
        self.drawn_rectangle = None
        self.result_layer = None
        self.model_interface = None

        # 存储路径（测试）
        self.base_path = Path(__file__).parent.parent  # north_polar目录
        self.img_save_path = self.base_path / 'gee_modules' / 'results' / 'imgs'
        self.json_save_path = self.base_path / 'gee_modules' / 'results' / 'json'

        # 模型路径
        checkpoint_path = Path(__file__).parent.parent / 'model' / 'DeepLabV3Plus' / 'checkpoints'
        checkpoint_name = 'checkpoint_epoch30_20250731_181940.pth'
        self.model_interface = DeepLabV3PlusInterface(
            checkpoint_path= checkpoint_path / checkpoint_name
        )
        logger.info("Model interface loaded successfully")
        # except Exception as e:
            # logger.error(f"Failed to load model interface: {e}")

        self.draw_control = None  # 添加
        self.rectangle_layer = None  # 添加
        
    def activate(self) -> None:
        """Activate the area analysis tool."""
        if self.is_drawing:
            logger.warning("Already drawing a rectangle")
            return
            
        self.is_active = True
        self.is_drawing = True
        
        # Disable the Area AI button while drawing
        area_button = self.map.button_container.buttons.get('area_ai')
        if area_button:
            area_button.widget.disabled = True
            
        # Show drawing instruction
        self.map.status_panel.update(MapConfig.MESSAGES['draw_rectangle'], 'info')
        
        # Activate rectangle drawing
        self._start_rectangle_drawing()
        
        logger.info("Area AI tool activated")
        
    def deactivate(self) -> None:
        """Deactivate the area analysis tool."""
        self.is_active = False
        self.is_drawing = False
        
        # Re-enable the Area AI button
        area_button = self.map.button_container.buttons.get('area_ai')
        if area_button and self.map.gmap.zoom >= MapConfig.MIN_ZOOM_FOR_TOOLS:
            area_button.widget.disabled = False
            
        # Hide status panel
        self.map.status_panel.hide()
        
        logger.info("Area AI tool deactivated")
        
    def _start_rectangle_drawing(self) -> None:
        """Start the rectangle drawing mode."""
        # try:
        # Create draw control for rectangle
        # Create draw control for rectangle
        self.draw_control = ipyleaflet.DrawControl(  # 保存到self
            marker={},
            rectangle={'shapeOptions': {'color': '#ff0000'}},
            circle={},
            circlemarker={},
            polyline={},
            polygon={}
        )
        
        # Add event handler
        self.draw_control.on_draw(self._on_rectangle_drawn)
        
        # Add control to map
        self.map.gmap.add(self.draw_control)
            
        # except Exception as e:
        #     logger.error(f"Failed to start rectangle drawing: {e}")
        #     self.map.status_panel.update(f"绘制功能启动失败: {str(e)}", 'danger')
        #     self.deactivate()
            
    def _on_rectangle_drawn(self, feature, action, geo_json) -> None:
        """Handle rectangle drawing completion."""
        if action == 'created' and geo_json['geometry']['type'] == 'Polygon':
            # Get rectangle coordinates
            coords = geo_json['geometry']['coordinates'][0]
            
            if len(coords) == 5:
                self.drawn_rectangle = coords
                self.is_drawing = False
                
                # 移除绘图控件
                if self.draw_control:
                    self.map.gmap.remove(self.draw_control)
                
                # 创建矩形图层
                rectangle = ipyleaflet.Polygon(
                    locations=[[coord[1], coord[0]] for coord in coords],
                    color="red",
                    fill_color="red", 
                    fill_opacity=0.1,
                    weight=2,
                    name="Area Selection"  # 添加这一行，给图层命名
                )
                
                # 添加到地图并保存引用
                self.map.gmap.add_layer(rectangle)
                self.rectangle_layer = rectangle
                
                # Process the rectangle
                self._process_rectangle(coords)
                    
            # except Exception as e:
            #     logger.error(f"Error processing drawn rectangle: {e}")
            #     self.map.status_panel.update(f"处理矩形时出错: {str(e)}", 'danger')
                
    def _process_rectangle(self, coords: List) -> None:
        """Process the drawn rectangle through the AI model."""
        # try:
        self.map.status_panel.update(MapConfig.MESSAGES['processing'], 'info')
        
        # Convert coordinates to ee.Geometry
        ee_coords = [[coord[0], coord[1]] for coord in coords]
        ee_geometry = ee.Geometry.Polygon([ee_coords])
        
        # Export image from GEE
        logger.info("Exporting image from GEE...")
        image_data = self.map.gee_instance.exportSingleAreaForModel(
            ee_geometry,
            start_date='2022-01-01',
            end_date='2023-12-31'
        )
        
        if not image_data:
            raise ValueError("Failed to export image from GEE")
        
        # 添加：保存输入图像和坐标
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存RGB图像
        img_filename = f"input_area_{timestamp}.png"
        img_filepath = self.img_save_path / img_filename
        image_data['image'].save(img_filepath)
        logger.info(f"Saved input image to: {img_filepath}")
        
        # 保存坐标信息
        json_filename = f"input_area_{timestamp}.json"
        json_filepath = self.json_save_path / json_filename
        coord_data = {
            'timestamp': timestamp,
            'rectangle_coords': coords,
            'ee_geometry': image_data['geometry'],
            'bounds': image_data['bounds'],
            'image_file': img_filename
        }
        with open(json_filepath, 'w') as f:
            json.dump(coord_data, f, indent=2)
        logger.info(f"Saved coordinates to: {json_filepath}")
            
        # Run model inference
        logger.info("Running model inference...")
        if self.model_interface:
            mask = self.model_interface.predict_area(image_data)
            
            # Display result on map
            self._display_result(mask, ee_geometry)
        else:
            raise ValueError("Model interface not available")
                
        # except Exception as e:
        #     logger.error(f"Error in processing rectangle: {e}")
        #     self.map.status_panel.update(MapConfig.MESSAGES['error'].format(str(e)), 'danger')
        #     self.deactivate()
            
    def _display_result(self, mask: np.ndarray, geometry: ee.Geometry) -> None:
        """Display the mask result on the map."""
        # try:
            # Convert mask to image layer
            # This is a simplified version - you might need to adjust based on your needs

        # 隐藏选择矩形
        if self.rectangle_layer:
            self.map.gmap.remove_layer(self.rectangle_layer)
            self.rectangle_layer = None
                
        # Convert mask to image layer
        # Create a temporary file for the mask
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        mask_image.save(temp_file.name)
        
        # Get geographic bounds from EE geometry
        # EE coordinates are [lon, lat], ImageOverlay expects ((min_lat, min_lon), (max_lat, max_lon))
        bounds_info = geometry.bounds().getInfo()['coordinates'][0]
        min_lon, min_lat = bounds_info[0][0], bounds_info[0][1]
        max_lon, max_lat = bounds_info[2][0], bounds_info[2][1]
        overlay_bounds = ((min_lat, min_lon), (max_lat, max_lon))
        
        # Create ImageOverlay
        overlay = geemap.ImageOverlay(
            url=temp_file.name,  # Local file path (works in Jupyter)
            bounds=overlay_bounds,
            name='AI Mask Overlay',
            opacity=0.5  # Adjustable transparency
        )
        
        # Add to map
        self.map.gmap.add_layer(overlay)
        self.result_layer = overlay  # Store for later removal
        
        # Update status and show panel
        self.map.status_panel.update(MapConfig.MESSAGES['complete'], 'success')
        self.map.result_panel.show()
        
        # Store result for saving
        self.current_mask = mask
        self.current_geometry = geometry
            
        # except Exception as e:
        #     logger.error(f"Error displaying result: {e}")
        #     self.map.status_panel.update(f"显示结果时出错: {str(e)}", 'danger')
            
    def save_result(self) -> None:
        """Save the current result."""
        # try:
        # Create save directory if not exists
        os.makedirs(MapConfig.MASK_SAVE_PATH, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"area_ai_mask_{timestamp}.npy"
        filepath = os.path.join(MapConfig.MASK_SAVE_PATH, filename)
        
        # Save mask
        np.save(filepath, self.current_mask)
        
        # Save geometry info
        geo_filename = f"area_ai_geometry_{timestamp}.json"
        geo_filepath = os.path.join(MapConfig.MASK_SAVE_PATH, geo_filename)
        
        import json
        with open(geo_filepath, 'w') as f:
            json.dump({
                'geometry': self.current_geometry.getInfo(),
                'mask_file': filename
            }, f)
            
        logger.info(f"Result saved to {filepath}")
        self.map.status_panel.update(f"结果已保存到: {filename}", 'success')
            
        # except Exception as e:
        #     logger.error(f"Error saving result: {e}")
        #     self.map.status_panel.update(f"保存失败: {str(e)}", 'danger')
            
    def cancel_result(self) -> None:
        """Cancel the current result."""
        # Clear current result
        self.current_mask = None
        self.current_geometry = None
        
        # Remove result layer from map if exists
        if self.result_layer:
            self.map.gmap.remove_layer(self.result_layer)
            self.result_layer = None

        # 移除矩形图层
        if self.rectangle_layer:
            self.map.gmap.remove_layer(self.rectangle_layer)
            self.rectangle_layer = None
            
        self.map.status_panel.update("已取消", 'info')


class BuildingAITool(MapTool):
    """AI-powered building identification tool."""
    
    def __init__(self, map_instance: 'GMap'):
        self.map = map_instance
        self.is_active = False
        
    def activate(self) -> None:
        """Activate the building identification tool."""
        self.is_active = True
        logger.info("Building AI tool activated")
        # TODO: Implement building detection functionality
        
    def deactivate(self) -> None:
        """Deactivate the building identification tool."""
        self.is_active = False
        logger.info("Building AI tool deactivated")


# ============================================================================
# Tool Factory
# ============================================================================

class ToolFactory:
    """Factory for creating map tools and their associated buttons."""
    
    @staticmethod
    def create_default_tools() -> List[ToolButton]:
        """Create the default set of tool buttons."""
        return [
            ToolButton(
                id='area_ai',
                text='Area AI',
                icon='square-o',
                style=ButtonStyle.PRIMARY,
                tooltip='区域AI分析工具'
            ),
            ToolButton(
                id='building_ai',
                text='Building AI',
                icon='building-o',
                style=ButtonStyle.SUCCESS,
                tooltip='建筑物AI识别工具'
            )
        ]
    
    @staticmethod
    def create_tool_instance(tool_id: str, map_instance: 'GMap') -> Optional[MapTool]:
        """Create a tool instance based on tool ID."""
        tool_mapping = {
            'area_ai': AreaAITool,
            'building_ai': BuildingAITool
        }
        
        tool_class = tool_mapping.get(tool_id)
        return tool_class(map_instance) if tool_class else None


# ============================================================================
# Event System
# ============================================================================

class EventType(Enum):
    """Types of events in the map system."""
    ZOOM_CHANGED = 'zoom_changed'
    TOOL_ACTIVATED = 'tool_activated'
    TOOL_DEACTIVATED = 'tool_deactivated'
    MAP_CLICKED = 'map_clicked'


class EventManager:
    """Manages events and their handlers."""
    
    def __init__(self):
        self._handlers: Dict[EventType, List[Callable]] = {
            event_type: [] for event_type in EventType
        }
        
    def subscribe(self, event_type: EventType, handler: Callable) -> None:
        """Subscribe a handler to an event type."""
        self._handlers[event_type].append(handler)
        
    def unsubscribe(self, event_type: EventType, handler: Callable) -> None:
        """Unsubscribe a handler from an event type."""
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)
            
    def emit(self, event_type: EventType, **kwargs) -> None:
        """Emit an event to all subscribed handlers."""
        for handler in self._handlers[event_type]:
            # try:
            handler(**kwargs)
            # except Exception as e:
            #     logger.error(f"Error in event handler for {event_type}: {e}")


# ============================================================================
# Main Map Class
# ============================================================================

class GMap:
    """
    Main map class that orchestrates all components and tools.
    
    This class follows the Facade pattern to provide a simple interface
    to the complex map system.
    """
    
    def __init__(self, gee_instance=None):
        """Initialize the GMap instance."""
        logger.debug("Initializing GMap")
        
        # Store GEE instance
        self.gee_instance = gee_instance
        
        # Core components
        self.gmap = geemap.Map()
        self.event_manager = EventManager()
        
        # UI components
        self.zoom_indicator = ZoomIndicator()
        self.button_container = ButtonContainer()
        self.status_panel = StatusPanel()
        self.result_panel = None
        
        # Tools
        self.tools: Dict[str, MapTool] = {}
        self.active_tool: Optional[MapTool] = None
        
        # State
        self._initialized = False
        
        # Initialize
        self._setup_tools()
        self._setup_event_handlers()
        self._setup_result_panel()
        
        logger.debug("GMap initialization complete")
        
    def _setup_tools(self) -> None:
        """Set up the default tools and their buttons."""
        tool_configs = ToolFactory.create_default_tools()
        
        for config in tool_configs:
            # Create tool instance
            tool = ToolFactory.create_tool_instance(config.id, self)
            if tool:
                self.tools[config.id] = tool
                
                # Set up button handler
                config.handler = lambda _, tool_id=config.id: self._on_tool_click(tool_id)
                
                # Create and add button
                button_widget = ToolButtonWidget(config)
                self.button_container.add_button(button_widget)
                
        logger.debug(f"Set up {len(self.tools)} tools")
        
    def _setup_event_handlers(self) -> None:
        """Set up event handlers for map interactions."""
        # Subscribe to internal events
        self.event_manager.subscribe(EventType.ZOOM_CHANGED, self._on_zoom_changed)
        
        # Set up geemap observers
        self.gmap.observe(self._handle_zoom_change, names=['zoom'])
        
        # Try to set up click handler if supported
        if hasattr(self.gmap, 'on_interaction'):
            self.gmap.on_interaction(self._handle_map_interaction)
            
    def _setup_result_panel(self) -> None:
        """Set up the result panel with save/cancel handlers."""
        def on_save():
            if self.active_tool and hasattr(self.active_tool, 'save_result'):
                self.active_tool.save_result()
            self.result_panel.hide()
            
        def on_cancel():
            if self.active_tool and hasattr(self.active_tool, 'cancel_result'):
                self.active_tool.cancel_result()
            self.result_panel.hide()
            
        self.result_panel = ResultPanel(on_save=on_save, on_cancel=on_cancel)
            
    def _handle_zoom_change(self, change: Dict[str, Any]) -> None:
        """Handle zoom change from geemap."""
        if change['name'] == 'zoom':
            self.event_manager.emit(EventType.ZOOM_CHANGED, zoom_level=change['new'])
            
    def _handle_map_interaction(self, **kwargs) -> None:
        """Handle map interactions."""
        if kwargs.get('type') == 'click':
            self.event_manager.emit(
                EventType.MAP_CLICKED, 
                coordinates=kwargs.get('coordinates', [])
            )
            
    def _on_zoom_changed(self, zoom_level: int) -> None:
        """Handle zoom level changes."""
        # Update zoom indicator
        self.zoom_indicator.update(zoom_level=zoom_level)
        
        # Update button states
        tools_enabled = zoom_level >= MapConfig.MIN_ZOOM_FOR_TOOLS
        self.button_container.update(enabled=tools_enabled)
        
        # Deactivate active tool if zoom is too low
        if not tools_enabled and self.active_tool:
            self.active_tool.deactivate()
            self.active_tool = None
            
    def _on_tool_click(self, tool_id: str) -> None:
        """Handle tool button clicks."""
        tool = self.tools.get(tool_id)
        if not tool:
            return
            
        # Don't allow activation if tool is already drawing
        if hasattr(tool, 'is_drawing') and tool.is_drawing:
            logger.warning(f"Tool {tool_id} is already in use")
            return
            
        # Deactivate current tool if different
        if self.active_tool and self.active_tool != tool:
            self.active_tool.deactivate()
            self.event_manager.emit(
                EventType.TOOL_DEACTIVATED, 
                tool_id=tool_id
            )
            
        # Toggle tool activation
        if self.active_tool == tool:
            tool.deactivate()
            self.active_tool = None
            self.event_manager.emit(EventType.TOOL_DEACTIVATED, tool_id=tool_id)
        else:
            tool.activate()
            self.active_tool = tool
            self.event_manager.emit(EventType.TOOL_ACTIVATED, tool_id=tool_id)
            
    def add_custom_tool(self, tool_config: ToolButton, tool_instance: MapTool) -> None:
        """Add a custom tool to the map."""
        self.tools[tool_config.id] = tool_instance
        
        # Set up button handler
        tool_config.handler = lambda _, tool_id=tool_config.id: self._on_tool_click(tool_id)
        
        # Create and add button
        button_widget = ToolButtonWidget(tool_config)
        self.button_container.add_button(button_widget)
        
        # Update button state based on current zoom
        current_zoom = self.gmap.zoom
        button_widget.update(enabled=current_zoom >= tool_config.min_zoom)
        
        logger.info(f"Added custom tool: {tool_config.id}")
        
    def display(self) -> None:
        """Display the map and all UI components."""
        if not self._initialized:
            # try:
            # Add native geemap tools
            self._add_native_tools()
            
            # Initialize button states
            self._on_zoom_changed(self.gmap.zoom)
            
            self._initialized = True
            logger.debug("Map display initialized")
                
            # except Exception as e:
            #     logger.error(f"Error initializing map display: {e}")
            #     raise
                
        # Display map
        display(self.gmap)
        
        # Display UI panel
        ui_panel = widgets.VBox([
            self.zoom_indicator.create(),
            self.button_container.create(),
            self.status_panel.create(),
            self.result_panel.create()
        ])
        display(ui_panel)
        
    def _add_native_tools(self) -> None:
        """Add native geemap tools to the map."""
        # try:
        self.gmap.add_search_control(position='topright')
        self.gmap.add_layer_control()
        logger.debug("Native tools added successfully")
        # except Exception as e:
        #     logger.warning(f"Could not add some native tools: {e}")
            
    def start(self) -> None:
        """Start the map application."""
        logger.info("Starting GMap application")
        self.display()