from typing import Optional

from gee_modules.config import *



class FeatColl:
    def __init__(
            self,
            path: Optional[str] = None,
            data: Optional[ee_featureCollection] = None
        ):
        self.path = path
        self.data = data if data is not None else self.load()
        self.getInfo()


    def validPath(self):
        """
        Make sure path is valid
        1. must be string
        2. must start with users/ or projects/
        """

        if not isinstance(self.path, str):
            raise ValueError("Path must be a string")
        
        if not (self.path.startswith("users/") or self.path.startswith("projects/")):
            raise ValueError("Path must start with 'users/' or 'projects/'")
        
        return True


    def load(self) -> ee_featureCollection:
        """
        If path is valid, then load.
        """

        if self.validPath():
            try:
                fc = ee_featureCollection(self.path)
                return fc
            except Exception as e:
                raise ValueError(f"Failed to load FeatureCollection from path {self.path}") from e
        else:
            raise ValueError("Invalid path for FeatureCollection")
            
            
    def getInfo(self):
        """
        Get basic info of this collection
        + bounds
        + number of features
        """

        self.rectBounds = self.data.geometry().bounds()
        self.num_features = self.data.size()
            
    
    """
    -------------------------------------------------------------
    Note: Remote data transmission is required for methods below.

    Do not use these methods in a loop or repeatedly.
    -------------------------------------------------------------
    """
    def __get__(self, n:int):
        """
        Get n-th feature from this collection
        """
        # Check if num_features is defined
        if not hasattr(self, 'num_features'):
            raise ValueError("FeatureCollection size is not defined. Call getInfo() first.")
        
        # Ensure num_features is an integer
        if not isinstance(self.num_features, int):
            if self.num_features is None:
                raise ValueError("FeatureCollection size is not defined")
            else:
                info = self.num_features.getInfo()
                if info is None:
                    raise ValueError("FeatureCollection size could not be retrieved (got None)")
                self.num_features = int(info)

        # Check if n is within the valid range
        if n < 0 or n >= self.num_features:
            raise IndexError("Index out of range")

        # Get the n-th feature
        return self.data.toList(1, n).get(0)
    


class Platform:
    """
    Manage platform specifications for Earth Engine data.
    """

    def __init__(self, name: str):
        self.name = name

        if name.lower() not in PLATFORM_SPECS:
            print("Available platforms are:", \
                  '\n\t+'.join(PLATFORM_SPECS.keys()))
            raise ValueError(f"Platform '{name}' is not supported.")
        else:
            self.specs = PLATFORM_SPECS[name]


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


    def get_bands(self, band_type: str):
        """
        Get the list of bands for a specific band type.
        """

        bands = self.specs['bands'].get(band_type, [])

        if not bands:
            raise ValueError(f"Band type '{band_type}' is not defined in platform '{self.name}'.")
        
        return bands


    def get_resolution(self, band_type: str):
        """
        Get the resolution for a specific band type.
        """

        resolution = self.specs['resolution'].get(band_type, None)

        if resolution is None:
            raise ValueError(f"Resolution for band type '{band_type}' is not defined in platform '{self.name}'.")

        return ee_number(resolution)
    

    def get_collection(self):
        """
        Get the collection of the platform
        """

        collection = self.specs.get('collection', None)

        if collection is None:
            raise ValueError(f"Collection for platform '{self.name}' is not defined.")

        return collection
    

    def get_filter(self):
        """
        Get the filter for the platform
        """

        filters = self.specs.get('filters', None)

        if filters is None:
            raise ValueError(f"Filter for platform '{self.name}' is not defined.")

        return filters


    def get_vis_params(self):
        """
        Get the visualization parameters for the platform
        """

        vis_params = self.specs.get('vis_params', None)

        if vis_params is None:
            raise ValueError(f"Visualization parameters for platform '{self.name}' are not defined.")

        vis_params['min'] = vis_params.get('min', 0) / 10000
        vis_params['max'] = vis_params.get('max', 3000) / 10000
        vis_params['gamma'] = vis_params.get('gamma', 1.2)

        return vis_params