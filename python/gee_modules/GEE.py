import ee

from gee_modules.config import *
from gee_modules.data import FeatColl, Platform
from annotation.utils import Tool

from typing import List, Dict

class GEE:
    def __init__(self, project_id, platform_name):
        self.project_id = project_id
        self.connect()
        self.platform = Platform(platform_name)
    

    def connect(self):
        """
        Connect to Google Earth Engine
        """
        ee.Authenticate()
        ee.Initialize(project=self.project_id)


    def makeGrids(self,
                  rect_bounds,
                  num_rows,
                  num_cols):
        """
        Create a grid of rectangles within the specified bounds.

        num_rows x num_cols:
        1. 140 x 60 when 300m/pixel resolution is required

        Args:
            rect_bounds (ee.geometry): The bounding rectangle for the grid.
            num_rows (int): Number of rows in the grid.
            num_cols (int): Number of columns in the grid.
        """

        if rect_bounds is None:
            raise ValueError("rect_bounds must be provided")


        coords = ee_list(rect_bounds.coordinates().get(0)) # Gain coordinates
        get_val = lambda coord_idx, point_idx: ee_number(ee_list(coords.get(coord_idx)).get(point_idx))
        
        num_rows = ee_number(num_rows)
        num_cols = ee_number(num_cols)

        x_min = get_val(0, 0)
        y_min = get_val(0, 1)
        x_max = get_val(2, 0)
        y_max = get_val(2, 1)

        width = (x_max.subtract(x_min)).divide(num_cols)
        height = (y_max.subtract(y_min)).divide(num_rows)

        # Create a list of all rectangles in the grid
        col_seq = ee_list.sequence(0, num_cols.subtract(1))
        row_seq = ee_list.sequence(0, num_rows.subtract(1))
        
        def create_row(row_idx):
            """ Create a row of rectangles."""
            def create_col(col_idx):
                x1 = x_min.add(width.multiply(col_idx))
                y1 = y_min.add(height.multiply(row_idx))
                x2 = x1.add(width)
                y2 = y1.add(height)

                rect = ee_geometry.Rectangle([x1, y1, x2, y2])

                return ee_feature(rect, {
                    'row': row_idx,
                    'col': col_idx
                })
            
            return ee_list(col_seq.map(create_col))
        
        
        all_cells = ee_list(row_seq.map(create_row))
        grid_fc = ee_featColl(ee_list(all_cells).flatten())


        return grid_fc
    

    def calScale(self,
                coords,
                output_size=(256, 256),
                debug=False):
        """
        Calculate the scale for the given polygon coordinates and output size.
        """

        lat_log = ee_list(coords.coordinates().get(0))
        get_val = lambda coord_idx, point_idx: ee_number(ee_list(lat_log.get(coord_idx)).get(point_idx))

        x_min = get_val(0, 0)
        y_min = get_val(0, 1)
        x_max = get_val(2, 0)
        y_max = get_val(2, 1)

        # Calculate width and height in pixels
        width_pixels = ee_number(output_size[0])
        height_pixels = ee_number(output_size[1])

        # Calculate scale
        scale_x = x_max.subtract(x_min).divide(width_pixels)
        scale_y = y_max.subtract(y_min).divide(height_pixels)
        scale = ee_number.min(scale_x, scale_y)

        if debug: print(f"Calculated scale: {scale.getInfo()}")

        return scale


    def exportPic(self,
              coords,
              output_size=(256, 256),
              band_type='rgb',
              start_date='2022-01-01',
              end_date='2023-12-31'):
    
        """
        Export a picture from Google Earth Engine.
        
        Returns:
            An image
        """

        # Make sure coords is Geometry
        if isinstance(coords, ee_feature):
            coords = coords.geometry()
        elif isinstance(coords, ee_featColl):
            coords = coords.geometry()
        
        # Filter image collections
        plat_coll = self.platform.get_collection()
        sentinel_coll = ee_imgColl(plat_coll) \
                        .filterBounds(coords) \
                        .filterDate(start_date, end_date) \
    
        # Cloud filtering
        filters = self.platform.get_filter()
        if filters: sentinel_coll = sentinel_coll.filter(ee_filter.lt(filters['cloudCover'], filters['maxCloud']))

        # Cloud masking
        bands = self.platform.get_bands(band_type) # rbg by default

        if self.platform.name == 'sentinel2':
            def maskClouds(image):
                qa = image.select('QA60')
                mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
                return image.updateMask(mask).divide(10000).select(bands)
            sentinel_coll = sentinel_coll.map(maskClouds)

        # Create Composite
        composite = sentinel_coll.median()

        # Clip to region
        composite = composite.clip(coords)

        # Prepare visualization params
        vis_params = self.platform.get_vis_params()
        if self.platform.name == 'sentinel2':
            vis_params = {
                'bands': bands,
                'min': vis_params.get('min'),
                'max': vis_params.get('max'),
                'gamma': vis_params.get('gamma')
            }

        return {
            'image': composite,
            'vis_params': vis_params,
            'bounds': coords,
            'platform': self.platform.name
        }
    

    def importData(self, dir_path):
        """
        Import JSON data from given directory path, and give back
            1. json data of type ee.FeatureCollection
            2. bounds of each feature

        Return:
            Dict[
                ee.FeatureCollection,
                ee.Geometry.polygon
            ]

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
    


# test
if __name__ == "__main__":
    project_id = 'planar-compass-462105-e3'
    platform_name = 'sentinel2'
    gee = GEE(project_id, platform_name)
    
    iceland = FeatColl('users/liuhsuu6/iceland')

    img = gee.exportPic(
        iceland.rectBounds,
        output_size=(1280, 1280/3.99),
    )

    print(type(img))