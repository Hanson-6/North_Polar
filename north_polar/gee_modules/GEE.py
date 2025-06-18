import ee
import json
import os

from config import *



class GEE:
    def __init__(self):
        self.project_id = Config.project_id
        self.connect()
        
    
    def connect(self):
        """
        Connect to Google Earth Engine
        """
        try:
            ee.Authenticate()
            ee.Initialize(project=self.project_id)
        except:
            pass


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
    

    def readJSON(self, json_path):
        """
        Read a JSON file from the specified path.

        Args:
            json_path (str): Path to the JSON file.
        
        Returns:
            (ee_featColl): Feature collection loaded from the JSON file.
        """

        # Validate the JSON file path
        if isinstance(json_path, str) is False or \
            json_path.endswith('.json') is False or \
            not os.path.exists(json_path):

            raise ValueError("Please provide a valid JSON file path.")

        # Extract coordinates
        coords = []

        with open(json_path, 'r') as file:
            data = json.load(file)

            # import annotations
            if 'annotations' in data:
                coords = data['annotations'][0]['polygons'][0]['coordinates']
            else:
                print("No annotations found in JSON file.")

        return ee_featColl(ee_list(coords))
    

    def exportPic(self,
                  coords,
                  output_size=(256, 256),
                  platform='sentinel2',
                  band_type='rgb'
                  ):
        
        """
        Export a picture from Google Earth Engine.

        Args:
            coords (GEE Type): Coordinates of the area to export.
            output_size (tuple): Size of the output image (width, height).
            platform (str): Platform to use for the export (e.g., 'sentinel2').
            band_type (str): Type of bands to export (e.g., 'rgb', 'nir').
        """

        # Get platform configuration
        platform_config = PLATFORM_SPECS.get(platform, {})
        if not platform_config:
            raise ValueError(f"Platform '{platform}' is not supported.")
        
        # Get resolution of the specified band type
        platform_resolution = platform_config['resolution'].get(band_type, None)
        if platform_resolution is None:
            raise ValueError(f"Band type '{band_type}' is not supported for platform '{platform}'.")
        
        # Calculate best scale
        coords =  ee_list(coords.coordinates().get(0))

        sw = ee_list(coords.get(0))
        ne = ee_list(coords.get(2))

        POINT_W = ee_geometry.Point([sw.get(0), sw.get(1)])
        POINT_E = ee_geometry.Point([ne.get(0), sw.get(1)])
        POINT_S = ee_geometry.Point([sw.get(0), sw.get(1)])
        POINT_N = ee_geometry.Point([sw.get(0), ne.get(1)])

        width_m = POINT_E.distance(POINT_W)  # Convert to kilometers
        height_m = POINT_N.distance(POINT_S)  # Convert to kilometers

        required_scale_w = width_m.divide(output_size[0])
        required_scale_h = height_m.divide(output_size[1])
        required_scale = ee_number(ee_number.min(required_scale_w, required_scale_h))

        optimal_scale = ee_number.max(platform_resolution, required_scale)

        print(f"Optimal scale for export: {optimal_scale.getInfo()} meters/pixel") # Debug

        # # Create the image collection
        # try:
        #     coords = ee_geometry.Polygon(coords)

        #     collection = ee_imgColl(platform_config['collection']) \
        #         .filterBounds(coords) \
        #         .filterDate(platform_config['start_date'], platform_config['end_date'])
            
        # except:
        #     pass