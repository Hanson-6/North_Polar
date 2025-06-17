import ee
import geemap

import GEE



# alias
ee_feature = ee.feature.Feature
ee_featColl = ee.featurecollection.FeatureCollection
ee_geometry = ee.geometry.Geometry
ee_list = ee.ee_list.List
ee_number = ee.ee_number.Number

class GMap:
    def __init__(self):
        self.map = geemap.Map()


    def makeGrids(self,
                  rect_bounds,
                  num_rows,
                  num_cols,
                  showOnMap):
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

        # show the grid on the map if requested
        if showOnMap:
            vis_params = {
                'color': 'blue',
                'fillColor': '00000000',  # Transparent fill
                'width': 1
            }
            self.map.addLayer(grid_fc, vis_params, "Grid Rectangles")
            self.map.centerObject(grid_fc)

        return grid_fc