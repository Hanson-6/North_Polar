import geemap

from config import *
from GEE import GEE

class GMap:
    def __init__(self):
        self.map = geemap.Map()


    def showGrids(self,
                  rect_bounds,
                  num_rows,
                  num_cols):
        """
        Show grids of rectangles within the specified bounds.
        """

        grid_fc = GEE().makeGrids(rect_bounds, num_rows, num_cols)

        vis_params = {
            'color': 'blue',
            'fillColor': '00000000',  # Transparent fill
            'width': 1
        }
        self.map.addLayer(grid_fc, vis_params, "Grid Rectangles")
        self.map.centerObject(grid_fc)

        return grid_fc