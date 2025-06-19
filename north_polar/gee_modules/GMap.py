import geemap
from config import *



class GMap:
    def __init__(self, gee):
        self.map = geemap.Map()
        self.gee = gee


    def addLayer(self, data, style, name):
        self.map.addLayer(data, style, name)

    
    def setCenter(self, data, zoom=7):
        self.map.centerObject(data, zoom)


    def showGrids(self,
                  rect_bounds,
                  num_rows,
                  num_cols):
        """
        Show grids of rectangles within the specified bounds.
        """

        grid_fc = self.gee.makeGrids(rect_bounds, num_rows, num_cols)

        vis_params = {
            'color': 'blue',
            'fillColor': '00000000',  # Transparent fill
            'width': 1
        }
        self.map.addLayer(grid_fc, vis_params, "Grid Rectangles")
        self.map.centerObject(grid_fc)

        return grid_fc

    
    def showImg(self, result, zoom):
        self.setCenter(result['bounds'], zoom)

        self.map.addLayer(
            result['image'], 
            result['vis_params'], 
            f"{result['platform']}"
        )
