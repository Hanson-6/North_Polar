import ee



# alias
ee_feature = ee.feature.Feature
ee_featColl = ee.featurecollection.FeatureCollection
ee_geometry = ee.geometry.Geometry
ee_list = ee.ee_list.List
ee_number = ee.ee_number.Number
ee_imgColl = ee.imagecollection.ImageCollection
ee_filter = ee.filter.Filter


# Platform Specifications
PLATFORM_SPECS = {
    'sentinel2': {
        'collection': 'COPERNICUS/S2_SR_HARMONIZED',
        'bands': {
            'rgb': ['B4', 'B3', 'B2'],
            'nir': ['B8'],
            'red_edge': ['B5', 'B6', 'B7'],
            'swir': ['B11', 'B12']
        },
        'resolution': {
            'rgb': 10,      # RGB波段10米
            'nir': 10,      # 近红外10米
            'red_edge': 20, # 红边20米
            'swir': 20      # 短波红外20米
        },
        'vis_params': {
            'min': 0,
            'max': 3000,
            'gamma': 1.2
        },
        'filters': {
            'cloudCover': 'CLOUDY_PIXEL_PERCENTAGE',
            'maxCloud': 10
        }
    },
}