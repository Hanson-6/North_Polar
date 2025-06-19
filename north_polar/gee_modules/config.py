import ee



# alias
ee_feature = ee.feature.Feature
ee_featColl = ee.featurecollection.FeatureCollection
ee_geometry = ee.geometry.Geometry
ee_list = ee.ee_list.List
ee_number = ee.ee_number.Number
ee_imgColl = ee.imagecollection.ImageCollection


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


    'landsat9': {
        'collection': 'LANDSAT/LC09/C02/T1_L2',
        'bands': {
            'rgb': ['SR_B4', 'SR_B3', 'SR_B2'],
            'nir': ['SR_B5'],
            'swir': ['SR_B6', 'SR_B7'],
            'pan': ['SR_B8']  # 全色波段15米
        },
        'resolution': {
            'rgb': 30,
            'nir': 30,
            'swir': 30,
            'pan': 15  # 全色波段分辨率更高
        },
        'vis_params': {
            'min': 7000,
            'max': 30000,
            'gamma': 1.4
        },
        'scale_factor': 0.0000275,
        'offset': -0.2,
        'filters': {
            'cloudCover': 'CLOUD_COVER',
            'maxCloud': 20
        }
    },
}