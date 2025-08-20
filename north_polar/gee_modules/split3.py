#!/usr/bin/env python3
# -*- coding: utf-8 -*-（适用于Russia）

import os
import math
import geopandas as gpd
import pandas as pd
from shapely.geometry import box, LineString, MultiPolygon, Polygon
from shapely.ops import split, unary_union

def split_east_west(gdf):
    """
    将跨180度的国家切分为东西两部分
    """
    country = gdf.unary_union  # 合并为单个几何

    # 构造180度经线
    meridian = LineString([(180, 90), (180, -90)])

    # 切分
    result = split(country, meridian)

    west_parts = []
    east_parts = []

    for geom in result.geoms:  # 遍历 GeometryCollection
        if isinstance(geom, (Polygon, MultiPolygon)):
            # 计算质心判断属于东西半球
            centroid_x = geom.centroid.x
            if centroid_x > 0:  # 东半球
                east_parts.append(geom)
            else:               # 西半球
                west_parts.append(geom)

    west = unary_union(west_parts) if west_parts else None
    east = unary_union(east_parts) if east_parts else None

    return west, east

def generate_grid(geom, dx, dy):
    """
    根据输入 polygon (geom)，生成覆盖的矩形格网
    """
    minx, miny, maxx, maxy = geom.bounds
    tiles = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            tile = box(x, y, min(x+dx, maxx), min(y+dy, maxy))
            if tile.intersects(geom):  # 只保留和国家相交的格子
                tiles.append(tile)
            y += dy
        x += dx
    return tiles

def main():
    shp_path = "/share/home/u20169/North_Polar/examples/shp2/russia.shp"  # 你的俄罗斯边界 shp
    output_path = "/share/home/u20169/North_Polar/examples/shp2/russia_grid.shp"

    gdf_country = gpd.read_file(shp_path)
    gdf_country = gdf_country.to_crs("EPSG:4326")

    print("切分俄罗斯东西部分...")
    west, east = split_east_west(gdf_country)

    if west is None or east is None:
        raise RuntimeError("❌ 国家切分失败，请检查输入 shp 是否跨180度")

    # 这里格网大小可调整 (经纬度度数单位)
    dx, dy = 2.0, 1.0

    print("生成西部格网...")
    tiles_west = generate_grid(west, dx, dy)

    print("生成东部格网...")
    tiles_east = generate_grid(east, dx, dy)

    # 合并
    all_tiles = gpd.GeoDataFrame(pd.DataFrame({"geometry": tiles_west + tiles_east}), crs="EPSG:4326")

    print(f"保存到 {output_path}")
    all_tiles.to_file(output_path)

if __name__ == "__main__":
    main()
