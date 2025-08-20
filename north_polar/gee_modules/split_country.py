#!/usr/bin/env python3（适用于Alaska、Sweden、Norway）
# -*- coding: utf-8 -*-

import os
import math
import time
import sys
from shapely.geometry import box
from shapely.ops import unary_union
import geopandas as gpd

# 可选：进度条库，若未安装则使用简单打印
try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False

# ------------ 配置区域（按需修改） ------------
country_shp = r"D:\Download\North_Polar\examples\shp2\canada.shp"
tile_size_m = 5120          # tile 边长（米）
overlap = 0.2               # tile 重叠比例（0-1）
metric_crs = "EPSG:3413"    # 北极立体投影e:\InSAR论文\shp2
keep_min_intersect = 0.01   # tile 与国家相交面积阈值（比例）
out_geojson = r"D:\Download\North_Polar\examples\shp2\canada_tiles.geojson"
out_shp = r"D:\Download\North_Polar\examples\shp2\canada_tiles.shp"
# ------------------------------------------------

def load_country_from_shp(shp_path):
    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"国家边界文件未找到: {shp_path}")
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError("读取的国家边界文件为空或无有效几何。")
    return gdf

def create_metric_grid(gdf_country, tile_size_m=5000, overlap=0.2, metric_crs="EPSG:3413", keep_min_intersect=0.01):
    """
    在 metric_crs 下生成规则方格，并只保留与国家相交比例 >= keep_min_intersect 的 tile。
    返回 GeoDataFrame（crs = metric_crs），包含字段 tile_id, row, col, intersect_pct, geometry。
    过程中会显示进度。
    """
    t0 = time.perf_counter()
    g_metric = gdf_country.to_crs(metric_crs)
    union_geom = unary_union(g_metric.geometry)
    minx, miny, maxx, maxy = g_metric.total_bounds

    step = tile_size_m * (1 - overlap)
    cols = int(math.ceil((maxx - minx) / step))
    rows = int(math.ceil((maxy - miny) / step))

    total_steps = cols * rows
    print(f"格网参数: tile_size_m={tile_size_m}, overlap={overlap}, step={step:.2f}")
    print(f"网格行列: cols={cols}, rows={rows}, 总格数候选={total_steps}")

    boxes = []
    metas = []
    tile_area = float(tile_size_m) * float(tile_size_m)

    if total_steps == 0:
        print("⚠️ 总候选格数为 0（可能 bbox 太小或参数不合理），返回空 GeoDataFrame。")
        return gpd.GeoDataFrame(columns=["tile_id", "row", "col", "intersect_pct", "geometry"], crs=metric_crs)

    # 进度显示：优先使用 tqdm，否则每若干步打印一次
    if HAS_TQDM:
        iterator = tqdm(range(total_steps), desc="生成 tiles", unit="tile")
    else:
        iterator = range(total_steps)
        progress_step = max(1, total_steps // 20)  # 20 次打印

    try:
        for idx in iterator:
            # 将 idx 映射回 (i, j) ： i 是列索引，j 是行索引（与原代码的循环顺序等价）
            i = idx // rows
            j = idx % rows

            x0 = minx + i * step
            y0 = miny + j * step
            x1 = x0 + tile_size_m
            y1 = y0 + tile_size_m
            b = box(x0, y0, x1, y1)

            inter_area = b.intersection(union_geom).area
            intersect_pct = 0.0 if tile_area == 0 else (inter_area / tile_area)

            if intersect_pct >= keep_min_intersect:
                boxes.append(b)
                metas.append({"row": int(j), "col": int(i), "intersect_pct": float(intersect_pct)})

            # 若没有 tqdm，则定期打印进度
            if not HAS_TQDM and (idx % progress_step == 0 or idx == total_steps - 1):
                found = len(boxes)
                pct = (idx + 1) / total_steps
                print(f"[{idx+1}/{total_steps}] 处理进度 {pct:.1%}，已找到符合阈值的 tiles: {found}")
    except KeyboardInterrupt:
        print("\n✋ 检测到人工中断 (KeyboardInterrupt)。已停止生成。")
        # 返回当前已生成的部分结果（如果有）
        if len(boxes) == 0:
            return gpd.GeoDataFrame(columns=["tile_id", "row", "col", "intersect_pct", "geometry"], crs=metric_crs)
        tiles_gdf = gpd.GeoDataFrame(metas, geometry=boxes, crs=metric_crs)
        tiles_gdf["tile_id"] = [f"tile_{r['col']}_{r['row']}" for r in tiles_gdf[["col","row"]].to_dict(orient="records")]
        tiles_gdf = tiles_gdf[["tile_id", "row", "col", "intersect_pct", "geometry"]]
        return tiles_gdf

    elapsed = time.perf_counter() - t0
    print(f"生成格网完成，用时 {elapsed:.1f} 秒，符合阈值的 tiles 数量：{len(boxes)}")

    if len(boxes) == 0:
        return gpd.GeoDataFrame(columns=["tile_id", "row", "col", "intersect_pct", "geometry"], crs=metric_crs)

    tiles_gdf = gpd.GeoDataFrame(metas, geometry=boxes, crs=metric_crs)
    tiles_gdf["tile_id"] = [f"tile_{r['col']}_{r['row']}" for r in tiles_gdf[["col","row"]].to_dict(orient="records")]
    tiles_gdf = tiles_gdf[["tile_id", "row", "col", "intersect_pct", "geometry"]]
    return tiles_gdf

def save_geojson_wgs84(gdf_metric, out_path_wgs84):
    t0 = time.perf_counter()
    gdf_wgs84 = gdf_metric.to_crs("EPSG:4326")
    out_dir = os.path.dirname(out_path_wgs84)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    # 覆盖已有文件时 geopandas 会报错或写入多个文件；这里直接调用 to_file
    gdf_wgs84.to_file(out_path_wgs84, driver="GeoJSON")
    elapsed = time.perf_counter() - t0
    print(f"✅ 已导出 GeoJSON: {out_path_wgs84} （用时 {elapsed:.1f} 秒）")
    return gdf_wgs84

def save_shapefile(gdf_metric, out_path_shp):
    t0 = time.perf_counter()
    gdf_wgs84 = gdf_metric.to_crs("EPSG:4326")
    out_dir = os.path.dirname(out_path_shp)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    # ESRI Shapefile 会生成多个关联文件，确保路径目录存在
    gdf_wgs84.to_file(out_path_shp, driver="ESRI Shapefile")
    elapsed = time.perf_counter() - t0
    print(f"✅ 已导出 Shapefile: {out_path_shp} （用时 {elapsed:.1f} 秒）")
    return gdf_wgs84

def main():
    print("开始生成规则格网...")
    t_main = time.perf_counter()

    try:
        gdf_country = load_country_from_shp(country_shp)
    except Exception as e:
        print("❌ 读取国家边界出错：", e)
        return

    print("国家原始 CRS:", gdf_country.crs)
    try:
        tiles_metric = create_metric_grid(
            gdf_country,
            tile_size_m=tile_size_m,
            overlap=overlap,
            metric_crs=metric_crs,
            keep_min_intersect=keep_min_intersect
        )
    except Exception as e:
        print("❌ 生成格网出错：", e)
        return

    print(f"生成的 tiles 数量（在 metric_crs={metric_crs} 下）: {len(tiles_metric)}")
    if len(tiles_metric) == 0:
        print("⚠️ 没有生成任何 tile，请检查 keep_min_intersect / tile 大小 / country_shp 是否有效几何。")
        return

    # 保存 GeoJSON
    try:
        tiles_wgs84 = save_geojson_wgs84(tiles_metric, out_geojson)
    except Exception as e:
        print("❌ 保存 GeoJSON 出错：", e)
        return

    # 保存 Shapefile
    try:
        save_shapefile(tiles_metric, out_shp)
    except Exception as e:
        print("❌ 保存 Shapefile 出错：", e)
        return

    # 打印前几条示例
    print("示例 tiles（前 10 条）:")
    # 使用 to_string 避免 pandas 的省略
    try:
        print(tiles_wgs84.head(10).to_string(index=False))
    except Exception:
        # 若 to_string 出错，降级为简单打印
        print(tiles_wgs84.head(10))

    print(f"✅ 总共生成的矩形数量: {len(tiles_metric)}")
    print(f"全部流程用时：{time.perf_counter() - t_main:.1f} 秒")

if __name__ == "__main__":
    if not HAS_TQDM:
        print("提示：未检测到 tqdm，建议 `pip install tqdm` 以获得更好的进度显示（可选）。")
    main()
