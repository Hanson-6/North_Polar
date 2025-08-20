#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更稳健、带进度与加速的格网生成脚本（适用于Canada）。
关键点：
 - 使用 shapely.prepared 先做快速 intersects 过滤
 - 按列分块生成格子，避免一次性创建超大列表
 - 超大候选格数时自动放大 tile_size（保守策略），并打印提示
"""

import os
import math
import time
import sys
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.prepared import prep
import geopandas as gpd
import shapely
try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False

# -------------------- 用户配置（按需修改） --------------------
country_shp = r"D:\Download\North_Polar\examples\shp2\russia.shp"
tile_size_m = 5120          # 初始 tile 边长（米）
overlap = 0.2               # 重叠比例
metric_crs = "EPSG:3413"    # 投影（环北极常用）
keep_min_intersect = 0.01   # 保留阈值（>=）
out_geojson = r"D:\Download\North_Polar\examples\shp2\russia_tiles.geojson"
out_shp = r"D:\Download\North_Polar\examples\shp2\russia_tiles.shp"
# 性能保护参数
MAX_TOTAL_TILES = 200000    # 超过这个候选格数将尝试放大 tile_size_m
CHUNK_COLS = 50             # 每次生成多少列作为一个 chunk（可调）
# -------------------------------------------------------------

def diagnose_gdf(gdf):
    print("=== 文件诊断 ===")
    print("记录数:", len(gdf))
    print("CRS:", gdf.crs)
    # total bounds
    tb = gdf.total_bounds
    print("总 bounds (minx,miny,maxx,maxy):", tb)
    # approximate area (警告：若为经纬度需先投影)
    try:
        # 若为度，要临时投影到 EPSG:3857 计算近似面积（仅作参考）
        if gdf.crs is None:
            print("⚠️ 警告: 输入 shapefile 无 CRS，假设为 EPSG:4326（经纬度）")
        if str(gdf.crs).lower().find("4326") >= 0 or gdf.crs is None:
            area_approx = gdf.to_crs("EPSG:3857").geometry.area.sum() / 1e6
            print(f"近似面积（km²，via EPSG:3857）: {area_approx:.0f}")
        else:
            area = gdf.geometry.area.sum() / 1e6
            print(f"总面积（km²，按当前 CRS 计算）: {area:.0f}")
    except Exception as e:
        print("面积计算失败:", e)
    print("=================\n")

def create_metric_grid_fast(gdf_country,
                            tile_size_m=5000,
                            overlap=0.2,
                            metric_crs="EPSG:3413",
                            keep_min_intersect=0.01,
                            max_total_tiles_protect=MAX_TOTAL_TILES,
                            chunk_cols=CHUNK_COLS):
    """
    更快的格网生成：
      - 把 country 投影到 metric_crs
      - 计算 cols/rows 与总候选格数；若过大则自动扩大 tile_size_m（保守策略）
      - 按列块生成格子，用 prepared geometry 快速过滤，再做精确面积计算
    返回：GeoDataFrame (crs=metric_crs) 包含 tile_id,row,col,intersect_pct,geometry
    """
    t0 = time.perf_counter()
    # 投影
    g_metric = gdf_country.to_crs(metric_crs)
    # 修复无效几何（若有）
    invalid_count = (~g_metric.is_valid).sum()
    if invalid_count > 0:
        print(f"注意: 检测到 {invalid_count} 个无效几何，尝试用 buffer(0) 修复...")
        g_metric['geometry'] = g_metric['geometry'].buffer(0)
    # union 并准备 geometry
    union_geom = unary_union(g_metric.geometry)
    prepared = prep(union_geom)  # 用于快速 intersects 判断
    minx, miny, maxx, maxy = g_metric.total_bounds

    # 计算 cols/rows
    step = tile_size_m * (1 - overlap)
    if step <= 0:
        raise ValueError("overlap 必须小于 1.0")
    cols = int(math.ceil((maxx - minx) / step))
    rows = int(math.ceil((maxy - miny) / step))
    total_tiles = cols * rows
    print(f"初始 grid 参数: tile_size_m={tile_size_m}, step={step:.1f}, cols={cols}, rows={rows}, 候选总格数={total_tiles}")

    # 保护机制：如果 total_tiles 太大，自动放大 tile_size_m 直到小于阈值
    adjusted = False
    while total_tiles > max_total_tiles_protect:
        adjusted = True
        tile_size_m = tile_size_m * 1.5  # 放大 1.5 倍（保守）
        step = tile_size_m * (1 - overlap)
        cols = int(math.ceil((maxx - minx) / step))
        rows = int(math.ceil((maxy - miny) / step))
        total_tiles = cols * rows
        print(f"⚠ 候选格数过大，已自动将 tile_size_m 放大为 {tile_size_m:.0f}，新候选格数={total_tiles}")
        if tile_size_m > 1000000:
            # 太大了说明可能投影不合适或边界异常
            raise RuntimeError("自动调整 tile_size 失败（已放大到极高值），请检查投影或手动裁剪区域。")

    if adjusted:
        print("⚠ 自动调整后请确认新的 tile_size_m 是否满足你的需求！")

    # 开始分块生成
    boxes = []
    metas = []
    tile_area = float(tile_size_m) * float(tile_size_m)
    total_steps = cols * rows
    print(f"开始按列块生成：每块 {chunk_cols} 列，预计候选格子循环次数 {total_steps}")

    # 进度显示
    if HAS_TQDM:
        outer_iter = range(0, cols, chunk_cols)
        outer_iter = tqdm(outer_iter, desc="列块")
    else:
        outer_iter = range(0, cols, chunk_cols)

    processed = 0
    for i0 in outer_iter:
        i1 = min(i0 + chunk_cols, cols)
        # 为这个列块生成所有 boxes（列 i0..i1-1, 行 0..rows-1）
        chunk_boxes = []
        chunk_meta_idx = []
        for i in range(i0, i1):
            x0 = minx + i * step
            x1 = x0 + tile_size_m
            # 生成该列所有行的 box (减少 Python 层循环开销)
            for j in range(rows):
                y0 = miny + j * step
                y1 = y0 + tile_size_m
                b = box(x0, y0, x1, y1)
                # 快速过滤（prepared.intersects 是很快的）
                if prepared.intersects(b):
                    # 精确计算面积比
                    inter_area = b.intersection(union_geom).area
                    intersect_pct = 0.0 if tile_area == 0 else (inter_area / tile_area)
                    if intersect_pct >= keep_min_intersect:
                        boxes.append(b)
                        metas.append({"row": int(j), "col": int(i), "intersect_pct": float(intersect_pct)})
                # else: 直接跳过
                processed += 1
        # 打印进度（非 tqdm 时）
        if not HAS_TQDM:
            pct = processed / total_steps
            print(f"已处理 {processed}/{total_steps} ({pct:.1%})，已找到 tiles: {len(boxes)}", end="\r", flush=True)

    print()  # 换行

    elapsed = time.perf_counter() - t0
    print(f"生成候选并筛选完成，用时 {elapsed:.1f}s，符合阈值的 tiles 数量：{len(boxes)}")

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
    gdf_wgs84.to_file(out_path_wgs84, driver="GeoJSON")
    elapsed = time.perf_counter() - t0
    print(f"✅ 已导出 GeoJSON: {out_path_wgs84} （用时 {elapsed:.1f}s）")
    return gdf_wgs84

def save_shapefile(gdf_metric, out_path_shp):
    t0 = time.perf_counter()
    gdf_wgs84 = gdf_metric.to_crs("EPSG:4326")
    out_dir = os.path.dirname(out_path_shp)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    gdf_wgs84.to_file(out_path_shp, driver="ESRI Shapefile")
    elapsed = time.perf_counter() - t0
    print(f"✅ 已导出 Shapefile: {out_path_shp} （用时 {elapsed:.1f}s）")
    return gdf_wgs84

def main():
    t_all = time.perf_counter()
    print("开始格网生成流程...")
    try:
        gdf_country = gpd.read_file(country_shp)
    except Exception as e:
        print("❌ 读取 shapefile 失败：", e)
        return

    diagnose_gdf(gdf_country)

    try:
        tiles_metric = create_metric_grid_fast(
            gdf_country,
            tile_size_m=tile_size_m,
            overlap=overlap,
            metric_crs=metric_crs,
            keep_min_intersect=keep_min_intersect,
            max_total_tiles_protect=MAX_TOTAL_TILES,
            chunk_cols=CHUNK_COLS
        )
    except Exception as e:
        print("❌ 生成格网出错：", e)
        return

    print(f"生成的 tiles 数量（metric_crs={metric_crs}）: {len(tiles_metric)}")
    if len(tiles_metric) == 0:
        print("⚠️ 没有生成任何 tile。")
        return

    try:
        tiles_wgs84 = save_geojson_wgs84(tiles_metric, out_geojson)
    except Exception as e:
        print("❌ 保存 GeoJSON 出错：", e)
        return

    try:
        save_shapefile(tiles_metric, out_shp)
    except Exception as e:
        print("❌ 保存 Shapefile 出错：", e)
        return

    print("示例 tiles（前 10 条）:")
    try:
        print(tiles_wgs84.head(10).to_string(index=False))
    except Exception:
        print(tiles_wgs84.head(10))

    print(f"✅ 总共生成的矩形数量: {len(tiles_metric)}")
    print(f"全部流程用时: {time.perf_counter() - t_all:.1f} 秒")

if __name__ == "__main__":
    if not HAS_TQDM:
        print("提示：未检测到 tqdm；可通过 `pip install tqdm` 获得更好进度显示。")
    main()
