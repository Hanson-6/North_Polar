import os
import shutil
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import rasterio
from scipy.ndimage import laplace
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from tqdm import tqdm


def compute_black_ratio(img_arr, black_thresh=30):
    # img_arr: HxW or HxWxC in [0,255]
    gray = np.mean(img_arr, axis=-1) if img_arr.ndim==3 else img_arr
    return np.mean(gray < black_thresh)


def compute_lap_variance(img_arr):
    gray = np.mean(img_arr, axis=-1) if img_arr.ndim==3 else img_arr
    return np.var(laplace(gray))


def filter_and_sync(images_dir, polys_dir, out_images, out_polys,
                    black_thresh, lap_thresh):
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_polys, exist_ok=True)

    scores = []  # list of (path, black_ratio, lap_var)
    for root, _, files in os.walk(images_dir):
        for fn in files:
            if not fn.lower().endswith(('tif','tiff','jpg','png','jpeg')):
                continue
            path = os.path.join(root, fn)
            with rasterio.open(path) as src:
                arr = src.read([1,2,3]).transpose(1,2,0)
            br = compute_black_ratio(arr)
            lv = compute_lap_variance(arr)
            scores.append((path, br, lv))

    kept, dropped = [], []
    for path, br, lv in scores:
        if br > black_thresh or lv < lap_thresh:
            dropped.append((path, br, lv))
        else:
            kept.append((path, br, lv))

    # copy kept and sync polygons
    for path, _, _ in kept:
        rel = os.path.relpath(path, images_dir)
        dst_img = os.path.join(out_images, rel)
        os.makedirs(os.path.dirname(dst_img), exist_ok=True)
        shutil.copy2(path, dst_img)

        region = rel.split(os.sep)[0]
        geojson_in = os.path.join(polys_dir, f"{region}.json")
        geojson_out = os.path.join(out_polys, f"{region}.json")
        if not os.path.exists(geojson_out):
            shutil.copy2(geojson_in, geojson_out)

    # quant evaluation
    for name, group in [('kept', kept), ('dropped', dropped)]:
        brs = [br for _, br, _ in group]
        lvs = [lv for _, _, lv in group]
        print(f"{name}: count={len(group)}; black_ratio mean={np.mean(brs):.3f}, var={np.var(brs):.3f}")
        print(f"{name}: lap_var mean={np.mean(lvs):.3f}, var={np.var(lvs):.3f}")

    # plot distributions
    plt.figure()
    plt.hist([br for _, br, _ in kept], bins=50, alpha=0.6, label='kept')
    plt.hist([br for _, br, _ in dropped], bins=50, alpha=0.6, label='dropped')
    plt.title('Black ratio distribution')
    plt.legend()
    plt.savefig('black_ratio_dist.png')

    plt.figure()
    plt.hist([lv for _, _, lv in kept], bins=50, alpha=0.6, label='kept')
    plt.hist([lv for _, _, lv in dropped], bins=50, alpha=0.6, label='dropped')
    plt.title('Laplace variance distribution')
    plt.legend()
    plt.savefig('lap_variance_dist.png')

    print("Plots saved: black_ratio_dist.png, lap_variance_dist.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir',
                        # default='/share/home/u20169/code/North_Polar/python/model/dataset/sentinel_2',
                        default=r'D:\HKU\OneDrive - The University of Hong Kong - Connect\项目\测绘-TJ\north_polar\python\model\dataset\sentinel_2',
                    )
    parser.add_argument('--polys_dir',
                        # default='/share/home/u20169/code/North_Polar/python/model/dataset/polygons',
                        default=r'D:\HKU\OneDrive - The University of Hong Kong - Connect\项目\测绘-TJ\north_polar\python\model\dataset\polygons',
                    )
    parser.add_argument('--out_images',
                        # default='/share/home/u20169/code/North_Polar/python/model/filtered_sentinel_2',
                        default=r'D:\HKU\OneDrive - The University of Hong Kong - Connect\项目\测绘-TJ\north_polar\python\model\filtered_dataset\sentinel_2',
                    )
    parser.add_argument('--out_polys',
                        # default='/share/home/u20169/code/North_Polar/python/model/filtered_polygons',
                        default=r'D:\HKU\OneDrive - The University of Hong Kong - Connect\项目\测绘-TJ\north_polar\python\model\filtered_dataset\polygons',
                    )
    parser.add_argument('--black_thresh', type=float, default=10.0,
                        help='max black_ratio to keep')
    parser.add_argument('--lap_thresh', type=float, default=20.0,
                        help='min lap variance to keep')
    args = parser.parse_args()


    # countries = ['Alaska', 'Canada', 'Norway', 'Russia', 'Sweden']

    # sentinel_dir = r'D:\HKU\OneDrive - The University of Hong Kong - Connect\项目\测绘-TJ\north_polar\python\model\dataset\sentinel_2'
    # polygon_dir = r'D:\HKU\OneDrive - The University of Hong Kong - Connect\项目\测绘-TJ\north_polar\python\model\dataset\polygons'

    # filtered_sentinel_dir = r'D:\HKU\OneDrive - The University of Hong Kong - Connect\项目\测绘-TJ\north_polar\python\model\filtered_dataset\sentinel_2'
    # filtered_polygon_dir = r'D:\HKU\OneDrive - The University of Hong Kong - Connect\项目\测绘-TJ\north_polar\python\model\filtered_dataset\polygons'

    # for country in countries:
        
    #     images_dir = sentinel_dir
    #     polys_dir = polygon_dir
    #     out_images = filtered_sentinel_dir
    #     out_polys = filtered_polygon_dir

    filter_and_sync(
        args.images_dir, args.polys_dir,
        args.out_images, args.out_polys,
        args.black_thresh, args.lap_thresh
    )

if __name__=='__main__':
    main()
