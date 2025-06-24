# nohup python add_regional_purify_2_dataset.py > test.log 2>&1 &
import numpy as np
import random
import os
import h5py
import argparse
from sklearn.neighbors import KDTree
import open3d as o3d
import matplotlib.pyplot as plt

g_colors = np.array([
    [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
    [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212],
    [0, 128, 128], [220, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
    [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128],
    [255, 255, 0], [0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255],
    [255, 0, 255], [192, 192, 192], [65, 105, 225], [127, 255, 212], [218, 165, 32],
    [210, 105, 30], [245, 245, 220], [250, 250, 210], [112, 128, 144], [255, 140, 0],
    [144, 238, 144], [255, 182, 193], [119, 136, 153], [106, 90, 205], [72, 61, 139],
    [75, 0, 130], [139, 69, 19], [165, 42, 42], [255, 69, 0], [188, 143, 143],
    [85, 107, 47], [255, 140, 105], [204, 204, 255], [233, 150, 122], [143, 188, 143]
]) / 255.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/mnt/data/rh/data/data_parsenet/train_data.h5')
    parser.add_argument('--save_path', type=str, default='./inst_label/')
    parser.add_argument('--visualize_begin', type=int, default=0)
    parser.add_argument('--visualize_end', type=int, default=50)
    args = parser.parse_args()
    for arg, value in sorted(vars(args).items()):
        print("[INFO] Argument {}: {}".format(arg, value))
        
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    has_edge = False
    with h5py.File(args.data_path, 'r') as hf:
        gt_points = np.array(hf.get("points"))
        gt_labels = np.array(hf.get("labels"))
        gt_normals = np.array(hf.get("normals"))
        gt_primitives = np.array(hf.get("prim"))
        if 'edge' in hf:
            gt_edges = np.array(hf.get("edge"))
            has_edge = True
            print("[INFO] has edge label")
        else:
            print("[INFO] no edge label")

    begin = args.visualize_begin
    end = args.visualize_end
    len_ = len(gt_points)
    print("[INFO] The data set contains {} data".format(len_))
    for i in range(len_):
        if i < begin:
            continue
        if i > end:
            break
        point_cloud = gt_points[i]
        labels = gt_labels[i]
        selected_colors = g_colors[labels]

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(point_cloud)
        pc.colors = o3d.utility.Vector3dVector(selected_colors)
        ply_file = args.save_path + '{}.ply'.format(i)
        o3d.io.write_point_cloud(ply_file, pc, write_ascii=True)
        
        print("[info] handle {} model".format(i))
        
