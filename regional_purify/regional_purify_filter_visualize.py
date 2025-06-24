# nohup python add_regional_purify_2_dataset.py > test.log 2>&1 &
import numpy as np
import random
import os
import h5py
import argparse
from sklearn.neighbors import KDTree
import open3d as o3d
import matplotlib.pyplot as plt


def save_ply(point_cloud, weight, postfix, color_map, save_path):
    N = point_cloud.shape[0]
    colors = np.zeros((N, 3))
    colormap = plt.get_cmap(color_map)  
    colors = colormap(weight)
    colors = colors[:, :3]
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(point_cloud)
    pc.colors = o3d.utility.Vector3dVector(colors)
    ply_file = save_path + '{}_{}.ply'.format(i, postfix)
    o3d.io.write_point_cloud(ply_file, pc, write_ascii=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/mnt/data/rh/data/data_parsenet/train_data.h5')
    parser.add_argument('--save_path', type=str, default='./regional_purify_filter/')
    parser.add_argument('--neighborhood_size', type=int, default=10)
    parser.add_argument('--is_filter', type=int, default=1)
    parser.add_argument('--filter_neighborhood_size', type=int, default=5)
    parser.add_argument('--visualize_begin', type=int, default=0)
    parser.add_argument('--visualize_end', type=int, default=50)
    parser.add_argument('--is_sed_weight', type=int, default=0)
    parser.add_argument('--is_density_based_weight', type=int, default=1)
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
        regional_purify = []
        point_cloud = gt_points[i]
        label = gt_labels[i]
        purify_point = 0

        tree = KDTree(point_cloud)
        distances, indices = tree.query(point_cloud, k=args.neighborhood_size)
        for j in range(len(indices)):
            query_label = label[j]
            total_count = indices[j].size
            match_count = np.sum(label[indices[j]] == query_label)
            regional_purify.append(match_count/total_count)
            # print("[DEBUG] ID: {}, total_count: {}, match_count: {}, regional_purify: {}".format(j,
            #                                                                                      total_count,
            #                                                                                      match_count,
            #                                                                                      regional_purify[j]))
            if regional_purify[j] == 1:
                purify_point += 1
        
        if args.is_filter == 1:
            regional_purify_arr = np.array(regional_purify)
            filter_distances, filter_indices = tree.query(point_cloud, k=args.filter_neighborhood_size)
            for j in range(len(filter_indices)):
                if all(regional_purify_arr[filter_indices[j][1:]] == 1.0):
                    regional_purify[j] = 1
        
        N = len(regional_purify)
        # Default weight
        purify_weight = [1.0 for _ in range(N)]
        regional_purify_arr = np.array(regional_purify)
        purify_point_cnt = np.sum(regional_purify_arr == 1.0)
        if args.is_sed_weight == 1:
            if purify_point_cnt != 0:
                indexes = np.where(regional_purify_arr == 1.0)[0]
                purify_point_weight = (N - purify_point_cnt) / purify_point_cnt
                print("[INFO] purify_point_weight: {}".format(purify_point_weight))
                for index in indexes:
                    purify_weight[index] = purify_point_weight
            else:
                print("[INFO] purify_point_cnt is zero, purify_point_weight: {}".format(1))
        if args.is_density_based_weight == 1:
            purify_weight_arr = np.array(purify_weight)
            bins = 10  # Number of bins
            hist, bin_edges = np.histogram(regional_purify_arr, bins=bins, range=(0, 1))
            bin_edges[-1] += 0.1
            bin_indices = np.digitize(regional_purify_arr, bins=bin_edges, right=False) - 1
            for m in range(bins):
                if hist[m] != 0:
                    # Inverse Proportional Weighting
                    # purify_weight_arr[bin_indices == m] = (1.0 / hist[m])
                    # Logarithmic Inverse Proportional Weighting
                    purify_weight_arr[bin_indices == m] = 1.0 / np.log(1.02 + (hist[m] / N))
                else:
                    purify_weight_arr[bin_indices == m] = 0
            if np.any(purify_weight_arr == 0):
                print("[ERROR] weight has zero")
                exit
            print("[INFO] max weight: {}, min weight: {}, multiple: {}".format(np.max(purify_weight_arr), np.min(purify_weight_arr), (np.max(purify_weight_arr)/np.min(purify_weight_arr))))
            # Normalize weights
            purify_weight_arr = purify_weight_arr / np.max(purify_weight_arr)
            # purify_weight_arr = purify_weight_arr * 10
            purify_weight = purify_weight_arr.tolist()
        
        # 'viridis', 'plasma', 'inferno', 'magma'
        # 'autumn', 'cool', 'copper', 'hot', 'spring', 'summer', 'winter'
        # 'GnBu', 'BuGn', 'YiGn'
        save_ply(point_cloud, regional_purify, 'regional_purify', 'GnBu', args.save_path)
        save_ply(point_cloud, purify_weight, 'purify_weight', 'GnBu', args.save_path)
        print("[INFO] model {} purify point cnt: {}".format(i, purify_point))
        
