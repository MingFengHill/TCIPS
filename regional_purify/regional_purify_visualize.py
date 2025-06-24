# nohup python add_regional_purify_2_dataset.py > test.log 2>&1 &
import numpy as np
import random
import os
import h5py
import argparse
from sklearn.neighbors import KDTree
import open3d as o3d
import matplotlib.pyplot as plt

def normalize_point_cloud(point_cloud):
    point_cloud = point_cloud.copy()
    centroid = np.mean(point_cloud, axis=0)
    point_cloud -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(point_cloud ** 2, axis=1)))
    if furthest_distance == 0:
        print("[ERROR] furthest_distance is zero")
        exit
    point_cloud /= furthest_distance
    return point_cloud


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/mnt/data/rh/data/data_parsenet/train_data.h5')
    parser.add_argument('--save_path', type=str, default='./regional_purify_visulize/')
    parser.add_argument('--radius', type=float, default=0.08)
    parser.add_argument('--is_use_kdtree', type=int, default=1)
    parser.add_argument('--is_normalize', type=int, default=0)
    parser.add_argument('--neighborhood_size', type=int, default=80)
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
    radius = args.radius
    gt_regional_purify = []
    len_ = len(gt_points)
    print("[INFO] The data set contains {} data".format(len_))
    for i in range(len_):
        if i < begin:
            continue
        if i > end:
            break
        regional_purify = []
        point_cloud = gt_points[i]
        if args.is_normalize == 1:
            point_cloud = normalize_point_cloud(point_cloud)
        label = gt_labels[i]
        purify_point = 0
        if args.is_use_kdtree == 1:
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
        else:
        # ball query
            for j in range(len(point_cloud)):
                query_point = point_cloud[j]
                distances = np.linalg.norm(point_cloud - query_point, axis=1)
                indices = np.where(distances <= radius)[0]

                query_label = label[j]
                total_count = indices.size
                match_count = np.sum(label[indices] == query_label)
                regional_purify.append(match_count/total_count)
                # print("[DEBUG] ID: {}, total_count: {}, match_count: {}, regional_purify: {}".format(j,
                #                                                                                      total_count,
                #                                                                                      match_count,
                #                                                                                      regional_purify[j]))
                if regional_purify[j] == 1:
                    purify_point += 1
        
        N = point_cloud.shape[0]
        colors = np.zeros((N, 3))
        # 'viridis', 'plasma', 'inferno', 'magma'
        # 'autumn', 'cool', 'copper', 'hot', 'spring', 'summer', 'winter'
        # 'GnBu', 'BuGn', 'YiGn'
        colormap = plt.get_cmap('viridis')  
        colors = colormap(regional_purify)
        colors = colors[:, :3]
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(point_cloud)
        pc.colors = o3d.utility.Vector3dVector(colors)
        ply_file = args.save_path + '{}.ply'.format(i)
        o3d.io.write_point_cloud(ply_file, pc, write_ascii=True)
        print("[INFO] model {} purify point cnt: {}".format(i, purify_point))
        
