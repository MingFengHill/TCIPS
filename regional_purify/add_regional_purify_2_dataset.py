# nohup python add_regional_purify_2_dataset.py > test.log 2>&1 &
import numpy as np
import random
import os
import h5py
import argparse
from sklearn.neighbors import KDTree


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
    parser.add_argument('--data_path', type=str, default='/mnt/data/rh/data/data/train_data_withEdge.h5')
    # parser.add_argument('--data_path', type=str, default='/mnt/data/rh/data/data_parsenet/train_data.h5')
    parser.add_argument('--save_path', type=str, default='/mnt/data/rh/data/data_save')
    parser.add_argument('--save_file_name', type=str, default='train_data_withEdge_sed_10_5.h5')
    parser.add_argument('--radius', type=float, default=0.08)
    parser.add_argument('--is_use_kdtree', type=int, default=1)
    parser.add_argument('--is_normalize', type=int, default=0)
    parser.add_argument('--neighborhood_size', type=int, default=30)
    parser.add_argument('--is_filter', type=int, default=0)
    parser.add_argument('--filter_neighborhood_size', type=int, default=5)
    parser.add_argument('--is_add_weight', type=int, default=0)
    parser.add_argument('--is_uniform_weight', type=int, default=0)
    parser.add_argument('--is_sed_weight', type=int, default=0)
    parser.add_argument('--is_density_based_weight', type=int, default=1)
    args = parser.parse_args()
    for arg, value in sorted(vars(args).items()):
        print("[INFO] Argument {}: {}".format(arg, value))

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

    radius = args.radius
    gt_regional_purify = []
    gt_purify_weight = []
    len_ = len(gt_points)
    print("[INFO] The data set contains {} data".format(len_))
    for model_id in range(len_):
        regional_purify = []
        purify_weight = []
        point_cloud = gt_points[model_id]
        if args.is_normalize == 1:
            point_cloud = normalize_point_cloud(point_cloud)
        label = gt_labels[model_id]
        purify_point_cnt = 0
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
                    purify_point_cnt += 1
            # Outlier removal
            if args.is_filter == 1:
                filter_distances, filter_indices = tree.query(point_cloud, k=args.filter_neighborhood_size)
                regional_purify_arr = np.array(regional_purify)
                for j in range(len(filter_indices)):
                    if all(regional_purify_arr[filter_indices[j][1:]] == 1.0):
                        regional_purify[j] = 1
            if args.is_add_weight == 1:
                N = len(regional_purify)
                # Default weight
                purify_weight = [1.0 for _ in range(N)]
                if args.is_uniform_weight == 1:
                    pass
                # Refer to sednet
                if args.is_sed_weight == 1:
                    if purify_point_cnt != 0:
                        regional_purify_arr = np.array(regional_purify)
                        indexes = np.where(regional_purify_arr == 1)[0]
                        purify_point_weight = (N - purify_point_cnt) / purify_point_cnt
                        print("[INFO] purify_point_weight: {}".format(purify_point_weight))
                        for index in indexes:
                            purify_weight[index] = purify_point_weight
                    else:
                        print("[INFO] purify_point_cnt is zero, purify_point_weight: {}".format(1))
                if args.is_density_based_weight == 1:
                    regional_purify_arr = np.array(regional_purify)
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
                    # purify_weight_arr = purify_weight_arr / np.max(purify_weight_arr)
                    # purify_weight_arr = purify_weight_arr * 30
                    # print("[INFO] max weight: {}, min weight: {}, multiple: {}".format(np.max(purify_weight_arr), np.min(purify_weight_arr), (np.max(purify_weight_arr)/np.min(purify_weight_arr))))
                    purify_weight = purify_weight_arr.tolist()
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
                    purify_point_cnt += 1
        print("[INFO] model {} purify point cnt: {}".format(model_id, purify_point_cnt))

        gt_regional_purify.append(regional_purify)
        gt_purify_weight.append(purify_weight)
    gt_regional_purify = np.array(gt_regional_purify, dtype=np.float32)
    gt_purify_weight = np.array(gt_purify_weight, dtype=np.float32)

    with h5py.File(os.path.join(args.save_path, args.save_file_name), 'w') as new_file:
        new_file.create_dataset('points', data=gt_points)
        new_file.create_dataset('labels', data=gt_labels)
        new_file.create_dataset('normals', data=gt_normals)
        new_file.create_dataset('prim', data=gt_primitives)
        new_file.create_dataset('purity', data=gt_regional_purify)
        if has_edge:
            new_file.create_dataset('edge', data=gt_edges)
        if args.is_add_weight == 1:
            new_file.create_dataset('purify_weight', data=gt_purify_weight)

    print("[INFO] The new data set is created successfully!")
