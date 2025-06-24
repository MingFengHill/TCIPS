# nohup python add_regional_purity_2_abc_primitive.py > test.log 2>&1 &
import numpy as np
import random
import os
import h5py
import argparse
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import open3d as o3d


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/wang/final/bgpseg_data/ABCPrimitive/train/')
    parser.add_argument('--save_path', type=str, default='/home/wang/final/bgpseg_data/ABCPrimitive_purity/train/')
    parser.add_argument('--save_vis_path', type=str, default='/home/wang/recon/new_dataset/sed-net/regional_purify/vis')
    parser.add_argument('--is_vis', type=int, default=0)
    parser.add_argument('--neighborhood_size', type=int, default=30)
    args = parser.parse_args()
    for arg, value in sorted(vars(args).items()):
        print("[INFO] Argument {}: {}".format(arg, value))
        
    # os.makedirs(args.save_path, exist_ok=True)
    data_list = [f for f in os.listdir(args.data_path) if os.path.isfile(os.path.join(args.data_path, f))]
    print("[INFO] total file cnt: {}".format(len(data_list)))
    for index in range(len(data_list)):        
        file_name = data_list[index]
        if not file_name.endswith(".npz"):
            print("[WARR] file name not end with npz: {}".format(file_name))
            continue
        file_path = os.path.join(args.data_path, file_name)
        data = np.load(file_path)
        # coord, normals, boundary, label, semantic, param, F, edges, dse_edges = data['V'],data['N'],data['B'],data['L'],data['S'],data['T_param'],data['F'],data['edges'],data['dse_edges']
        coord      = data['V']
        normals    = data['N']
        boundary   = data['B']
        label      = data['L']
        semantic   = data['S']
        param      = data['T_param']
        F          = data['F']
        edges      = data['edges']
        dse_edges  = data['dse_edges']
        
        regional_purify = []
        purify_point_cnt = 0
        tree = KDTree(coord)
        distances, indices = tree.query(coord, k=args.neighborhood_size)
        
        for j in range(len(indices)):
            query_label = label[j]
            total_count = indices[j].size
            match_count = np.sum(label[indices[j]] == query_label)
            regional_purify.append(match_count/total_count)
            if regional_purify[j] == 1:
                purify_point_cnt += 1
                
        
        regional_purify = np.array(regional_purify, dtype=np.float32)
        out_file = os.path.join(args.save_path, file_name)
        np.savez(
            out_file,
            V=coord,
            N=normals,
            B=boundary,
            L=label,
            S=semantic,
            T_param=param,
            F=F,
            edges=edges,
            dse_edges=dse_edges,
            RP=regional_purify
        )
        
        if args.is_vis == 1:
            colormap = plt.get_cmap('viridis')
            colors = colormap(regional_purify)
            colors = colors[:, :3]
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(coord)
            pc.colors = o3d.utility.Vector3dVector(colors)
            ply_file = os.path.join(args.save_vis_path, "{}_purity.ply".format(index))
            o3d.io.write_point_cloud(ply_file, pc, write_ascii=True)

        print("[INFO] saved augmented file to: {}".format(out_file))
        print("[INFO] model {} purify point cnt: {}".format(index, purify_point_cnt))
    print("[INFO] add purity label success!!!")


        
    