# nohup ./run_predictions_abc_primitive_vis.sh > test.log 2>&1 &
import sys
import logging
import json
import os
from read_config import Config
config = Config("configs/config_MTLNet_normal.yml")
GPU = config.gpu
os.environ['CUDA_VISIBLE_DEVICES'] = GPU
from shutil import copyfile
import numpy as np
from gen_test_vis import COLORS_TYPE, visual_labels
from src.dataset_segments import ori_simple_data
import h5py
from metrics import ConfusionMatrix, get_mious
from src.ABCPrimitive import ABCPrimitive_Dataset, collate_fn, collate_fn_region
from src.mean_shift_new import mean_shift_gpu
import open3d as o3d
import matplotlib.pyplot as plt


def guard_mean_shift(ms, embedding, quantile, iterations, kernel_type="gaussian"):

        while True:
            _, center, bandwidth, cluster_ids = ms.mean_shift(
                embedding, 10000, quantile, iterations, kernel_type=kernel_type
            )
            if torch.unique(cluster_ids).shape[0] > 49:
                quantile *= 1.2
            else:
                break
        return center, bandwidth, cluster_ids

# def continuous_labels(labels_):
#     new_labels = np.zeros_like(labels_)
#     for index, value in enumerate(np.sort(np.unique(labels_))):
#         new_labels[labels_ == value] = index
#     return new_labels

def farthest_point_sampling(coord, num_points):
    """
    Select points using Farthest Point Sampling.
    coord: Point cloud coordinates (N, 3)
    num_points: The number of points to select
    Returns: The indices of the selected points
    """
    N = coord.shape[0]
    farthest_pts = torch.zeros(num_points, dtype=torch.long)
    dist = torch.ones(N) * 1e10  # Initialize distance with a very large number

    farthest_pts[0] = torch.randint(0, N, (1,))  # Randomly select a point
    for i in range(1, num_points):
        # Calculate the distance of each point to the closest selected point
        new_dist = torch.norm(coord - coord[farthest_pts[i-1]], dim=1)
        dist = torch.min(dist, new_dist)  # Update the distance with the minimum distance to the selected points
        farthest_pts[i] = torch.argmax(dist)
    
    return farthest_pts

program_root = os.path.dirname(os.path.abspath(__file__)) + "/"
sys.path.append(program_root + "src")
sys.path.append(program_root + "models")
import torch
from models.tcips import MTLNetBase
from src.segment_loss import EmbeddingLoss
from src.segment_utils import SIOU_matched_segments
from src.segment_utils import to_one_hot, SIOU_matched_segments
from src.mean_shift import MeanShift
from src.segment_utils import SIOU_matched_segments

def save_ply(data, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6] / 255.0)
    o3d.io.write_point_cloud(filename, pcd)

def continuous_labels(labels_):
    unique_labels, counts = np.unique(labels_, return_counts=True)
    sorted_indices = np.argsort(-counts)
    sorted_labels = unique_labels[sorted_indices]
    
    new_labels = np.zeros_like(labels_)
    for index, value in enumerate(sorted_labels):
        new_labels[labels_ == value] = index
    return new_labels

COLORS_TYPE_EDGE = np.array([[120, 120, 120], [200, 60, 60]])
COLORS_TYPE_EDGE = COLORS_TYPE_EDGE.astype(np.float32)

COLORS_TYPE_PRIMITIVE = np.array([[234, 123, 33],
                                  [81, 189, 133],
                                  [195, 48, 44],
                                  [62, 87, 141],
                                  [50, 135, 202],
                                  [147, 127, 183]])
COLORS_TYPE_PRIMITIVE = COLORS_TYPE_PRIMITIVE.astype(np.float32)

is_save = True
is_save_gt = True
save_dir = "primitime_abc_edge"

if_normals = config.normals
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)


model = MTLNetBase(batch_size=1)
model = model.cuda( )

split_dict = {"train": config.num_train, "val": config.num_val, "test": config.num_test}
ms = MeanShift()


test_data = ABCPrimitive_Dataset(split='val', data_root="/home/rh/final/bgpseg_data/ABCPrimitive_purity/", loop=1)

loader_test = torch.utils.data.DataLoader(
    test_data, batch_size=1, num_workers=0, shuffle=False, drop_last=False, collate_fn=collate_fn
)

model.eval()

iterations = 50
quantile = 0.015

state_dict = torch.load(config.pretrain_model_path)
state_dict = {k[k.find(".")+1:]: state_dict[k] for k in state_dict.keys()} if list(state_dict.keys())[0].startswith("module.") else state_dict
model.load_state_dict(state_dict)

test_ResN = []
test_s_iou = []
test_p_iou = []
PredictedLabels = []
PredictedPrims = []
test_s_recall = []
PredictedPurity = []
PredictedOffset = []
save_gt = False

cm_all = ConfusionMatrix(6)
max_point_count = 45000

for val_b_id, data in enumerate(loader_test):
    # if val_b_id == 5:
    #     break
    fn, coord, normals, boundary, label, semantic, param, offset, edges, dse_edges, regional_purity, center_offset = data
    
    if coord.shape[0] > max_point_count:
        fps_idx = farthest_point_sampling(coord, max_point_count)
        coord = coord[fps_idx]
        normals = normals[fps_idx]
        label = label[fps_idx]
        semantic = semantic[fps_idx]
        boundary = boundary[fps_idx]
        offset = torch.tensor([max_point_count])
    
    points_numpy = coord.numpy()
    points = coord.cuda()
    normals = normals.cuda()
    offset = offset.cuda()
    labels = label.numpy()
    primitives_tensor = semantic
    primitives_ = semantic.numpy()
    boundary = boundary.numpy()
    
    primitives_tensor[primitives_tensor == 9] = 0
    primitives_tensor[primitives_tensor == 6] = 0
    primitives_tensor[primitives_tensor == 7] = 0
    primitives_tensor[primitives_tensor == 8] = 2
    
    with torch.no_grad():
        _input = torch.cat([points, normals], 1)
        embedding, primitives_log_prob, purity_pre, offset_pre = model(feat=_input, offset=offset)           

    pred_primitives_tensor = torch.argmax(primitives_log_prob, dim=1).data.cpu() 
    pred_primitives_tensor[pred_primitives_tensor == 9] = 0
    pred_primitives_tensor[pred_primitives_tensor == 6] = 0
    pred_primitives_tensor[pred_primitives_tensor == 7] = 0
    pred_primitives_tensor[pred_primitives_tensor == 8] = 2
    pred_primitives = pred_primitives_tensor.numpy()   

    cm = ConfusionMatrix(6)
    cm.update(pred_primitives_tensor, primitives_tensor)
    cm_all.update(pred_primitives_tensor, primitives_tensor)

    primitives_prob_total = None
    index = None
    total_type_pred = None
  
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

    _, _, cluster_ids = guard_mean_shift(
            ms, embedding, quantile, iterations, kernel_type="gaussian"
        )
    cluster_ids = cluster_ids.data.cpu().numpy()
    
    # # new mean_shift
    # cluster_ids = mean_shift_gpu(embedding)
    
    cluster_ids = continuous_labels(cluster_ids)
    weights = to_one_hot(cluster_ids)
    
    
    ResN = np.abs(np.unique(labels[0]).size - np.unique(cluster_ids).size)
    test_ResN.append(ResN)

    s_iou, p_iou, _, _ = SIOU_matched_segments(    # ================= default is SIOU_matched_segments
        labels,
        cluster_ids,
        pred_primitives,
        primitives_,
        weights,
    )

    tp, union, count = cm.tp, cm.union, cm.count
    miou, macc, oa, ious, accs = get_mious(tp, union, count)
    logger.info(f"ID:{val_b_id} | inst_iou: "+str(s_iou) + " type_iou: "+str(p_iou)+ " miou: "+str(miou)+ " macc: "+str(macc)+ " oa: "+str(oa)) 
    test_s_iou.append(s_iou)
    test_p_iou.append(p_iou)
    # test_s_recall.append(s_recall)
    PredictedLabels.append(cluster_ids)
    PredictedPrims.append(pred_primitives)
    purity_pre = purity_pre.detach().cpu().numpy()
    offset_pre = offset_pre.detach().cpu().numpy()
    PredictedPurity.append(purity_pre)
    PredictedOffset.append(offset_pre)
    
    if is_save == True:
        if is_save_gt:
            if not os.path.exists("/mnt/data/rh/data/{}/gt/".format(save_dir)):
                os.makedirs("/mnt/data/rh/data/{}/gt/".format(save_dir))
            type_vis_file = os.path.join("/mnt/data/rh/data/{}/gt/".format(save_dir),"{}_type_gt.ply".format(val_b_id))
            inst_vis_file = os.path.join("/mnt/data/rh/data/{}/gt/".format(save_dir),"{}_inst_gt.ply".format(val_b_id))
            edge_vis_file = os.path.join("/mnt/data/rh/data/{}/gt/".format(save_dir),"{}_edge_gt.ply".format(val_b_id))
            cluster_ids_gt = labels
            cluster_ids_gt = continuous_labels(cluster_ids_gt)
            inst_vis = visual_labels(points_numpy, cluster_ids_gt.astype(np.compat.long), COLORS_TYPE)
            save_ply(inst_vis, inst_vis_file)
            
            primitives_[primitives_ == 9] = 0
            primitives_[primitives_ == 6] = 0
            primitives_[primitives_ == 7] = 0
            primitives_[primitives_ == 8] = 2

            type_vis = visual_labels(points_numpy, primitives_.astype(np.compat.long), COLORS_TYPE_PRIMITIVE)
            save_ply(type_vis, type_vis_file)
            
            edge_vis = visual_labels(points_numpy, boundary.astype(np.compat.long), COLORS_TYPE_EDGE)
            save_ply(edge_vis, edge_vis_file)
        
        if not os.path.exists("/mnt/data/rh/data/{}/prediction/".format(save_dir)):
            os.makedirs("/mnt/data/rh/data/{}/prediction/".format(save_dir))

        pred_primitives[(pred_primitives==7) | (pred_primitives==6) | (pred_primitives==9)] = 0
        pred_primitives[pred_primitives==8] = 2
        type_vis = visual_labels(points_numpy, pred_primitives.astype(np.compat.long), COLORS_TYPE_PRIMITIVE)
        type_vis_file = os.path.join("/mnt/data/rh/data/{}/prediction/".format(save_dir),"{}_type.ply".format(val_b_id))
        save_ply(type_vis, type_vis_file)

        inst_vis = visual_labels(points_numpy, cluster_ids.astype(np.compat.long), COLORS_TYPE)
        inst_vis_file = os.path.join("/mnt/data/rh/data/{}/prediction/".format(save_dir),"{}_inst.ply".format(val_b_id))
        save_ply(inst_vis, inst_vis_file)
        
        purity_pre = purity_pre.squeeze(axis=1)
        colormap = plt.get_cmap('viridis')
        colors = colormap(purity_pre)
        colors = colors[:, :3]
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points_numpy)
        pc.colors = o3d.utility.Vector3dVector(colors)
        ply_file = os.path.join("/mnt/data/rh/data/{}/prediction/".format(save_dir),"{}_purity.ply".format(val_b_id))
        o3d.io.write_point_cloud(ply_file, pc, write_ascii=True)


# PredictedLabels = np.array(PredictedLabels, dtype=np.int64)
# PredictedPrims = np.array(PredictedPrims, dtype=np.int64)
# PredictedPurity = np.array(PredictedPurity, dtype=np.float32)
# PredictedOffset = np.array(PredictedOffset, dtype=np.float32)
# with h5py.File("/mnt/data/rh/data/prediction_all/predictions_tcips.h5", 'w') as new_file:
#     new_file.create_dataset('seg_id', data=PredictedLabels)
#     new_file.create_dataset('pred_primitives', data=PredictedPrims)
#     new_file.create_dataset('pred_purity', data=PredictedPurity)
#     new_file.create_dataset('pred_offset', data=PredictedOffset)

logger.info("===========> inst_iou: "+str(np.mean(test_s_iou))+"  type_iou: "+str(np.mean(test_p_iou)) +"\n")

tp, union, count = cm_all.tp, cm_all.union, cm_all.count
miou, macc, oa, ious, accs = get_mious(tp, union, count)
logger.info("===========> miou: "+str(miou)+"  macc: "+str(macc)+"  oa: "+str(oa) +"\n")

logger.info("===========> ResN: "+str(np.mean(test_ResN)))
