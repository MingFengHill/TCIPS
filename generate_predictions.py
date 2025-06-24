# nohup ./run_predictions.sh > test.log 2>&1 &
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
from src.dataset_segments import ori_simple_data
from metrics import ConfusionMatrix, get_mious
program_root = os.path.dirname(os.path.abspath(__file__)) + "/"
sys.path.append(program_root + "src")
sys.path.append(program_root + "models")
import torch
from models.tcips import MTLNetBase
from src.segment_utils import to_one_hot, SIOU_matched_segments
from src.mean_shift import MeanShift

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

def continuous_labels(labels_):
    new_labels = np.zeros_like(labels_)
    for index, value in enumerate(np.sort(np.unique(labels_))):
        new_labels[labels_ == value] = index
    return new_labels

# test configs
starts = 0  # default 0
iterations = 50
quantile = 0.015
if_normals = config.normals

# logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")

fn = "TEST_TCIPS_{}".format(config.pretrain_model_path.split("/")[-1])
file_handler = logging.FileHandler(
    f"./predictions/logs/{fn}.log", mode="a"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(handler)

with open(
        "./predictions/config/cfg_{}.json".format(fn), "w"
) as file:
    json.dump(vars(config), file)
source_file = __file__
destination_file = "./predictions/config/code_{}_{}".format(
    fn, __file__.split("/")[-1]
)
copyfile(source_file, destination_file)

model = MTLNetBase(batch_size=1)
model = model.cuda()

ms = MeanShift()
mix_test_dataset = ori_simple_data(prefix="/home/rh/recon/", if_normals=if_normals, if_train=False, aug=True, starts=starts)
loader_test = torch.utils.data.DataLoader(
    mix_test_dataset, batch_size=1, num_workers=0, shuffle=False, drop_last=False
)
model.eval()
state_dict = torch.load(config.pretrain_model_path)
state_dict = {k[k.find(".")+1:]: state_dict[k] for k in state_dict.keys()} if list(state_dict.keys())[0].startswith("module.") else state_dict
model.load_state_dict(state_dict)

test_ResN = []
test_s_iou = []
test_p_iou = []
PredictedLabels = []
PredictedPrims = []
PredictedPurity = []
PredictedOffset = []
save_gt = False

cm = ConfusionMatrix(6)

for val_b_id, data in enumerate(loader_test):
    # if val_b_id == 19:
    #     break
    points_, labels, normals_, primitives_ = data[:4]
    
    points = points_.cuda( )
    normals = normals_.cuda( )
    labels = labels.numpy()
    primitives_tensor = primitives_[0]
    primitives_ = primitives_.numpy()
    
    with torch.no_grad():
        _input = torch.cat([points, normals], 2)
        embedding, primitives_log_prob, purity_pre, offset_pre = model(_input.permute(0, 2, 1))           

    pred_primitives_tensor = torch.max(primitives_log_prob[0], 0)[1].data.cpu()
    pred_primitives = pred_primitives_tensor.numpy()
    
    primitives_tensor[primitives_tensor == 9] = 0
    primitives_tensor[primitives_tensor == 6] = 0
    primitives_tensor[primitives_tensor == 7] = 0
    primitives_tensor[primitives_tensor == 8] = 2
    cm.update(pred_primitives_tensor, primitives_tensor)

    embedding = torch.nn.functional.normalize(embedding[0].T, p=2, dim=1)
    _, _, cluster_ids = guard_mean_shift(
            ms, embedding, quantile, iterations, kernel_type="gaussian"
        )
    weights = to_one_hot(cluster_ids, np.unique(cluster_ids.data.data.cpu().numpy()).shape[
        0])
    cluster_ids = cluster_ids.data.cpu().numpy()
    cluster_ids = continuous_labels(cluster_ids)
    
    ResN = np.abs(np.unique(labels[0]).size - np.unique(cluster_ids).size)
    test_ResN.append(ResN)

    s_iou, p_iou, _, _ = SIOU_matched_segments(    # ================= default is SIOU_matched_segments
        labels[0],
        cluster_ids,
        pred_primitives,
        primitives_[0],
        weights,
    )

    logger.info(f"ID:{val_b_id+starts} | inst_iou: "+str(s_iou) + " type_iou: "+str(p_iou)) 
    test_s_iou.append(s_iou)
    test_p_iou.append(p_iou)
    PredictedLabels.append(cluster_ids)
    PredictedPrims.append(pred_primitives)
    purity_pre = purity_pre.detach().cpu().numpy()
    purity_pre = np.squeeze(purity_pre)
    offset_pre = offset_pre.detach().cpu().numpy()
    offset_pre = np.squeeze(offset_pre)
    PredictedPurity.append(purity_pre)
    PredictedOffset.append(offset_pre)

PredictedLabels = np.array(PredictedLabels, dtype=np.int64)
PredictedPrims = np.array(PredictedPrims, dtype=np.int64)
PredictedPurity = np.array(PredictedPurity, dtype=np.float32)
PredictedOffset = np.array(PredictedOffset, dtype=np.float32)
# with h5py.File("/mnt/data/rh/data/prediction_all/predictions_tcips.h5", 'w') as new_file:
#     new_file.create_dataset('seg_id', data=PredictedLabels)
#     new_file.create_dataset('pred_primitives', data=PredictedPrims)
#     new_file.create_dataset('pred_purity', data=PredictedPurity)
#     new_file.create_dataset('pred_offset', data=PredictedOffset)

logger.info("===========> inst_iou: "+str(np.mean(test_s_iou))+"  type_iou: "+str(np.mean(test_p_iou)) +"\n")

tp, union, count = cm.tp, cm.union, cm.count
miou, macc, oa, ious, accs = get_mious(tp, union, count)
logger.info("===========> miou: "+str(miou)+"  macc: "+str(macc)+"  oa: "+str(oa) +"\n")

logger.info("===========> ResN: "+str(np.mean(test_ResN)))
