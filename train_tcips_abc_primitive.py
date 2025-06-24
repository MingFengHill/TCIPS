"""
This scrip trains model to predict per point primitive type.
"""
# nohup ./run_train_tcips_abc_primitive.sh > test.log 2>&1 &
import json
import logging
import nntplib
import os
import sys
from shutil import copyfile
from torch.utils.tensorboard import SummaryWriter
program_root = os.path.dirname(os.path.abspath(__file__)) + "/"
sys.path.append(program_root + "src")
sys.path.append(program_root + "models")
from models.tcips import MTLNetBase
import os
from read_config import Config
config = Config("./configs/config_MTLNet_normal.yml")
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from read_config import Config
from src.segment_loss import (
    EmbeddingLoss,
    LabelSmoothingLoss,
    evaluate_miou,
    primitive_loss,
)
import torch.nn.functional as F
import torch.nn as nn
from src.ABCPrimitive import ABCPrimitive_Dataset, collate_fn, collate_fn_region

model_name = config.model_path.format(
    config.batch_size,
    config.lr,
    config.mode,
    config.knn
)
print(model_name)


if not os.path.exists("trains/{}".format(model_name)):
    os.mkdir("trains/{}/".format(model_name))
    os.mkdir("trains/{}/config".format(model_name))
    os.mkdir("trains/{}/ckpts".format(model_name))

userspace = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")
file_handler = logging.FileHandler(
    "trains/{}".format(model_name)+"/{}.log".format(model_name), mode="a"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(handler)

with open(
        "trains/{}/config".format(model_name)+"/config.json", "w"
) as file:
    json.dump(vars(config), file)
source_file = __file__
destination_file = "trains/{}/config".format(model_name)+"/{}".format(__file__.split("/")[-1])
copyfile(source_file, destination_file)
if_normals = config.normals

if_normal_noise = True

if_jitter_points = config.dataset == "noise"
if if_jitter_points:
    print("USE jitter NOISE!")

print("logs prepared!")


def on_load_checkpoint(model, state_dict) -> None:
        model_state_dict = model.state_dict()
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    logger.info(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    '''
                    state_dict[k] = model_state_dict[k]
                    is_changed = True'''
            else:
                '''
                logger.info(f"Dropping parameter {k}")
                is_changed = True'''
                model_state_dict[k] = state_dict[k]
        return model_state_dict


Loss = EmbeddingLoss(margin=1.0, if_mean_shift=False)

type_smoothCE_loss = torch.nn.NLLLoss()


model = MTLNetBase(batch_size=config.batch_size)



print("model got!")
model = model.cuda()

module_lr = config.lr
transformer_lr = config.lr/10
pediction_head_lr_dict = ('head.embedding_module', 'head.edge_module', 'head.primitives_module')
module_lr_dict = ('block', 'head')
params_lr_modified = [dict(names=[], params=[], lr=module_lr),
                      dict(names=[], params=[], lr=transformer_lr)]
for para_name, model_para in model.named_parameters():
    flag = False
    for n in pediction_head_lr_dict:
        if n in para_name:
            params_lr_modified[0]["names"].append(para_name)
            params_lr_modified[0]["params"].append(model_para)
            flag = True
    if flag:
        continue
    for n in module_lr_dict:
        if n in para_name:
            params_lr_modified[1]["names"].append(para_name)
            params_lr_modified[1]["params"].append(model_para)
            flag = True
    if flag:
        continue
    params_lr_modified[0]["names"].append(para_name)
    params_lr_modified[0]["params"].append(model_para)
    
# for debug
if True:
    output_file = "lr_debug.log"
    with open(output_file, 'w') as file:
        for param_dict in params_lr_modified:
            file.write("=================== learning rate: {} ====================\n".format(param_dict['lr']))
            for name in param_dict['names']:
                file.write(f"{name}\n")

if config.optim=="adam":
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
else:
    print("USE AdamW! L2 weight decay {}!".format(config.weight_decay))
    optimizer = optim.AdamW(params_lr_modified, lr=config.lr, weight_decay=config.weight_decay)



if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

print("model to cuda!")
# ==== load ckpt 
if config.preload_model:
    print("loading from ckpt:", config.pretrain_model_path)

    state_dict = torch.load(config.pretrain_model_path)
    if torch.cuda.device_count() > 1:
        state_dict = {"module."+k: state_dict[k] for k in state_dict.keys()} if not list(state_dict.keys())[0].startswith("module.") else state_dict
    else:
        state_dict = {k[7:]: state_dict[k] for k in state_dict.keys()} if list(state_dict.keys())[0].startswith("module.") else state_dict
    try:
        model.load_state_dict(state_dict)
    except Exception as e: 
        print(e)
        print("load error!")
        new_dict = on_load_checkpoint(model, state_dict)
        model.load_state_dict(new_dict, strict=False)

if config.preload_model and config.pretrain_opti_path != "":
    print("loading from ckpt optimizer:", config.pretrain_opti_path)
    optimizer.load_state_dict(
        torch.load(config.pretrain_opti_path)
    )
    for g in optimizer.param_groups:
        g['lr'] = config.lr

print("model ckpt load!")


# origin ABC parsenet dataset + ours edge combined dataset for train

train_data = ABCPrimitive_Dataset(split='train', data_root="/home/rh/final/bgpseg_data/ABCPrimitive_80/", loop=2)

loader_train = torch.utils.data.DataLoader(
    train_data, batch_size=config.batch_size, num_workers=4, shuffle=True, drop_last=True, persistent_workers=True, collate_fn=collate_fn
)

print("get mixed train data")

# origin ABC parsenet dataset for test

# test_data = ABCPrimitive_Dataset(split='val', data_root="/home/rh/final/bgpseg_data/ABCPrimitive_purity/", loop=1)

# loader_test = torch.utils.data.DataLoader(
#     test_data, batch_size=config.batch_size, num_workers=4, shuffle=False, drop_last=True, persistent_workers=True, collate_fn=collate_fn
# )

print("get mixed test data")

cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
print("current LR: ", cur_lr)


if config.sche == "cos":
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=cur_lr / 20, verbose=True)
elif config.sche == "reduce":
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=config.patience, verbose=True, min_lr=5e-5
    )
elif config.sche == "one":
    scheduler = OneCycleLR(optimizer,
                           max_lr=[module_lr, transformer_lr],
                           pct_start=0.05,
                           anneal_strategy="cos",
                           div_factor=10.0,
                           final_div_factor=100.0,
                           cycle_momentum=True,
                           base_momentum=0.85,
                           max_momentum=0.95,
                           three_phase=False,
                           last_epoch=-1,
                           verbose=False,
                           total_steps=len(loader_train)*config.epochs,
                          )


prev_test_loss = 1e4
prev_inst_embed_loss = 1e4
prev_type_bce_loss = 1e4

eval_inter = config.eval_T

todebug = False

writer = None
if config.is_monitor == True:
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()

cur_inter = 0
for e in range(config.epochs):
    train_emb_losses = []
    train_prim_losses = []
    train_iou = [] 
    train_offset_losses = []
    train_losses = []
    model.train()

    num_iter = 1

    for train_b_id, data in enumerate(loader_train):   # ====================================> 1000 
        # ================== My ABC Edge train
        optimizer.zero_grad()
        losses = 0
        ious = 0
        p_losses = 0
        embed_losses = 0
        offset_losses = 0
        for _ in range(num_iter):
            fn, coord, normals, boundary, label, semantic, param, offset, edges, dse_edges, regional_purity, center_offset = data
            coord, normals, boundary, label, semantic, param, offset, edges, dse_edges, regional_purity, center_offset = coord.cuda(), normals.cuda(), boundary.cuda(), \
                        label.cuda(), semantic.cuda(), param.cuda(), offset.cuda(), edges.cuda(), dse_edges.cuda(), regional_purity.cuda(), center_offset.cuda()
            aux_prim_logprob = None

            input = torch.cat([coord, normals], 1)
            embedding, primitives_log_prob, purity_pred, offset_pred = model(feat=input, offset=offset)
            
            start_idx = 0
            embed_loss = 0
            for index in range(len(offset)):
                end_idx = offset[index]
                segment_labels = label[start_idx:end_idx]
                segment_labels = segment_labels.unsqueeze(0)
                segment_embedding = embedding[start_idx:end_idx]
                segment_embedding = segment_embedding.transpose(0, 1)
                segment_embedding = segment_embedding.unsqueeze(0)
                segment_embed_loss = torch.mean(Loss.triplet_loss(segment_embedding, segment_labels.cpu().numpy()))
                embed_loss += segment_embed_loss
                start_idx = end_idx
            embed_loss = embed_loss / len(offset)
            

            offset_loss = F.l1_loss(offset_pred, center_offset)
            cos_sim = nn.functional.cosine_similarity(offset_pred, center_offset, dim=-1)
            # 求每个batch的平均余弦相似度
            cosine_loss = 1 - cos_sim.mean()
            
            purity_loss = F.l1_loss(purity_pred.squeeze(), regional_purity)

            p_loss = type_smoothCE_loss(primitives_log_prob, semantic)  
            iou = 0
            loss = embed_loss + p_loss*1.5 + offset_loss*8 + purity_loss + cosine_loss*2
            loss.backward()

            losses += loss.data.cpu().numpy() / num_iter
            p_losses += p_loss.data.cpu().numpy() / num_iter
            ious += iou / num_iter
            offset_losses += offset_loss.data.cpu().numpy() / num_iter
            embed_losses += embed_loss.data.cpu().numpy() / num_iter
        optimizer.step()
        if isinstance(scheduler, OneCycleLR):
            scheduler.step()
        train_iou.append(ious)
        train_losses.append(losses)
        train_prim_losses.append(p_losses)
        train_emb_losses.append(embed_losses)
        train_offset_losses.append(offset_losses)

        if config.is_monitor == True:
            learning_rate = optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning Rate', learning_rate,  e * len(loader_train) + train_b_id)
            writer.add_scalar('Loss', loss.item(), e * len(loader_train) + train_b_id)
            writer.add_scalar('Embedding Loss', embed_loss.item(), e * len(loader_train) + train_b_id)
            writer.add_scalar('Primitive Loss', p_loss.item(), e * len(loader_train) + train_b_id)
            writer.add_scalar('Offset Loss', offset_loss.item(), e * len(loader_train) + train_b_id)
            writer.add_scalar('Purity Loss', purity_loss.item(), e * len(loader_train) + train_b_id)
            writer.add_scalar('Cosine Loss', cosine_loss.item(), e * len(loader_train) + train_b_id)

        cur_inter += 1
        print(
            "\rEpoch: {} iter: {}, prim loss: {}, emb loss: {}, iou: {}, offset: {}".format(
                e, train_b_id, p_losses, embed_losses, iou, offset_losses,
            ),
            end="",
        )    

    if isinstance(scheduler, CosineAnnealingLR):
        scheduler.step()

    if config.is_monitor == True:
        print("\n!==== Check model parameters ====!")
        for name, param in model.named_parameters():
            if param.grad is None:
                print(f"Gradient for {name} is None.")
                print(f"Parameter name: {name}, Requires grad: {param.requires_grad}")
                continue
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"Gradient has NaN or Inf values in {name}")
                continue
            grad_max = param.grad.abs().max().item()
            grad_min = param.grad.abs().min().item()
            if grad_max > 1e6:
                print(f"Gradient is too large in {name}")
            if grad_min < -1e6:
                print(f"Gradient is too small in {name}")

        for name, param in model.named_parameters():
            if torch.isnan(param.data).any():
                print(f"NaN values in {name}")
                continue
            if torch.isinf(param.data).any():
                print(f"Inf values in {name}")
                continue
            param_norm = param.norm()
            if param_norm > 1e6:
                print(f"Model parameter is too large in {name}")

        # for name, param in model.named_parameters():
        #     writer.add_histogram(f'{name}/grads', param.grad, e)
        #     writer.add_histogram(f'{name}/weights', param.data, e)
        
    # Save the model at the end of each epoche
    logger.info("save latest, saving model at epoch: {}".format(e))
    torch.save(
        model.state_dict(),
        "trains/{}/ckpts".format(model_name)+"/{}_latest.pth".format(model_name),
    )

if config.is_monitor == True:
    writer.close()



