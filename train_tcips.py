"""
This scrip trains model to predict per point primitive type.
"""
# nohup ./run_train_tcips.sh > test.log 2>&1 &
import json
import logging
import nntplib
import os
import sys
from shutil import copyfile
from tabnanny import verbose

from torch import cosine_embedding_loss, index_put
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
from torch.utils.data import DataLoader

from read_config import Config
from src.dataset_segments import ori_simple_data
from src.dataset_segments_purity import ori_simple_data as ori_simple_data_purity

from src.segment_loss import (
    EmbeddingLoss,
    LabelSmoothingLoss,
    evaluate_miou,
    primitive_loss,
)
###
from src.My_edge_loss import compute_embedding_loss, edge_cls_loss, compute_edge_embedding_loss   # HPNet
import torch.nn.functional as F
import torch.nn as nn

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


try:
    my_knn = config.knn
except:
    my_knn = 64
print("dgcnn knn {}".format(my_knn))

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

# The learning rate is configured with reference to TPv3.
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

mix_train_dataset = ori_simple_data_purity(prefix="/home/rh/recon/", if_normals=if_normals, if_train=True, aug=True, if_offset=True, if_purity=True, neighborhood_size='30', if_weight=False, 
                                           noise=False, noise_level=0)  # ==== 

loader_train = torch.utils.data.DataLoader(
    mix_train_dataset, batch_size=config.batch_size, num_workers=4, shuffle=True, drop_last=True, persistent_workers=True
)

print("get mixed train data")

# origin ABC parsenet dataset for test

mix_test_dataset = ori_simple_data(prefix="/home/rh/recon/", if_normals=if_normals, if_train=False, if_offset=True)

loader_test = torch.utils.data.DataLoader(
    mix_test_dataset, batch_size=config.batch_size, num_workers=4, shuffle=False, drop_last=True, persistent_workers=True
)

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
            points, labels, normals, primitives, edges, edges_W, offset_label, purity_label = data
            points, labels, normals, primitives, edges, edges_W, offset_label, purity_label = points.cuda(), labels.cuda(), normals.cuda(), primitives.cuda(), edges.cuda(), edges_W.cuda(), offset_label.cuda(), purity_label.cuda()
            aux_prim_logprob = None
            if if_normals:
                input = torch.cat([points, normals], 2).transpose(1,2)
                embedding, primitives_log_prob, purity_pred, offset_pred = model(feat=input)
            else:
                embedding, primitives_log_prob, purity_pred, offset_pred = model(points.transpose(1,2))

            embed_loss = torch.mean(Loss.triplet_loss(embedding, labels.cpu().numpy()))
            
            primitives[(primitives==9) | (primitives==6) | (primitives==7)] = 0
            primitives[primitives==8] = 2
            B, N, _ = offset_label.shape
            offset_loss = F.l1_loss(offset_pred.transpose(1, 2), offset_label, reduction='sum') / (B * N)
            
            cos_sim = nn.functional.cosine_similarity(offset_pred.transpose(1, 2), offset_label, dim=-1)
            # 求每个batch的平均余弦相似度
            cosine_loss = 1 - cos_sim.mean()
            
            B, N = purity_label.shape
            purity_pred = purity_pred.transpose(1, 2)
            purity_pred = purity_pred.view(B, N)
            purity_loss = F.l1_loss(purity_pred, purity_label, reduction='sum') / (B * N)

            p_loss = type_smoothCE_loss(primitives_log_prob, primitives)  
            iou = 0
            loss = embed_loss + p_loss*1.5 + offset_loss*5 + purity_loss + cosine_loss
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
        if cur_inter == eval_inter or todebug:
            todebug = False
            cur_inter = 0
            test_emb_losses = []
            test_prim_losses = []
            test_losses = []
            test_iou = []
            model.eval()

            for val_b_id, data in enumerate(loader_test):
                points, labels, normals, primitives, edges, edges_W, offset_label = data
                points, labels, normals, primitives, edges, edges_W, offset_label = points.cuda(), labels.cuda(), normals.cuda(), primitives.cuda(), edges.cuda(), edges_W.cuda(), offset_label.cuda()
                
                with torch.no_grad():
                    aux_prim_logprob = None
                    if if_normals:
                        input = torch.cat([points, normals], 2).transpose(1,2)
                        embedding, primitives_log_prob, _, offset_pred = model(input)

                    else:
                        embedding, primitives_log_prob, _, offset_pred = model(points.transpose(1,2))

                    embed_loss = torch.mean(compute_embedding_loss(embedding.transpose(1, 2), labels)[0])

                    primitives[(primitives==9) | (primitives==6) | (primitives==7)] = 0
                    primitives[primitives==8] = 2
                    p_loss = primitive_loss(primitives_log_prob, primitives)

                    loss = embed_loss + p_loss
                # 计算测试集 prim类别的iou
                iou = evaluate_miou(
                    primitives.data.cpu().numpy(),
                    primitives_log_prob.permute(0, 2, 1).data.cpu().numpy(),
                )
                test_iou.append(iou)
                test_prim_losses.append(p_loss.data.cpu().numpy())
                test_emb_losses.append(embed_loss.data.cpu().numpy())
                test_losses.append(loss.data.cpu().numpy())

                if config.is_monitor == True:
                    writer.add_scalar('Test Loss', loss.item(), e * len(loader_train) + train_b_id)
                    writer.add_scalar('Test Embedding Loss', embed_loss.item(), e * len(loader_train) + train_b_id)
                    writer.add_scalar('Test Primitive Loss', p_loss.item(), e * len(loader_train) + train_b_id)
                    writer.add_scalar('Test IOU', iou.item(), e * len(loader_train) + train_b_id)
            # fix spcov with none gradient
            model.train()

            # torch.cuda.empty_cache()
            print("\n")
            logger.info(
                "Epoch: {}/{} => TrL:{}, TsL:{}, TrP:{}, TsP:{}, TrE:{}, TsE:{}, TrI:{}, TsI:{}, TrOffset {}".format(
                    e,
                    config.epochs,
                    np.mean(train_losses),
                    np.mean(test_losses),
                    np.mean(train_prim_losses),
                    np.mean(test_prim_losses),
                    np.mean(train_emb_losses),
                    np.mean(test_emb_losses),
                    np.mean(train_iou),
                    np.mean(test_iou),
                    np.mean(train_offset_losses),
                )
            )

            my_crition = np.mean(test_emb_losses) + 0.15 * np.mean(test_prim_losses)

            test_emb_losses = np.mean(test_emb_losses)
            test_prim_losses = np.mean(test_prim_losses)

            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(my_crition)
            
            if prev_test_loss > my_crition:
                logger.info("total improvement, saving model at epoch: {}".format(e))
                prev_test_loss = my_crition
                torch.save(
                    model.state_dict(),
                    "trains/{}/ckpts".format(model_name)+"/{}.pth".format(model_name),
                )
            
            if prev_inst_embed_loss > test_emb_losses:
                logger.info("inst improvement, saving model at epoch: {}".format(e))
                prev_inst_embed_loss = test_emb_losses
                torch.save(
                    model.state_dict(),
                    "trains/{}/ckpts".format(model_name)+"/{}_InstBest.pth".format(model_name),
                )

            if prev_type_bce_loss > test_prim_losses:
                logger.info("type improvement, saving model at epoch: {}".format(e))
                prev_type_bce_loss = test_prim_losses
                torch.save(
                    model.state_dict(),
                    "trains/{}/ckpts".format(model_name)+"/{}_TypeBest.pth".format(model_name),
                )

            else:
                torch.save(
                    model.state_dict(),
                    "trains/{}/ckpts".format(model_name)+"/{}_latest.pth".format(model_name),
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



