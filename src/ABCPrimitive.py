import os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
import math
from collections import Counter
from src.augment_utils import PointTransformerAugment, MyAugment, rotate_perturbation_point_cloud, jitter_point_cloud, shift_point_cloud, \
    random_scale_point_cloud, rotate_point_cloud, PointTransformerValidateAugment

def collate_fn(batch):
    fn, coord, normals, boundary, label, semantic, param, F, edges, dse_edges, regional_purity, center_offset = list(zip(*batch))
    offset, count = [], 0
    # print("coord:", len(coord))
    for item in coord:
        # print("item shape:",item.shape)
        count += item.shape[0]
        offset.append(count)

    return fn, torch.cat(coord), torch.cat(normals), torch.cat(boundary), torch.cat(label), torch.cat(semantic), torch.cat(param), torch.IntTensor(offset), torch.cat(edges), torch.cat(dse_edges), torch.cat(regional_purity), torch.cat(center_offset)

def collate_fn_region(batch):
    fn, coord, normals, boundary, label, semantic, param, F, edges, dse_edges = list(zip(*batch))
    offset, count = [], 0
    # print("coord:", len(coord))
    for item in coord:
        # print("item shape:",item.shape)
        count += item.shape[0]
        offset.append(count)

    F_offset, count = [], 0
    for item in F:
        # print("item shape:",item.shape)
        count += item.shape[0]
        F_offset.append(count)
    return fn, torch.cat(coord), torch.cat(normals), torch.cat(boundary), torch.cat(label), torch.cat(semantic), torch.cat(param), torch.IntTensor(offset), torch.cat(edges), torch.cat(dse_edges), torch.cat(F), torch.IntTensor(F_offset)


def data_prepare_abcprimitive(coord, normals, boundary, label, semantic, param, F, edges, dse_edges):

    coord_min = np.min(coord, 0)
    coord -= coord_min
    label -= 1  
    # set small number primitive as background
    counter = Counter(label)
    mapper = np.ones([label.max() + 1]) * -1
    keys = [k for k, v in counter.items() if v > 100]
    if len(keys):
        mapper[keys] = np.arange(len(keys))
    label = mapper[label]
    clean_primitives = np.ones_like(semantic) * -1
    valid_mask = label != -1
    clean_primitives[valid_mask] = semantic[valid_mask]
    semantic = clean_primitives.astype(int)
    label = label.astype(int)
    coord = torch.FloatTensor(coord)
    normals = torch.FloatTensor(normals)
    boundary = torch.LongTensor(boundary)
    semantic = torch.LongTensor(semantic)
    param = torch.FloatTensor(param)
    label = torch.LongTensor(label)
    F = torch.LongTensor(F)
    edges = torch.IntTensor(edges)
    dse_edges = torch.IntTensor(dse_edges)
    return coord, normals, boundary, label, semantic, param, F, edges, dse_edges

def data_prepare_abcprimitive_val(coord, normals, boundary, label, semantic, param, F, edges, dse_edges, regional_purity, center_offset):

    coord_min = np.min(coord, 0)
    coord -= coord_min
    
    coord = torch.FloatTensor(coord)
    normals = torch.FloatTensor(normals)
    boundary = torch.LongTensor(boundary)
    semantic = torch.LongTensor(semantic)
    param = torch.FloatTensor(param)
    label = torch.LongTensor(label)
    F = torch.LongTensor(F)
    edges = torch.IntTensor(edges)
    dse_edges = torch.IntTensor(dse_edges)
    regional_purity = torch.FloatTensor(regional_purity)
    center_offset = torch.FloatTensor(center_offset)
    
    return coord, normals, boundary, label, semantic, param, F, edges, dse_edges, regional_purity, center_offset


class ABCPrimitive_Dataset(Dataset):
    def __init__(self, split='train', data_root='trainval', loop=1):
        super().__init__()
        self.split, self.loop = split, loop
        if split == 'train':
            data_root += '/train/'
        elif split == 'val' or split == 'test':
            data_root += '/val/'
        data_list = sorted(os.listdir(data_root))
        self.data_list = [item[:-4] for item in data_list]
        self.data_root = data_root
        
        self.data_idx = np.arange(len(self.data_list))
        print("Totally {} samples in {} set.".format(len(self.data_idx), split))
        
        if split == "train":
            self.if_train = True
        else:
            self.if_train = False
        self.aug = True
        self.myAug = PointTransformerAugment()
        self.myValAug = PointTransformerValidateAugment()

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]

        item = self.data_list[data_idx]
        data_path = os.path.join(self.data_root, item + '.npz')
        data = np.load(data_path)

        coord, normals, boundary, label, semantic, param, F, edges, dse_edges = data['V'],data['N'],data['B'],data['L'],data['S'],data['T_param'],data['F'],data['edges'],data['dse_edges']
        
        regional_purity = data['RP']
        
        N = coord.shape[0]
        if self.if_train and self.aug:
            tmp = self.myAug.augment([coord.reshape(1, N, 3), normals.reshape(1, N, 3)])
            coord = tmp[0][0]
            normals = tmp[1][0]

        if not self.if_train and self.aug:
            tmp = self.myValAug.augment([coord.reshape(1, N, 3), normals.reshape(1, N, 3)])
            coord = tmp[0][0]
            normals = tmp[1][0]
        
        
        pt_mean = np.ones((coord.shape[0], 3), dtype=np.float32) * -10000.0
        instance_pointnum = []
        instance_cls = []
        instance_num = max(int(label.max()) + 1, 0)
        for i_ in range(instance_num):
            inst_idx_i = np.where(label == i_)
            xyz_i = coord[inst_idx_i]
            inst_pt_size = len(inst_idx_i[0])
            if inst_pt_size == 0:
                continue
            pt_mean[inst_idx_i] = xyz_i.mean(0)
            instance_pointnum.append(inst_idx_i[0].size)
            cls_idx = inst_idx_i[0][0]
            instance_cls.append(label[cls_idx])
        has_negative = np.any(np.isin(pt_mean, -10000.0))
        if has_negative:
            exit
        pt_offset_label = pt_mean - coord
        pt_offset_label = pt_offset_label.astype(np.float32)

        # noise = normals * np.clip(
        #     np.random.randn(coord.shape[0], 1) * 0.01,
        #     a_min=-0.01,
        #     a_max=0.01)
        # coord_noise = coord + noise.astype(np.float32)
        # coord_min = np.min(coord_noise, 0)
        # coord_noise -= coord_min
        # coord_noise = torch.FloatTensor(coord_noise)

        if self.split == 'train':
            # coord, normals, boundary, label, semantic, param, F, edges, dse_edges = data_prepare_abcprimitive(coord, normals, boundary, label, semantic, param, F, edges, dse_edges)
            coord, normals, boundary, label, semantic, param, F, edges, dse_edges, regional_purity, center_offset = data_prepare_abcprimitive_val(coord, normals, boundary, label, semantic, param, F, edges, dse_edges, regional_purity, pt_offset_label)
        else:
            coord, normals, boundary, label, semantic, param, F, edges, dse_edges, regional_purity, center_offset = data_prepare_abcprimitive_val(coord, normals, boundary, label, semantic, param, F, edges, dse_edges, regional_purity, pt_offset_label)

        return item, coord, normals, boundary, label, semantic, param, F, edges, dse_edges, regional_purity, center_offset

    def __len__(self):
        return round(len(self.data_idx) * self.loop)