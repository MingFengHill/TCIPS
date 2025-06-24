import torch
import torch.nn as nn
import spconv.pytorch as spconv
import copy

import ptf_model


class FeatureExtractor(ptf_model.PointModule):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.fe = ptf_model.PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                in_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key=None,
            )
        )
        self.fe.add(nn.BatchNorm1d(num_features=in_channels, eps=1e-3, momentum=0.01), name="norm")
        self.fe.add(nn.GELU(), name="act")
        self.fe.add(spconv.SubMConv3d(
                        in_channels,
                        out_channels,
                        kernel_size=5,
                        padding=1,
                        bias=False,
                        indice_key=None,),
                    name="Conv1")
        self.fe.add(nn.BatchNorm1d(num_features=out_channels, eps=1e-3, momentum=0.01), name="norm1")
        self.fe.add(nn.GELU(), name="act1")
        # resnet
        self.act = nn.ReLU()

    def forward(self, point):
        shortcut = point.feat
        point = self.fe(point)
        point.feat = shortcut + point.feat
        point.feat = self.act(point.feat)
        # TODO: Whether this step is necessary
        # point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class MTLHead(ptf_model.PointModule):
    def __init__(self,
                 num_primitives=10,
                 batch_size=8,
                 in_channels=64,
                 feat_channels=64,
                 attn_channels=(64, 256),
                 attn_num_head=(4, 16),
                 attn_patch_size=(512, 512),
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 order_index=0,
                 enable_rpe=False,
                 enable_flash=True,
                 upcast_attention=False,
                 upcast_softmax=False,
                 order=("z", "z-trans", "hilbert", "hilbert-trans"),
                 shuffle_orders=True,
                 drop_out=0.0):
        super().__init__()
        self.tasks = ("offset", "instance", "semantic", "purity")
        self.batch_size = batch_size
        self.order = [order] if isinstance(order, str) else order
        self.shuffle_orders = shuffle_orders

        # feature extractor
        self.feature_extractors = nn.ModuleDict({task : FeatureExtractor(in_channels, 
                                                                         feat_channels) 
                                                 for task in self.tasks})

        # feature fusion module
        self.feat_fusion_attn = ptf_model.SerializedAttention(
            channels=attn_channels[1],
            patch_size=attn_patch_size[1],
            num_heads=attn_num_head[1],
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=True,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.ln1 = nn.LayerNorm(attn_channels[1])
        self.mlp = ptf_model.MLP(
            in_channels=attn_channels[1],
            hidden_channels=int(attn_channels[1] * 3),
            out_channels=attn_channels[1],
            act_layer=nn.GELU,
            drop=proj_drop,
        )
        self.ln2 = nn.LayerNorm(attn_channels[1])
        self.dim_reduction_module = nn.Sequential(
            nn.Linear(attn_channels[1], feat_channels),
            nn.BatchNorm1d(feat_channels),
            nn.ReLU(True),
            nn.Dropout(drop_out),
        )

        # task query module
        self.task_querys = nn.ModuleDict({task : ptf_model.TaskQueryAttention(
                                                    channels=attn_channels[0],
                                                    patch_size=attn_patch_size[0],
                                                    num_heads=attn_num_head[0],
                                                    qkv_bias=qkv_bias,
                                                    qk_scale=qk_scale,
                                                    attn_drop=attn_drop,
                                                    proj_drop=proj_drop,
                                                    order_index=order_index,
                                                    enable_rpe=enable_rpe,
                                                    upcast_attention=upcast_attention,
                                                    upcast_softmax=upcast_softmax,
                                                    enable_flash=True,
                                                ) 
                                                for task in self.tasks})
        self.task_mlp = nn.ModuleDict({task : ptf_model.MLP(
                in_channels=feat_channels,
                hidden_channels=int(feat_channels * 3),
                out_channels=feat_channels,
                act_layer=nn.GELU,
                drop=proj_drop,
            )
            for task in self.tasks})
        self.task_norm1 = nn.ModuleDict({task : nn.LayerNorm(feat_channels)
            for task in self.tasks})
        self.task_norm2 = nn.ModuleDict({task : nn.LayerNorm(feat_channels)
            for task in self.tasks})
        
        # task prediction head
        # embedding
        self.embedding_module = nn.Sequential(
            nn.Linear(feat_channels*2, feat_channels*2),
            nn.BatchNorm1d(feat_channels*2),
            nn.ReLU(True),
            nn.Dropout(drop_out),
            nn.Linear(feat_channels*2, feat_channels*2),
        )
        # offset prediction
        self.offset_module = nn.Sequential(
            nn.Linear(feat_channels, feat_channels),
            nn.BatchNorm1d(feat_channels),
            nn.ReLU(True),
            nn.Dropout(drop_out),
            nn.Linear(feat_channels, 3),
        )
        # primitives prediction
        self.primitives_module = nn.Sequential(
            nn.Linear(feat_channels, feat_channels),
            nn.BatchNorm1d(feat_channels),
            nn.ReLU(True),
            nn.Dropout(drop_out),
            nn.Linear(feat_channels, num_primitives),
            nn.LogSoftmax(dim=1)
        )
        # purity prediction
        self.purity_module = nn.Sequential(
            nn.Linear(feat_channels, feat_channels),
            nn.BatchNorm1d(feat_channels),
            nn.ReLU(True),
            nn.Dropout(drop_out),
            nn.Linear(feat_channels, 1),
        )

    def forward(self, feat):
        # Preparation: Replication feature
        ori_feats = {}
        ori_point_dict = dict(
            coord=feat.coord,
            feat=feat.feat,
            grid_size=feat.grid_size,
            offset=feat.offset,
            batch=feat.batch,
            serialized_depth=feat.serialized_depth,
            serialized_code=feat.serialized_code,
            serialized_order=feat.serialized_order,
            serialized_inverse=feat.serialized_inverse,
            sparse_shape=feat.sparse_shape,
            sparse_conv_feat=feat.sparse_conv_feat,
            # pad=feat.pad,
            # unpad=feat.unpad,
            # cu_seqlens_key=feat.cu_seqlens_key,
        )
        ori_feats["offset"] = ptf_model.Point(ori_point_dict)
        ori_feats["instance"] = ptf_model.Point(ori_point_dict)
        ori_feats["semantic"] = ptf_model.Point(ori_point_dict)
        ori_feats["purity"] = ptf_model.Point(ori_point_dict)
        
        # phase 1: Extract the primary feature of each task
        primary_feats = {}
        for name, feature_extractor in self.feature_extractors.items():
            primary_feats[name] = feature_extractor(ori_feats[name])

        # phase 2: Fuse the features of different tasks
        point_dict = dict(
            coord=primary_feats[self.tasks[0]].coord,
            feat=torch.cat((primary_feats[self.tasks[0]].feat, 
                                      primary_feats[self.tasks[1]].feat, 
                                      primary_feats[self.tasks[2]].feat,
                                      primary_feats[self.tasks[3]].feat,), 
                                      dim=1),
            grid_size=primary_feats[self.tasks[0]].grid_size,
            offset=primary_feats[self.tasks[0]].offset,
            batch=primary_feats[self.tasks[0]].batch,
            serialized_depth=primary_feats[self.tasks[0]].serialized_depth,
            serialized_code=primary_feats[self.tasks[0]].serialized_code,
            serialized_order=primary_feats[self.tasks[0]].serialized_order,
            serialized_inverse=primary_feats[self.tasks[0]].serialized_inverse,
        )
        fusion_feat = ptf_model.Point(point_dict)
        shortcut = fusion_feat.feat
        fusion_feat.feat = self.ln1(fusion_feat.feat)
        fusion_feat = self.feat_fusion_attn(fusion_feat)
        fusion_feat.feat = shortcut + fusion_feat.feat
        shortcut = fusion_feat.feat
        fusion_feat.feat = self.ln2(fusion_feat.feat)
        fusion_feat.feat = self.mlp(fusion_feat.feat)
        fusion_feat.feat = shortcut + fusion_feat.feat
        fusion_feat.feat = self.dim_reduction_module(fusion_feat.feat)

        # phase 3: task query
        # TODO: whether it is necessary?
        primary_feats_bak = {}
        for task in self.tasks:
            primary_feats_bak[task] = primary_feats[task].feat
        final_feats = {}
        for name, task_query in self.task_querys.items():
            primary_feats[name].feat = self.task_norm1[name](primary_feats[name].feat)
            final_feat = task_query(primary_feats[name], fusion_feat)
            final_feat.feat = final_feat.feat + primary_feats_bak[name]
            shortcut = final_feat.feat
            final_feat.feat = self.task_norm2[name](final_feat.feat)
            final_feat.feat = self.task_mlp[name](final_feat.feat)
            final_feat.feat = final_feat.feat + shortcut
            final_feats[name] = final_feat

        # phase 4: Task related head
        # embedding
        embedding_pre = torch.cat((final_feats["offset"].feat, 
                                   final_feats["instance"].feat),
                                   dim=1)
        embedding_pre = self.embedding_module(embedding_pre)
        # offset prediction
        offset_pre = final_feats["offset"].feat
        offset_pre = self.offset_module(offset_pre)
        # primitives prediction
        primitives_pre = final_feats["semantic"].feat
        primitives_pre = self.primitives_module(primitives_pre)
        # purity prediction
        purity_pre = final_feats["purity"].feat
        purity_pre = self.purity_module(purity_pre)

        return [embedding_pre, primitives_pre, purity_pre, offset_pre]

class MTLNetBase(ptf_model.PointModule):
    def __init__(self, 
                 batch_size=8):
        super().__init__()
        self.batch_size = batch_size
        self.backbone = ptf_model.PointTransformerV3(stride=(2, 2, 2),
                                                    enc_depths=(2, 2, 6, 2),
                                                    enc_channels=(64, 128, 256, 512),
                                                    enc_num_head=(4, 8, 16, 32),
                                                    enc_patch_size=(512, 512, 512, 512),
                                                    dec_depths=(2, 2, 2),
                                                    dec_channels=(64, 128, 256),
                                                    dec_num_head=(4, 8, 16),
                                                    dec_patch_size=(512, 512, 512),
                                                    mlp_ratio=4,
                                                    qkv_bias=True,
                                                    qk_scale=None,
                                                    attn_drop=0.0,
                                                    proj_drop=0.0,
                                                    drop_path=0.3,
                                                    shuffle_orders=True,
                                                    pre_norm=True,
                                                    enable_rpe=False,
                                                    enable_flash=True,
                                                    upcast_attention=False,
                                                    upcast_softmax=False,
                                                    cls_mode=False,
                                                    pdnorm_bn=False,
                                                    pdnorm_ln=False,
                                                    pdnorm_decouple=False,
                                                    pdnorm_adaptive=False,
                                                    pdnorm_affine=True,)
        self.head = MTLHead(batch_size=batch_size)

    def forward(self, feat, offset):
        data_dict = self.abc_primitive_adaptation(feat, offset)
        feat = self.backbone(data_dict)
        out = self.head(feat)
        return out

    def abc_primitive_adaptation(self, feat, offset, grid_size=0.01):
        coord = feat[:, 0:3]

        data_dict = dict(
            coord=coord,
            feat=feat,
            grid_size=grid_size,
            offset=offset,
        )
        return data_dict
