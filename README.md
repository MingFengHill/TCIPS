<div style="text-align: center;">
  <img src="./img/logo.png" width="280" />
</div>

![Python](https://img.shields.io/badge/python-3.8-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.1-red)
![CUDA](https://img.shields.io/badge/CUDA-11.8-green)
---
This repository hosts the implementation of TCIPS which is an innovative Transformer-based Cross-task Interactive Primitive Segmentation method.

The code will be uploaded once the paper is accepted for publication.
<div style="text-align: center;">
  <img src="./img/framework.jpg" width="800" />
</div>

## 🛠️ Preparation
(1) Following [PTv3 guidelines](https://github.com/Pointcept/PointTransformerV3),  you need to first initialize a Conda environment and install the dependencies::
```
conda create -n pointcept python=3.8 -y
conda activate pointcept
conda install ninja -y
# Choose version you want here: https://pytorch.org/get-started/previous-versions/
# We use CUDA 11.8 and PyTorch 2.1.0 for our development of PTv3
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric

# spconv (SparseUNet)
# refer https://github.com/traveller59/spconv
pip install spconv-cu118  # choose version match your local cuda version

# Open3D (visualization, optional)
pip install open3d
```
(2) Following [README](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features) in Flash Attention repo and install Flash Attention.

(3) Install the dependencies for TCIPS.
```
./install.sh
```

## 📦 Pretrained Model  
Get our pretrained ABCPrimitive model here → [Download pretrained model](https://drive.google.com/file/d/1JdNndo7-BObTm7mE8ite8rwECIzwFSX_/view?usp=drive_link)

## 📊 Dataset
**Pre-labeled download**  
Get the ABCPrimitive dataset with *regional purity* labels here → [Download ABCPrimitive](https://drive.google.com/file/d/1EXBz08Qh2rkZGJabJDZbZ2R4qRmYSI2i/view?usp=drive_link)

**Generate labels yourself**  
To add regional-purity labels to the raw ABCPrimitive dataset, run the script below:
```
python ./regional_purity/add_regional_purity_2_abc_primitive.py
```
>  When processing the validation dataset, set the argument `is_del_small_ins` to 0.

## 🚀 Train
You can run the training script in the foreground using:
```
./run_train_tcips_abc_primitive.sh
```
To run the script in the background and redirect the output to a log file (test.log), use:
```
nohup ./run_train_tcips_abc_primitive.sh > test.log 2>&1 &
```

## ✔️ Test
You can run the test script in the foreground using:
```
./run_predictions_abc_primitive.sh
```
To run the script in the background and redirect the output to a log file (test.log), use:
```
nohup ./run_predictions_abc_primitive.sh > test.log 2>&1 &
```

## 🙏 Acknowledgment
This work was inspired by the following project: [PointTransformerV3](https://github.com/Pointcept/PointTransformerV3), [ParSeNet](https://github.com/Hippogriff/parsenet-codebase), [HPNet](https://github.com/SimingYan/HPNet), [SED-Net](https://github.com/yuanqili78/SED-Net), [BGPSeg](https://github.com/fz-20/BGPSeg), [DeMT](https://github.com/yangyangxu0/DeMT), [Pointcept](https://github.com/Pointcept/Pointcept).

## 📄 Citation
If you find our work useful in your research, please consider citing:
```
@article{wang2025enhancing,
  title={Enhancing primitive segmentation through transformer-based cross-task interaction},
  author={Wang, Tao and Xi, Weibin and Cheng, Yong and Zhang, Jun and Yin, Ruochen and Yang, Yang},
  journal={Engineering Applications of Artificial Intelligence},
  volume={158},
  pages={111307},
  year={2025},
  publisher={Elsevier}
}
```