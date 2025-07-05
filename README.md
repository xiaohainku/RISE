# Beyond Single Images: Retrieval Self-Augmented Unsupervised Camouflaged Object Detection

> [**Beyond Single Images: Retrieval Self-Augmented Unsupervised Camouflaged Object Detection**](https://github.com/xiaohainku/RISE)
>
> **ICCV 2025**
>
> **NKU & PolyU**


## Installation

1. Create the Virtual Environment

```bash
conda create -n RISE python=3.8
conda activate RISE
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
git clone https://github.com/xiaohainku/RISE.git
cd RISE
pip install -r requirement.txt
```

2. Install FAISS package for efficient retrieval

Please refer to [Installing FAISS](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) for more details.

Install gpu version of faiss:

```bash
conda install -c pytorch -c nvidia faiss-gpu=1.7.3
```

If you have difficulty in installing the gpu version of faiss, try to install the cpu verison:

```bash
pip install faiss-cpu==1.7.4
```

If you have installed the cpu version, remember to change the "--faiss_device" argument in [retrieval.py](/retrieval.py)  from "cuda" to "cpu".



## Prepare datasets

Cluster maps, pseudo masks, and predicted maps are available at [Google](https://drive.google.com/file/d/1ZxKU6AekCHQCyNAo2bPIptHX-m3eNvDT/view?usp=drive_link) | [Quark](https://pan.quark.cn/s/9f0a30f67b84?pwd=cE8u) | [Baidu](https://pan.baidu.com/s/1VoBZp0DzQCdfgvBsas1QaQ?pwd=6uks).

## Evaluation

RISE consists of two stages: generating pseudo masks and training [SINet](https://github.com/GewelsJI/SINet-V2) based on pseudo masks.

### 1. Generating pseudo masks

Step 1: Spectral clustering

```shell
python spectral-clustering.py \
--data_path /your/train/dataset/path \
--save_path /your/cluster/map/save/path
```

Step 2: Generate prototypes

```shell
python gen-proto.py \
--data_path /your/train/dataset/path \
--cluster_path /your/cluster/map/save/path \
--save_path /your/prototype/save/path
```

Step 3: Retrieval (generate pseudo mask)

```shell
python retrieval.py \
--data_path /your/train/dataset/path \
--prototype_path /your/prototype/save/path \
-save_path /your/pseudo/mask/save/path
```

### 2. Train SINet

Step 1: Training

```shell
cd SINet
python MyTrain_Val.py \
--img_root /your/train/image/path/ \
--gt_root /your/pseudo/mask/path \
--val_root /your/validation/dataset/path/ \
--save_path /your/checkpoint/save/path/
```

Step 2: Evaluation

```shell
python MyTesting.py \
--pth_path /your/checkpoint/save/path/ \
--data_path /your/test/dataset/path \
--save_dir /your/predicted/map/save/path
```

## Acknowledgements

RISE builds upon self-supervised vision foundation model [DINOv2](https://github.com/facebookresearch/dinov2) and retrieval-based unsupervised COD method [EASE](https://github.com/xiaohainku/EASE). Thanks for their elegant work.

## Citing

If you find our work interesting, please consider using the following BibTeX entry:

```latex
@INPROCEEDINGS{RISE,
  title={Beyond Single Images: Retrieval Self-Augmented Unsupervised Camouflaged Object Detection},
  author={Du, Ji and Wang, Xin and Hao, Fangwei and Yu, Mingyang and Chen, Chunyuan and Wu, Jiesheng and Wang, Bin and Xu, Jing and Li, Ping},
  booktitle={ICCV},
  year={2025}
}
```

