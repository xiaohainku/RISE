import faiss
import torch
import torchvision.transforms as T
from PIL import Image
from hubconf import dinov2_vitl14, dinov2_vits14, dinov2_vitb14
import os
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from vis import *

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--top_k", type=int, default=512)
parser.add_argument("--dino", type=str, default="vit-l14")
parser.add_argument("--imgsz", type=int, default=476)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/dataset/COD/TrainDataset/Imgs")
parser.add_argument("--prototype_path", type=str, default="./prototype")
parser.add_argument("--save_path", type=str, default="/root/autodl-tmp/dataset/COD/TrainDataset/PseudoMask")
parser.add_argument("--faiss_device", type=str, default='cuda')

args = parser.parse_args()


def cosine_similarity_batch(a, b):
    dot_product = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    similarity = dot_product / (norm_a * norm_b + 1e-8)
    return similarity


def create_gpu_index_use_single_gpu(fore, back):
    res = faiss.StandardGpuResources()  # use a single GPU
    assert fore.shape[1] == back.shape[1]
    index_flat = faiss.IndexFlatIP(fore.shape[1])
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index_flat.add(fore)
    gpu_index_flat.add(back)
    return gpu_index_flat


def create_cpu_index(fore, back):
    cpu_index_flat = faiss.IndexFlatIP(fore.shape[1])
    cpu_index_flat.add(fore)
    cpu_index_flat.add(back)
    return cpu_index_flat


if args.faiss_device == 'cuda':
    create_index = create_gpu_index_use_single_gpu
elif args.faiss_device == 'cpu':
    create_index = create_cpu_index

dino_dim = {
    'vit-s14': 384,
    'vit-b14': 768,
    'vit-l14': 1024,
}

dinotype = args.dino
imgsz = args.imgsz
feat_h = int(imgsz / 14)
embed_dim = dino_dim[dinotype]
device = args.device

topk = args.top_k

transform = T.Compose([
    T.Resize((imgsz, imgsz)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

if dinotype == 'vit-s14':
    dinov2 = dinov2_vits14().to(device)
elif dinotype == 'vit-b14':
    dinov2 = dinov2_vitb14().to(device)
elif dinotype == 'vit-l14':
    dinov2 = dinov2_vitl14().to(device)

source_dir = args.data_path
mask_source_dir = args.save_path
os.makedirs(mask_source_dir, exist_ok=True)

imgs = [i for i in os.listdir(source_dir) if i.endswith('.jpg')]

import torchvision.transforms.functional as TF

aug_transforms = {
    "normal": lambda x: x,
    "r90": lambda x: TF.rotate(x, 90, expand=True),
    "r180": lambda x: TF.rotate(x, 180, expand=True),
    "r270": lambda x: TF.rotate(x, 270, expand=True),
    "hflip": lambda x: TF.hflip(x),
    "vflip": lambda x: TF.vflip(x),
}

inv_aug_transforms = {
    "normal": lambda x: x,
    "r90": lambda x: np.rot90(x, k=3),
    "r180": lambda x: np.rot90(x, k=2),
    "r270": lambda x: np.rot90(x, k=1),
    "hflip": lambda x: np.fliplr(x),
    "vflip": lambda x: np.flipud(x),
}

pixel_fore_embed = np.load(f"{args.prototype_path}/fore.npy")
pixel_back_embed = np.load(f"{args.prototype_path}/back.npy")

faiss.normalize_L2(pixel_fore_embed)
faiss.normalize_L2(pixel_back_embed)

pixel_fore_proto_num = pixel_fore_embed.shape[0]

assert topk <= pixel_fore_proto_num

if topk <= 2048:
    pixel_index = create_index(pixel_fore_embed, pixel_back_embed)
else:
    pixel_index = create_cpu_index(pixel_fore_embed, pixel_back_embed)

for img_name in tqdm(imgs):
    img_path = os.path.join(source_dir, img_name)
    save_path = os.path.join(mask_source_dir, img_name.split('.')[0] + '.png')

    fusion_mask = 0

    for aug_name, aug in aug_transforms.items():
        img = Image.open(img_path).convert("RGB")

        img = aug(img)
        width, height = img.size
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            feats = dinov2.get_intermediate_layers(img_tensor, reshape=True)[0]  # bs x dim x h x w
        local_query = feats.permute(2, 3, 1, 0).reshape(feat_h * feat_h, embed_dim,
                                                        -1).contiguous().squeeze().cpu().numpy()

        faiss.normalize_L2(local_query)

        # pixel-level
        _, pixel_indices = pixel_index.search(local_query, topk)
        pixel_mask = np.sum(pixel_indices < pixel_fore_proto_num, axis=1) / topk
        pixel_mask = pixel_mask.astype(np.float32).reshape(feat_h, feat_h)
        pixel_mask = F.interpolate(torch.tensor(pixel_mask).unsqueeze(0).unsqueeze(0),
                                   size=(height, width),
                                   mode='bilinear', align_corners=True).squeeze().numpy()

        aug_mask = inv_aug_transforms[aug_name](pixel_mask)
        fusion_mask += aug_mask

    fusion_mask /= len(aug_transforms.keys())
    fusion_mask = (fusion_mask > 0.5).astype(np.uint8)
    Image.fromarray(fusion_mask * 255, mode='L').save(save_path)

