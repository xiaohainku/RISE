import torch
import torchvision.transforms as T
from PIL import Image
from hubconf import dinov2_vitl14, dinov2_vits14, dinov2_vitb14
import os
import numpy as np
from tqdm import tqdm
import faiss

from utils.adaptive_threshold import HIST_RISE
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dino", type=str, default="vit-l14")
parser.add_argument("--imgsz", type=int, default=476)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/dataset/COD/TrainDataset/Imgs")
parser.add_argument("--cluster_path", type=str, default="/root/autodl-tmp/dataset/COD/TrainDataset/SpectralClustering")
parser.add_argument("--save_path", type=str, default="./prototype")
parser.add_argument("--faiss_device", type=str, default='cuda')

args = parser.parse_args()


def cosine_similarity_batch(a, b):
    dot_product = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    similarity = dot_product / (norm_a * norm_b + 1e-8)
    return similarity


def create_gpu_index_use_single_gpu(datas_embedding):
    res = faiss.StandardGpuResources()  # use a single GPU
    index_flat = faiss.IndexFlatIP(datas_embedding.shape[1])
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index_flat.add(datas_embedding)
    return gpu_index_flat


def create_cpu_index(datas_embedding):
    cpu_index_flat = faiss.IndexFlatIP(datas_embedding.shape[1])
    cpu_index_flat.add(datas_embedding)
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

source_dir = args.data_path
cluster_source_dir = args.cluster_path

img_ls = [i for i in os.listdir(source_dir) if i.endswith('.jpg')]

if dinotype == 'vit-s14':
    dinov2 = dinov2_vits14().to(device)
elif dinotype == 'vit-b14':
    dinov2 = dinov2_vitb14().to(device)
elif dinotype == 'vit-l14':
    dinov2 = dinov2_vitl14().to(device)

transform = T.Compose([
    T.Resize((imgsz, imgsz)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

fore_protos = np.zeros((len(img_ls), embed_dim), dtype='float32')
back_protos = np.zeros((len(img_ls), embed_dim), dtype='float32')

for i, img_name in enumerate(tqdm(img_ls)):
    img_path = os.path.join(source_dir, img_name)
    cluster_path = os.path.join(cluster_source_dir, img_name.split('.')[0] + '.png')

    img = Image.open(img_path).convert('RGB')

    mask = Image.open(cluster_path).convert('L').resize((feat_h, feat_h))
    mask = np.array(mask)
    fore = mask / 255.
    fore = (fore > 0.5).astype(np.float32)
    back = 1. - fore

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feats = dinov2.get_intermediate_layers(img_tensor, reshape=True)[0]
    fore_proto = np.sum(feats.cpu().numpy() * fore, axis=(2, 3)) / (np.sum(fore) + 1e-8).astype(np.float32)
    back_proto = np.sum(feats.cpu().numpy() * back, axis=(2, 3)) / (np.sum(back) + 1e-8).astype(np.float32)

    fore_protos[i, :] = fore_proto
    back_protos[i, :] = back_proto

cos_simi = cosine_similarity_batch(fore_protos, back_protos)
cos_simi_thr_hist = HIST_RISE((cos_simi))
print(f"Cosine Similarity Threshold: {cos_simi_thr_hist}")

os.makedirs(args.save_path, exist_ok=True)
fore_save_path = f"{args.save_path}/fore.npy"
back_save_path = f"{args.save_path}/back.npy"

pixel_fore_protos_ls = []
pixel_back_protos_ls = []

for i, img_name in enumerate(tqdm(img_ls)):
    img_path = os.path.join(source_dir, img_name)
    cluster_path = os.path.join(cluster_source_dir, img_name.split('.')[0] + '.png')

    img = Image.open(img_path).convert('RGB')

    mask = Image.open(cluster_path).convert('L').resize((feat_h, feat_h))
    mask = np.array(mask)
    fore = mask / 255.
    fore = (fore > 0.5).astype(np.float32)
    back = 1. - fore

    if fore.mean() == 0 or fore.mean() == 1:
        print(f"mean for {img_name} is {fore.mean()}")
        continue

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feats = dinov2.get_intermediate_layers(img_tensor, reshape=True)[0]
    fore_proto = np.sum(feats.cpu().numpy() * fore, axis=(2, 3)) / (np.sum(fore) + 1e-8).astype(np.float32)
    back_proto = np.sum(feats.cpu().numpy() * back, axis=(2, 3)) / (np.sum(back) + 1e-8).astype(np.float32)

    cos_simi = np.dot(fore_proto, back_proto.T) / (
            np.linalg.norm(fore_proto, axis=1) * np.linalg.norm(back_proto, axis=1) + 1e-8)
    if cos_simi < cos_simi_thr_hist:
        new_feats = feats.squeeze(0).permute(1, 2, 0).cpu().numpy().astype('float32')

        pixel_fore_feats = new_feats[fore == 1]
        pixel_back_feats = new_feats[back == 1]

        faiss.normalize_L2(pixel_fore_feats)
        faiss.normalize_L2(pixel_back_feats)

        fore_feat_index = create_index(pixel_fore_feats)
        back_feat_index = create_index(pixel_back_feats)

        faiss.normalize_L2(fore_proto)
        faiss.normalize_L2(back_proto)

        _, fore_indices = fore_feat_index.search(back_proto, new_feats.shape[0])
        _, back_indices = back_feat_index.search(fore_proto, new_feats.shape[0])

        pixel_fore_proto = pixel_fore_feats[fore_indices[0][-1:]]
        pixel_back_proto = pixel_back_feats[back_indices[0][-1:]]

        pixel_fore_protos_ls.append(pixel_fore_proto)
        pixel_back_protos_ls.append(pixel_back_proto)

pixel_fore_protos = np.concatenate(pixel_fore_protos_ls, axis=0)
pixel_back_protos = np.concatenate(pixel_back_protos_ls, axis=0)

np.save(fore_save_path, pixel_fore_protos)
np.save(back_save_path, pixel_back_protos)
