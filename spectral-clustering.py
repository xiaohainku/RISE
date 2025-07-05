import torchvision.transforms as T
from PIL import Image
from hubconf import dinov2_vitl14, dinov2_vits14, dinov2_vitb14
import os
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from vis import *
import torch

from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dino", type=str, default="vit-l14")
parser.add_argument("--imgsz", type=int, default=476)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/dataset/COD/TrainDataset/Imgs")
parser.add_argument("--save_path", type=str, default="/root/autodl-tmp/dataset/COD/TrainDataset/SpectralClustering")

args = parser.parse_args()


def Spectral_Cluster(embeding, affinity=None):
    spectral = SpectralClustering(n_clusters=2, affinity=affinity)
    labels = spectral.fit_predict(embeding)
    return labels


def count_edge_point(mask):
    zero_nums = 0
    one_nums = 0

    zero_nums += (mask[0, :] == 0).sum()
    one_nums += (mask[0, :] == 1).sum()

    zero_nums += (mask[-1, :] == 0).sum()
    one_nums += (mask[-1, :] == 1).sum()

    zero_nums += (mask[1:-1, 0] == 0).sum()
    one_nums += (mask[1:-1, 0] == 1).sum()

    zero_nums += (mask[1:-1, -1] == 0).sum()
    one_nums += (mask[1:-1, -1] == 1).sum()

    assert zero_nums + one_nums == (2 * (mask.shape[0] + mask.shape[1]) - 4)

    return zero_nums, one_nums


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

imgs = [i for i in os.listdir(source_dir) if i.endswith(".jpg")]

for img_name in tqdm(imgs):
    img_path = os.path.join(source_dir, img_name)
    save_path = os.path.join(mask_source_dir, img_name.split('.')[0] + '.png')

    img = Image.open(img_path).convert("RGB")

    width, height = img.size
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feats = dinov2.get_intermediate_layers(img_tensor, reshape=True)[0]
    local_query = feats.permute(2, 3, 1, 0).reshape(feat_h * feat_h, embed_dim,
                                                    -1).contiguous().squeeze().cpu().numpy()

    cosine_sim_matrix = cosine_similarity(local_query)
    cosine_sim_matrix = np.maximum(cosine_sim_matrix, 0)
    labels = Spectral_Cluster(cosine_sim_matrix, affinity="precomputed")

    labels = labels.reshape(-1, 1)

    labels = labels.reshape(feat_h, feat_h)
    zero_nums_at_edge, one_nums_at_edge = count_edge_point(labels)

    if one_nums_at_edge > zero_nums_at_edge:
        labels = 1 - labels

    labels = labels.astype(np.float32)
    labels = F.interpolate(torch.from_numpy(labels).unsqueeze(0).unsqueeze(0),
                           size=(height, width),
                           mode='bilinear', align_corners=True).squeeze().numpy()
    labels = (labels > 0.5).astype(np.uint8)

    Image.fromarray(labels * 255, mode="L").save(save_path)
