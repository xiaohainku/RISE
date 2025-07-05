import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import cv2
from lib.Network_Res2Net_GRA_NCD import Network
from utils.data_val import test_dataset
from PIL import Image
from vis import *
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./snapshot/baseline/Net_epoch_best.pth')
parser.add_argument('--save_dir', type=str, default='./pred/baseline')
parser.add_argument('--data_path', type=str, default="/root/autodl-tmp/dataset/COD/dataset/TestDataset")
opt = parser.parse_args()

model = Network(imagenet_pretrained=False)
model.load_state_dict(torch.load(opt.pth_path))
model.cuda()
model.eval()

for _data_name in ['CHAMELEON', 'CAMO', 'COD10K', 'NC4K']:

    metric = EvaluationMetricsV2()

    data_path = f"{opt.data_path}/{_data_name}"
    save_path = os.path.join(opt.save_dir, _data_name)
    os.makedirs(save_path, exist_ok=True)

    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in tqdm(range(test_loader.size)):
        image, gt, name, _ = test_loader.load_data()
        gt = np.array(gt)
        image = image.cuda()

        res5, res4, res3, res2 = model(image)
        res = res2
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        metric.step(pred=res, gt=gt)

        Image.fromarray((res*255).astype(np.uint8)).save(os.path.join(save_path, name))

    metric_dic = metric.get_results()

    sm = float(metric_dic['sm'])
    emMean = float(metric_dic['emMean'])
    emAdp = float(metric_dic['emAdp'])
    emMax = float(metric_dic['emMax'])
    fmMean = float(metric_dic['fmMean'])
    fmAdp = float(metric_dic['fmAdp'])
    fmMax = float(metric_dic['fmMax'])
    wfm = float(metric_dic['wfm'])
    mae = float(metric_dic['mae'])

    print(_data_name)

    print('sm:', sm)
    print('emMean:', emMean)
    print('emAdp:', emAdp)
    print('emMax:', emMax)
    print('fmMean:', fmMean)
    print('fmAdp:', fmAdp)
    print('fmMax:', fmMax)
    print('wfm:', wfm)
    print('mae:', mae)


