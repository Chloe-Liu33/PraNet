import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from datetime import datetime

from scipy import misc
import imageio
# from scipy import imageio
from lib.PraNet_Res2Net import PraNet
# from utils.dataloader import test_dataset
from utils.data_val import test_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./snapshots/PraNet_800-39.pth')
start1 = datetime.now()
end1 = datetime.now()

t = end1-start1
for _data_name in ['Locust-mini']:
    data_path = 'D:/imgR/imgR2/SINet/Dataset/TestDataset/{}/'.format(_data_name)
    save_path = './res_800/PraNet/{}/'.format(_data_name)

    opt = parser.parse_args()
    model = PraNet()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/Image/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)

    test_loader = test_dataset(image_root, gt_root, opt.testsize)


    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        start = datetime.now()

        res5, res4, res3, res2 = model(image)
        end = datetime.now()
        t = t + end - start
        res = res2
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)


        # misc.imsave(save_path+name, res)
        # imageio.imwrite(save_path+name, res)

# f = 120/t
print(t)