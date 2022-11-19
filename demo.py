import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import argparse
import time
import torchvision.utils as vutils
from model import LMFANet
from data_utils import *
parser = argparse.ArgumentParser()
parser.add_argument('--dir_test', type=str, default='./test', help='path_input')
parser.add_argument('--dir_base', type=str, default='./best.pk', help='path_model')
parser.add_argument('--dir_save', type=str, default='./result', help='path_save')

parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", help='cuda device')
parser.add_argument('--num_workers', type=int, default=0, help='num_workers')
# parser.add_argument('--device', default="cuda:0", help='cuda device')
opt = parser.parse_args()

loader_test = Test_Dataset(path=opt.dir_test, img_ytype="png", patch=11)
loader_test = DataLoader(dataset=loader_test, batch_size=1, shuffle=False, num_workers=opt.num_workers)

if len(opt.dir_save)>0:
    if not os.path.exists(opt.dir_save):
        os.makedirs(opt.dir_save) 
    dir_result="{}/{}".format(opt.dir_save, time.strftime("%Y-%m-%d-%H-%M", time.localtime()))
    if not os.path.exists(dir_result):
        os.makedirs(dir_result) 

since = time.time()
net = LMFANet()
if opt.device=="cpu":
    ckp = torch.load(opt.dir_base,map_location='cpu')      
else:
    ckp = torch.load(opt.dir_base) 
print(opt.device)

net.load_state_dict(ckp['model'])
net = net.to(opt.device)
since = time.time()
with torch.no_grad():
    net.eval()
    for i, [image_x] in enumerate(loader_test):
        image_x = image_x.to(opt.device)
        image_x = torch.unsqueeze(image_x, dim=0)
        print(image_x.shape)
        pred = net(image_x)
        pred = torch.squeeze(pred, dim=0)
        vutils.save_image(pred.cpu(),'{}/pred_{}.jpg'.format(dir_result, i))
print(time.time()-since)