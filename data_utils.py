import torch
import random,os,time,math
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.utils import make_grid


def split_train_val(trainset, val_fraction=0): 
    n_train = int((1. - val_fraction) * len(trainset))
    n_val =  len(trainset) - n_train
    train_subset, val_subset = torch.utils.data.random_split(trainset, (n_train, n_val))
    return train_subset, val_subset

def lr_schedule_cosdecay(t,T,init_lr):
	lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
	return lr

def tensorShow(tensors,titles=None):
        '''
        t:BCWH
        '''
        fig=plt.figure()
        for tensor,tit,i in zip(tensors,titles,range(len(tensors))):
            img = make_grid(tensor)
            npimg = img.numpy()
            ax = fig.add_subplot(211+i)
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(tit)
        plt.show()

class RESIDE_Dataset(data.Dataset):
    def __init__(self,path,dir_name=["haze","clear"],img_ytype="jpg",size=-1,patch=-1,augSet=[False,False,False],is_train=False):
        super(RESIDE_Dataset,self).__init__()
        self.img_ytype=img_ytype
        self.size=size
        self.patch=patch
        self.RHFlip=augSet[0]
        self.RRotate=augSet[1]
        self.normal_=augSet[2]
        self.is_train=is_train

        self.haze_imgs_dir=os.listdir(os.path.join(path,dir_name[0]))
        self.haze_imgs=[os.path.join(path,dir_name[0],img) for img in self.haze_imgs_dir]
        self.clear_dir=os.path.join(path,dir_name[1])
    def __getitem__(self, index):
        haze=Image.open(self.haze_imgs[index])
        if self.is_train and self.size>0:
            while haze.size[0]<self.size or haze.size[1]<self.size :
                index=random.randint(0,len(self.haze_imgs_dir)-1)
                haze=Image.open(self.haze_imgs[index])
        img=self.haze_imgs[index]
        id_=img.split('\\')[-1].split('_')[0]
        clear_name=id_+'.'+self.img_ytype
        # print(img, id_, clear_name)
        clear=Image.open(os.path.join(self.clear_dir,clear_name))
        clear=tfs.CenterCrop(haze.size[::-1])(clear)
        if self.is_train and self.size>0:
            i,j,h,w=tfs.RandomCrop.get_params(haze,output_size=(self.size,self.size))
            haze=FF.crop(haze,i,j,h,w)
            if self.patch>0:
                clear=FF.crop(clear,i+self.patch//2,j+self.patch//2,h-self.patch+1,w-self.patch+1)
            else:
                clear=FF.crop(clear,i,j,h,w)

        haze,clear=self.augData(haze.convert("RGB") ,clear.convert("RGB") )
        return haze,clear
    def augData(self,data,target):
        if self.is_train:
            if self.RHFlip:
                rand_hor=random.randint(0,1)
                data=tfs.RandomHorizontalFlip(rand_hor)(data)
                target=tfs.RandomHorizontalFlip(rand_hor)(target)

            if self.RRotate:
                rand_rot=random.randint(0,3)
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)
        data=tfs.ToTensor()(data)
        if self.normal_:
            data=tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(data)
        if not self.is_train and self.patch>0:
            data=tfs.Pad(padding=self.patch//2, fill=0, padding_mode='reflect')(data)

        target=tfs.ToTensor()(target)
        return  data ,target
    def __len__(self):
        return len(self.haze_imgs)

class Test_Dataset(data.Dataset):
    def __init__(self,path,img_ytype="jpg",patch=-1):
        super(Test_Dataset,self).__init__()
        self.img_ytype=img_ytype
        self.patch=patch

        self.haze_imgs_dir=os.listdir(path)
        self.haze_imgs=[os.path.join(path,img) for img in self.haze_imgs_dir]
    def __getitem__(self, index):
        haze=Image.open(self.haze_imgs[index])
        haze=self.augData(haze.convert("RGB"))
        return haze
    def augData(self,data):
        data=tfs.ToTensor()(data)
        if self.patch>0:
            data=tfs.Pad(padding=self.patch//2, fill=0, padding_mode='reflect')(data)
        return  data
    def __len__(self):
        return len(self.haze_imgs)

def make_new_dir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_) 
    dir_today="{}/{}".format(dir_, time.strftime("%Y-%m-%d", time.localtime()))
    if not os.path.exists(dir_today):
        os.makedirs(dir_today) 
    dir_now="{}/{}".format(dir_today, time.strftime("%H-%M", time.localtime()))
    if not os.path.exists(dir_now):
        os.makedirs(dir_now)
    dir_images="{}/images".format(dir_now)
    if not os.path.exists(dir_images):
        os.makedirs(dir_images)
    dir_numpy_files="{}/numpy_files".format(dir_now)
    if not os.path.exists(dir_numpy_files):
        os.makedirs(dir_numpy_files)
    return dir_now, dir_numpy_files, dir_images

def make_test_dir(dir_):
    name_ = dir_.split("/")[-1]
    dir_ = dir_.split("/{}".format(name_))[0]
    dir_now="{}/{}".format(dir_, time.strftime("%Y-%m-%d-%H-%M", time.localtime()))
    if not os.path.exists(dir_now):
        os.makedirs(dir_now)
    return dir_now

def name_(img_path):
    img_name = img_path.split("/")[-1]
    img_type = img_name.split(".")[-1]
    img_res = "{}_result.{}".format(img_name[:-1*(1+len(img_type))], img_type)
    return img_res

def loss_hol(loss_old, loss):
    if loss<loss_old:
        return "▼", "down"
    else:
        return "▲", "up"




if __name__ == "__main__":
    print(name_("../test.jpg"))