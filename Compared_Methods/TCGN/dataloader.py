from torch.utils.data import Dataset, DataLoader # 载入数据专用
from torchvision import transforms # 图像变换专用
import torch # pytorch
import os
import numpy as np
import pandas as pd
from PIL import Image # used for read graph
from .dataset import ViT_HER2ST

use_gpu = torch.cuda.is_available() # gpu加速
#use_gpu=False
torch.cuda.empty_cache() # 清除显卡缓存

class ST_HER2_Dataset(Dataset):
    def __init__(self,HE_image_paths, adata_list,transforms,train_or_test='train',names=None):
        self.train_or_test=train_or_test
        her2_set = ViT_HER2ST(HE_image_paths, adata_list,train=(train_or_test=='train'), names=names)
        if train_or_test=='train':
            shuffle=True
        else:
            shuffle=False
        her2_loader = DataLoader(her2_set, batch_size=1, num_workers=0, shuffle=shuffle)
        self.imgs=[]
        self.exps=[]
        self.transforms=transforms
        for stepi, input in enumerate(her2_loader, 1):
            '''
            img <class 'torch.Tensor'> torch.Size([1, 325, 3, 112, 112])
            position <class 'torch.Tensor'> torch.Size([1, 325, 2])
            exp <class 'torch.Tensor'> torch.Size([1, 325, 785])
            <class 'torch.Tensor'> torch.Size([1, 325, 325])
            <class 'torch.Tensor'> torch.Size([1, 325, 785])
            <class 'torch.Tensor'> torch.Size([1, 325])
            <class 'torch.Tensor'> torch.Size([1, 325, 2])
            '''
            self.imgs.append(self.transforms[0](input[0].squeeze(dim=0)/255))
            self.exps.append(input[2].squeeze(dim=0))
        self.imgs=torch.concat(self.imgs,dim=0)
        self.exps=torch.concat(self.exps,dim=0)

    def __getitem__(self, index):
        if self.train_or_test=='train':
            return self.transforms[1](self.imgs[index]),self.exps[index]
        elif self.train_or_test=='test':
            return self.imgs[index],self.exps[index]

    def __len__(self):
        return self.exps.shape[0]

def make_HER2_dataset(new_adata_list, HE_image_paths, names, train_index, test_index,transform,batch_size=32):
    train_names=[names[i] for i in train_index]
    train_HE_image_paths=[HE_image_paths[i] for i in train_index]
    train_adata_list=[new_adata_list[i] for i in train_index]

    test_names=[names[i] for i in test_index]
    test_HE_image_paths=[HE_image_paths[i] for i in test_index]
    test_adata_list=[new_adata_list[i] for i in test_index]
    train_dataset=ST_HER2_Dataset(train_HE_image_paths, train_adata_list,transform,train_or_test='train',names=train_names)
    test_dataset = ST_HER2_Dataset(test_HE_image_paths, test_adata_list,transform,train_or_test='test',names=test_names)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,drop_last = True)
    test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,drop_last = False)
    return train_loader,test_loader

if __name__=="__main__":
    import time

    time_begin = time.time()
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    train_transform = transforms.Compose(
        [transforms.RandomRotation(180),
         transforms.RandomHorizontalFlip(0.5),  # randomly 水平旋转
         transforms.RandomVerticalFlip(0.5),  # 随机竖直翻转
         ]
    )
    basic_transform = transforms.Compose(
        [transforms.Resize((224, 224), antialias=True),  # resize to 256x256 square
         transforms.ConvertImageDtype(torch.float),
         transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)  # 归一化
         ]
    )
    my_transforms = [basic_transform, train_transform]
    # /export/home/bs2021
    train_loader, test_loader, test_sample = make_HER2_dataset(0,my_transforms)
    print(len(test_loader), len(train_loader))
    print("finish loading")
    for stepi, (img, gene) in enumerate(train_loader, 1):
        imgs = img.cuda()
        if stepi % 20 == 0:
            print(stepi)
            print(imgs.shape,gene.shape)
        pass
    time_end = time.time()
    print("improved:", time_end - time_begin)
