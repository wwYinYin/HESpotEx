import glob
import os

import numpy as np
import pandas as pd
import scanpy as sc
import scprep as scp
import torch
import torchvision.transforms as transforms
from PIL import ImageFile, Image
from scipy.sparse import issparse
import cv2
from .graph_construction import calcADJ
import tifffile
Image.MAX_IMAGE_PIXELS = 2300000000
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ViT_HER2ST(torch.utils.data.Dataset):

    def __init__(self, image_path, ST_adata, train, names):
        super(ViT_HER2ST, self).__init__()

        self.names = names
        self.image_path=image_path
        self.ST_adata=ST_adata
        self.train=train
        # self.r = 224 // 4
        # self.r = 30
        # self.r = (ST_adata.uns['block_r']//2).astype(int)
        self.r_dict = {name: (ST_adata[i].uns['block_r']//2) for i, name in
                            enumerate(self.names)}
        # self.img_dict = {name: torch.Tensor(self.get_img(self.image_path[i])) for i, name in enumerate(self.names)}
        self.img_dict = {name: self.get_img(self.image_path[i]) for i, name in enumerate(self.names)}

        self.exp_dict = {name: scp.transform.log(scp.normalize.library_size_normalize(self.ST_adata[i].layers['counts'])) for i, name in
                         enumerate(self.names)}
        
        self.center_dict = {name: np.floor(self.ST_adata[i].obsm['spatial']).astype(int) for i, name in
                            enumerate(self.names)}
        self.loc_dict = {name: self.ST_adata[i].obs[['array_row', 'array_col']].values.astype(float) for i, name in
                         enumerate(self.names)}
        
        self.patch_dict = {}
        self.id2name = dict(enumerate(self.names))
        self.transforms = transforms.Compose([transforms.ColorJitter(0.5, 0.5, 0.5), transforms.RandomHorizontalFlip(),
                                              transforms.RandomRotation(degrees=180), transforms.ToTensor()])
        self.adj_dict = {name: calcADJ(coord=m, k=4, pruneTag='NA') for name, m in self.center_dict.items()}
    
    def transform(self, image):
        w = image.shape[0]
        dim = (112, 112)
        if w>=112:
            image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) #适用于图像缩小
        else:
            image = cv2.resize(image, dim, interpolation = cv2.INTER_CUBIC) #适用于图像放大

        return image
    
    def __getitem__(self, index):
        i = index
        name = self.id2name[i]
        im = self.img_dict[name]
        # im = im.permute(1, 0, 2)
        im = im.transpose(1, 0, 2)
        exps = self.exp_dict[name]
        centers = self.center_dict[name]
        loc = self.loc_dict[name]
        positions = torch.LongTensor(loc)

        if name in self.patch_dict:
            patches = self.patch_dict[name]
        else:
            patches = None

        adj = self.adj_dict[name]
        self.r = int(self.r_dict[name])

       
        n_patches = len(centers)
        if issparse(exps):
            exps=exps.todense()
        exps = torch.Tensor(exps)
        adj = torch.Tensor(adj)
        centers = torch.LongTensor(centers)
        w,h= im.shape[0], im.shape[1]
        if patches is None:
            # patches = torch.zeros((n_patches, 3, 2 * self.r, 2 * self.r))
            patches = torch.zeros((n_patches, 3, 112, 112)).float()
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[max((x - self.r),0):min((x + self.r),w), max((y - self.r),0):min((y + self.r),h), :]
                patch = self.transform(patch)
                # patches[i] = patch.permute(2, 0, 1)
                patches[i] = torch.tensor(patch).permute(2, 0, 1).float()
            self.patch_dict[name] = patches
        patches = torch.Tensor(patches)

        if self.train:
            return patches, positions, exps, adj
        else:
            return patches, positions, exps, torch.Tensor(centers), adj

    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, image_path):
        if type(image_path)==np.ndarray:
            im = image_path.copy()
        else:
            if image_path.endswith('tif'):
                im=tifffile.imread(image_path)
            else:
                # im = cv2.imread(image_path)
                im = Image.open(image_path)
                im = np.array(im)
                im = im[:,:,0:3]
        
        return im
