import os
import glob
import torch
import torchvision
import numpy as np
import scanpy as sc
import pandas as pd 
import scprep as scp
import anndata as ad
import cv2
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import ImageFile, Image
from .utils import read_tiff, get_data
from .graph_construction import calcADJ
from collections import defaultdict as dfd
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from scipy.sparse import issparse
import tifffile

class ViT_HER2ST(torch.utils.data.Dataset):

    def __init__(self, image_path, ST_adata, train, names):
        super(ViT_HER2ST, self).__init__()

        self.names = names
        self.image_path=image_path
        self.ST_adata=ST_adata
        self.train=train
        # self.r = 224 // 4
        # self.r = 30
        self.r_dict = {name: (self.ST_adata[i].uns['block_r']//2) for i, name in
                            enumerate(self.names)}
        self.img_dict = {name: self.get_img(self.image_path[i]) for i, name in enumerate(self.names)}
        self.ori_dict = {name: self.ST_adata[i].layers['counts'] for i, name in enumerate(self.names)}
        self.counts_dict={}
        for i,m in self.ori_dict.items():
                n_counts=m.sum(1)
                sf = n_counts / np.median(n_counts)
                self.counts_dict[i]=sf

        self.exp_dict = {name: scp.transform.log(scp.normalize.library_size_normalize(self.ST_adata[i].layers['counts'])) for i, name in
                         enumerate(self.names)}
        
        self.center_dict = {name: np.floor(self.ST_adata[i].obsm['spatial']).astype(int) for i, name in
                            enumerate(self.names)}
        self.loc_dict = {name: self.ST_adata[i].obs[['array_row', 'array_col']].values.astype(float) for i, name in
                         enumerate(self.names)}
        
        self.patch_dict=dfd(lambda :None)
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
        ID=self.id2name[index]
        im = self.img_dict[ID]
        im = im.transpose(1, 0, 2)
        # im = torch.Tensor(np.array(self.im))
        exps = self.exp_dict[ID]
        oris = self.ori_dict[ID]
        sfs = self.counts_dict[ID]
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        adj = self.adj_dict[ID]
        self.r = int(self.r_dict[ID])
        patches = self.patch_dict[ID]
        positions = torch.LongTensor(loc)
        if issparse(exps):
            exps=exps.todense()
        exps = torch.Tensor(exps)
        w,h= im.shape[0], im.shape[1]
        if issparse(oris):
            oris=oris.todense()
        if patches is None:
            n_patches = len(centers)
            patches = torch.zeros((n_patches,3,112,112))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[max((x - self.r),0):min((x + self.r),w), max((y - self.r),0):min((y + self.r),h), :]
                patch = self.transform(patch)
                # patches[i]=patch.permute(2,0,1)
                patches[i] = torch.tensor(patch).permute(2, 0, 1).float()
            self.patch_dict[ID]=patches
        data=[patches, positions, exps, adj,torch.Tensor(oris.astype(np.float32)),torch.Tensor(sfs.astype(np.float32))]

        data.append(torch.Tensor(centers))
        return data
        
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


class ViT_SKIN(torch.utils.data.Dataset):
    """Some Information about ViT_SKIN"""
    def __init__(self,train=True,r=4,norm=False,fold=0,flatten=True,ori=False,adj=False,prune='NA',neighs=4):
        super(ViT_SKIN, self).__init__()

        self.dir = './data/GSE144240_RAW/'
        self.r = 224//r

        patients = ['P2', 'P5', 'P9', 'P10']
        reps = ['rep1', 'rep2', 'rep3']
        names = []
        for i in patients:
            for j in reps:
                names.append(i+'_ST_'+j)
        gene_list = list(np.load('data/skin_hvg_cut_1000.npy',allow_pickle=True))

        self.ori = ori
        self.adj = adj
        self.norm = norm
        self.train = train
        self.flatten = flatten
        self.gene_list = gene_list
        samples = names
        te_names = [samples[fold]]
        tr_names = list(set(samples)-set(te_names))

        if train:
            self.names = tr_names
        else:
            self.names = te_names

        print(te_names)
        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in self.names}
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in self.names}

        self.gene_set = list(gene_list)
        if self.norm:
            self.exp_dict = {
                i:sc.pp.scale(scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)))
                for i,m in self.meta_dict.items()
            }
        else:
            self.exp_dict = {
                i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) 
                for i,m in self.meta_dict.items()
            }
        if self.ori:
            self.ori_dict = {i:m[self.gene_set].values for i,m in self.meta_dict.items()}
            self.counts_dict={}
            for i,m in self.ori_dict.items():
                n_counts=m.sum(1)
                sf = n_counts / np.median(n_counts)
                self.counts_dict[i]=sf
        self.center_dict = {
            i:np.floor(m[['pixel_x','pixel_y']].values).astype(int)
            for i,m in self.meta_dict.items()
        }
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}
        self.adj_dict = {
            i:calcADJ(m,neighs,pruneTag=prune)
            for i,m in self.loc_dict.items()
        }
        self.patch_dict=dfd(lambda :None)
        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))


    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i,exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp>0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:,j])


    def __getitem__(self, index):
        ID=self.id2name[index]
        im = self.img_dict[ID].permute(1,0,2)

        exps = self.exp_dict[ID]
        if self.ori:
            oris = self.ori_dict[ID]
            sfs = self.counts_dict[ID]
        adj=self.adj_dict[ID]
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        patches = self.patch_dict[ID]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4
        exps = torch.Tensor(exps)
        if patches is None:
            n_patches = len(centers)
            if self.flatten:
                patches = torch.zeros((n_patches,patch_dim))
            else:
                patches = torch.zeros((n_patches,3,2*self.r,2*self.r))

            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                if self.flatten:
                    patches[i] = patch.flatten()
                else:
                    patches[i]=patch.permute(2,0,1)
            self.patch_dict[ID]=patches
        data=[patches, positions, exps]
        if self.adj:
            data.append(adj)
        if self.ori:
            data+=[torch.Tensor(oris),torch.Tensor(sfs)]
        data.append(torch.Tensor(centers))
        return data
        
    def __len__(self):
        return len(self.exp_dict)

    def get_img(self,name):
        path = glob.glob(self.dir+'*'+name+'.jpg')[0]
        im = Image.open(path)
        return im

    def get_cnt(self,name):
        path = glob.glob(self.dir+'*'+name+'_stdata.tsv')[0]
        df = pd.read_csv(path,sep='\t',index_col=0)
        return df

    def get_pos(self,name):
        path = glob.glob(self.dir+'*spot*'+name+'.tsv')[0]
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join(pos.set_index('id'),how='inner')

        return meta

    def get_overlap(self,meta_dict,gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set&set(i.columns)
        return list(gene_set)