import os
import cv2
import pandas as pd
import torch
# from scipy.sparse import csr_matrix
import numpy as np
from skimage.feature.texture import graycomatrix, graycoprops
import random
from PIL import Image
import scprep as scp
Image.MAX_IMAGE_PIXELS = 2800000000
from scipy.spatial import distance
from scipy.sparse import issparse
from cellpose import models

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, ST_adata, nuclei_mask_path=None):
        super(CLIPDataset, self).__init__()
        # self.whole_image = cv2.imread(image_path)
        self.spatial_pos_csv =  ST_adata.obsm['spatial'].astype(int)
        self.block_r = (ST_adata.uns['block_r']//2).astype(int)
        # self.block_r = 112

        if issparse(ST_adata.X):
            self.expression = ST_adata.X.todense()
        else:
            self.expression = ST_adata.X
        self.latent = ST_adata.obsm['embedding_rep']
        # self.loc_dict = ST_adata.obs[['array_row', 'array_col']].values.astype(float)         
        
        if type(image_path)==np.ndarray:
            self.whole_image = image_path.copy()
        else:
            # self.whole_image = cv2.imread(image_path)
            self.whole_image = Image.open(image_path)
            self.whole_image = np.array(self.whole_image)
            self.whole_image = self.whole_image[:,:,0:3]
        if nuclei_mask_path is not None:
            self.nuclei_masks = np.load(nuclei_mask_path,allow_pickle=True)
        else:
            model_cellpose = models.Cellpose(gpu=True, model_type='nuclei')
            self.nuclei_masks,_,_,_=model_cellpose.eval(self.whole_image, flow_threshold=0.8, diameter=None, 
                                                       min_size=15,channels=[1,0], invert=True)
    def transform(self, image):
        w = image.shape[0]
        dim = (224, 224)
        if w>=224:
            image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) #适用于图像缩小
        else:
            image = cv2.resize(image, dim, interpolation = cv2.INTER_CUBIC) #适用于图像放大

        return image

    def __getitem__(self, idx):
        item = {}
        n_patches=self.spatial_pos_csv.shape[0]
        image_patches = torch.zeros((n_patches, 3, 224, 224)).float()
        h=self.whole_image.shape[0]
        w=self.whole_image.shape[1]
        # print(h,w)
        nuclei_number = np.zeros((n_patches, 1))
        for i in range(n_patches):
            v1 = self.spatial_pos_csv[i,0]
            v2 = self.spatial_pos_csv[i,1]

            image = self.whole_image[max((v2-self.block_r),0):min((v2+self.block_r),h),max((v1-self.block_r),0):min((v1+self.block_r),w),:]

            nuclei_mask=self.nuclei_masks[max((v2-self.block_r),0):min((v2+self.block_r),h),max((v1-self.block_r),0):min((v1+self.block_r),w)]
            unique_values = np.unique(nuclei_mask)
            nuclei_number[i,:]=len(unique_values) 
            
            # nuclei_number[i,:]=int(np.mean(nuclei_mask)/(len(unique_values))) 
            image = self.transform(image)

            image_patches[i]=torch.tensor(image).permute(2, 0, 1).float() #[N,3,224,224]
        nuclei_number = np.minimum(nuclei_number, 20)
        nuclei_number = np.maximum(nuclei_number, 1)
        nuclei_number=nuclei_number/nuclei_number.max() + 1
        item['expression'] = torch.tensor(self.expression).float()  #[N x features] (3467)
        item['latent'] = torch.tensor(self.latent).float()  #[N x features] (30)
        # item['loc'] = torch.LongTensor(self.loc_dict) #[N x 2]
        item['adj'] = calcADJ(self.spatial_pos_csv)     #[N x N]
        item['nuclei_number'] = torch.tensor(nuclei_number).float()

        item['image'] = image_patches
        items = {'meta': item}
        return items


    def __len__(self):
        return len([1])
    
class CLIPDataset_ST1K(torch.utils.data.Dataset):
    def __init__(self, image_path, ST_adata):
        super(CLIPDataset_ST1K, self).__init__()
        self.spatial_pos_csv =  ST_adata.obsm['spatial'].astype(int)
        self.block_r = (ST_adata.uns['block_r']//2).astype(int)
        # self.block_r = 448
        if issparse(ST_adata.X):
            self.expression = ST_adata.X.todense()
        else:
            self.expression = ST_adata.X
        self.latent = ST_adata.obsm['embedding_rep']      
    
        self.whole_image = Image.open(image_path)
        self.whole_image = np.array(self.whole_image)
        self.whole_image = self.whole_image[:,:,0:3]
        
    def transform(self, image):
        w = image.shape[0]
        dim = (224, 224)
        try:
            if w>=224:
                image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) #适用于图像缩小
            else:
                image = cv2.resize(image, dim, interpolation = cv2.INTER_CUBIC) #适用于图像放大
        except:
            print(image.shape)
            raise

        return image

    def __getitem__(self, idx):
        item = {}
        n_patches=self.spatial_pos_csv.shape[0]
        image_patches = torch.zeros((n_patches, 3, 224, 224)).float()
        h=self.whole_image.shape[0]
        w=self.whole_image.shape[1]
        
        for i in range(n_patches):
            v1 = self.spatial_pos_csv[i,0].astype(int)
            v2 = self.spatial_pos_csv[i,1].astype(int)
            try:
                image = self.whole_image[max((v2-self.block_r),0):min((v2+self.block_r),h),max((v1-self.block_r),0):min((v1+self.block_r),w),:]
                image = self.transform(image)
            except:
                print(h,w)
                print(v1,v2)
                raise

            image_patches[i]=torch.tensor(image).permute(2, 0, 1).float() #[N,3,224,224]

        item['expression'] = torch.tensor(self.expression).float()  #[N x features] (3467)
        item['latent'] = torch.tensor(self.latent).float()  #[N x features] (30)
        item['adj'] = calcADJ(self.spatial_pos_csv)     #[N x N]
        item['nuclei_number'] = torch.tensor(0).float()
        item['image'] = image_patches
        items = {'meta': item}
        return items

    def __len__(self):
        return len([1])

class CLIPDataset_reference(torch.utils.data.Dataset):
    def __init__(self, image_path, ST_adata, nuclei_mask_path=None, reference_sc=None):
        super(CLIPDataset_reference, self).__init__()
        # self.whole_image = cv2.imread(image_path)
        self.spatial_pos_csv =  ST_adata.obsm['spatial'].astype(int)
        # self.block_r = ST_adata.uns['block_r']
        # self.block_r = 56
        self.block_r = 56

        if issparse(ST_adata.X):
            self.expression = ST_adata.X.todense()
        else:
            self.expression = ST_adata.X

        if issparse(reference_sc.X):
            self.reference_sc = reference_sc.X.todense()
        else:
            self.reference_sc = reference_sc.X
        # self.expression=scp.transform.log(scp.normalize.library_size_normalize(ST_adata.layers['counts']))
        self.latent = ST_adata.obsm['embedding_rep']
        self.loc_dict = ST_adata.obs[['array_row', 'array_col']].values.astype(float)         
        
        if type(image_path)==np.ndarray:
            self.whole_image = image_path.copy()
        else:
            # self.whole_image = cv2.imread(image_path)
            self.whole_image = Image.open(image_path)
            self.whole_image = np.array(self.whole_image)
            self.whole_image = self.whole_image[:,:,0:3]
        if nuclei_mask_path is not None:
            self.nuclei_masks = np.load(nuclei_mask_path,allow_pickle=True)
        else:
            model_cellpose = models.Cellpose(gpu=True, model_type='nuclei')
            self.nuclei_masks,_,_,_=model_cellpose.eval(self.whole_image, flow_threshold=0.8, diameter=None, 
                                                       min_size=15,channels=[1,0], invert=True)
    def transform(self, image):
        w = image.shape[0]
        dim = (224, 224)
        if w>=224:
            image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) #适用于图像缩小
        else:
            image = cv2.resize(image, dim, interpolation = cv2.INTER_CUBIC) #适用于图像放大

        return image

    def __getitem__(self, idx):
        item = {}
        
        n_patches=self.spatial_pos_csv.shape[0]
        image_patches = torch.zeros((n_patches, 3, 224, 224)).float()
        h=self.whole_image.shape[0]
        w=self.whole_image.shape[1]
        # print(h,w)
        nuclei_number = np.zeros((n_patches))
        for i in range(n_patches):
            v1 = self.spatial_pos_csv[i,0]
            v2 = self.spatial_pos_csv[i,1]

            image = self.whole_image[max((v2-self.block_r),0):min((v2+self.block_r),h),max((v1-self.block_r),0):min((v1+self.block_r),w),:]

            nuclei_mask=self.nuclei_masks[max((v2-self.block_r),0):min((v2+self.block_r),h),max((v1-self.block_r),0):min((v1+self.block_r),w)]
            unique_values = np.unique(nuclei_mask)
            nuclei_number[i]=len(unique_values)
                     
            image = self.transform(image)

            image_patches[i]=torch.tensor(image).permute(2, 0, 1).float() #[N,3,224,224]
        nuclei_number = np.minimum(nuclei_number, 20)
        nuclei_number = np.maximum(nuclei_number, 1)
        item['expression'] = torch.tensor(self.expression).float()  #[N x features] (3467)
        item['reference_sc'] = torch.tensor(self.reference_sc).float() 
        item['latent'] = torch.tensor(self.latent).float()  #[N x features] (30)
        item['nuclei_number'] = torch.tensor(nuclei_number).float()
        item['image'] = image_patches
        items = {'meta': item}
        return items

    def __len__(self):
        return len([1])
    
def calcADJ(coord, k=4, distanceType='euclidean', pruneTag='NA'):
    r"""
    Calculate spatial Matrix directly use X/Y coordinates
    """
    spatialMatrix=coord#.cpu().numpy()
    nodes=spatialMatrix.shape[0]
    Adj=torch.zeros((nodes,nodes))
    for i in np.arange(spatialMatrix.shape[0]):
        tmp=spatialMatrix[i,:].reshape(1,-1)
        distMat = distance.cdist(tmp,spatialMatrix, distanceType)
        if k == 0:
            k = spatialMatrix.shape[0]-1
        res = distMat.argsort()[:k+1]
        tmpdist = distMat[0,res[0][1:k+1]]
        boundary = np.mean(tmpdist)+np.std(tmpdist) #optional
        for j in np.arange(1,k+1):
            # No prune
            if pruneTag == 'NA':
                Adj[i][res[0][j]]=1.0
            elif pruneTag == 'STD':
                if distMat[0,res[0][j]]<=boundary:
                    Adj[i][res[0][j]]=1.0
            # Prune: only use nearest neighbor as exact grid: 6 in cityblock, 8 in euclidean
            elif pruneTag == 'Grid':
                if distMat[0,res[0][j]]<=2.0:
                    Adj[i][res[0][j]]=1.0
    return Adj
