import os
import cv2
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
# from scipy.sparse import csr_matrix
import numpy as np
import torchvision.transforms.functional as TF
import random
from PIL import Image
from scipy.sparse import issparse
import tifffile

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, ST_adata):

        self.spatial_pos_csv =  ST_adata.obsm['spatial'].astype(int)

        if issparse(ST_adata.X):
            self.expression = ST_adata.X.todense()
        else:
            self.expression = ST_adata.X
        if type(image_path)==np.ndarray:
            self.whole_image = image_path.copy()
        else:
            if image_path.endswith('tif'):
                self.whole_image=tifffile.imread(image_path)
            else:
                # self.whole_image = cv2.imread(image_path)
                self.whole_image = Image.open(image_path)
                self.whole_image = np.array(self.whole_image)
                self.whole_image = self.whole_image[:,:,0:3]

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
        h=self.whole_image.shape[0]
        w=self.whole_image.shape[1]
        v1 = self.spatial_pos_csv[idx,0]
        v2 = self.spatial_pos_csv[idx,1]
        image = self.whole_image[max((v2-112),0):min((v2+112),h),max((v1-112),0):min((v1+112),w),:]
        image = self.transform(image)
        item['image'] = torch.tensor(image).permute(2, 0, 1).float() #color channel first, then XY
        item['reduced_expression'] = torch.tensor(self.expression[idx,:]).float().squeeze(0)  #cell x features (3467)

        return item


    def __len__(self):
        return len(self.spatial_pos_csv)