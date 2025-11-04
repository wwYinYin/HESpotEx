# coding:utf-8 
import random

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from .predict import model_predict
from .utils import *
from .vis_model import THItoGene
from .dataset import ViT_HER2ST
from pynvml import *
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def get_free_gpu() -> int:
    r"""
    Get index of GPU with least memory usage
    
    Ref
    ----------
    https://stackoverflow.com/questions/58216000/get-total-amount-of-free-gpu-memory-and-available-using-pytorch
    """
    nvmlInit()
    index = 0
    max = 0
    for i in range(torch.cuda.device_count()):
        h = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(h)
        index = i if info.free > max else index
        max = info.free if info.free > max else max
        
    # seed = np.random.randint(1000)
    # os.system(f'nvidia-smi -q -d Memory |grep Used > gpu-{str(seed)}.tmp')
    # memory_available = [int(x.split()[2]) for x in open('gpu.tmp', 'r').readlines()]
    # os.system(f'rm gpu-{str(seed)}.tmp')
    # print(memory_available)
    return index

def train(train_loader,test_loader,out_embedding,epochs=60):
    try:
        gpu_index = get_free_gpu()
        print(f"Choose GPU:{gpu_index} as device")
    except:
        print('GPU is not available')
    model = THItoGene(n_genes=out_embedding, learning_rate=1e-5, route_dim=64, caps=20, heads=[16, 8], n_layers=4)
    trainer = pl.Trainer(accelerator="gpu", devices=[2], max_epochs=epochs)
    # trainer = pl.Trainer(accelerator="cpu",max_epochs=epochs)
    trainer.fit(model, train_loader)

    pred = trainer.predict(model=model, dataloaders=test_loader)
    # R, p_val = get_R(pred, gt)
    # pred.var["p_val"] = p_val
    # pred.var["-log10p_val"] = -np.log10(p_val)

    # print('Mean Pearson Correlation:', np.nanmean(R))
    trainer.save_checkpoint("./data/her2st/THItoGene_B4.ckpt")
    return pred

def main(new_adata_list, HE_image_paths, names, train_index, test_index,out_embedding):
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_names=[names[i] for i in train_index]
    train_HE_image_paths=[HE_image_paths[i] for i in train_index]
    train_adata_list=[new_adata_list[i] for i in train_index]

    test_names=[names[i] for i in test_index]
    test_HE_image_paths=[HE_image_paths[i] for i in test_index]
    test_adata_list=[new_adata_list[i] for i in test_index]

    dataset = ViT_HER2ST(train_HE_image_paths, train_adata_list, train=True, names=train_names)
    train_loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)

    dataset_test = ViT_HER2ST(test_HE_image_paths, test_adata_list, train=False, names=test_names)
    test_loader = DataLoader(dataset_test, batch_size=1, num_workers=0)

    pred = train(train_loader,test_loader,out_embedding)

    return pred
