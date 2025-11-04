import os
import torch
import random
import argparse
import pickle as pk
import pytorch_lightning as pl
from .utils import *
from .HIST2ST import *
from .predict import *
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

def main(new_adata_list, HE_image_paths, names, train_index, test_index,out_embedding):

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    tag='5-7-2-8-4-16-32'
    kernel,patch,depth1,depth2,depth3,heads,channel=map(lambda x:int(x),tag.split('-'))

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

    model = Hist2ST(
        depth1=depth1, depth2=depth2, depth3=depth3,
        n_genes=out_embedding, learning_rate=1e-5, 
        kernel_size=kernel, patch_size=patch,
        heads=heads, channel=channel, 
        zinb=0.25, bake=0, lamb=0, 
    )
    trainer = pl.Trainer(
        accelerator="gpu", devices=[3], max_epochs=60
    )

    trainer.fit(model, train_loader, test_loader)
    # torch.save(model.state_dict(),f"../data/her2st/Hist2ST_B2.ckpt")
    # model.load_state_dict(torch.load(f"./model/{args.fold}-Hist2ST{'_cscc' if args.data=='cscc' else ''}.ckpt"),)
    pred, gt = test(model, test_loader,'cuda')
    R=get_R(pred,gt)[0]
    print('Pearson Correlation:',np.nanmean(R))

    return pred, gt

