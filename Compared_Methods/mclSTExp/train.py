import argparse
import torch
import os
from .dataset import SKIN, HERDataset, TenxDataset
from .model import mclSTExp_Attention
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils import AvgMeter, get_lr
import random
import numpy as np

def train(model, train_dataLoader, optimizer, device):
    loss_meter = AvgMeter()
    tqdm_train = tqdm(train_dataLoader, total=len(train_dataLoader))
    for batch in tqdm_train:
        batch = {k: v.to(device) for k, v in batch.items() if
                 k == "image" or k == "expression" or k == "position"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_train.set_postfix(train_loss=loss_meter.avg)
    return loss_meter

def test_epoch(model, test_loader, device):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(test_loader, total=len(test_loader))
    for batch in tqdm_object:
        batch = {k: v.to(device) for k, v in batch.items() if
                 k == "image" or k == "expression" or k == "position"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter

def main(new_adata_list, HE_image_paths, names, train_index, test_index,out_embedding, save_path=None):
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True   
    
    batch_size=256
    max_epochs=10
    
    train_names=[names[i] for i in train_index]
    train_HE_image_paths=[HE_image_paths[i] for i in train_index]
    train_adata_list=[new_adata_list[i] for i in train_index]

    test_names=[names[i] for i in test_index]
    test_HE_image_paths=[HE_image_paths[i] for i in test_index]
    test_adata_list=[new_adata_list[i] for i in test_index]
    train_dataset = HERDataset(train_HE_image_paths, train_adata_list, train=True, names=train_names)
    train_dataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = HERDataset(test_HE_image_paths, test_adata_list, train=False, names=test_names)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model = mclSTExp_Attention(encoder_name='densenet121',
                                spot_dim=out_embedding,
                                temperature=1,
                                image_dim=1024,
                                projection_dim=256,
                                heads_num=8,
                                heads_dim=64,
                                head_layers=2,
                                dropout=0)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, weight_decay=1e-3
    )
    best_loss = float('inf')
    best_epoch = 0
    for epoch in range(max_epochs):
        model.train()
        train_loss=train(model, train_dataLoader, optimizer, device)
        # Evaluate the model
        model.eval()
        with torch.no_grad():
            test_loss = test_epoch(model, test_loader, device)

        if test_loss.avg < best_loss:
            best_loss = test_loss.avg
            best_epoch = epoch

            torch.save(model.state_dict(), save_path + "/mclSTExp_best.pt")
            print("Saved Best Model! Loss: {}".format(best_loss))
    print("Done!, final loss: {}".format(best_loss))
    print("Best epoch: {}".format(best_epoch))
