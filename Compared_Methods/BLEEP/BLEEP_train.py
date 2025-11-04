import os
from tqdm import tqdm

import torch
from torch import nn
import torch.distributed as dist
import torch.utils.data.distributed
import numpy as np
import random
from .models import CLIPModel
from torch.utils.data import DataLoader
import argparse

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text
    
def train_epoch(model, train_loader, optimizer, device):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(device) for k, v in batch.items() if k == "image" or k == "reduced_expression"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg)
    return loss_meter

def test_epoch(model, test_loader, device):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(test_loader, total=len(test_loader))
    for batch in tqdm_object:
        batch = {k: v.to(device) for k, v in batch.items() if k == "image" or k == "reduced_expression"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter

def main(dataset_list, train_index, test_index,out_embedding, save_path=None):
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    batch_size=256
    max_epochs=10
    num_workers=0
    device = torch.device("cuda:3")
    print("GPU available:", device)

    train_dataset = torch.utils.data.ConcatDataset([dataset_list[i] for i in train_index])
    test_dataset = torch.utils.data.ConcatDataset([dataset_list[i] for i in test_index])
    print(len(train_dataset), len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)

    model = CLIPModel(spot_embedding=out_embedding).to(device)
    print("Image encoder is ResNet50")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-3
    )

    best_loss = float('inf')
    best_epoch = 0
    for epoch in range(max_epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()

        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            test_loss = test_epoch(model, val_loader, device)

        if test_loss.avg < best_loss:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            best_loss = test_loss.avg
            best_epoch = epoch

            torch.save(model.state_dict(), save_path + "/BLEEP_best.pt")
            print("Saved Best Model! Loss: {}".format(best_loss))

    print("Done!, final loss: {}".format(best_loss))
    print("Best epoch: {}".format(best_epoch))







