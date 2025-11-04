import torch
from torchvision import datasets, transforms, models
import torch.nn as nn

import pandas as pd
import datetime
import random
from .dataloader import make_HER2_dataset
from .model import TCGN
import numpy as np
import psutil
import os
import gc
import anndata

use_gpu = torch.cuda.is_available() # gpu加速
#use_gpu=False
torch.cuda.empty_cache() # 清除显卡缓存


def ST_TCGN(new_adata_list, HE_image_paths, names, train_index, test_index,out_embedding):
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    batch_size=256
    epoch=20
    if use_gpu:
        device = torch.device("cuda:0")
        print("GPU available:", device)
    # load data
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
    #/export/home/bs2021
    train_loader, test_loader = make_HER2_dataset(new_adata_list, HE_image_paths, names, 
                                                               train_index, test_index,my_transforms,batch_size)
    print("finish loading")

    # initialize model
    import os
    model_name = "TCGN"
    dirs = "./TCGN/record-" + model_name
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    my_model = TCGN(num_classes=out_embedding,device=device)
    if use_gpu:
        my_model = my_model.to(device)
        my_model.load_state_dict(torch.load('./TCGN/pretrained/cmt_tiny.pth'), strict=False)

    # train the model
    # v5: remove the first GNN
    optimizer = torch.optim.Adam(my_model.parameters(),lr=1e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    loss_func = nn.MSELoss()
    dfhistory = pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "train_median_pcc", "val_median_pcc"])
    print("Start Training...")
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_step_freq = 20

    print("==========" * 8 + "%s" % nowtime)
    from .real_metric import compare_prediction_label_list
    best_val_median_pcc = -1
    record_file = open(dirs + '/'+'best_epoch.csv', mode='w')
    record_file.write("epoch,best_val_median_pcc\n")
    for epoch in range(1, epoch):
        my_model.train()
        loss_train_sum = 0.0
        epoch_median_pcc_val = None
        epoch_median_pcc_train = None

        epoch_real_record_train = []
        epoch_predict_record_train = []
        step_train = 0
        for stepi, (imgs, genes) in enumerate(train_loader, 1):
            #print(stepi,end="")
            step_train = stepi
            optimizer.zero_grad()
            if use_gpu:
                imgs = imgs.to(device)
                genes = genes.to(device)

            predictions = my_model(imgs)
            loss = loss_func(predictions, genes)
            ## 反向传播求梯度
            loss.backward()  # 反向传播求各参数梯度
            optimizer.step()  # 用optimizer更新各参数

            if use_gpu:
                predictions = predictions.cpu().detach().numpy()
            else:
                predictions = predictions.detach().numpy()
            epoch_real_record_train += list(genes.cpu().numpy())
            epoch_predict_record_train += list(predictions)

            if use_gpu:
                loss_train_sum += loss.cpu().item()  # 返回数值要加.item
            else:
                loss_train_sum += loss.item()

            gc.collect()
        epoch_median_pcc_train = compare_prediction_label_list(epoch_predict_record_train, epoch_real_record_train,
                                                        gene_num=out_embedding)
        
        if epoch % 2 == 0:  # 当多少个batch后打印结果
            print(("training: [epoch = %d, step = %d, images = %d] loss: %.3f, " + "median pearson coefficient" + ": %.3f") %
                    (epoch, stepi, stepi*batch_size,loss_train_sum / stepi, epoch_median_pcc_train))


        my_model.eval()
        loss_val_sum = 0.0
        epoch_real_record_val = []
        epoch_predict_record_val = []
        step_val = 0
        for stepi, (imgs, genes) in enumerate(test_loader, 1):
            #print(stepi, end="")
            step_val = stepi
            with torch.no_grad():
                if use_gpu:
                    imgs = imgs.to(device)
                    genes = genes.to(device)
                predictions = my_model(imgs)
                loss = loss_func(predictions, genes)

                if use_gpu:
                    loss_val_sum += loss.cpu().item()  # 返回数值要加.item
                else:
                    loss_val_sum += loss.item()

                if use_gpu:
                    predictions = predictions.cpu().detach().numpy()
                else:
                    predictions = predictions.detach().numpy()

            epoch_real_record_val += list(genes.cpu().numpy())
            epoch_predict_record_val += list(predictions)
        epoch_median_pcc_val = compare_prediction_label_list(epoch_predict_record_val, epoch_real_record_val,
                                                                gene_num=out_embedding)

        if epoch % 2 == 0:  # 当多少个batch后打印结果
            print(("validation: [step = %d] loss: %.3f, " + "median pearson coefficient" + ": %.3f") %
                    (stepi, loss_val_sum / stepi, epoch_median_pcc_val))

        historyi = (
            epoch, loss_train_sum / step_train, loss_val_sum / step_val, epoch_median_pcc_train, epoch_median_pcc_val)
        dfhistory.loc[epoch - 1] = historyi
        
        header = list(range(out_embedding))
        prediction_df = pd.DataFrame(columns=header, data=epoch_predict_record_val)
        real_df = pd.DataFrame(columns=header, data=epoch_real_record_val)
        print(model_name)
        print((
                  "\nEPOCH = %d, loss_train_avg = %.3f, loss_val_avg = %.3f, epoch_median_pcc_train = %.3f, epoch_median_pcc_val = %.3f")
              % historyi)

        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n" + "==========" * 8 + "%s" % nowtime)
        if epoch >= 1:
            if epoch_median_pcc_val > best_val_median_pcc:
                best_val_median_pcc = epoch_median_pcc_val
                print("best epoch now:", epoch)
                record_file.write(str(epoch) + "," + str(epoch_median_pcc_val) + "\n")
                record_file.flush()
                torch.save(my_model.state_dict(),
                           dirs + "/"  + "ST_Net-" + model_name + "-best_B4.pth")
                adata_pred = anndata.AnnData(prediction_df)
                adata_real = anndata.AnnData(real_df)

    record_file.close()

    return adata_pred, adata_real
