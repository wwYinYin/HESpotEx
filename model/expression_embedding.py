import random
from typing import Optional, List, Union
from joblib import Parallel, delayed

import scanpy as sc
import numpy as np
import pandas as pd
import scipy
import scipy.sparse
from anndata import AnnData
import torch
from torch_geometric.data import Data
from .preprocess import scanpy_workflow, Cal_Spatial_Net, scanpy_workflow_ST1K
from .utils import get_free_gpu
from .exp_module import STAGATE,stAA
from tqdm import tqdm
import torch.nn.functional as F
from scipy.sparse import issparse
from torch_geometric.utils import train_test_split_edges

seed=0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
    
def load_anndatas(adatas:List[AnnData],
                feature:Optional[str]='hvg',
                self_loop:Optional[bool]=True,
                join:Optional[str]='inner',
                rad_cutoff:Optional[Union[None,int]]=None,
                k_cutoff:Optional[Union[None,int]]=None, 
                model:Optional[str]='KNN',
                n_top_genes:Optional[int]=1000,
                method='GATE',
    ) -> List[Data]:

    assert feature.lower() in ['raw','hvg','pca','harmony']

    adatas = [adata.copy() for adata in adatas ] # May consume more memory
    
    # Edge
    edgeLists = []
    for adata in adatas:
        adata = Cal_Spatial_Net(adata, rad_cutoff=rad_cutoff, k_cutoff=k_cutoff, model=model)
        G_df = adata.uns['Spatial_Net'].copy()
        cells = np.array(adata.obs_names)
        cells_id_tran = dict(zip(cells, range(cells.shape[0])))
        G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
        G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

        # build adjacent matrix 一个稀疏矩阵
        G = scipy.sparse.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), 
                                    shape=(adata.n_obs, adata.n_obs))
        if self_loop:
            G = G + scipy.sparse.eye(G.shape[0])
        edgeList = np.nonzero(G)
        edgeLists.append(edgeList)

    # Feature
    datas = []
    print(f'Use {feature} feature to format graph')
    if feature.lower() == 'raw':
        for i, adata in enumerate(adatas):
            if type(adata.X) == np.ndarray:
                data = Data(edge_index=torch.LongTensor(np.array([edgeLists[i][0], edgeLists[i][1]])),
                            x=torch.FloatTensor(adata.X))  # .todense()
            else:
                data = Data(edge_index=torch.LongTensor(np.array(
                    [edgeLists[i][0], edgeLists[i][1]])), x=torch.FloatTensor(adata.X))
            datas.append(data)
    elif feature.lower() in ['hvg']:
        if len(adatas[0].var_names) > n_top_genes:
            if method == 'ST1K':
                all_highly_variable_genes = scanpy_workflow_ST1K(adatas, n_top_genes=n_top_genes)
            else:
                all_highly_variable_genes = scanpy_workflow(adatas, n_top_genes=n_top_genes)
            #all_highly_variable_genes = scanpy_workflow_insection(adatas, n_top_genes=n_top_genes)
        else:
            all_highly_variable_genes = adatas[0].var_names
        print(f'Use {len(all_highly_variable_genes)} genes to format graph')
        adata_all = adatas[0].concatenate(*adatas[1:], join=join) # join can not be 'outer'!
        adata_all = adata_all[:, all_highly_variable_genes]
        if n_top_genes<1100:
            sc.pp.filter_genes(adata_all, min_cells=1000)
        # sc.pp.filter_genes(adata_all, min_cells=1000)
        print(adata_all)
        if 'counts' not in adata_all.layers.keys():
            if issparse(adata_all.X):
                adata_all.layers["counts"] = adata_all.X.todense()
            else:
                adata_all.layers["counts"] = adata_all.X.copy()
        sc.pp.normalize_total(adata_all, target_sum=1e6)
        # sc.pp.normalize_total(adata_all, target_sum=1)
        sc.pp.log1p(adata_all)

        for i in range(len(adatas)):
            adata = adata_all[adata_all.obs['batch'] == str(i)]
            if issparse(adata.X):
                data = Data(edge_index=torch.LongTensor(np.array([edgeLists[i][0], edgeLists[i][1]])), 
                            x=torch.FloatTensor(adata.X.todense()))
            else:
                data = Data(edge_index=torch.LongTensor(np.array([edgeLists[i][0], edgeLists[i][1]])), 
                            x=torch.FloatTensor(adata.X))
            datas.append(data)

    edges = [dataset.edge_index for dataset in datas]
    features = [dataset.x for dataset in datas]
    adata_list = [adata_all[adata_all.obs['batch'] == str(i)] for i in range(len(adatas))]
    return adata_list, edges, features

def expression_GVAE(adatas:List[AnnData],
                    method:Optional[str]='GATE',
                    feature_method:Optional[str]='hvg',
                    k_cutoff:Optional[Union[None,int]]=8,
                    n_top_genes:Optional[int]=1000,
    ) -> AnnData:

    try:
        gpu_index = get_free_gpu()
        print(f"Choose GPU:{gpu_index} as device")
    except:
        print('GPU is not available')
    # device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')
    device = 'cuda:2'
    adata_all, edges, features=load_anndatas(adatas, feature=feature_method, k_cutoff=k_cutoff, n_top_genes=n_top_genes,method=method)

    for i in range(len(features)):
        features[i] = features[i].to(device)
    for j in range(len(edges)):
        edges[j] = edges[j].to(device)
    model = STAGATE(hidden_dims = [features[0].size(1)] + [512, 64]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    new_adatas=[]
    STAGATE_epochs=500
    gradient_clipping=5.0
    for adata_idx in range(len(features)):
        feature = features[adata_idx]
        edge = edges[adata_idx]

        for epoch in tqdm(range(1, STAGATE_epochs+1)):
        # for epoch in range(1, STAGATE_epochs+1):
            model.train()
            optimizer.zero_grad()
            z, out = model(feature, edge)
            loss = F.mse_loss(feature, out) #F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
            # if epoch % 10 == 0:
            #     print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

        model.eval()
        z, out = model(feature, edge)
        STAGATE_rep = z.to('cpu').detach().numpy()
        adata_all[adata_idx].obsm['embedding_rep'] = STAGATE_rep
        new_adatas.append(adata_all[adata_idx])

    return new_adatas  

