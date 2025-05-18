r"""
Data preprocess and build graph
"""
from typing import Optional, Union, List

import pandas as pd
import numpy as np
import scanpy as sc
import torch
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.utils import to_undirected
from anndata import AnnData
import sklearn.neighbors
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from .exp_module import refine_label, dopca


def Cal_Spatial_Net(adata:AnnData,
                    rad_cutoff:Optional[Union[None,int]]=None,
                    k_cutoff:Optional[Union[None,int]]=None, 
                    model:Optional[str]='Radius',
                    return_data:Optional[bool]=True,
                    verbose:Optional[bool]=True
    ) -> AnnData:
    r"""
    构建空间邻居网络。

    参数
    ----------
    adata
        scanpy包的AnnData对象。
    rad_cutoff
        当model='Radius'时的半径截止值
    k_cutoff
        当model='KNN'时的最近邻居数量
    model
        网络构建模型。当model=='Radius'时，点与距离小于rad_cutoff的点相连。 
        当model=='KNN'时，点与其前k_cutoff个最近邻居相连。
    
    返回
    -------
    空间网络保存在adata.uns['Spatial_Net']中
    """

    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('Calculating spatial neighbor graph ...')

    if model == 'KNN':
        # 使用knn_graph函数计算K最近邻图
        edge_index = knn_graph(x=torch.tensor(adata.obsm['spatial']), flow='target_to_source',
                                k=k_cutoff, loop=True, num_workers=8)
        # 确保图是无向的
        edge_index = to_undirected(edge_index, num_nodes=adata.shape[0]) 
    elif model == 'Radius':
        # 使用radius_graph函数计算半径图
        edge_index = radius_graph(x=torch.tensor(adata.obsm['spatial']), flow='target_to_source',
                                    r=rad_cutoff, loop=True, num_workers=8) 

    # 将edge_index转换为DataFrame
    graph_df = pd.DataFrame(edge_index.numpy().T, columns=['Cell1', 'Cell2'])
    # 创建一个映射，将索引映射到观察名
    id_cell_trans = dict(zip(range(adata.n_obs), adata.obs_names))
    # 使用映射更新Cell1和Cell2的值
    graph_df['Cell1'] = graph_df['Cell1'].map(id_cell_trans)
    graph_df['Cell2'] = graph_df['Cell2'].map(id_cell_trans)
    # 将结果保存在adata.uns['Spatial_Net']中
    adata.uns['Spatial_Net'] = graph_df
    
    if verbose:
        print(f'The graph contains {graph_df.shape[0]} edges, {adata.n_obs} cells.')
        print(f'{graph_df.shape[0]/adata.n_obs} neighbors per cell on average.')

    if return_data:
        return adata

from collections import Counter
def scanpy_workflow(adatas:List[AnnData],
                    n_top_genes:Optional[int]=1000
    ) -> AnnData:
    all_highly_variable_genes=[]
    for raw_adata in adatas:
        adata = raw_adata.copy()
        if "highly_variable" not in adata.var_keys() and adata.n_vars > n_top_genes:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3")

        highly_variable_genes = adata.var[adata.var['highly_variable']].index.tolist()
        print(f'Highly variable genes: {len(highly_variable_genes)}')
        # gene_counts.update(highly_variable_genes)
        all_highly_variable_genes+=highly_variable_genes
    all_highly_variable_genes = list(set(all_highly_variable_genes))
    # 仅保留在一半以上的 adata 中都是高变基因的基因
    # all_highly_variable_genes = [gene for gene, count in gene_counts.items() if count >= total_adatas / 6]
    return all_highly_variable_genes

def scanpy_workflow_ST1K(adatas:List[AnnData],
                    n_top_genes:Optional[int]=1000
    ) -> AnnData:
    all_highly_variable_genes=[]
    for raw_adata in adatas:
        adata = raw_adata.copy()
        if "highly_variable" not in adata.var_keys() and adata.n_vars > n_top_genes:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3",span=0.8)

        highly_variable_genes = adata.var[adata.var['highly_variable']].index.tolist()
        print(f'Highly variable genes: {len(highly_variable_genes)}')
        # gene_counts.update(highly_variable_genes)
        all_highly_variable_genes+=highly_variable_genes
    all_highly_variable_genes = list(set(all_highly_variable_genes))
    return all_highly_variable_genes

def Spatial_Dis_Cal(adata, rad_dis=None, knn_dis=None, model='Radius', verbose=True):
    """\
    Calculate the spatial neighbor networks, as the distance between two spots.
    Parameters
    ----------
    adata:  AnnData object of scanpy package.
    rad_dis:  radius distance when model='Radius' 
    knn_dis:  The number of nearest neighbors when model='KNN' 
    model:
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_dis. 
        When model=='KNN', the spot is connected to its first knn_dis nearest neighbors.
    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """
    assert(model in ['Radius', 'KNN']) 
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index 
    # coor.columns = ['imagerow', 'imagecol']
    coor.columns = ['Spatial_X', 'Spatial_Y'] 

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_dis).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
      
        KNN_list = []
        for spot in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([spot]*indices[spot].shape[0], indices[spot], distances[spot])))
    
    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=knn_dis+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for spot in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([spot]*indices.shape[1],indices[spot,:], distances[spot,:])))

    KNN_df = pd.concat(KNN_list) 
    KNN_df.columns = ['Spot1', 'Spot2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    id_spot_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Spot1'] = Spatial_Net['Spot1'].map(id_spot_trans) 
    Spatial_Net['Spot2'] = Spatial_Net['Spot2'].map(id_spot_trans) 
    if verbose:
        print('The graph contains %d edges, %d spots.' %(Spatial_Net.shape[0], adata.n_obs)) 
        print('%.4f neighbors per spot on average.' %(Spatial_Net.shape[0]/adata.n_obs)) 
    adata.uns['Spatial_Net'] = Spatial_Net

def Adata2Torch_data(adata): 
    G_df = adata.uns['Spatial_Net'].copy() 
    spots = np.array(adata.obs_names) 
    spots_id_tran = dict(zip(spots, range(spots.shape[0]))) 
    G_df['Spot1'] = G_df['Spot1'].map(spots_id_tran) 
    G_df['Spot2'] = G_df['Spot2'].map(spots_id_tran) 

    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Spot1'], G_df['Spot2'])), 
        shape=(adata.n_obs, adata.n_obs))

    G = G + sp.eye(G.shape[0]) 
    edgeList = np.nonzero(G) 

    if type(adata.X) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X))  
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense())) 
    data = train_test_split_edges(data)
    return data

def process_adata(adata):
    adata.var_names_make_unique()
    #Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000)
    sc.pp.normalize_total(adata, target_sum=1e4) ##normalized data
    sc.pp.log1p(adata)  #log-transformed data
    adata = adata[:, adata.var['highly_variable']]
    return adata

def get_initial_label(adata, n_clusters, refine=True):
    adata.var_names_make_unique()
    #Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4) ##normalized data
    sc.pp.log1p(adata)  #log-transformed data
    adata = adata[:, adata.var['highly_variable']]
    features = adata.X.copy()
    if type(features) == np.ndarray:
        features = features
    else:
        features = features.todense()
    features=np.array(features)
    pca_input = dopca(features, dim = 20)
    adata.obsm["pca"] = pca_input
    sc.pp.neighbors(adata, n_neighbors=50, use_rep="pca")
    sc.tl.louvain(adata, resolution=n_clusters, random_state=0)
    pred=adata.obs['louvain'].astype(int).to_numpy()
    if refine:
        pred = refine_label(pred, adata.obsm["spatial"], radius=60)
    pred = list(map(int, pred))
    return np.array(pred)