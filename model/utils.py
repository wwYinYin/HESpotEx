r"""
Miscellaneous utilities
"""
import random

from pynvml import *
from anndata import AnnData
from typing import List, Optional, Union

import numpy as np
import torch


def norm_to_raw(
    adata: AnnData, 
    library_size: Optional[Union[str,np.ndarray]] = 'total_counts',
    check_size: Optional[int] = 100
) -> AnnData:
    r"""
    将标准化的adata.X转换为原始计数
    
    参数
    ----------
    adata
        需要转换的adata
    library_size
        每个细胞的原始库大小，可以是`adata.obs`的一个键或一个数组
    check_size
        检查头部`[0:check_size]`行和列以判断adata是否已标准化
    
    注意
    ----------
    Adata必须遵循scanpy官方的标准化步骤 
    """
    # 获取数据的一个子集，用于检查数据是否已经标准化
    check_chunk = adata.X[0:check_size,0:check_size].todense()
    # 确保数据已经标准化，即数据中的所有元素都不是整数
    assert not all(isinstance(x, int) for x in check_chunk)
    
    from scipy import sparse
    # 计算每个细胞的标准化因子
    scale_size = np.array(adata.X.expm1().sum(axis=1).round()).flatten() 
    # 根据`library_size`参数的类型，计算每个细胞的缩放因子
    if isinstance(library_size, str):
        scale_factor = np.array(adata.obs[library_size])/scale_size
    elif isinstance(library_size, np.ndarray):
        scale_factor = library_size/scale_size
    else:
        try:
            scale_factor = np.array(library_size)/scale_size
        except:
            raise ValueError('Invalid `library_size`')
    # 将缩放因子调整为正确的形状
    scale_factor.resize((scale_factor.shape[0],1))
    # 计算原始计数
    raw_count = sparse.csr_matrix.multiply(sparse.csr_matrix(adata.X).expm1(), sparse.csr_matrix(scale_factor))
    # 将原始计数四舍五入到最接近的整数
    raw_count = sparse.csr_matrix(np.round(raw_count))
    # 将原始计数存储回AnnData对象
    # adata.X = raw_count
    return adata


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

def get_two_free_gpus() -> List[int]:
    r"""
    Get indices of two GPUs with least memory usage
    
    Ref
    ----------
    https://stackoverflow.com/questions/58216000/get-total-amount-of-free-gpu-memory-and-available-using-pytorch
    """
    nvmlInit()
    memory_info = []
    for i in range(torch.cuda.device_count()):
        h = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(h)
        memory_info.append((info.free, i))
        
    # Sort GPUs by free memory and get indices of two GPUs with most free memory
    memory_info.sort(reverse=True)
    indices = [memory_info[i][1] for i in range(2)]
    
    return indices


def global_seed(seed: int) -> None:
    r"""
    Set seed
    
    Parameters
    ----------
    seed 
        int
    """
    if seed > 2**32 - 1 or seed <0:
        seed = 0

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Global seed set to {seed}.")