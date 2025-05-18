import pandas as pd
from scipy.io import mmread
import os
import scanpy as sc
import anndata
import numpy as np
import json

def normalize_data(data):
    data = np.nan_to_num(data).astype(float)
    data *= 10**6 / np.sum(data, axis=0, dtype=float)
    np.log2(data + 1, out=data)
    np.nan_to_num(data, copy=False)
    return data

#估计每个spot中的细胞数量
def estimate_cell_number_RNA_reads(st_df, mean_cell_numbers):
    # Read data
    expressions = st_df.values.astype(float)

    # Data normalization
    expressions_tpm_log = normalize_data(expressions)

    # Set up fitting problem
    RNA_reads = np.sum(expressions_tpm_log, axis=0, dtype=float)
    mean_RNA_reads = np.mean(RNA_reads)
    min_RNA_reads = np.min(RNA_reads)

    min_cell_numbers = 1 if min_RNA_reads > 0 else 0
    fit_parameters = np.polyfit(np.array([min_RNA_reads, mean_RNA_reads]),
                                np.array([min_cell_numbers, mean_cell_numbers]), 1)
    polynomial = np.poly1d(fit_parameters)
    cell_number_to_node_assignment = polynomial(RNA_reads).astype(int)
    if min_cell_numbers==1:
        cell_number_to_node_assignment[cell_number_to_node_assignment == 0] += 1
    return cell_number_to_node_assignment

def data_preprocessing_ST1K(input_path):
    st_adatas=[]
    HE_image_paths=[]
    nuclei_mask_paths=[]
    h5ad_data_paths = [d for d in os.listdir(input_path) if d.endswith('.h5ad')]

    for h5ad_filename in h5ad_data_paths:
        h5ad_path = os.path.join(input_path, h5ad_filename)
        print(h5ad_path)
        st_adata=sc.read_h5ad(h5ad_path)
        st_adata = st_adata[~st_adata.obs.index.duplicated(keep='first')]
        st_adata = st_adata[:, ~st_adata.var.index.duplicated(keep='first')]
        if 'in_tissue' in st_adata.obs.columns:
            st_adata=st_adata[st_adata.obs['in_tissue']==1]
        
        name_orig = list(st_adata.uns['spatial'])[0]
        block_r=np.ceil(st_adata.uns['spatial'][name_orig]['scalefactors']['spot_diameter_fullres'])
        st_adata = st_adata.copy()
        st_adata.uns['block_r'] = block_r
                
        st_adatas.append(st_adata)
        image_path=os.path.join(input_path, h5ad_filename.replace('.h5ad', '.tif'))
        if not os.path.exists(image_path):
            print('Image not found:', image_path)
        HE_image_paths.append(image_path)
        nuclei_mask_path=os.path.join(input_path, h5ad_filename.replace('.h5ad', '.npy'))
        if not os.path.exists(nuclei_mask_path):
            print('Nuclei mask not found:', nuclei_mask_path)
        nuclei_mask_paths.append(nuclei_mask_path)
                        
    return st_adatas, HE_image_paths, nuclei_mask_paths

def exp_data_preprocessing(input_path):
    st_adatas=[]
    HE_image_paths=[]
    nuclei_mask_paths=[]

    sub_dir = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]
    print(sub_dir)

    for d in sub_dir:
        fn_dir = os.path.join(input_path, d)
        print(fn_dir)
        if os.path.exists(os.path.join(fn_dir, 'filtered_feature_bc_matrix')):
        # 这里是如果存在'filtered_feature_bc_matrix'子文件夹时要执行的代码
            mtx = mmread(
                os.path.join(fn_dir,'filtered_feature_bc_matrix/matrix.mtx.gz')
            )
            genes = pd.read_csv(
                os.path.join(fn_dir,'filtered_feature_bc_matrix/features.tsv.gz'), 
                sep='\t', header=None,
                ).iloc[:,1].values
            barcodes = pd.read_csv(
                os.path.join(fn_dir,'filtered_feature_bc_matrix/barcodes.tsv.gz'),
                sep='\t', header=None).iloc[:,0].values    

            mtx = mtx.toarray()
            st_adata=sc.AnnData(mtx.T, obs=pd.DataFrame(index=barcodes), var=pd.DataFrame(index=genes))    
               
            position = pd.read_csv(
                os.path.join(fn_dir,'spatial/tissue_positions_list.csv'),
                sep=',', header=None,index_col=0)
            position.columns = [
                "in_tissue",
                "array_row",
                "array_col",
                "pxl_col_in_fullres",
                "pxl_row_in_fullres",
            ]
            tmp = position.sort_values(["array_row", "array_col"])
            block_y = int(np.median(tmp.pxl_row_in_fullres.values[2:-1] - tmp.pxl_row_in_fullres.values[1:-2]) // 2)
            tmp = position.sort_values(["array_col", "array_row"])
            block_x = int(np.median(tmp.pxl_col_in_fullres.values[2:-1] - tmp.pxl_col_in_fullres.values[1:-2]) // 2)
            block_r = min(block_x, block_y)

            position = position[position.index.isin(st_adata.obs_names)]
            intro_spots=position.iloc[:,0].where(position.iloc[:,0] == 1).dropna().index.tolist()
            # with open(os.path.join(fn_dir,'spatial/scalefactors_json.json')) as f:
            #     img_json = json.load(f)
            # try:
            #     # Need to use spot diamater here
            #     radius = int(img_json['fiducial'][0]['dia'] // 2)
            # except:
            #     radius = int(img_json['spot_diameter_fullres']// 2)
            st_adata.obs['array_row'] = position['array_row']
            st_adata.obs['array_col'] = position['array_col']
            st_adata.obsm['spatial']=position[["pxl_row_in_fullres", "pxl_col_in_fullres"]].to_numpy()
            st_adata.uns['block_r'] = block_r
            # st_adata.obs = st_adata.obs.join(position.iloc[:,1:])
            # st_adata.obs.columns = ['Row','Col','X','Y']
            # st_adata.obs['Spot_radius'] = radius
            st_adata=st_adata[intro_spots]
            st_adata = st_adata[~st_adata.obs.index.duplicated(keep='first')]
            st_adata = st_adata[:, ~st_adata.var.index.duplicated(keep='first')]

            # st_df=st_adata.to_df().T
            # cell_number_each_spot = estimate_cell_number_RNA_reads(st_df, 5) 
            # st_adata.obs['cell_number'] = cell_number_each_spot

            st_adatas.append(st_adata)
            if os.path.exists(os.path.join(fn_dir, 'spatial/image.tif')):
                HE_image_paths.append(os.path.join(fn_dir,'spatial/image.tif'))
                nuclei_mask_paths.append(os.path.join(fn_dir,'spatial/image.npy'))
            else:
                HE_image_paths.append(os.path.join(fn_dir,'spatial/image.jpg'))
                nuclei_mask_paths.append(os.path.join(fn_dir,'spatial/image.npy'))

        elif os.path.exists(os.path.join(fn_dir, 'filtered_feature_bc_matrix.h5')):
            st_adata=sc.read_10x_h5(os.path.join(fn_dir,'filtered_feature_bc_matrix.h5'))
            if os.path.exists(os.path.join(fn_dir, 'spatial/tissue_positions.csv')):
                position = pd.read_csv(
                    os.path.join(fn_dir,'spatial/tissue_positions.csv'),
                    sep=',', index_col=0)
            if os.path.exists(os.path.join(fn_dir, 'spatial/tissue_positions_list.txt')):
                position = pd.read_csv(
                    os.path.join(fn_dir,'spatial/tissue_positions_list.txt'),
                    sep=',', header=None,index_col=0)                
            # print(position)
            position.columns = [
                "in_tissue",
                "array_row",
                "array_col",
                "pxl_col_in_fullres",
                "pxl_row_in_fullres",
            ]
            tmp = position.sort_values(["array_row", "array_col"])
            # print(tmp)
            block_y = int(np.median(tmp.pxl_col_in_fullres.values[2:-1] - tmp.pxl_col_in_fullres.values[1:-2]) // 2)
            tmp = position.sort_values(["array_col", "array_row"])
            block_x = int(np.median(tmp.pxl_row_in_fullres.values[2:-1] - tmp.pxl_row_in_fullres.values[1:-2]) // 2)
            block_r = min(block_x, block_y)
            print(block_x, block_y, block_r)

            position = position[position.index.isin(st_adata.obs_names)]
            intro_spots=position.iloc[:,0].where(position.iloc[:,0] == 1).dropna().index.tolist()

            st_adata.uns['block_r'] = block_r
            st_adata.obs['array_row'] = position['array_row']
            st_adata.obs['array_col'] = position['array_col']
            st_adata.obsm['spatial']=position[["pxl_row_in_fullres", "pxl_col_in_fullres"]].to_numpy()

            st_adata=st_adata[intro_spots]
            st_adata = st_adata[~st_adata.obs.index.duplicated(keep='first')]
            st_adata = st_adata[:, ~st_adata.var.index.duplicated(keep='first')]
            st_adatas.append(st_adata)
            if os.path.exists(os.path.join(fn_dir, 'spatial/image.tif')):
                HE_image_paths.append(os.path.join(fn_dir,'spatial/image.tif'))
            else:
                HE_image_paths.append(os.path.join(fn_dir,'spatial/image.jpg'))
            # else:
            #     st_adata=sc.read_visium(path=fn_dir,count_file='filtered_feature_bc_matrix.h5',
            #                             load_images=True,source_image_path=os.path.join(fn_dir,'spatial/'))
            #     if 'in_tissue' in st_adata.obs.columns:
            #         st_adata=st_adata[st_adata.obs['in_tissue']==1]
            #     name_orig = list(st_adata.uns['spatial'])[0]
            #     scale_factor=st_adata.uns['spatial'][name_orig]['scalefactors']['tissue_hires_scalef']
            #     st_adata.uns['block_r'] = int(st_adata.uns["spatial"][name_orig]['scalefactors']['spot_diameter_fullres']*scale_factor*0.75)
            #     if 'spatial' not in st_adata.obsm.keys() :
            #         st_adata.obsm['spatial']=st_adata.obsm['X_spatial']
            #     st_adata = st_adata[~st_adata.obs.index.duplicated(keep='first')]
            #     st_adata = st_adata[:, ~st_adata.var.index.duplicated(keep='first')]
            #     st_adata.obsm['spatial']=(st_adata.obsm['spatial'] * scale_factor).astype(int)
            #     image = st_adata.uns["spatial"][name_orig]['images']['hires']
            #     if image.max() <= 1.0:
            #         image = image*255
            #     image = image.astype(np.uint8)
            #     st_adatas.append(st_adata)
            #     HE_image_paths.append(image)

        elif any(os.path.splitext(filename)[1] == '.h5ad' for filename in os.listdir(fn_dir)):
            if os.path.exists(os.path.join(fn_dir, 'image.jpg')) and os.path.exists(os.path.join(fn_dir, 'st_adata_allgene.h5ad')):
                # h5ad_filename = [filename for filename in os.listdir(fn_dir) if filename.endswith("adata.h5ad")][0]
                h5ad_filename = [filename for filename in os.listdir(fn_dir) if filename.endswith("allgene.h5ad")][0]
                st_adata=sc.read_h5ad(os.path.join(fn_dir,h5ad_filename))
                st_adata.obs['array_row'] = st_adata.obs['x']
                st_adata.obs['array_col'] = st_adata.obs['y']  
                st_adata = st_adata[~st_adata.obs.index.duplicated(keep='first')]
                st_adata = st_adata[:, ~st_adata.var.index.duplicated(keep='first')] 

                position=st_adata.obs.copy()
                position['pxl_row_in_fullres']=st_adata.obsm['spatial'][:,0].tolist()
                position['pxl_col_in_fullres']=st_adata.obsm['spatial'][:,1].tolist()
                tmp = position.sort_values(["array_row", "array_col"])
                block_y = int(np.median(tmp.pxl_row_in_fullres.values[2:-1] - tmp.pxl_row_in_fullres.values[1:-2]) // 2)
                tmp = position.sort_values(["array_col", "array_row"])
                block_x = int(np.median(tmp.pxl_col_in_fullres.values[2:-1] - tmp.pxl_col_in_fullres.values[1:-2]) // 2)
                block_r = min(block_x, block_y)

                st_adata.uns['block_r'] = block_r
                
                st_adatas.append(st_adata)
                HE_image_paths.append(os.path.join(fn_dir, 'image.jpg'))  
                nuclei_mask_paths.append(os.path.join(fn_dir,'image.npy')) 

            else:
                h5ad_filename = [filename for filename in os.listdir(fn_dir) if filename.endswith("adata.h5ad")][0]
                st_adata=sc.read_h5ad(os.path.join(fn_dir,h5ad_filename))
                st_adata = st_adata[~st_adata.obs.index.duplicated(keep='first')]
                st_adata = st_adata[:, ~st_adata.var.index.duplicated(keep='first')] 
                if input_path.endswith('V_HD_Mouse_Brain_bin16/'):
                    import json
                    with open(os.path.join(input_path,'sub_1/scalefactors_json.json'), 'r') as f:
                        json_data = json.load(f)
                    HE_image_paths.append(os.path.join(input_path, 'sub_1/image.tif')) 
                    st_adata.uns['block_r'] = int(json_data['spot_diameter_fullres']*0.5)

                if input_path.endswith('V_HD_Small_Intestine_bin16/'):
                    import json
                    with open(os.path.join(input_path,'sub_1/scalefactors_json.json'), 'r') as f:
                        json_data = json.load(f)
                    HE_image_paths.append(os.path.join(input_path, 'sub_1/image.btf')) 
                    st_adata.uns['block_r'] = int(json_data['spot_diameter_fullres']*0.5)
                    nuclei_mask_paths.append(os.path.join(input_path, 'sub_1/image.npy'))

                if input_path.endswith('lymphoma_spatial/'):
                    position=st_adata.obsm['spatial']
                    st_adata.uns['block_r'] = 112
                    HE_image_paths.append(os.path.join(fn_dir,'spatial/image2.tif'))
                    nuclei_mask_paths.append(os.path.join(fn_dir,'spatial/image2.npy'))

                if input_path.endswith('skin/'):
                    st_adata.uns['block_r'] = 112
                    st_adata.obs['array_row'] = st_adata.obs['x']
                    st_adata.obs['array_col'] = st_adata.obs['y'] 
                    HE_image_paths.append(os.path.join(fn_dir,'image.jpg'))
                    nuclei_mask_paths.append(os.path.join(fn_dir,'image.npy'))
                
                st_adatas.append(st_adata)                           

    return st_adatas, HE_image_paths, nuclei_mask_paths