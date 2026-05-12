# HESpotEx
Whole-slide histopathological images (WSIs) constitute a fundamental approach in cancer diagnosis and prognosis. Recently emerging spatial transcriptomics (ST) methods can reveal the spatial gene expression landscape behind the WSI. Despite this merit, the adoption of ST methods is limited by their high costs, especially in contrast to the cost-effectiveness and wide availability of WSIs. Therefore, here we propose HESpotEx, a dual-stream multimodal deep embedding framework to predict the spatial gene expression patterns from WSI images. Leveraging graph attention autoencoders, an image encoder and a GCN decoder, HESpotEx could predict spot-level expressions of 33 to 5457 genes solely from histological images of various disease tissue samples. Validation on external ST datasets and large-scale TCGA WSI dataset demonstrates its superior performance and better robustness, underscoring HESpotEx's potential in deciphering the molecular characteristics underlying tissue histological patterns.
![](https://github.com/wwYinYin/HESpotEx/blob/main/HESpotEx_Overview.png)
## Install Guidelines
* We recommend you to use an Anaconda virtual environment. Install PyTorch >= 1.12 according to your GPU driver and Python >= 3.9, and run：

```
pip install -r requirements.txt
```
Normally, this installation takes about 10 to 20 minutes.

## Data Availability
The publicly available expression datasets analyzed in this work are available in previous studies. 
* The HER2+ dataset can be obtained from [her2st](https://github.com/almaan/her2st/)
* The external breast cancer validation cohort can be available at [breast cancer](https://www.spatialresearch.org/resources-published-datasets/doi-10-1126science-aaf2403/)
* The cSCC dataset is available from the Gene Expression Omnibus (GEO) with accession numbers [GSE144240](https://www.ncbi.xyz/geo/query/acc.cgi?acc=GSE144240)
* The TCGA-BRCA datasets are obtained from the National Cancer Institute Genomic Data Commons Portal [TCGA-BRCA](https://portal.gdc.cancer.gov/)
* The Xenium colorectal cancer dataset is available from the 10x Visium platform [Xenium CRC](https://www.10xgenomics.com/products/visium-hd-spatial-gene-expression/dataset-human-crc)
* The Visium HD mouse small intestine is available from the 10x Visium HD platform [Visium HD](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-mouse-intestine)
* The Human cancerous and healthy colon tissues ST datasets is available from [HEST-1k](https://huggingface.co/datasets/MahmoodLab/hest)
* The non-communicable inflammatory skin diseases (ncISDs) dataset is available from the Gene Expression Omnibus (GEO) with accession numbers [GSE206391](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE206391)

Meanwhile, we have uploaded the processed datasets to the Zenodo platform [HESpotEx1](https://zenodo.org/records/18281197) and [HESpotEx2](https://zenodo.org/records/18281215). These datasets can be directly used to run the code.
## Tutorial
Before running the tutorial, you need to download the weight file of [Quilt-Net](https://huggingface.co/wisdomik/QuiltNet-B-32/blob/main/open_clip_pytorch_model.bin) and put it in the folder ./model/QuiltNet-B-32/.
### Input files
* A spatial transcriptomics (ST) data with spot space coordinates (.h5ad)
* A high-resolution H&E-stained image corresponding to the ST data (.jpg, .tif, .png)
* Use Cellpose to segment the nucleus of the H&E-stained image and save the segmentation results as .npy files
  ```
  cellpose_calculate_nuclir.ipynb
  ```
* We recommend running the Cellpose script first, as this step is time-consuming. It takes about an hour to process a 20,000 x 20,000 H&E image.
* The data file structure of this project should be organized as follows:

```
data/
├── HEST
│   ├── COLON-CANCER-HEALTHY
│   │   ├── TENX89.h5ad
│   │   ├── TENX89.tif
│   │   ├── TENX89.npy
│   │   ├── TENX90.h5ad
│   │   ├── TENX90.tif
│   │   ├── TENX90.npy
          ...
│   │   ├── TENX92.h5ad
│   │   ├── TENX92.tif
│   │   └── TENX92.npy
│   ├── COLON-CANCER_Xenium
│   ├── SKIN-AD
        ...
│   └── SKIN-LP
```
### Output files
* my_pre.h5ad: Predicted expression profile
* my_gt.h5ad: Ground truth after normalization and high-variable gene filtering
### Run
This is a leave-one-out cross-validation script with max_steps=30. This sample data has 10 samples and 42212 spots in total. It takes about 15 hours to run on an 80G A100. You can save time by setting a smaller max_steps value. Sample data comes from [HEST-1K](https://github.com/mahmoodlab/hest/?tab=readme-ov-file)
```
tutorials_ST1K.ipynb
```
For other datasets mentioned in the article, simply replace the data input section in the tutorials with the folder containing the other datasets.

