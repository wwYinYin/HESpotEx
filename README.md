# HESpotEx
Whole-slide histopathological images (WSIs) constitute a fundamental approach in cancer diagnosis and prognosis. Recently emerging spatial transcriptomics (ST) methods can reveal the spatial gene expression landscape behind the WSI. Despite this merit, the adoption of ST methods is limited by their high costs, especially in contrast to the cost-effectiveness and wide availability of WSIs. Therefore, here we propose HESpotEx, a dual-stream multimodal deep embedding framework to predict the spatial gene expression patterns from WSI images. Leveraging graph attention autoencoders, an image encoder and a GCN decoder, HESpotEx could predict spot-level expressions of 33 to 5457 genes solely from histological images of various disease tissue samples. Validation on external ST datasets and large-scale TCGA WSI dataset demonstrates its superior performance and better robustness, underscoring HESpotEx's potential in deciphering the molecular characteristics underlying tissue histological patterns.
## Install Guidelines
* We recommend you to use an Anaconda virtual environment. Install PyTorch >= 1.12 according to your GPU driver and Python >= 3.9, and run：

```
pip install -r requirements.txt
```
Normally, this installation takes about 10 to 20 minutes.
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
### Run
This is a leave-one-out cross-validation script with max_steps=30. This sample data has 10 samples and 42212 spots in total. It takes about 15 hours to run on an 80G A100. You can save time by setting a smaller max_steps value. Sample data comes from [HEST-1K](https://github.com/mahmoodlab/hest/?tab=readme-ov-file)
```
tutorials_ST1K.ipynb
```
