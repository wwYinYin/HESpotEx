# HESpotEx
Whole-slide histopathological images (WSIs) constitute a fundamental approach in cancer diagnosis and prognosis. Recently emerging spatial transcriptomics (ST) methods can reveal the spatial gene expression landscape behind the WSI. Despite this merit, the adoption of ST methods is limited by their high costs, especially in contrast to the cost-effectiveness and wide availability of WSIs. Therefore, here we propose HESpotEx, a dual-stream multimodal deep embedding framework to predict the spatial gene expression patterns from WSI images. Leveraging graph attention autoencoders, an image encoder and a GCN decoder, HESpotEx could predict spot-level expressions of 33 to 5457 genes solely from histological images of various disease tissue samples. Validation on external ST datasets and large-scale TCGA WSI dataset demonstrates its superior performance and better robustness, underscoring HESpotEx's potential in deciphering the molecular characteristics underlying tissue histological patterns.
## Install Guidelines
* We recommend you to use an Anaconda virtual environment. Install PyTorch >= 1.12 according to your GPU driver and Python >= 3.9, and run：

```
pip install -r requirements.txt
```
Before running the tutorial, you need to download the weight file of [Quilt-Net](https://huggingface.co/wisdomik/QuiltNet-B-32/blob/main/open_clip_pytorch_model.bin) and put it in the folder ./model/QuiltNet-B-32/.

## Tutorial
