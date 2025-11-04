import numpy as np
from PIL import Image
from skimage.transform import rescale
import pandas as pd
import tifffile

def main(prefix):
    with open(prefix + 'pixel-size-raw.txt', 'r') as file:
        raw_pix_size = float([line.rstrip() for line in file][0])
    scale_pix_size = 0.5
    scale = raw_pix_size / scale_pix_size

    loc = pd.read_csv(prefix+'locs-raw.csv', header=0, index_col=0)
    loc = loc * scale
    loc = loc.round().astype(int)
    loc.columns = ['x', 'y']
    loc.to_csv(prefix+'locs.csv')

    img = np.array(Image.open(prefix+'he-raw.jpg'))
    img = rescale(img, [scale, scale, 1], preserve_range=True)
    img = img.astype(np.uint8)
    H, W, _ = img.shape
    img = img[:H // 224 * 224, :W // 224 * 224]
    Image.fromarray(img).save(prefix+'he.jpg')

def load_image(filename, verbose=True):
    if filename.endswith('tif'):
        img= tifffile.imread(filename)
    else:
        img = Image.open(filename)
        img = np.array(img)
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]  # remove alpha channel
    return img

def crop_image(img, extent, mode='edge', constant_values=None):
    extent = np.array(extent)
    pad = np.zeros((img.ndim, 2), dtype=int)
    for i, (lower, upper) in enumerate(extent):
        if lower < 0:
            pad[i][0] = 0 - lower
        if upper > img.shape[i]:
            pad[i][1] = upper - img.shape[i]
    if (pad != 0).any():
        kwargs = {}
        if mode == 'constant' and constant_values is not None:
            kwargs['constant_values'] = constant_values
        img = np.pad(img, pad, mode=mode, **kwargs)
        extent += pad[:extent.shape[0], [0]]
    for i, (lower, upper) in enumerate(extent):
        img = img.take(range(lower, upper), axis=i)
    return img

def adjust_margins(img, pad, pad_value=None):
    extent = np.stack([[0, 0], img.shape[:2]]).T  #extent 的形状为 (2, 2)，其中第一列为 0，表示图像的起始坐标，第二列为图像的形状（即高度和宽度），表示图像的结束坐标
    # make size divisible by pad without changing coords
    remainder = (extent[:, 1] - extent[:, 0]) % pad
    complement = (pad - remainder) % pad
    extent[:, 1] += complement
    if pad_value is None:
        mode = 'edge'
    else:
        mode = 'constant'
    img = crop_image(
            img, extent, mode=mode, constant_values=pad_value)
    return img
    
def align_image_pos_main(image_path,locs):
    pixel_size_raw = 0.25
    pixel_size = 0.5
    scale = pixel_size_raw / pixel_size  
    
    locs = locs * scale  
    locs = locs.round().astype(int)
    if type(image_path)==np.ndarray:
        img=image_path
    else:
        img = load_image(image_path)
    img = rescale(img, [scale, scale, 1], preserve_range=True)
    img = img.astype(np.uint8)
    img = adjust_margins(img, pad=224, pad_value=255)
    # H, W, _ = img.shape
    # img = img[:H // 224 * 224, :W // 224 * 224]
    return img, locs

def align_image(image_path):
    pixel_size_raw = 0.25
    pixel_size = 0.5
    scale = pixel_size_raw / pixel_size  
    
    if type(image_path)==np.ndarray:
        img=image_path
    else:
        img = load_image(image_path)
    img = rescale(img, [scale, scale, 1], preserve_range=True)
    img = img.astype(np.uint8)
    img = adjust_margins(img, pad=224, pad_value=255)
    return img

def align_pos(locs):
    pixel_size_raw = 0.25
    pixel_size = 0.5
    scale = pixel_size_raw / pixel_size  
    
    locs = locs * scale  
    locs = locs.round().astype(int)
    return locs