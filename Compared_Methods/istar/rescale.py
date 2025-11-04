import argparse
import os
from time import time
import pandas as pd
from skimage.transform import rescale
import numpy as np
from .image import crop_image
from .utils import (
        load_image, save_image, read_string, write_string,
        load_tsv, save_tsv)


def get_image_filename(prefix):
    file_exists = False
    for suffix in ['.jpg', '.png', '.tiff']:
        filename = prefix + suffix
        if os.path.exists(filename):
            file_exists = True
            break
    if not file_exists:
        raise FileNotFoundError('Image not found')
    return filename


# def rescale_image(img, scale):
#     if img.ndim == 2:
#         img = rescale(img, scale, preserve_range=True)
#     elif img.ndim == 3:
#         channels = img.transpose(2, 0, 1)
#         channels = [rescale_image(c, scale) for c in channels]
#         img = np.stack(channels, -1)
#     else:
#         raise ValueError('Unrecognized image ndim')
#     return img


def rescale_image(img, scale):
    if img.ndim == 2:
        scale = [scale, scale]
    elif img.ndim == 3:
        scale = [scale, scale, 1]
    else:
        raise ValueError('Unrecognized image ndim')
    img = rescale(img, scale, preserve_range=True)
    return img


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--image', action='store_true')
    parser.add_argument('--mask', action='store_true')
    parser.add_argument('--locs', action='store_true')
    parser.add_argument('--radius', action='store_true')
    args = parser.parse_args()
    return args


def main():

    args = get_args()

    pixel_size_raw = float(read_string(args.prefix+'pixel-size-raw.txt'))
    pixel_size = float(read_string(args.prefix+'pixel-size.txt'))
    scale = pixel_size_raw / pixel_size # rescale each image such that the size of one pixel is 0.5 × 0.5 μm2

    if args.image:
        img = load_image(get_image_filename(args.prefix+'he-raw'))
        img = img.astype(np.float32)
        print(f'Rescaling image (scale: {scale:.3f})...')
        t0 = time()
        img = rescale_image(img, scale)
        print(int(time() - t0), 'sec')
        img = img.astype(np.uint8)
        save_image(img, args.prefix+'he-scaled.jpg')

    if args.mask:
        mask = load_image(args.prefix+'mask-raw.png')
        mask = mask > 0
        if mask.ndim == 3:
            mask = mask.any(2)
        print(f'Rescaling mask (scale: {scale:.3f})...')
        t0 = time()
        mask = rescale_image(mask.astype(np.float32), scale)
        print(int(time() - t0))
        mask = mask > 0.5
        save_image(mask, args.prefix+'mask-scaled.png')

    if args.locs:
        locs = load_tsv(args.prefix+'locs-raw.tsv')
        locs = locs * scale
        locs = locs.round().astype(int)
        save_tsv(locs, args.prefix+'locs.tsv')

    if args.radius:
        radius = float(read_string(args.prefix+'radius-raw.txt'))
        radius = radius * scale
        radius = np.round(radius).astype(int)
        write_string(radius, args.prefix+'radius.txt')

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

def rescale_img(image_path):
    pixel_size_raw = 0.25
    pixel_size = 0.5
    scale = pixel_size_raw / pixel_size # rescale each image such that the size of one pixel is 0.5 × 0.5 μm2

    if type(image_path)==np.ndarray:
        img=image_path
    else:
        img = load_image(image_path)
    img = img.astype(np.float32)
    print(f'Rescaling image (scale: {scale:.3f})...')
    img = rescale_image(img, scale)
    img = img.astype(np.uint8)
    img = adjust_margins(img, pad=256, pad_value=255)
    return img

def rescale_locs(locs):
    pixel_size_raw = 0.25
    pixel_size = 0.5
    scale = pixel_size_raw / pixel_size # rescale each image such that the size of one pixel is 0.5 × 0.5 μm2

    locs = locs * scale
    locs = locs.round().astype(int)
    return locs

def rescale_main(image_path,locs):
    pixel_size_raw = 0.25
    pixel_size = 0.5
    scale = pixel_size_raw / pixel_size # rescale each image such that the size of one pixel is 0.5 × 0.5 μm2

    if type(image_path)==np.ndarray:
        img=image_path
    else:
        img = load_image(image_path)
    img = img.astype(np.float32)
    print(f'Rescaling image (scale: {scale:.3f})...')
    img = rescale_image(img, scale)
    img = img.astype(np.uint8)
    img = adjust_margins(img, pad=256, pad_value=255)

    # if args.mask:
    #     mask = load_image(args.prefix+'mask-raw.png')
    #     mask = mask > 0
    #     if mask.ndim == 3:
    #         mask = mask.any(2)
    #     print(f'Rescaling mask (scale: {scale:.3f})...')
    #     t0 = time()
    #     mask = rescale_image(mask.astype(np.float32), scale)
    #     print(int(time() - t0))
    #     mask = mask > 0.5
    #     save_image(mask, args.prefix+'mask-scaled.png')
    locs = locs * scale
    locs = locs.round().astype(int)
    return img, locs

    # if args.radius:
    #     radius = float(read_string(args.prefix+'radius-raw.txt'))
    #     radius = radius * scale
    #     radius = np.round(radius).astype(int)
    #     write_string(radius, args.prefix+'radius.txt')
    

if __name__ == '__main__':
    main()
