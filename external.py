#!/usr/bin/env python3
import numpy as np
from osgeo import gdal

import torch

def normalise(arr):
    normalized = (arr - arr.min()) / (arr.max() - arr.min())

    normalized = 2*normalized - 1

    return normalized

def tile(file, kernel_size):

    dem = gdal.Open(file)

    crs = dem.GetProjection()
    geo_transform = dem.GetGeoTransform()

    image = dem.ReadAsArray()

    img_height, img_width = image.shape
    tile_height, tile_width = kernel_size

    # If cant divide perfectly
    if (img_height % tile_height != 0 or img_width % tile_width != 0):
        new_height = img_height - (img_height % tile_height)
        new_width = img_width - (img_width % tile_width)

        image = image[:new_height, :new_width]

    tiles_high = img_height // tile_height
    tiles_wide = img_width // tile_width

    tiled_array = image.reshape(tiles_high,
                                tile_height,
                                tiles_wide,
                                tile_width )

    tiled_array = tiled_array.swapaxes(1, 2)

    tiled_array = tiled_array.reshape(tiles_high * tiles_wide, tile_height, tile_width)

    # GC should get this, but just to be safe
    dem = None

    min_max = []
    for arr in tiled_array:
        min_max.append((arr.min(), arr.max()))

    vectorized_normalise = np.vectorize(normalise, signature='(n,m)->(n,m)')

    tiled_array = vectorized_normalise(tiled_array)

    # Slope
    cellsize = geo_transform[1]
    px, py = np.gradient(tiled_array, cellsize, axis=(1,2))
    slope = np.arctan(np.sqrt(px ** 2 + py ** 2))
    slope = vectorized_normalise(slope)

    all = np.stack((tiled_array, slope), axis=3)

    # tiled_array = np.expand_dims(tiled_array, axis=3)
    all = np.transpose(all, (0, 3, 1, 2))

    # H,W,C to C,H,W
    return torch.from_numpy(all), min_max, crs, geo_transform

def process_losses(log):
    epoch = []
    l1 = []
    ae = []
    wgan_g = []
    wgan_d = []
    wgan_gp = []
    g = []
    d = []

    for line in log:
        s = line.split(',')

        epoch.append(int(s[0].split(':')[1][1:]))
        l1.append(float(s[2].split(':')[1][1:]))
        ae.append(float(s[3].split(':')[1][1:]))
        wgan_g.append(float(s[4].split(':')[1][1:]))
        wgan_d.append(float(s[5].split(':')[1][1:]))
        wgan_gp.append(float(s[6].split(':')[1][1:]))
        g.append(float(s[7].split(':')[1][1:]))
        d.append(float(s[8].split(':')[1][1:]))

    return (epoch, l1, ae, wgan_g, wgan_d, wgan_gp, g, d)
