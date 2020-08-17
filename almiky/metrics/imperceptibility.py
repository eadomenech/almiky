'''
Imperceptibility performance metrics
'''

from math import log10
import numpy as np


def mse(cover_work, ws_work):
    '''
    Calculate an return Mean Square Error (MSE)
    between a cover work an wattermaked work (or stego work)
    '''

    if cover_work.shape != ws_work.shape:
        raise ValueError('Cover and watermak/stego work have different shape')

    dims_cover = cover_work.shape
    m = dims_cover[0]
    n = dims_cover[1]
    C, S = cover_work * 1.0, ws_work * 1.0
    diff_pow = (np.abs(C) - np.abs(S)) ** 2
    if len(dims_cover) == 2:
        return sum(sum(diff_pow)) / (m * n)
    elif len(dims_cover) == 3:
        return sum(sum(sum(diff_pow))) / (m * n * dims_cover[2])


def psnr(cover_array, stego_array, max=255):
    '''
    Peak Signal-to-Noise Ratio (PSNR)
    '''
    RMSE = mse(cover_array, stego_array) or 1 / cover_array.size
    return 10 * log10(max ** 2 / RMSE) if RMSE else 100
