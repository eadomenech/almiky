import cv2
import numpy as np
from scipy import ndimage
from numpy import fft


class HCFCOM:
    '''
    Center of mass (COM) of histogram characteristic function (HCF) of
    color images. HCF is a representation of the image histogram in
    the frequency domain
    '''

    def histogram(self, image):
        '''
        color_histogram(image) => numpy array: return histogram of color image
        @image: np array representative of image

        Calculate histogram of image per channel (R, G, B)
        using opencv calHist function. Array returned has (3, N) shape where
        N is bean of histogram
        '''
        # FIXME: arbritarly 8 bit image (256 posibles values) is used
        red_histogram = cv2.calcHist(
            [image], [0], None, [256], [0, 256]).reshape(-1)
        green_histogram = cv2.calcHist(
            [image], [1], None, [256], [0, 256]).reshape(-1)
        blue_histogram = cv2.calcHist(
            [image], [2], None, [256], [0, 256]).reshape(-1)

        normalize_histogram = np.array([
            red_histogram,
            green_histogram,
            blue_histogram
        ]) / image.size

        return normalize_histogram

    def __call__(self, image):
        '''
        center_mass(histogram) => numpy array: return color image`s center of
        mass. Array returned has (3,) shape with one component for each
        imagen color channels.
        '''

        # Characteristic function
        hchf = fft.ifft(self.histogram(image))
        # Only [0, N/2 -1 ] coeficients of DFT are used
        # to calculate center of mass
        # FIXME: arbritarly 8 bit image (256 posibles values) is used
        hchf_modified = np.absolute(hchf[:,:127])

        return np.array([
            ndimage.center_of_mass(channel)[0]
            for channel in hchf_modified
        ])
