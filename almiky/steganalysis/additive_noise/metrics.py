import cv2
import numpy as np
from scipy import ndimage
from scipy.fftpack import dct

class HCFCOM:

    def histogram(self, image):
        '''
        color_histogram(image) => numpy array: return histogram of color image
        @image: np array representative of image

        Calculate histogram of image per channel (R, G, B)
        using opencv calHist function. Array returned has (3, N) shape where
        N is bean of histogram
        '''

        red_histogram = cv2.calcHist([image],[0],None,[256],[0,256]).reshape(-1)
        green_histogram = cv2.calcHist([image],[1],None,[256],[0,256]).reshape(-1)
        blue_histogram = cv2.calcHist([image],[2],None,[256],[0,256]).reshape(-1)

        return np.array([red_histogram, green_histogram, blue_histogram])

    def hchf(self, image):
        '''
        HISTOGRAM CHARACTERISTIC FUNCTION
        '''
        histogram = self.histogram(image)
        N = int(histogram.shape[1] / 2)
        return [
            dct(chanel, n=N)
            for chanel in histogram
        ]

    def __call__(self, image):
        '''
        center_mass(histogram) => numpy array: Calculate center of mass of
        histogram characteristic function (a representation of the color image
        histogram in the frequency domain)
        '''

        return np.array([
            ndimage.center_of_mass(channel)[0]
            for channel in self.hchf(image)
        ])
