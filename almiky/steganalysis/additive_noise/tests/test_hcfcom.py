import imageio
from pathlib import Path
import unittest
from unittest.mock import Mock

import cv2
import numpy as np
from scipy import ndimage
from scipy import fftpack

from almiky.steganalysis.additive_noise import metrics

IMAGE_DIR = Path(__file__).parent.joinpath('images')


class ColorImageHistogramTest(unittest.TestCase):

    def test_histogram_value(self):
        red_histogram = np.random.rand(1, 256)
        green_histogram = np.random.rand(1, 256)
        blue_histogram = np.random.rand(1, 256)
        histogram = np.array([
            red_histogram.reshape(-1),
            green_histogram.reshape(-1),
            blue_histogram.reshape(-1)]
        )

        def side_effect(img, channel, x, dean, range):
            if channel == [0]:
                value = red_histogram
            elif channel == [1]:
                value = green_histogram
            elif channel == [2]:
                value = blue_histogram
            return value

        cv2.calcHist = Mock(side_effect=side_effect)
        image = imageio.imread('{}/01.bmp'.format(IMAGE_DIR))

        hcfcom = metrics.HCFCOM()
        hist = hcfcom.histogram(image)
        np.testing.assert_array_almost_equal(hist, histogram)

    def test_histogram_shape(self):
        image = imageio.imread('{}/01.bmp'.format(IMAGE_DIR))
        hcfcom = metrics.HCFCOM()
        hist = hcfcom.histogram(image)
        np.testing.assert_array_equal(hist.shape, (3, 256))


class HistogramCharacteristicFunction(unittest.TestCase):
    def test_hcf(self):
        image = imageio.imread('{}/01.bmp'.format(IMAGE_DIR))

        transform1 = np.random.rand(128)
        transform2 = np.random.rand(128)
        transform3 = np.random.rand(128)
        output = np.array([transform1, transform2, transform3])
        fftpack.dct = Mock(side_effect=[transform1, transform2, transform3])

        hcfcom = metrics.HCFCOM()
        hcf = hcfcom.hchf(image)
        np.testing.assert_array_equal(hcf, output)


class CenterOfMassTest(unittest.TestCase):
    def test_center_mass(self):
        image = imageio.imread('{}/01.bmp'.format(IMAGE_DIR))

        ndimage.center_of_mass = Mock(side_effect=[[2], [3], [4]])
        hcfcom = metrics.HCFCOM()

        com = hcfcom(image)
        np.testing.assert_array_equal(com, [2, 3, 4])
