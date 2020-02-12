import imageio
from pathlib import Path
import unittest
from unittest.mock import Mock

import cv2
import numpy as np
from scipy import ndimage
from numpy import fft

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
        histogram /= image.size

        hcfcom = metrics.HCFCOM()
        hist = hcfcom.histogram(image)
        np.testing.assert_array_almost_equal(hist, histogram)

    def test_histogram_shape(self):
        image = imageio.imread('{}/01.bmp'.format(IMAGE_DIR))
        hcfcom = metrics.HCFCOM()
        hist = hcfcom.histogram(image)
        np.testing.assert_array_equal(hist.shape, (3, 256))


class CenterOfMassTest(unittest.TestCase):
    def test_center_mass(self):
        image = imageio.imread('{}/01.bmp'.format(IMAGE_DIR))
        value1 = 3.5+1j
        value2 = 4.2+0j
        value3 = 5.6+4j

        ndimage.center_of_mass = Mock(
            side_effect=[[value1], [value2], [value3]])
        hcfcom = metrics.HCFCOM()

        com = hcfcom(image)
        np.testing.assert_array_equal(com, [value1, value2, value3])
