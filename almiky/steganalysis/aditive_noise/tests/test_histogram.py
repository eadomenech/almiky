import imageio
from pathlib import Path
import unittest
from unittest.mock import Mock

from almiky.steganalysis.aditive_noise import histogram
import cv2
import numpy as np

IMAGE_DIR = Path(__file__).parent.joinpath('images')


class ColorImageHistogramTest(unittest.TestCase):

    def test_histogram_value(self):
        def side_effect(img, channel, x, dean, range):
            if channel == [0]:
                value = 0
            elif channel == [1]:
                value = 1
            elif channel == [2]:
                value = 2
            return value

        cv2.calcHist = Mock(side_effect=side_effect)
        image = imageio.imread('{}/01.bmp'.format(IMAGE_DIR))

        hist = histogram.color_histogram(image)
        np.testing.assert_array_equal(hist, np.array([0, 1, 2]))

    def test_histogram_shape(self):
        image = imageio.imread('{}/01.bmp'.format(IMAGE_DIR))
        hist = histogram.color_histogram(image)
        np.testing.assert_array_equal(hist.shape, (3, 256))

