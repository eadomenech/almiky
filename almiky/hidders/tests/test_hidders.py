import unittest
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np

from almiky.hidders import frequency
from almiky.utils import ortho_matrix


class HidderFrequencyLeastSignificantBit(unittest.TestCase):
    def test_with_dct_8x8(self):
        from almiky.moments.matrix import Transform
        from almiky.utils.ortho_matrix import dct

        transform = Transform(dct)
        cover_array = np.random.rand(64, 64)
        hidder = frequency.HidderFrequencyLeastSignificantBit(transform)

        watermarked_array = hidder.insert(cover_array, 'anier', coeficient_index=10)
        msg = hidder.extract(watermarked_array, coeficient_index=10)

        self.assertTrue(msg.startswith('anier'))


class HidderEightFrequencyCoeficients(unittest.TestCase):
    def test_with_dct_8x8(self):
        from almiky.moments.matrix import Transform
        from almiky.utils.ortho_matrix import dct

        trasform = Transform(dct)
        cover_array = np.random.rand(32, 32)
        hidder = frequency.HidderEightFrequencyCoeficients(trasform)

        watermarked_array = hidder.insert(cover_array, 'anier')
        msg = hidder.extract(watermarked_array)

        self.assertTrue(msg.startswith('anier'))
