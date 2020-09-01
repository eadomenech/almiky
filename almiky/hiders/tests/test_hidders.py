import unittest
from unittest.mock import Mock
from unittest.mock import patch
from pathlib import Path

import numpy as np

from almiky.hidders import frequency
from almiky.utils import ortho_matrix

base = Path(__file__).parent.parent.parent

# class HidderFrequencyLeastSignificantBit(unittest.TestCase):
#     def test_with_dct_8x8(self):
#         from almiky.moments.matrix import Transform
#         from almiky.utils.ortho_matrix import dct

#         transform = Transform(dct)
#         cover_array = np.random.rand(64, 64)
#         hidder = frequency.HidderFrequencyLeastSignificantBit(transform)

#         watermarked_array = hidder.insert(cover_array, 'anier', coeficient_index=10)
#         msg = hidder.extract(watermarked_array, coeficient_index=10)

#         self.assertTrue(msg.startswith('anier'))


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

    def test_with_charlier_8x8(self):
        from almiky.moments.matrix import Transform
        from almiky.utils.ortho_matrix import charl

        trasform = Transform(charl)
        cover_array = np.random.rand(32, 32)
        hidder = frequency.HidderEightFrequencyCoeficients(trasform)

        watermarked_array = hidder.insert(cover_array, 'anier')
        msg = hidder.extract(watermarked_array)

        self.assertTrue(msg.startswith('anier'))

    def test_with_qkrawtchouk_8x8(self):
        from almiky.moments.matrix import QKrawtchoukMatrix

        path = base.joinpath("messages/msg64bytes.txt")
        with open(str(path), "r") as file:
            msg = file.read()

        trasform = QKrawtchoukMatrix(8, p=707, q=0.77, N=7)
        cover_array = np.random.rand(72, 72)
        hidder = frequency.HidderEightFrequencyCoeficients(trasform)

        watermarked_array = hidder.insert(cover_array, msg)
        msg = hidder.extract(watermarked_array)

        self.assertTrue(msg.startswith(msg))

    def test_with_qhahn_8x8(self):
        from almiky.moments.matrix import QHahnMatrix

        path = base.joinpath("messages/msg64bytes.txt")
        with open(str(path), "r") as file:
            msg = file.read()

        trasform = QHahnMatrix(8, q=0.5, alpha=0.5, beta=0.5, N=7)
        cover_array = np.random.rand(72, 72)
        hidder = frequency.HidderEightFrequencyCoeficients(trasform)

        watermarked_array = hidder.insert(cover_array, msg)
        msg = hidder.extract(watermarked_array)

        self.assertTrue(msg.startswith(msg))
