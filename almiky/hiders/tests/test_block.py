"""Test for hiders in transform domain"""

import unittest
from unittest import TestCase
from unittest.mock import Mock, MagicMock, patch

import numpy as np

from almiky.hiders import block as hiders
from almiky.utils.scan.scan import ScanMapping


class BlockBitHidderTest(TestCase):
    """
    Test for BlockHidder class
    """

    def mock_insert(self, scan, bit, index):
        scan.data[0, 0] = 0

    def test_insert(self):
        msg = '1010'
        cover = np.array([
            [0.72953648, 0.62911616, 0.51911824, 0.87013322],
            [0.17510781, 0.84696329, 0.98087834, 0.38006197],
            [0.74479979, 0.13571776, 0.15937805, 0.16483562],
            [0.86981138, 0.49378362, 0.50166927, 0.11741198]
        ])

        bit_hidder = Mock()
        bit_hidder.insert = Mock(side_effect=self.mock_insert)

        ws = np.array([
            [0, 0.62911616, 0, 0.87013322],
            [0.17510781, 0.84696329, 0.98087834, 0.38006197],
            [0, 0.13571776, 0, 0.16483562],
            [0.86981138, 0.49378362, 0.50166927, 0.11741198]
        ])

        hider = hiders.BlockBitHider(bit_hidder)
        hider.insert(cover, msg, block_shape=(2, 2), index=0)

        self.assertEqual(bit_hidder.insert.call_count, 4)

        np.testing.assert_array_equal(cover, ws)

    def test_extract(self):

        bit_hidder = Mock()
        bit_hidder.extract = Mock(side_effect=[1, 0, 0, 1])

        cover = np.array([
            [0.72953648, 0.62911616, 0.51911824, 0.87013322],
            [0.17510781, 0.84696329, 0.98087834, 0.38006197],
            [0.74479979, 0.13571776, 0.15937805, 0.16483562],
            [0.86981138, 0.49378362, 0.50166927, 0.11741198]
        ])

        hider = hiders.BlockBitHider(bit_hidder)
        msg = hider.extract(cover, index=0, block_shape=(2, 2))

        self.assertEqual(bit_hidder.extract.call_count, 4)
        
        self.assertEqual(msg, '1001')


if __name__ == '__main__':
    unittest.main()