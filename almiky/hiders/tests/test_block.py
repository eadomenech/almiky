"""Test for hiders in transform domain"""

import unittest
from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from almiky.hiders import block as hiders


class BlockBitHidderTest(TestCase):
    """
    Test for BlockHidder class
    """

    def test_insert(self):
        cover = np.array([
            [0.72953648, 0.62911616, 0.51911824, 0.87013322],
            [0.17510781, 0.84696329, 0.98087834, 0.38006197],
            [0.74479979, 0.13571776, 0.15937805, 0.16483562],
            [0.86981138, 0.49378362, 0.50166927, 0.11741198]
        ])
        expected_ws_work = np.array([
            [0.72153848, 0.62911616, 0.51171824, 0.87013322],
            [0.17510781, 0.84696329, 0.98087834, 0.38006197],
            [0.74479979, 0.13571776, 0.15937805, 0.16483562],
            [0.86981138, 0.49378362, 0.50166927, 0.11741198]
        ])
        ws_base_works = [
            np.array([
                [0.72153848, 0.62911616],
                [0.17510781, 0.84696329],
            ]),
            np.array([
                [0.51171824, 0.87013322],
                [0.98087834, 0.38006197],
            ]),
        ]

        base_hider = Mock()
        base_hider.insert = Mock(side_effect=ws_base_works)

        hider = hiders.BlockBitHider(base_hider)
        ws_work = hider.insert(cover, '01', block_shape=(2, 2), index=0)

        self.assertEqual(base_hider.insert.call_count, 2)

        np.testing.assert_array_equal(ws_work, expected_ws_work)

    def test_less_blocks_than_bits(self):
        cover = np.array([
            [0.72953648, 0.62911616, 0.51911824, 0.87013322],
            [0.17510781, 0.84696329, 0.98087834, 0.38006197],
            [0.74479979, 0.13571776, 0.15937805, 0.16483562],
            [0.86981138, 0.49378362, 0.50166927, 0.11741198]
        ])

        ws_work = np.array([
            [0.72953648, 0.62911616],
            [0.17510781, 0.84696329],
        ])

        base_hider = Mock()
        base_hider.insert = Mock(return_value=ws_work)
        hider = hiders.BlockBitHider(base_hider)

        with self.assertRaises(ValueError):
            hider.insert(cover, '01100', block_shape=(2, 2), index=0)

        self.assertEqual(base_hider.insert.call_count, 4)

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
