"""Test for hiders in transform domain"""

import unittest
from unittest import TestCase
from unittest.mock import Mock, MagicMock, patch

import numpy as np

from almiky.hiders import base as hiders
from almiky.utils.scan.scan import ScanMapping



class SingleBitHidderTest(TestCase):
    """
    Test for SingleBitHidder class
    """

    def test_insert(self):

        embedder = MagicMock()
        embedder.embed = Mock(return_value=0.52259635)

        cover = np.array([
            [0.72104381, 0.3611912],
            [0.54423469, 0.99351504],
        ])

        ws = np.array([
            [0.52259635, 0.3611912 ],
            [0.54423469, 0.99351504]
        ])

        scan = ScanMapping()
        hider = hiders.SingleBitHider(scan, embedder)
        hider.insert(cover, bit=1, index=0)

        embedder.embed.assert_called_once_with(0.72104381, 1)

        np.testing.assert_array_equal(cover, ws)

    def test_extract(self):

        embedder = MagicMock()
        embedder.extract = Mock(return_value=0.52259635)

        cover = np.array([
            [0.72104381, 0.3611912],
            [0.54423469, 0.99351504],
        ])

        scan = ScanMapping()
        hider = hiders.SingleBitHider(scan, embedder)
        amplitude = hider.extract(cover, index=0)

        embedder.extract.assert_called_once_with(.72104381)

        self.assertEqual(amplitude, .52259635)


if __name__ == '__main__':
    unittest.main()