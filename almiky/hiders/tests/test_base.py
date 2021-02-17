"""Test for hiders in transform domain"""

import unittest
from unittest import TestCase
from unittest.mock import Mock, MagicMock

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

        cover_work = np.array([
            [0.72104381, 0.3611912],
            [0.54423469, 0.99351504],
        ])

        ws_work_expected = np.array([
            [0.52259635, 0.3611912],
            [0.54423469, 0.99351504],
        ])

        scan = ScanMapping()
        hider = hiders.SingleBitHider(scan, embedder)
        ws_work = hider.insert(cover_work, bit=1, index=0)

        embedder.embed.assert_called_once_with(0.72104381, 1)

        np.testing.assert_array_equal(ws_work, ws_work_expected)

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


class TranformHiderTest(TestCase):

    def test_insert(self):
        cover_work = np.array([
            [0.72104381, 0.3611912],
            [0.54423469, 0.99351504],
        ])

        def mock_insert(cover_work, data, **kwargs):
            return cover_work * 2

        def mock_direct(cover_work):
            return cover_work * 3

        def mock_inverse(cover_work):
            return cover_work * -1

        base_hider = Mock()
        base_hider.insert = Mock(side_effect=mock_insert)

        transform = Mock()
        transform.direct = Mock(side_effect=mock_direct)
        transform.inverse = Mock(side_effect=mock_inverse)

        hider = hiders.TransformHider(base_hider, transform=transform)
        ws_work = hider.insert(cover_work, data=1, index=0)
        ws_work_expected = ((cover_work * 3) * 2) * -1

        base_hider.insert.assert_called_once()
        transform.direct.assert_called_once()
        transform.inverse.assert_called_once()

        np.testing.assert_array_equal(ws_work, ws_work_expected)

    def test_extract(self):
        data = np.array([
            [0.72104381, 0.3611912],
            [0.54423469, 0.99351504],
        ])

        def mock_extract(cover_work, **kwargs):
            return np.sum(cover_work)

        def mock_direct(cover_work):
            return cover_work * 3

        base_hider = Mock()
        base_hider.extract = Mock(side_effect=mock_extract)

        transform = Mock()
        transform.direct = Mock(side_effect=mock_direct)

        hider = hiders.TransformHider(base_hider, transform=transform)
        msg = hider.extract(data, bit=1, index=0)
        msg_expected = np.sum(data * 3)

        base_hider.extract.assert_called_once()
        transform.direct.assert_called_once()

        np.testing.assert_array_equal(msg, msg_expected)


if __name__ == '__main__':
    unittest.main()
