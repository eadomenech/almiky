"""Test for hiders in transform domain"""

import unittest
from unittest import TestCase
from unittest.mock import Mock, MagicMock, patch

import numpy as np

from almiky.hiders import frequency


class TestBlockHider(TestCase):
    """
    Test for BlockHider class
    """

    def test_initialization(self):
        embedder = Mock()
        transform = Mock()
        hider = frequency.BlockHider(embedder, transform)

        self.assertEqual(embedder, hider.embedder)
        self.assertEqual(transform, hider.transform)

    def test_empty_msg(self):
        embedder = Mock()
        transform = Mock()
        cover = np.random.rand(8, 8)
        hider = frequency.BlockHider(embedder, transform)

        ws_work = hider.hide(cover, '')

        np.testing.assert_almost_equal(ws_work, cover)

    def test_embedding(self):
        # Setup
        embedder = Mock()
        embedder.embed = Mock(return_value=25)
        transform = Mock()
        
        cover = np.random.rand(64, 64)
        hider = frequency.BlockHider(embedder, transform)

        ws_work = hider.hide(cover, '')

        np.testing.assert_almost_equal(ws_work, np.full((64, 64), 25))



if __name__ == '__main__':
    unittest.main()