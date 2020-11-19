from pathlib import Path
import unittest
from unittest.mock import Mock

import numpy as np
import imageio

from almiky.steganalysis.additive_noise import features
from almiky.steganalysis.additive_noise import metrics

class DirectoryLoadTest(unittest.TestCase):
    def test_folder_loading(self):
        # Fake data
        folder = Mock()
        files = (Path('imagen1'), Path('imagen2'), Path('imagen3'))
        target = np.array([0, 1, 0])
        com = np.random.rand(3)
        com1 = np.random.rand(3)
        com2 = np.random.rand(3)
        coms = (com, com1, com2)
        image1 = np.random.rand(64, 64, 3)
        image2 = np.random.rand(64, 64, 3)
        image3 = np.random.rand(64, 64, 3)
        images = [image1, image2, image3]

        # Mocks
        folder.iterdir.return_value = files
        metric = Mock(side_effect=list(coms))
        process = features.ProcessImageFolder(metric)
        imageio.imread = Mock(side_effect=list(images))

        ft = process(folder)
        np.testing.assert_array_equal(list(ft), list(coms))
