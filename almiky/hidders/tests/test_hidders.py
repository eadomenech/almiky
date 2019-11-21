import unittest
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np

from almiky.hidders import frequency
from almiky.utils import ortho_matrix


class HiderEightFrequencyCoeficientsTest(unittest.TestCase):
    def test_insert(self):
        msg = ''
        hidder = frequency.HidderEightFrequencyCoeficients(ortho_matrix.dct)
        ws_work = hidder.insert_msg(msg)
        np.testing.assert_array_almost_equal(ws_work, X)
