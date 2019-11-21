import unittest
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np

from almiky.metrics import metrics


def mse_mock(cover_work, ws_work):
    '''
    Mocking metrics.mse function
    '''
    return 100.0


class TestMetrics(unittest.TestCase):

    def test_mse(self):
        '''
        Testing Mean Square Error (MSE)
        '''
        cover_work = np.array([[2.2, 3.3], [5.5, 6.6]])
        # watermaked work (stego_work)
        ws_work = np.array([[4.4, 7.7], [9.9, 10.0]])

        mse = metrics.mse(cover_work, ws_work)
        np.testing.assert_almost_equal(mse, 13.78, 3)

    @patch.object(metrics, 'mse', side_effect=mse_mock)
    def test_psnr(self, mock_mse):
        '''
        Testing Peak Signal-to-Noise Ratio (PSNR)
        '''
        cover_work = Mock()
        ws_work = Mock()

        psnr = metrics.psnr(cover_work, ws_work)
        np.testing.assert_almost_equal(psnr, 28.130, 3)


if __name__ == '__main__':
    unittest.main()