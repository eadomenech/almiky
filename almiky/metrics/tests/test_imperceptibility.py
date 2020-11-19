'''
Test for imperceptibility performance metrics
'''

import unittest
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np

from almiky.metrics import imperceptibility as metrics


class TestMSE(unittest.TestCase):
    '''
    Test for Mean Square Error (MSE)
    '''

    def test_images_different_shape(self):
        cover_work = Mock(shape=(2, 2))
        # watermaked work (stego_work)
        ws_work = Mock(shape=(2, 3))

        with self.assertRaises(ValueError):
            metrics.mse(cover_work, ws_work)

    def test_images_equals(self):
        cover_work = np.array([[2.2, 3.3], [5.5, 6.6]])
        # watermaked work (stego_work)
        ws_work = np.array([[2.2, 3.3], [5.5, 6.6]])

        mse = metrics.mse(cover_work, ws_work)
        self.assertEqual(mse, 0)

    def test_images_non_equals(self):
        cover_work = np.array([[2.2, 3.3], [5.5, 6.6]])
        # watermaked work (stego_work)
        ws_work = np.array([[4.4, 7.7], [9.9, 10.0]])

        mse = metrics.mse(cover_work, ws_work)
        np.testing.assert_almost_equal(mse, 13.78, 3)


def mse_non_zero_mock(cover_work, ws_work):
    '''
    Mocking metrics.mse function
    '''
    return 100.0


def mse_zero_mock(cover_work, ws_work):
    '''
    Mocking metrics.mse function
    '''
    return 0


class TestPSNR(unittest.TestCase):
    '''
    Testing Peak Signal-to-Noise Ratio (PSNR)
    '''

    def test_images_different_shape(self):
        cover_work = Mock(shape=(2, 2))
        # watermaked work (stego_work)
        ws_work = Mock(shape=(2, 3))

        with self.assertRaises(ValueError):
            metrics.psnr(cover_work, ws_work)

    @patch.object(metrics, 'mse', side_effect=mse_zero_mock)
    def test_with_zero_mse(self, mock_mse):
        cover_work = np.array([[2.2, 3.3], [5.5, 6.6]])
        # watermaked work (stego_work)
        ws_work = np.array([[4.4, 7.7], [9.9, 10.0]])

        psnr = metrics.psnr(cover_work, ws_work)
        
        np.testing.assert_almost_equal(psnr, 54.151403522, 3)

    @patch.object(metrics, 'mse', side_effect=mse_non_zero_mock)
    def test_with_non_zero_mse(self, mock_mse):
        '''
        Testing Peak Signal-to-Noise Ratio (PSNR)
        '''
        cover_work = Mock()
        ws_work = Mock()

        psnr = metrics.psnr(cover_work, ws_work)
        np.testing.assert_almost_equal(psnr, 28.130, 3)

    @patch.object(metrics, 'mse', side_effect=mse_non_zero_mock)
    def test_with_custom_max_value(self, mock_mse):
        '''
        Testing Peak Signal-to-Noise Ratio (PSNR)
        '''
        cover_work = Mock()
        ws_work = Mock()

        psnr = metrics.psnr(cover_work, ws_work, max=200)
        np.testing.assert_almost_equal(psnr, 26.020599913, 3)


if __name__ == '__main__':
    unittest.main()