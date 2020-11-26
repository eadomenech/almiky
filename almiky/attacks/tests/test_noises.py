'''Test for noisy attacks'''

import unittest
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from almiky.attacks import noises as attacks


class TestSaltAndPepperNoise(TestCase):
    '''Test for Salt and Pepper noise'''

    @patch('numpy.random.choice')
    @patch('numpy.random.uniform')
    def test_noise(self, uniform_mock, choice_mock):
        '''Test noise application with default maximun image value (255)'''

        uniform_mock.return_value = np.array([
            [0.7, 0.02, 0.3, 0.2],
            [0.4, 0.02, 0.5, 0.7],
            [0.2, 0.3, 0.03, 0.3],
            [0.4, 0.06, 0.6, 0.2]
        ])

        choice_mock.return_value = np.array([
            [-255, 255, -255, 255],
            [-255, 255, -255, 255],
            [-255, 255, -255, 255],
            [-255, -255, -255, 255]
        ])

        data = np.random.rand(4, 4)
        noisy = np.copy(data)
        noisy[1, 1] = noisy[0, 1] = 255
        noisy[3, 1] = noisy[2, 2] = 0

        np.testing.assert_almost_equal(
            attacks.salt_pepper_noise(data, density=0.1), noisy)


class TestGaussianNoise(TestCase):
    '''Test for gaussian noise'''

    @patch('numpy.std')
    @patch('numpy.random.normal')
    def test_default_max_value(self, gaussian_mock, std_mock):
        '''
        Test noise addition when default max intensity is used (255)'
        '''
        img_std = 54.5
        percent_noise = 0.1
        gaussian_mock.return_value = np.array([
            [4.0, -16.0, 25],
            [-10.6, 8.7, 4.8],
            [-10.6, 198.7, 4.8],
        ])
        std_mock.return_value = img_std

        data = np.array([
            [127, 12, 250],
            [14, 28, 155],
            [14, 8, 155],
        ])

        noisy = np.array([
            [131, 0, 255],
            [3, 36, 159],
            [3, 206, 159],
        ])

        ouput = attacks.gaussian_noise(data, percent_noise=percent_noise)

        gaussian_mock.called_with(0, img_std * percent_noise, data.shape)
        std_mock.called_once_with(data)
        np.testing.assert_almost_equal(ouput, noisy)

    @patch('numpy.std')
    @patch('numpy.random.normal')
    def test_custom_max_value(self, gaussian_mock, std_mock):
        '''
        Test noise addition when custom max intensity is used'
        '''
        img_std = 54.5
        percent_noise = 0.1
        gaussian_mock.return_value = np.array([
            [4.0, -16.0, 25],
            [-10.6, 8.7, 4.8],
            [-10.6, 198.7, 4.8],
        ])
        std_mock.return_value = img_std

        data = np.array([
            [127, 12, 180],
            [14, 28, 155],
            [14, 8, 155],
        ])

        noisy = np.array([
            [131, 0, 200],
            [3, 36, 159],
            [3, 200, 159],
        ])

        ouput = attacks.gaussian_noise(
            data, percent_noise=percent_noise, max_value=200)

        gaussian_mock.called_with(0, img_std * percent_noise, data.shape)
        std_mock.called_once_with(data)
        np.testing.assert_almost_equal(ouput, noisy)


if __name__ == '__main__':
    unittest.main()
