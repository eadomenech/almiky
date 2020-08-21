'''Test for noisy attacks'''


import random
import unittest
from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from almiky.attacks import noise as attacks


class TestSaltAndPepperNoise(TestCase):
    '''Test for Salt and Pepper noise'''

    def test_default_max_value(self):
        '''Test noise application with default maximun image value (255)'''

        random.uniform = Mock(side_effect=[
            0.7, 0.02, 0.3, 0.2,
            0.4, 0.02, 0.5, 0.7,
            0.2, 0.3, 0.03, 0.3,
            0.4, 0.06, 0.6, 0.2
        ])

        random.randint = Mock(side_effect=[
            0, 1, 0, 1
        ])

        data = np.random.rand(4, 4)
        noisy = np.copy(data)
        noisy[1, 1] = noisy[3, 1] = 255
        noisy[0, 1] = noisy[2, 2] = 0

        np.testing.assert_almost_equal(
            attacks.salt_paper_noise(data, density=0.1), noisy)

    def test_custom_max_value(self):
        '''Test noise application with a custom maximun image value'''

        random.uniform = Mock(side_effect=[
            0.7, 0.02, 0.1, 0.2,
            0.4, 0.02, 0.5, 0.7,
            0.2, 0.3, 0.03, 0.3,
            0.4, 0.06, 0.6, 0.2
        ])

        random.randint = Mock(side_effect=[
            0, 1, 0, 1
        ])

        data = np.random.rand(4, 4)
        noisy = np.copy(data)
        noisy[1, 1] = noisy[3, 1] = 512
        noisy[0, 1] = noisy[2, 2] = 0

        np.testing.assert_almost_equal(
            attacks.salt_paper_noise(data, density=0.1, max_value=512), noisy)


class TestGaussianNoise(TestCase):
    '''Test for gaussian noise'''

    def test_random_calls(self):
        '''Test number and arguments of random function calls'''

        random.uniform = Mock(return_value=0.05)
        random.gauss = Mock(return_value=0.05)

        attacks.gaussian_noise(np.random.rand(4, 4), density=0.1)

        expected_uniform_calls = [((0, 1),)] * 16
        self.assertEquals(
            random.uniform.call_args_list, expected_uniform_calls)
        expected_gauss_calls = [((0, 0.5),)] * 16
        self.assertListEqual(
            random.gauss.call_args_list, expected_gauss_calls)

    def test_random_calls_with_custom_arguments(self):
        '''Test number and arguments of random function calls'''

        random.uniform = Mock(return_value=0.05)
        random.gauss = Mock(return_value=0.05)

        attacks.gaussian_noise(
            np.random.rand(4, 4), density=0.1, mu=1, sigma=10)

        expected_gauss_calls = [((1, 10),)] * 16
        self.assertListEqual(
            random.gauss.call_args_list, expected_gauss_calls)

    def test_image_min_value_exceded(self):
        '''
        Test noise application when altered values excceds
        minimun image value
        '''

        random.uniform = Mock(side_effect=[
            0.7, 0.02, 0.3, 0.2,
            0.4, 0.02, 0.5, 0.7,
            0.2, 0.3, 0.03, 0.3,
            0.4, 0.06, 0.6, 0.2
        ])

        random.gauss = Mock(side_effect=[
            -18.5, -25.6, -19.4, -30.1
        ])

        data = np.random.rand(4, 4)
        noisy = np.copy(data)

        data[1, 1] = data[3, 1] = 21
        data[0, 1] = data[2, 2] = 17

        noisy[1, 1] = noisy[3, 1] = 0
        noisy[0, 1] = noisy[2, 2] = 0

        np.testing.assert_almost_equal(
            attacks.gaussian_noise(data, density=0.1), noisy)

    def test_image_max_value_exceded(self):
        '''
        Test noise application when altered values
        excceds maximun image value
        '''

        random.uniform = Mock(side_effect=[
            0.7, 0.02, 0.3, 0.2,
            0.4, 0.02, 0.5, 0.7,
            0.2, 0.3, 0.03, 0.3,
            0.4, 0.06, 0.6, 0.2
        ])

        random.gauss = Mock(side_effect=[
            4.0, 6.0, 10.6, 8.7
        ])

        data = np.random.rand(4, 4)
        noisy = np.copy(data)

        data[1, 1] = data[3, 1] = 250
        data[0, 1] = data[2, 2] = 252

        noisy[1, 1] = noisy[3, 1] = 255
        noisy[0, 1] = noisy[2, 2] = 255

        np.testing.assert_almost_equal(
            attacks.gaussian_noise(data, density=0.1), noisy)


    def test_image_custom_max_value_exceded(self):
        '''
        Test noise application when altered values excceds
        custom maximun image value'
        '''

        random.uniform = Mock(side_effect=[
            0.7, 0.02, 0.3, 0.2,
            0.4, 0.02, 0.5, 0.7,
            0.2, 0.3, 0.03, 0.3,
            0.4, 0.06, 0.6, 0.2
        ])

        random.gauss = Mock(side_effect=[
            4.0, 6.0, 10.6, 8.7
        ])

        data = np.random.rand(4, 4)
        noisy = np.copy(data)

        data[1, 1] = data[3, 1] = 65530
        data[0, 1] = data[2, 2] = 65537

        noisy[1, 1] = noisy[3, 1] = 65535
        noisy[0, 1] = noisy[2, 2] = 65535

        np.testing.assert_almost_equal(
            attacks.gaussian_noise(data, density=0.1, max_value=65535), noisy)

    def test_noise_addition(self):
        '''
        Test noise addition'
        '''

        random.uniform = Mock(side_effect=[
            0.7, 0.02, 0.3, 0.2,
            0.4, 0.02, 0.5, 0.7,
            0.2, 0.3, 0.03, 0.3,
            0.4, 0.06, 0.6, 0.2
        ])

        random.gauss = Mock(side_effect=[
            4.0, -6.0, -10.6, 8.7
        ])

        data = np.random.rand(4, 4)
        noisy = np.copy(data)

        data[1, 1] = data[3, 1] = 127
        data[0, 1] = data[2, 2] = 132

        noisy[1, 1], noisy[3, 1] = 121, 136
        noisy[0, 1], noisy[2, 2] = 136, 121

        np.testing.assert_almost_equal(
            attacks.gaussian_noise(data, density=0.1), noisy)


if __name__ == '__main__':
    unittest.main()

