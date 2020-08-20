'''Test for noisy attacks'''


import unittest
from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from almiky.attacks import noise as attacks


class TestSaltAndPepperNoise(TestCase):
    '''Test for Salt and Pepper noise'''

    def test_default_max_value(self):
        '''Test noise application with default maximun image value (255)'''

        import random

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

        import random

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


if __name__ == '__main__':
    unittest.main()

