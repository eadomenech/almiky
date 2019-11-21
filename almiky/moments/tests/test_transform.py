import unittest
from unittest.mock import Mock

import numpy as np

from almiky.moments.transform import Transform

class TransformTest(unittest.TestCase):

    def test_direct(self):
        ortho_matrix = np.random.rand(8, 8)
        data = np.random.rand(8, 8)
        result = np.dot(ortho_matrix, np.dot(data, ortho_matrix.T))

        transform = Transform(ortho_matrix)
        np.testing.assert_array_almost_equal(transform.direct(data), result)

    def test_inverse(self):
        ortho_matrix = np.random.rand(8, 8)
        data = np.random.rand(8, 8)
        result = np.dot(ortho_matrix.T, np.dot(data, ortho_matrix))

        transform = Transform(ortho_matrix)
        np.testing.assert_array_almost_equal(transform.inverse(data), result)


if __name__ == '__main__':
    unittest.main()
