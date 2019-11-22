# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

import unittest
import numpy as np


class OrtogonalMatrixTest(unittest.TestCase):

    def test_matrix(self):
        from almiky.moments.matrix import OrthogonalMatrix

        class OtherMatrix(OrthogonalMatrix):
            pass

        dimension = 2
        try:
            matrix = OtherMatrix(dimension, alpha=5)
        except TypeError:
            # Normal behavior. orthogonal_form_class must be set
            pass
        else:
            assert False, "Call to undefined get_values function in" \
                "OrthogonalMatrix derivated class without 'keval' method"


class CharlierMatrixTest(unittest.TestCase):

    def test_matrix(self):
        from almiky.moments.matrix import CharlierMatrix

        dimension = 2
        matrix = CharlierMatrix(dimension, alpha=5)

        np.testing.assert_array_almost_equal(
            matrix.values,
            np.asarray([
                [0.08208499862389880, -0.1835476368560144],
                [0.1835476368560144, -0.3283399944955952]
            ])
        )
    
    def test_direct_transform(self):
        from almiky.moments.matrix import CharlierMatrix

        dimension = 2
        matrix = CharlierMatrix(dimension, alpha=5)
        data = np.random.rand(2, 2)
        result = np.dot(matrix.values, np.dot(data, matrix.values.T))
        np.testing.assert_array_almost_equal(matrix.direct(data), result)
    
    def test_inverse(self):
        from almiky.moments.matrix import CharlierMatrix

        dimension = 2
        matrix = CharlierMatrix(dimension, alpha=5)
        data = np.random.rand(2, 2)
        result = np.dot(matrix.values.T, np.dot(data, matrix.values))

        np.testing.assert_array_almost_equal(matrix.inverse(data), result)


class CharlierSobolevMatrixTest(unittest.TestCase):

    def test_matrix(self):
        from almiky.moments.matrix import CharlierSobolevMatrix

        dimension = 2
        matrix = CharlierSobolevMatrix(dimension, alpha=0.5, beta=10, gamma=-2)

        np.testing.assert_array_almost_equal(
            matrix.values,
            np.asarray([
                [0.77880078, -0.55069531],
                [0.55069531, 0.38940039]
            ])
        )


class QHahnMatrixTest(unittest.TestCase):

    def test_matrix(self):
        from almiky.moments.matrix import QHahnMatrix

        dimension = 2
        matrix = QHahnMatrix(dimension, q=0.5, alpha=0.5, beta=0.5, N=2)

        np.testing.assert_array_almost_equal(
            matrix.values,
            np.asarray([
                [0.212512, 0.516398],
                [0.481932, 0.68313]
            ])
        )


if __name__ == '__main__':
    unittest.main()
