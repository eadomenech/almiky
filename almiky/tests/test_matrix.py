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
        matrix = OtherMatrix(alpha=5)
        try:
            matrix.get_values(dimension)
        except NotImplementedError:
            # Normal behavior. OrthogonalMatrix derivated
            # clasess must implement 'get_values' method because
            # has not implementation
            pass
        else:
            assert False, "Call to undefined get_values function in" \
                "OrthogonalMatrix derivated class without 'keval' method"


class CharlierMatrixTest(unittest.TestCase):

    def test_matrix(self):
        from almiky.moments.matrix import CharlierMatrix

        dimension = 2
        matrix = CharlierMatrix(alpha=5)
        values = matrix.get_values(dimension)

        np.testing.assert_array_almost_equal(
            values,
            np.asarray([
                [0.08208499862389880, -0.1835476368560144],
                [0.1835476368560144, -0.3283399944955952]
            ])
        )


class CharlierSobolevMatrixTest(unittest.TestCase):

    def test_matrix(self):
        from almiky.moments.matrix import CharlierSobolevMatrix

        dimension = 2
        matrix = CharlierSobolevMatrix(alpha=0.5, beta=10, gamma=-2)
        values = matrix.get_values(dimension)

        np.testing.assert_array_almost_equal(
            values,
            np.asarray([
                [0.77880078, -0.55069531],
                [0.55069531, 0.38940039]
            ])
        )


class QHahnMatrixTest(unittest.TestCase):

    def test_matrix(self):
        from almiky.moments.matrix import QHahnMatrix

        dimension = 2
        matrix = QHahnMatrix(q=0.5, alpha=0.5, beta=0.5, N=2)
        values = matrix.get_values(dimension)

        np.testing.assert_array_almost_equal(
            values,
            np.asarray([
                [0.212512, 0.516398],
                [0.481932, 0.68313]
            ])
        )


if __name__ == '__main__':
    unittest.main()
