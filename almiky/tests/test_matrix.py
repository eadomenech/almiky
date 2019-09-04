# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

import unittest
import numpy as np


class OrtogonalMatrixTest(unittest.TestCase):

    def test_matrix(self):
        from almiky.moments.matrix import OrthogonalMatrix
        from almiky.moments.orthogonal_forms import CharlierForm
        from almiky.moments.functions import CharlierFunction

        class OtherMatrix(OrthogonalMatrix):
            pass

        dimension = 2
        alpha = 5
        function = CharlierFunction(alpha)
        form = CharlierForm(function, alpha)
        matrix = OtherMatrix(form)
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
        from almiky.moments.orthogonal_forms import CharlierForm
        from almiky.moments.functions import CharlierFunction

        alpha = 5
        function = CharlierFunction(alpha)
        form = CharlierForm(function, alpha)
        matrix = CharlierMatrix(form)
        values = matrix.get_values(dimension=2)

        np.testing.assert_array_almost_equal(
            values,
            np.asarray([
                [0.08208499862389880, -0.1835476368560144],
                [0.1835476368560144, -0.3283399944955952]
            ])
        )


class CharlierSobolevMatrixTest(unittest.TestCase):

    def test_matrix(self):
        from almiky.moments.matrix import CharlierMatrix
        from almiky.moments.orthogonal_forms import CharlierForm
        from almiky.moments.functions import CharlierSobolevFunction

        alpha = 0.5
        function = CharlierSobolevFunction(alpha=alpha, beta=10, gamma=-2)
        form = CharlierForm(function, alpha=alpha)
        matrix = CharlierMatrix(form)
        values = matrix.get_values(dimension=2)

        np.testing.assert_array_almost_equal(
            values,
            np.asarray([
                [0.77880078, -0.55069531],
                [0.55069531, 0.38940039]
            ])
        )


class QHahnMatrixTest(unittest.TestCase):

    def test_matrix(self):
        from almiky.moments.matrix import CharlierMatrix
        from almiky.moments.orthogonal_forms import QHahnForm
        from almiky.moments.functions import QHahnFunction

        paremeters = {'q':0.5, 'alpha':0.5, 'beta':0.5, 'N':2}
        function = QHahnFunction(**paremeters)
        form = QHahnForm(function, **paremeters)
        matrix = CharlierMatrix(form)
        values = matrix.get_values(dimension=2)

        np.testing.assert_array_almost_equal(
            values,
            np.asarray([
                [0.212512, 0.516398],
                [0.481932, 0.68313]
            ])
        )


if __name__ == '__main__':
    unittest.main()
