# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

import unittest
import numpy as np


class TestOrtogonalFunctions(unittest.TestCase):
    '''
    Tests to verify the evaluation of ortogonal functions
    '''
    def test_moment_undefined(self):
        from almiky.moments.functions import OrtogonalFunction

        class OtherFunction(OrtogonalFunction):
            pass

        order = 8
        x = 4
        func = OtherFunction(order)
        try:
            func.eval(x)
        except NotImplementedError:
            # Normal behaivor. OrtogonalFunction derivated
            # clasess must implment 'expression' method because
            # is called by "eval" method
            pass
        else:
            assert False, "Evaluated OrtogonalFunction derivated class" \
                "without 'expression' method"


class TestCharlierFunctions(unittest.TestCase):
    '''
    Tests to verify the evaluation of ortogonal functions
    '''
    def test_charlier_keval(self):
        from almiky.moments.functions import CharlierFunction

        order = 8
        x = 4
        func = CharlierFunction(order, alpha=5)
        value = func.keval(x, 0, 8)

        self.assertEqual(value, 390625, "Incorrect evaluation")

    def test_charlier_evaluation(self):
        from almiky.moments.functions import CharlierFunction

        x, order = 3, 3
        func = CharlierFunction(order, alpha=0.5)
        value = func.eval(x)
        self.assertEqual(value, -0.875, "Incorrect evaluation")

    def test_norm(self):
        from almiky.moments.functions import CharlierFunction

        x, order, alpha  = 3, 7, 0.5
        func = CharlierFunction(order, alpha)
        value = func.norm()

        self.assertEqual(value, 39.375, "Incorrect evaluation")


class TestCharlierSobolevFunctions(unittest.TestCase):
    '''
    Tests to verify the evaluation of ortogonal functions
    '''

    def test_kernel(self):
        from almiky.moments.functions import CharlierFunction, CharlierSobolevFunction

        x, order, alpha, beta, gamma  = 3, 7, 0.5, 10, -2
        func = CharlierSobolevFunction(order, alpha, beta, gamma)
        value = func.kernel(x)

        np.testing.assert_almost_equal(value, 544.321, 3)


class TestOrtogonalForms(unittest.TestCase):
    '''
    Tests to verify the evaluation of ortogonal functions
    '''
    def test_charlier_eval(self):
        from almiky.moments.functions import CharlierFunction
        from almiky.moments.orthogonal_forms import CharlierForm

        order = 8
        x = 4
        form = CharlierForm(order, alpha=5)
        value = form.eval(x)

        self.assertEqual(value, -0.03129170161915745, "Incorrect evaluation")


class OrtogonalMatrixTest(unittest.TestCase):

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


if __name__ == '__main__':
    unittest.main()
