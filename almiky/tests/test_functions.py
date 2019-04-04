# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

import unittest
import numpy as np


class OrtogonalFunctionsTest(unittest.TestCase):
    '''
    Tests to verify the evaluation of ortogonal functions
    '''

    def test_moment_undefined(self):
        from almiky.moments.functions import OrtogonalFunction

        class OtherFunction(OrtogonalFunction):
            pass

        order = 8
        x = 4
        func = OtherFunction()
        try:
            func.eval(x, order)
        except NotImplementedError:
            # Normal behaivor. OrtogonalFunction derivated
            # clasess must implment 'expression' method because
            # is called by "eval" method
            pass
        else:
            assert False, "Evaluated OrtogonalFunction derivated class" \
                "without 'expression' method"


class CharlierFunctionsTest(unittest.TestCase):
    '''
    Tests to verify the evaluation of ortogonal functions
    '''

    def test_charlier_keval(self):
        from almiky.moments.functions import CharlierFunction

        order = 8
        x = 4
        func = CharlierFunction(alpha=5)
        value = func.keval(x, 0, order)

        self.assertEqual(value, 390625, "Incorrect evaluation")

    def test_charlier_evaluation(self):
        from almiky.moments.functions import CharlierFunction

        x, order = 3, 3
        func = CharlierFunction(alpha=0.5)
        value = func.eval(x, order)
        self.assertEqual(value, -0.875, "Incorrect evaluation")

    def test_norm(self):
        from almiky.moments.functions import CharlierFunction

        order, alpha = 7, 0.5
        func = CharlierFunction(alpha)
        value = func.norm(order)

        self.assertEqual(value, 39.375, "Incorrect evaluation")


class CharlierSobolevFunctionsTest(unittest.TestCase):
    '''
    Tests to verify the evaluation of ortogonal functions
    '''

    def test_kernel(self):
        from almiky.moments.functions import (
            CharlierFunction, CharlierSobolevFunction)

        x, order, alpha, beta, gamma = 3, 7, 0.5, 10, -2
        func = CharlierSobolevFunction(alpha, beta, gamma)
        value = func.kernel(x, order)

        np.testing.assert_almost_equal(value, 544.321, 3)


if __name__ == '__main__':
    unittest.main()
