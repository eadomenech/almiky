# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

import unittest
import numpy as np

class OrtogonalFunctionsTest(unittest.TestCase):
    '''
    Tests to verify the evaluation of ortogonal functions
    '''

    def test_keval(self):
        from almiky.moments.functions import OrtogonalFunction

        class OtherFunction(OrtogonalFunction):
            pass

        order = 8
        x = 4
        func = OtherFunction()
        try:
            func.keval(x, 6, order)
        except NotImplementedError:
            # Normal behavior. OtherFunction derivated
            # clasess must implement 'keval' method because
            # has not been implemented.
            pass
        else:
            assert False, "Evaluated OrtogonalFunction derivated class" \
                "without 'keval' method"

    def test_norm(self):
        from almiky.moments.functions import OrtogonalFunction

        class OtherFunction(OrtogonalFunction):
            pass

        order = 8
        x = 4
        func = OtherFunction()
        try:
            func.norm(x)
        except NotImplementedError:
            # Normal behavior. OtherFunction derivated
            # clasess must implement 'norm' method because
            # has not been implemented.
            pass
        else:
            assert False, "Evaluated OrtogonalFunction derivated class" \
                "without 'norm' method"


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

    def test_An(self):
        from almiky.moments.functions import (
            CharlierFunction, CharlierSobolevFunction)

        x, order, alpha, beta, gamma = 3, 7, 0.5, 10, -2
        func = CharlierSobolevFunction(alpha, beta, gamma)
        value = func.An(x, order)

        np.testing.assert_almost_equal(value, 2.491, 3)

    def test_Bn(self):
        from almiky.moments.functions import (
            CharlierFunction, CharlierSobolevFunction)

        x, order, alpha, beta, gamma = 3, 7, 0.5, 10, -2
        func = CharlierSobolevFunction(alpha, beta, gamma)
        value = func.Bn(x, order)

        np.testing.assert_almost_equal(value, 12.423, 3)

    def test_eval(self):
        from almiky.moments.functions import (
            CharlierFunction, CharlierSobolevFunction)

        x, order, alpha, beta, gamma = 3, 7, 0.5, 10, -2
        func = CharlierSobolevFunction(alpha, beta, gamma)
        value = func.eval(x, order)

        np.testing.assert_almost_equal(value, -99.581, 3)


if __name__ == '__main__':
    unittest.main()
