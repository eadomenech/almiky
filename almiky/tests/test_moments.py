# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

import unittest


class TestMoments(unittest.TestCase):
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
            func.moment(x, order)
        except NotImplementedError:
            # Normal behaivor. OrtogonalFunction derivated
            # clasess must implment 'expression' method because
            # is called by "eval" method
            pass
        else:
            assert False, "Evaluated OrtogonalFunction derivated class" \
                "without 'expression' method"

    def test_moment_defined(self):
        from almiky.moments.functions import CharlierFunction

        order = 8
        x = 4
        func = CharlierFunction(order, alpha=5)
        moment = func.moment(x)

        self.assertEqual(moment, 80, "Incorrect moment")


class OrtogonalMatrixTest(unittest.TestCase):

    def test_matrix(self):
        from almiky.moments.matrix import OrtogonalMatrix
        from almiky.moments.functions import OrtogonalFunction

        class OtherFunction(OrtogonalFunction):
            def _eval(self, x):
                return x

        dimension = 2
        func = OtherFunction(2)
        matrix = OrtogonalMatrix.get(func, dimension)

        self.assertEqual(
            matrix,
            [
                [1, 2],
                [3, 6]
            ]
        )


if __name__ == '__main__':
    unittest.main()
