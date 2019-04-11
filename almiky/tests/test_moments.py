# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

import unittest


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

    def test_charlier_eval(self):
        from almiky.moments.functions import CharlierFunction

        order = 8
        x = 4
        func = CharlierFunction(alpha=5)
        value = func.keval(x, 0, order)

        self.assertEqual(value, 390625, "Incorrect evaluation")

    def test_charlier_evaluation(self):
        from almiky.moments.functions import CharlierFunction

        order = 8
        x = 4
        func = CharlierFunction(alpha=5)
        value = func.eval(x, order)

        self.assertEqual(value, -9375, "Incorrect evaluation")


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
        import numpy as np
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
