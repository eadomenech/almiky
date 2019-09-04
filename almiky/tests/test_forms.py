# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

import unittest
import numpy as np


class OrtogonalFormsTest(unittest.TestCase):
    '''
    Tests to verify the evaluation of ortogonal functions
    '''

    def test_weight_eval(self):
        from almiky.moments.functions import CharlierFunction
        from almiky.moments.orthogonal_forms import OrthogonalForm

        class OtherOrthogonalForm(OrthogonalForm):
            pass

        function = CharlierFunction(alpha=5)

        form = OtherOrthogonalForm(function)
        try:
            form.weight(5)
        except NotImplementedError:
            # Normal behavior. OrthogonalForm derivated
            # clasess must implement 'weight' method because
            # has not been implemented.
            pass
        else:
            assert False, "Call to undefined weight function in orthogonal" \
                "form derivated class without 'keval' method"


class CharlierOrtogonalFormsTest(unittest.TestCase):
    '''
    Tests to verify the evaluation of ortogonal functions
    '''

    def test_charlier_eval(self):
        from almiky.moments.functions import CharlierFunction
        from almiky.moments.orthogonal_forms import CharlierForm

        alpha = 5
        function = CharlierFunction(alpha=5)
        form = CharlierForm(function, alpha=alpha)
        value = form.eval(4, order=8)

        self.assertEqual(value, -0.03129170161915745, "Incorrect evaluation")


class CharlierSobolevOrtogonalFormsTest(unittest.TestCase):
    '''
    Tests to verify the evaluation of ortogonal functions
    '''

    def test_charlierSobolev_eval(self):
        from almiky.moments.functions import CharlierSobolevFunction
        from almiky.moments.orthogonal_forms import CharlierForm

        alpha = 0.5
        function = CharlierSobolevFunction(alpha=alpha, beta=10, gamma=-2)
        form = CharlierForm(function, alpha=alpha)
        value = form.eval(3, order=7)

        np.testing.assert_almost_equal(value, -1.784, 3)


class QHahnOrtogonalFormsTest(unittest.TestCase):
    '''
    Tests to verify the evaluation of ortogonal functions
    '''

    def test_weight(self):
        from almiky.moments.functions import QHahnFunction
        from almiky.moments.orthogonal_forms import QHahnForm

        parameters = {'alpha':0.5, 'q':0.5, 'beta':0.5, 'N':8}
        function = QHahnFunction(**parameters)
        form = QHahnForm(function, **parameters)
        value = form.weight(4)

        np.testing.assert_almost_equal(value, 481.44, 3)

    def test_eval(self):
        from almiky.moments.functions import QHahnFunction
        from almiky.moments.orthogonal_forms import QHahnForm

        parameters = {'alpha':0.5, 'q':0.5, 'beta':0.5, 'N':8}
        function = QHahnFunction(**parameters)
        form = QHahnForm(function, **parameters)
        value = form.eval(4, order=8)

        np.testing.assert_almost_equal(value, 0.00166052, 8)


if __name__ == '__main__':
    unittest.main()
