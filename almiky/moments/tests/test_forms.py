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
            function_class = CharlierFunction

        order = 8
        x = 4
        form = OtherOrthogonalForm(order, alpha=5)
        try:
            form.weight(x)
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

        order = 8
        x = 4
        form = CharlierForm(order, alpha=5)
        value = form.eval(x)

        self.assertEqual(value, -0.03129170161915745, "Incorrect evaluation")


class CharlierSobolevOrtogonalFormsTest(unittest.TestCase):
    '''
    Tests to verify the evaluation of ortogonal functions
    '''

    def test_charlierSobolev_eval(self):
        from almiky.moments.functions import CharlierSobolevFunction
        from almiky.moments.orthogonal_forms import CharlierSobolevForm

        x, order = 3, 7
        form = CharlierSobolevForm(order, alpha=0.5, beta=10, gamma=-2)
        value = form.eval(x)

        np.testing.assert_almost_equal(value, -1.784, 3)


class QHahnOrtogonalFormsTest(unittest.TestCase):
    '''
    Tests to verify the evaluation of ortogonal functions
    '''

    def test_weight_eval(self):
        from almiky.moments.functions import QHahnFunction
        from almiky.moments.orthogonal_forms import QHahnForm

        x, order = 4, 8

        form = QHahnForm(order, q=0.5, alpha=0.5, beta=0.5, N=8)
        value = form.weight(x)

        np.testing.assert_almost_equal(value, 481.44, 3)

    def test_qhahnform_eval(self):
        from almiky.moments.functions import QHahnFunction
        from almiky.moments.orthogonal_forms import QHahnForm

        x, order = 4, 8

        form = QHahnForm(order, q=0.5, alpha=0.5, beta=0.5, N=8)
        value = form.eval(x)

        np.testing.assert_almost_equal(value, 0.00166052, 8)


if __name__ == '__main__':
    unittest.main()
