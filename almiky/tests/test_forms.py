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


if __name__ == '__main__':
    unittest.main()
