# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

import unittest
import numpy as np


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
