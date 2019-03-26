# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

from .functions import CharlierFunction
from .orthogonal_forms import CharlierForm

class OrthogonalMatrix:
    orthogonal_form_class=None
    function_class=None

    def __init__(self, **params):
        self.params = params

    def get(self, dimension=8):
        '''
        cls.get(func, n) => matrix, return ortogonal matrix
        of dimension 'n' from a polinomial function.
        '''
        matrix = []
        indices = range(dimension)
        for i in indices:
            row = []
            for j in indices:
                func = self.function_class(j, **self.params)
                form = self.orthogonal_form_class(func)
                row.append(form.eval_form(i))
            matrix.append(row)

        return matrix


class CharlierMatrix(OrthogonalMatrix):
    orthogonal_form_class=CharlierForm
    function_class=CharlierFunction
