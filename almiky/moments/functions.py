# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

import math
from scipy import special


class OrtogonalFunction:
    '''
    Represent a polinomial function used to calculate ortogonal moments.
    This class do not represent any particular polinomial function; should
    be derivated for set the evaluator by implementing _eval(...) method.
    '''

    def __init__(self, order, **kwargs):
        self.order = order
        self.params = kwargs

    def eval_poly(self, x):
        '''
        func.moment(x) => double, return the moment for value x
        '''
        values = (self._eval(x, k) for k in range(self.order + 1))
        return math.fsum(values)

    def _eval(self, x, k):
        '''
        func._eval(a) => double, return the result of evaluate func for value x
        '''
        raise NotImplementedError


class CharlierFunction(OrtogonalFunction):

    def _eval(self, x, k):
        aux = (
            (-1) ** (self.order - k) *
            special.poch(-self.order, k) *
            special.poch(-x, k) *
            self.params['alpha'] ** (self.order - k) /
            math.factorial(k)
        )
        return aux
