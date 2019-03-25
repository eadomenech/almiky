# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

import math
from functools import reduce


class OrtogonalFunction:
    '''
    Represent a polinomial function used to calculate ortogonal moments.
    This class do not represent any particular polinomial function; should
    be derivated for set the evaluator by implementing _eval(...) method.
    '''

    def __init__(self, orden, **kwargs):
        self.order = order
        self.kwargs = kwargs

    def moment(self, x):
        '''
        func.moment(x) => double, return the moment for value x
        '''
        values = (self._eval(x, o) for o in range(order + 1))
        return math.fsum(values)

    def _eval(self, x):
        '''
        func._eval(a) => double, return the result of evaluate func for value x
        '''
        raise NotImplementedError


class CharlierFunction(OrtogonalFunction):

    def _eval(self, x, order):
        return x * order * self.kwargs['alpha']
