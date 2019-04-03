# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

'''
It define orthogonal functions and provide operations for it evaluation. 
Each orthogonal function is defined in a derivated class of OrtogonalFunction.
'''

import math
from scipy import special


class OrtogonalFunction:
    '''
    Abastract class that represent a polinomial function used to calculate
    ortogonal moments. This class do not represent any particular polinomial
    function; should be derivated for set the evaluator by implementing _eval(...)
    method.

    class FunctionX(OrtogonalFunction)
        def keval(...)
            ...

    FunctionX(order, **parameters) => new orthogonal function with
    an specific order and parameters.

    For example: FunctionX(8, alpha=0.2, beta=0.3)
    '''

    def eval(self, x, order):
        '''
        func.eval(x) => double, return evaluation of the ortogonal function
        in x
        '''
        values = (self.keval(x, k, order) for k in range(order + 1))
        return sum(values)

    def keval(self, x, k, order):
        '''
        func.keval(a) => double, return evaluation of the ortogonal function
        in x for specific order
        '''
        raise NotImplementedError


class CharlierFunction(OrtogonalFunction):

    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def keval(self, x, k, order):
        return (
            (-1) ** (order - k) *
            special.poch(-order, k) *
            special.poch(-x, k) *
            self.alpha ** (order - k) /
            math.factorial(k)
        )
