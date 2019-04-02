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

    def __init__(self, order):
        self.order = order

    def eval(self, x, order=None):
        '''
        func.eval(x) => double, return evaluation of the ortogonal function
        in x
        '''
        order = self.order if order is None else order
        values = (self.keval(x, k, order) for k in range(order + 1))
        return sum(values)

    def keval(self, x, k, order):
        '''
        func.keval(a) => double, return evaluation of the ortogonal function
        in x for specific order
        '''
        raise NotImplementedError

    '''
    func.norm(x) => double, return the norm of orthogonal function
    '''
    def norm(self):
        raise NotImplementedError


class CharlierFunction(OrtogonalFunction):

    def __init__(self, order, alpha):
        super().__init__(order)
        self.alpha = alpha

    def keval(self, x, k, order):
        return (
            (-1) ** (order - k) *
            special.poch(-order, k) *
            special.poch(-x, k) *
            self.alpha ** (order - k) /
            math.factorial(k)
        )

    def norm(self, order=None):
        order = order or self.order
        if order < 0:
            return 0
        else:
            return math.factorial(order) * self.alpha ** order


class CharlierSobolevFunction(CharlierFunction):

    def __init__(self, order, alpha, beta, gamma):
        super().__init__(order, alpha)
        self.beta = beta
        self.gamma = gamma

    def kernel(self, x):
        return sum(
            [
                (i * super(CharlierFunction, self).eval(x, i - 1)) ** 2 /
                self.norm(i) for i in range(self.order)
            ]
        )


    def An(self, x):
        if self.order <= 0:
            return 1
        else:
            num = (
                self.beta *
                self.order *
                super().eval(self.gamma, self.order - 1) *
                (
                    eval(self.gamma, self.order - 1) + (x - self.gamma) *
                    (self.order - 1) *
                    super().eval(self.gamma, self.order - 2)
                )
            )
            den = (
                self.norm(self.order - 1) *
                (1 + self.order * self.kernel(self.gamma)) *
                (x - self.gamma) *
                (x - self.gamma - 1)
            )
            return 1 - num / den


    def Bn(self, x):
        if self.order <= 0:
            return 0
        else:
            num = (
                self.beta *
                self.order *
                super().eval(self.gamma, self.order - 1) *
                (
                    super().eval(self.gamma) + (x - self.gamma) *
                    self.order *
                    super().eval(self.gamma, self.order - 1)
                )
            )
            den = (
                self.norm(self.order - 1) *
                (1 + self.beta * self.kernel(self.gamma)) *
                (x - self.gamma) *
                (x - self.gamma- 1)
            )
            return num / den

    def eval(self, x):
        '''
        func.eval(x) => double, return evaluation of the ortogonal function
        in x
        '''
        return (
            self.An(x) * super().eval(x) +
            self.Bn(x) * super().eval(x, self.order - 1)
        )
