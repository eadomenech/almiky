# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

'''
It define orthogonal forms and provide operations for it evaluation.
Each ortogonal form is defined in a class and depend of an ortogonal function.
'''

import math
from .functions import CharlierFunction, CharlierSobolevFunction, QHahnFunction
from mpmath import *


class OrthogonalForm:
    '''
    Abstract class that represent an orthogonal form.
    Especific ortogonal form must define "function_class" class attribute
    and implement "weigth" and "norm" method in derivated classes.

    class FormX(OrtogonalForm)
        function_class = FunctionX

        def weigth(...)
            ...

        def weigth(...)
            ...

    FormX(order, **parameters) => new orthogonal form from
    orthogonal function FunctionX with n especific order and parameters.

    For example: FormX(8, alpha=0.2, beta=0.3)
    '''

    def __init__(self, function):
        self.function = function

    def eval(self, x, order):
        '''
        from.eval(x) => double, return evaluation of orthogonal form in x
        '''
        return (
            self.function.eval(x, order) *
            math.sqrt(self.weight(x) / self.function.norm(order))
        )

    def weight(self, x):
        '''
        from.weight(x) => double, return evaluation of weight function of
        orthogonal form in x
        '''
        raise NotImplementedError


class CharlierForm(OrthogonalForm):
    '''
    Class that represent a charlier ortogonal form.
    '''
    def __init__(self, function, alpha):
        super().__init__(function)
        self.alpha = alpha

    def weight(self, x):
        return math.exp(-self.alpha) * self.alpha ** x / math.factorial(x)


class QHahnForm(OrthogonalForm):
    '''
    Class that represent a q-hahn ortogonal form.
    '''
    def __init__(self, function, beta, q, alpha, N):
        super().__init__(function)
        self.beta = beta
        self.q = q
        self.alpha = alpha
        self.beta = beta
        self.N = N

    def weight(self, x):
        mp.dps = 25
        mp.pretty = True
        return (
            qp(self.alpha * self.q, self.q, x) *
            qp(self.q ** -self.N, self.q, x) *
            (self.alpha * self.beta * self.q) ** -x *
            qp(self.q, self.q, x) ** -1 *
            qp(self.beta ** -1 * self.q ** -self.N, self.q, x) ** -1
        )
