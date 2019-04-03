# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

'''
It define orthogonal forms and provide operations for it evaluation.
Each ortogonal form is defined in a class and depend of an ortogonal function.
'''

import math
from .functions import CharlierFunction


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
    function_class = None

    def __init__(self, order, **parameters):
        self.function = self.function_class(**parameters)
        self.parameters = parameters
        self.order = order
        self.alpha = parameters['alpha']

    def eval(self, x):
        '''
        from.eval(x) => double, return evaluation of orthogonal form in x
        '''
        return (
            self.function.eval(x, self.order) *
            math.sqrt(self.weight(x) / self.norm())
        )

    def weight(self, x):
        '''
        from.weight(x) => double, return evaluation of weight function of
        orthogonal form in x
        '''
        raise NotImplementedError

    def norm(self):
        '''
        from.norm(x) => double, return the norm of orthogonal form
        '''
        raise NotImplementedError


class CharlierForm(OrthogonalForm):
    '''
    Class that represent a charlier ortogonal form.
    '''
    function_class = CharlierFunction

    def weight(self, x):
        return math.exp(-self.alpha) * self.alpha ** x / math.factorial(x)

    def norm(self):
        if self.order < 0:
            return 0
        else:
            return math.factorial(self.order) * self.alpha ** self.order
