# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

'''
It define orthogonal matrix from orthogonal forms
'''

import numpy as np
from .orthogonal_forms import CharlierForm, CharlierSobolevForm, QHahnForm


class OrthogonalMatrix:
    '''
    Abstract class that represent an orthogonal matrix.
    Especific ortogonal matrix must define "othogonal_form__class" class
    attribute and implement "get_values" method in derivated classes.

    class MatrixX(OrthogonalMatrix)
        orthogonal_form_class = FromX

        def get_values(...)
            ...

    MatixX(**parameters) => new orthogonal matrix from orthogonal form FormX
    with an specific parameters.

    For example: MatrixX(alpha=0.2, beta=0.3)
    '''
    orthogonal_form_class = None

    def __init__(self, **parameters):
        self.parameters = parameters

    '''
    matrix.get_value(i, j) => double, return value of the matrix
    in the coefficients i,j
    '''

    def get_column(self, order, dimension):
        form = self.orthogonal_form_class(order, **self.parameters)
        return np.array([form.eval(i) for i in range(dimension)])

    def get_values(self, dimension=8):
        '''
        matrix.get_values(dimension) => matrix, return all values of an
        ortogonal matrix of the dimension especified.
        '''
        raise NotImplementedError


class CharlierMatrix(OrthogonalMatrix):

    orthogonal_form_class = CharlierForm

    def get_values(self, dimension=8):
        matrix = np.empty((dimension, dimension))
        indices = range(dimension)
        for i in indices:
            matrix[:, i] = self.get_column(order=i, dimension=dimension)

        return matrix


class CharlierSobolevMatrix(CharlierMatrix):

    orthogonal_form_class = CharlierSobolevForm


class QHahnMatrix(CharlierMatrix):

    orthogonal_form_class = QHahnForm
