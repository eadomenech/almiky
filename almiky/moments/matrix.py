# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

'''
It define orthogonal matrix from orthogonal forms
'''

import numpy as np
from .orthogonal_forms import CharlierForm, QHahnForm


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

    def __init__(self, ortogonal_form):
        self.form = ortogonal_form

    def get_value(self, i, j):
        '''
        matrix.get_value(i, j) => double, return value of the matrix
        in the coefficients i,j
        '''
        return self.form.eval(i, order=j)

    def get_values(self, dimension=8):
        '''
        matrix.get_values(dimension) => matrix, return all values of an
        ortogonal matrix of the dimension especified.
        '''
        raise NotImplementedError


class CharlierMatrix(OrthogonalMatrix):

    def get_values(self, dimension=8):
        matrix = np.zeros((dimension, dimension))
        indices = range(dimension)
        for i in indices:
            for j in indices:
                matrix[i][j] = self.get_value(i, j)

        return matrix

