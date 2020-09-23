# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

'''
It define orthogonal matrix from orthogonal forms
'''

import numpy as np
from .orthogonal_forms import (
    CharlierForm, CharlierSobolevForm, QHahnForm, QKrawtchoukForm)


class Transform:
    def __init__(self, ortho_matrix):
        self.values = ortho_matrix

    def direct(self, data):
        '''
        obj.direct(data) => (np.array): return direct matrix moments
        (data) is a (np.array) that it´s shape must match with (self.ortho_matrix shape)
        '''
        return np.dot(self.values, np.dot(data, self.values.T))

    def inverse(self, data):
        '''
        obj.direct(data) => (np.array): return inverse matrix moments
        (data) is a (np.array) that it´s shape must match with (self.ortho_matrix shape)
        '''
        return np.dot(self.values.T, np.dot(data, self.values))


class ImageTransform:
    def __init__(self, transform, max_amplitude=255):
        self.transform = transform
        self.max_amplitude = max_amplitude

    def direct(self, data):
        return self.transform.direct(data)

    def inverse(self, data):
        inverted = self.transform.inverse(data)
        return np.clip(np.rint(inverted), 0, self.max_amplitude)


class OrthogonalMatrix(Transform):
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

    def __init__(self, dimension, **parameters):
        self.dimension = dimension
        self.parameters = parameters
        self.set_values()

    def get_column(self, order):
        form = self.orthogonal_form_class(order, **self.parameters)
        return np.array([form.eval(i) for i in range(self.dimension)])

    def set_values(self):
        '''
        matrix.get_values(dimension) => matrix, return all values of an
        ortogonal matrix of the dimension especified.
        '''
        matrix = np.empty((self.dimension, self.dimension))
        indices = range(self.dimension)
        for i in indices:
            matrix[:, i] = self.get_column(order=i)

        self.values = matrix


class CharlierMatrix(OrthogonalMatrix):

    orthogonal_form_class = CharlierForm


class CharlierSobolevMatrix(CharlierMatrix):

    orthogonal_form_class = CharlierSobolevForm


class QHahnMatrix(CharlierMatrix):

    orthogonal_form_class = QHahnForm


class QKrawtchoukMatrix(OrthogonalMatrix):

    orthogonal_form_class = QKrawtchoukForm

    def __init__(self, dimension, **parameters):
        self.dimension = dimension
        self.parameters = parameters
        self.parameters['N'] = self.dimension - 1
        self.set_values()
