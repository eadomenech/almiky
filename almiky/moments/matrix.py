'''It define orthogonal matrix from orthogonal forms.'''

import numpy as np

from .orthogonal_forms import (
    CharlierForm, CharlierSobolevForm, QHahnForm, QKrawtchoukForm,
    QCharlierForm, TchebichefForm)


class SeparableTransform:
    def __init__(self, left_orthon_matrix, right_orthon_matrix):
        self.l_orthon_matrix = left_orthon_matrix
        self.r_orthon_matrix = right_orthon_matrix

    def direct(self, A):
        return np.dot(self.l_orthon_matrix.T, np.dot(A, self.r_orthon_matrix))

    def inverse(self, A):
        return np.dot(
            self.l_orthon_matrix,
            np.dot(A, self.r_orthon_matrix.T)
        )


class Transform:
    def __init__(self, ortho_matrix):
        self.transform = SeparableTransform(ortho_matrix, ortho_matrix)

    def direct(self, data):
        '''
        obj.direct(data) => (np.array): return direct matrix moments
        (data) is a (np.array) that it´s shape must match with
        (self.ortho_matrix shape)
        '''
        return self.transform.direct(data)

    def inverse(self, data):
        '''
        obj.direct(data) => (np.array): return inverse matrix moments
        (data) is a (np.array) that it´s shape must match with
        (self.ortho_matrix shape)
        '''
        return self.transform.inverse(data)


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

    MatixX(\**parameters) => new orthogonal matrix from orthogonal form FormX
    with an specific parameters.

    For example: MatrixX(alpha=0.2, beta=0.3)
    '''
    orthogonal_form_class = None

    def __init__(self, dimension, **parameters):
        self.dimension = dimension
        self.parameters = parameters
        matrix = self.get_values()
        super().__init__(matrix)

    def ident(self, matrix):
        identity = np.identity(matrix.shape[0])
        return sum(sum(abs(np.around(np.dot(matrix.T, matrix))) - identity))

    def get_column(self, order):
        form = self.orthogonal_form_class(order, **self.parameters)
        return np.array([form.eval(i) for i in range(self.dimension)])

    def get_values(self):
        '''
        matrix.get_values(dimension) => matrix, return all values of an
        ortogonal matrix of the dimension especified.
        '''
        matrix = np.empty((self.dimension, self.dimension))
        indices = range(self.dimension)
        for i in indices:
            matrix[:, i] = self.get_column(order=i)

        self.quasi_orthogonal = (
            self.ident(matrix) == 0 and
            self.ident(matrix.T) == 0
        )
        return matrix


class CharlierMatrix(OrthogonalMatrix):

    orthogonal_form_class = CharlierForm


class CharlierSobolevMatrix(CharlierMatrix):

    orthogonal_form_class = CharlierSobolevForm


class QHahnMatrix(CharlierMatrix):

    orthogonal_form_class = QHahnForm


class QKrawtchoukMatrix(OrthogonalMatrix):

    orthogonal_form_class = QKrawtchoukForm

    def __init__(self, dimension, **parameters):
        parameters['N'] = dimension - 1
        super().__init__(dimension, **parameters)


class QCharlierMatrix(OrthogonalMatrix):

    orthogonal_form_class = QCharlierForm


class TchebichefMatrix(OrthogonalMatrix):

    orthogonal_form_class = TchebichefForm
