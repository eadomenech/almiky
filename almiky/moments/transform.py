'''
Orthogonal direct and inverse moments
'''
import numpy as np

class Transform:
    def __init__(self, ortho_matrix):
        '''
        obj.__init__(ortho_matrix) => None: Set orthogonal matrix
        '''
        self.ortho_matrix = ortho_matrix

    def direct(self, data):
        '''
        obj.direct(data) => (np.array): return direct matrix moments
        (data) is a (np.array) that it´s shape must match with (self.ortho_matrix shape)
        '''
        return np.dot(self.ortho_matrix, np.dot(data, self.ortho_matrix.T))

    def inverse(self, data):
        '''
        obj.direct(data) => (np.array): return inverse matrix moments
        (data) is a (np.array) that it´s shape must match with (self.ortho_matrix shape)
        '''
        return np.dot(self.ortho_matrix.T, np.dot(data, self.ortho_matrix))
