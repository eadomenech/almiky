# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-


class OrtogonalMatrix:

    @staticmethod
    def get(function_class, dimension=8):
        '''
        cls.get(func, n) => matrix, return ortogonal matrix
        of dimension 'n' from a polinomial function.
        '''
        matrix = []
        indices = range(1, dimension + 1)
        for i in indices:
            row = []
            for j in indices:
                func = function_class(j)
                row.append(function.moment(i))
            matrix.append(row)

        return matrix
