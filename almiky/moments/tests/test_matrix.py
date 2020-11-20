# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

import unittest
from unittest.mock import Mock

import numpy as np

from almiky.moments import matrix


class ImageTransform(unittest.TestCase):
    def test_direct(self):
        data = np.random.rand(4, 4)
        expected = np.random.rand(4, 4)
        # Base transform mock
        transform = Mock()
        transform.direct = Mock(return_value=expected)

        itransform = matrix.ImageTransform(transform)

        # transform.direct.assert_called()
        np.testing.assert_equal(itransform.direct(data), expected)

    def test_inverse_defaul_max_amplitude(self):
        '''
        Testing inverse with max aplitude equal to 255
        '''
        inverse = np.array([
            [-1.8, 0, 58.78],
            [255.0, 255.1, 52.17]
        ])
        data = np.random.rand(4, 4)
        expected = np.array([
            [0, 0, 59],
            [255, 255, 52]
        ])
        # Base transform mock
        transform = Mock()
        transform.inverse = Mock(return_value=inverse)

        itransform = matrix.ImageTransform(transform)

        # transform.direct.assert_called()
        np.testing.assert_equal(itransform.inverse(data), expected)

    def test_inverse_custom_max_amplitude(self):
        max_amplitude = 16
        inverse = np.array([
            [-1.8, 0, 14.59],
            [16.0, 16.1, 12.17]
        ])
        data = np.random.rand(4, 4)
        expected = np.array([
            [0, 0, 15],
            [16, 16, 12]
        ])
        # Base transform mock
        transform = Mock()
        transform.inverse = Mock(return_value=inverse)

        itransform = matrix.ImageTransform(transform, max_amplitude)

        # transform.direct.assert_called()
        np.testing.assert_equal(itransform.inverse(data), expected)


class OrtogonalMatrixTest(unittest.TestCase):

    def test_matrix(self):
        from almiky.moments.matrix import OrthogonalMatrix

        class OtherMatrix(OrthogonalMatrix):
            pass

        dimension = 2
        try:
            OtherMatrix(dimension, alpha=5)
        except TypeError:
            # Normal behavior. orthogonal_form_class must be set
            pass
        else:
            assert False, "Call to undefined get_values function in" \
                "OrthogonalMatrix derivated class without 'keval' method"


class CharlierMatrixTest(unittest.TestCase):

    def test_matrix(self):
        from almiky.moments.matrix import CharlierMatrix

        dimension = 2
        matrix = CharlierMatrix(dimension, alpha=5)

        np.testing.assert_array_almost_equal(
            matrix.values,
            np.asarray([
                [0.08208499862389880, -0.1835476368560144],
                [0.1835476368560144, -0.3283399944955952]
            ])
        )

    def test_direct_transform(self):
        from almiky.moments.matrix import CharlierMatrix

        dimension = 2
        matrix = CharlierMatrix(dimension, alpha=5)
        data = np.random.rand(2, 2)
        result = np.dot(matrix.values, np.dot(data, matrix.values.T))
        np.testing.assert_array_almost_equal(matrix.direct(data), result)

    def test_inverse(self):
        from almiky.moments.matrix import CharlierMatrix

        dimension = 2
        matrix = CharlierMatrix(dimension, alpha=5)
        data = np.random.rand(2, 2)
        result = np.dot(matrix.values.T, np.dot(data, matrix.values))

        np.testing.assert_array_almost_equal(matrix.inverse(data), result)


class CharlierSobolevMatrixTest(unittest.TestCase):

    def test_matrix(self):
        from almiky.moments.matrix import CharlierSobolevMatrix

        dimension = 2
        matrix = CharlierSobolevMatrix(dimension, alpha=0.5, beta=10, gamma=-2)

        np.testing.assert_array_almost_equal(
            matrix.values,
            np.asarray([
                [0.77880078, -0.55069531],
                [0.55069531, 0.38940039]
            ])
        )


class QHahnMatrixTest(unittest.TestCase):

    def test_matrix(self):
        from almiky.moments.matrix import QHahnMatrix

        dimension = 2
        matrix = QHahnMatrix(dimension, q=0.5, alpha=0.5, beta=0.5, N=2)

        np.testing.assert_array_almost_equal(
            matrix.values,
            np.asarray([
                [0.212512, 0.516398],
                [0.481932, 0.68313]
            ])
        )


class QKrawtchoukMatrixTest(unittest.TestCase):

    def test_matrix(self):
        from almiky.moments.matrix import QKrawtchoukMatrix

        dimension = 8
        matrix = QKrawtchoukMatrix(dimension, p=0.7, q=0.75)

        np.testing.assert_array_almost_equal(
            matrix.values,
            np.asarray([
                [
                    0.00232851, 0.00682813, 0.0168077, 0.0394712, 0.0917658,
                    0.211195, 0.463611, 0.854487],
                [
                    0.0141819, 0.0383304, 0.085273, 0.174665, 0.331691,
                    0.543652, 0.564362, -0.486281],
                [
                    0.0550748, 0.131992, 0.250916, 0.407944, 0.518678,
                    0.288771, -0.605489, 0.176455],
                [
                    0.155209, 0.308613, 0.44707, 0.442106, 0.0548819,
                    -0.624575, 0.300124, -0.0464647],
                [
                    0.329795, 0.476248, 0.369815, -0.114016, -0.58576,
                    0.402597, -0.0968305, 0.00922526],
                [
                    0.52837, 0.379548, -0.200752, -0.535383, 0.475853,
                    -0.151373, 0.0219292, -0.00138102],
                [
                    0.614293, -0.153151, -0.541145, 0.516625, -0.195207,
                    0.0364955, -0.00345897, 0.000150025],
                [
                    0.455384, -0.701068, 0.509621, -0.198804, 0.0431869,
                    -0.00522401, 0.000339036, -0.0000103919]
            ])
        )


if __name__ == '__main__':
    unittest.main()
