
import imageio
import numpy as np
from pathlib import Path
import unittest
from unittest.mock import Mock

from scipy.spatial import distance

from almiky.steganalysis.additive_noise.model import AdditiveNoiseEstimator

class ModelTest(unittest.TestCase):
    def test_fit(self):
        data = np.random.rand(5, 3)
        mean = np.mean(data, axis=0)
        icov = np.linalg.inv(np.cov(data.transpose()))

        estimator = AdditiveNoiseEstimator()

        estimator.fit(data)
        np.testing.assert_array_almost_equal(mean, estimator.mean)
        np.testing.assert_array_almost_equal(icov, estimator.icovariance)

    def test_predict(self):
        data = np.random.rand(5, 3)
        estimator = AdditiveNoiseEstimator()
        estimator.mean = np.random.rand(3)
        estimator.icovariance = np.random.rand(3, 3)
        distance.mahalanobis = Mock(side_effect=[12, 40, 30, 86, 10])
        predictions = estimator.predict(data)
        self.assertEqual(predictions, [1, 0, 1, 0, 1])

