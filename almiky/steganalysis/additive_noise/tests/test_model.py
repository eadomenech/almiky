
import imageio
import numpy as np
from pathlib import Path
import unittest
from unittest.mock import Mock

from scipy.spatial import distance

from almiky.steganalysis.additive_noise.model import AdditiveNoiseEstimator
from almiky.steganalysis.additive_noise import metrics

class ModelTest(unittest.TestCase):
    def test_fit(self):
        data = np.random.rand(5, 3)
        mean = np.mean(data, axis=0)
        icov = np.linalg.inv(np.cov(data.transpose()))

        metric = Mock()
        estimator = AdditiveNoiseEstimator(metric)

        estimator.fit(data)
        np.testing.assert_array_almost_equal(mean, estimator.mean)
        np.testing.assert_array_almost_equal(icov, estimator.icovariance)

    def test_predict(self):
        data = np.random.rand(5, 3)
        metric = Mock()
        estimator = AdditiveNoiseEstimator(metric)
        estimator._distance = Mock(side_effect=[12, 40, 30, 86, 10])
        predictions = estimator.predict(data)
        self.assertEqual(predictions, [1, 0, 1, 0, 1])

    def test_distance(self):
        data = np.random.rand(5, 3)
        mean = np.mean(data, axis=0)
        icov = icov = np.linalg.inv(np.cov(data.transpose()))
        com = np.random.rand(3)

        metric = Mock(side_effect= lambda i: com)
        estimator = AdditiveNoiseEstimator(metric)
        estimator.mean = mean
        estimator.icovariance = icov

        target = distance.mahalanobis(com, mean, icov)
        value = estimator._distance(com)
        self.assertEqual(value, target)
