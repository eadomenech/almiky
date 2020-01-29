import numpy as np
from scipy import ndimage
from scipy.spatial import distance

from . import metrics

class AdditiveNoiseEstimator:
    def __init__(self, metric):
        self.metric = metric

    def _distance(self, image):
        cm = self.metric(image)
        return distance.mahalanobis(cm, self.mean, self.icovariance)

    def fit(self, images):
        centers_mass = np.array([
            self.metric(image)
            for image in images
        ])
        self.mean = np.mean(centers_mass, axis=0)
        self.icovariance = np.linalg.inv(np.cov(centers_mass.transpose()))

    def predict(self, images, threshold=40):
        return [
            1 if self._distance(image) > threshold else 0
            for image in images
        ]
