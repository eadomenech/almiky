import numpy as np
from scipy.spatial import distance


class AdditiveNoiseEstimator:
    '''
    One class classication for non stego images
    based on mahalanobis distance between center of mass
    of histogran characteristic funcion (hcf) of image and
    mean hcf of image stadistic ditribution.
    '''

    def __init__(self, threshold=40, mean=None, icovariance=None):
        self.threshold = threshold
        self.mean = mean
        self.icovariance = icovariance

    def fit(self, data):
        '''
        estimador.fit(data) => None: Calculate mean an covariance
        of data. data parameter must be an numpy array
        '''

        self.mean = np.mean(data, axis=0)
        self.icovariance = np.linalg.inv(np.cov(data.transpose()))

    def predict(self, data):
        '''
        estimador.predict(data) => numpy array: Classificate data using
        a distance (defined in self._distance method) and threshold.
        Data must be an numpy array.
        Return a numpy array where values are one if item belong to class
        but cero otherwise.
        '''
        return [
            1 if distance.mahalanobis(
                item, self.mean, self.icovariance
            ) < self.threshold else -1
            for item in data
        ]
