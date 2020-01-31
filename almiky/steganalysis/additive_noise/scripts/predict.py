import argparse
import json
import numpy as np
from pathlib import Path
import pickle
import sys

from almiky.steganalysis.additive_noise import metrics
from almiky.steganalysis.additive_noise import model

def main():
    parser = argparse.ArgumentParser(
        description='Predict image classification (cover or stego)'
    )
    parser.add_argument("dataset", metavar="dataset", help="dataset")
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="File where dataset will be saved")
    parser.add_argument(
        "-t",
        "--target",
        required=True,
        help="File where target will be saved")
    parser.add_argument(
        "-r",
        "--threshold",
        required=True,
        help="File where target will be saved")

    args = parser.parse_args()

    threshold = float(args.threshold)
    data = np.loadtxt(args.dataset, usecols=(0, 1, 2))
    model_data = np.loadtxt(args.model)
    mean = model_data[:3]
    icovariance = model_data[3:].reshape(3, 3)

    estimator = model.AdditiveNoiseEstimator()
    estimator.mean = mean
    estimator.icovariance = icovariance
    predictions = estimator.predict(data, threshold)

    np.savetxt(args.target, predictions)


if __name__ == "__main__":
    main()
