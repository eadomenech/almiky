import argparse
import numpy as np

from almiky.steganalysis.additive_noise import model


def main():
    parser = argparse.ArgumentParser(
        description='Train form original images'
    )
    parser.add_argument("dataset", metavar="dataset", help="Dataset file")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="File where dataset will be saved")

    args = parser.parse_args()

    data = np.loadtxt(args.dataset, usecols=(0, 1, 2))
    estimator = model.AdditiveNoiseEstimator()
    estimator.fit(data)

    icovariance = estimator.icovariance.reshape(-1)
    data = np.concatenate((estimator.mean, icovariance)).reshape(1, -1)
    np.savetxt(args.output, data)


if __name__ == "__main__":
    main()
