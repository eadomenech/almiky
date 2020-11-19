import argparse
import json
import numpy as np
from pathlib import Path
import sys

from almiky.steganalysis.additive_noise import metrics
from almiky.steganalysis.additive_noise import features

def main():
    parser = argparse.ArgumentParser(
        description='Create a dataset of center of mass o \
            histogram characteristic function of image'
    )
    parser.add_argument("indir", metavar="indir", help="Images directory")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="File where dataset will be saved")
    parser.add_argument(
        "-t",
        "--target",
        required=True,
        help="Target")
    parser.add_argument(
        "-s",
        "--size",
        required=True,
        help="Size of dataset")

    args = parser.parse_args()

    size = int(args.size)
    target = int(args.target)
    target = np.full((size, 1), target)
    indir = Path(args.indir)

    hchfcom = metrics.HCFCOM()
    load = features.ProcessImageFolder(hchfcom)

    data = list(load(indir))
    dataset = np.append(data, target, axis=1)
    np.savetxt(args.output, dataset)


if __name__ == "__main__":
    main()
