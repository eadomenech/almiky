import argparse
import json
import sys

from src.utils import process

def main():
    parser = argparse.ArgumentParser(
        description='Create ')
    parser.add_argument("indir", metavar="indir", help="Images directory")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="File where dataset will be saved")
    parser.add_argument(
        "-d",
        "--data",
        required=True,
        help="File with data to hide"
    )
    args = parser.parse_args()
    with open(args.data, 'r') as file:
        data = file.read()

    process.dct8x8(
        indir=args.indir, output=args.output, data=data)


if __name__ == "__main__":
    main()