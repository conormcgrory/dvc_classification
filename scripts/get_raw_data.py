"""Loads Iris dataset from sklearn and saves it in ASDF file."""

import argparse

import asdf
import numpy as np
from sklearn import datasets


def parse_args():
    """Parse arguments from command-line."""

    parser = argparse.ArgumentParser(
        description='Loads Iris data from sklearn and saves it in ASDF file.'
    )
    parser.add_argument(
        '-o', '--outfile', type=str, required=True,
        help='Path to output file (.asdf) containing Iris dataset'
    )
    return parser.parse_args()


def main():

    # Get command-line arguments
    args = parse_args()

    # Load Iris dataset from sklearn and copy to dictionary
    print('Loading Iris dataset...')
    iris_dset = datasets.load_iris()
    iris_dict = {
        'data': iris_dset.data,
        'target': iris_dset.target,
        'feature_names': iris_dset.feature_names,
        'target_names': iris_dset.target_names,
        'filename': iris_dset.filename,
        'data_module': iris_dset.data_module,
        'DESCR': iris_dset.DESCR,
    }
    print('Done.')

    # Save data to ASDF file
    print(f'Output file: {args.outfile}')
    print('Writing data...')
    out_file = asdf.AsdfFile(iris_dict)
    out_file.write_to(args.outfile)
    print('Done.')


if __name__ == '__main__':
    main()