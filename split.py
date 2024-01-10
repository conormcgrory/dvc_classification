"""Splits data into training and test sets."""


import argparse

import yaml
import asdf
import numpy as np
from sklearn.model_selection import train_test_split


def parse_args():
    """Parse arguments from command-line."""

    parser = argparse.ArgumentParser(
        description='Splits data into training and test sets.'
    )
    parser.add_argument(
        '-i', '--infile', type=str, required=True,
        help='Path to input file (.asdf) containing raw data'
    )
    parser.add_argument(
        '--trainfile', type=str, required=True,
        help='Path to output file for training data (.npz)'
    )
    parser.add_argument(
        '--testfile', type=str, required=True,
        help='Path to output file for test data (.npz)'
    )
    parser.add_argument(
        '-p', '--params', type=str, default='params.yaml',
        help='Parameters file'
    )

    return parser.parse_args()


def main():


    # Get arguments and parameters
    args = parse_args()
    params = yaml.safe_load(open(args.params))
    rand_seed = params['split']['rand_seed']
    train_size = params['split']['train_size']

    # Load data
    print(f'Loading data from {args.infile}...')
    with asdf.open(args.infile) as f:
        x = np.array(f['data'])
        y= np.array(f['target'])
    print('Done.')

    # Split into training and test sets
    print('Splitting data into training and test sets...')
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        train_size=train_size,
        random_state=rand_seed,
        stratify=y
    )
    print('Done.')
    print(f'Train size: {x_train.shape[0]}')
    print(f'Test size: {x_test.shape[0]}')

   # Save training set to file
    print(f'Saving training data to {args.trainfile}...')
    np.savez(args.trainfile, x=x_train, y=y_train)
    print('Done.')

   # Save test set to file
    print(f'Saving training data to {args.testfile}...')
    np.savez(args.testfile, x=x_test, y=y_test)
    print('Done.')


if __name__ == '__main__':
    main()