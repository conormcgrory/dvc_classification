"""Script for training classifier."""

import argparse

import yaml
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


def parse_args():
    """Parse arguments from command-line."""

    parser = argparse.ArgumentParser(
        description='Split data into training and test sets.'
    )
    parser.add_argument(
        '-i', '--infile', type=str, required=True,
        help='Path to input file (.npz) containing processed data'
    )
    parser.add_argument(
        '-o', '--outfile', type=str, required=True,
        help='Path to output file for model (.pkl)'
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

    # Load data
    print(f'Loading training data from {args.infile}...')
    in_data = np.load(args.infile)
    x = in_data['x']
    y = in_data['y']
    print('Done.')

    # Fit logistic regression model
    print('Fitting model...')
    model = LogisticRegression(**params['train'])
    model.fit(x, y)
    print('Done.')

    # Save model
    print(f'Saving model to {args.outfile}...')
    with open(args.outfile, 'wb') as f:
        joblib.dump(model, f)
    print('Done.')


if __name__ == '__main__':
    main()
