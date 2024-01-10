"""Script for evaluating classifier."""

import argparse
import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


def parse_args():
    """Parse arguments from command-line."""

    parser = argparse.ArgumentParser(
        description='Evaluate classifier on training and test sets.'
    )
    parser.add_argument(
        '--trainfile', type=str, required=True,
        help='Path to file (.npz) containing train data'
    )
    parser.add_argument(
        '--testfile', type=str, required=True,
        help='Path to file (.npz) containing test data'
    )
    parser.add_argument(
        '--modelfile', type=str, required=True,
        help='Path to file (.pkl) containing trained model'
    )
    parser.add_argument(
        '-o', '--outfile', type=str, required=True,
        help='Path to output file (.json) for metrics.'
    )

    return parser.parse_args()


def main():

    args = parse_args()

    # Load training data
    print(f'Loading training data from {args.trainfile}...')
    train_data = np.load(args.trainfile)
    x_train = train_data['x']
    y_train = train_data['y']
    print('Done.')

    # Load test data
    print(f'Loading test data from {args.testfile}...')
    test_data = np.load(args.testfile)
    x_test = test_data['x']
    y_test = test_data['y']
    print('Done.')

    # Load model
    print(f'Loading model from {args.modelfile}...')
    model = joblib.load(args.modelfile)
    print('Done.')

    # Compute accuracy on training data
    acc_train = model.score(x_train, y_train)
    print(f'Accuracy (train): {acc_train}')

    # Compute accuracy on test data
    acc_test = model.score(x_test, y_test)
    print(f'Accuracy (test): {acc_test}')

    metrics = {
        'acc_train': acc_train,
        'acc_test': acc_test,
    }

    # Save model
    print(f'Saving metrics to {args.outfile}...')
    with open(args.outfile, 'w') as f:
        json.dump(metrics, f)
    print('Done.')


if __name__ == '__main__':
    main()