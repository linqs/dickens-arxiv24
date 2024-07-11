#!/usr/bin/env python3
import argparse
import os
import sys

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(THIS_DIR, '..', '..', '..'))

import utils
import mnist_addition.experiments.experiment_utils as experiment_utils

EXPERIMENT_HYPERPARAMETER_SEARCH_DIR = os.path.join(experiment_utils.LEARNING_RESULTS_BASE_DIR, "unsupervised_hyperparameter_search")
EXPERIMENT_RESULTS_DIR = os.path.join(experiment_utils.LEARNING_RESULTS_BASE_DIR, "unsupervised")

CONFIG_FILE_NAME = "unsupervised_config.json"

SPLITS = ["0", "1", "2", "3", "4"]
HYPERPARAMETER_SEARCH_TRAIN_SIZES = ["00600"]
HYPERPARAMETER_SEARCH_UNLABELED_RATIOS = ["1.00"]
TRAIN_SIZES = ["00600"]
UNLABELED_RATIOS = ["1.00"]


def main(arguments):
    config = utils.load_json_file(os.path.join(THIS_DIR, arguments.learning_method, CONFIG_FILE_NAME))

    if arguments.run_hyperparameter_search:
        base_out_dir = os.path.join(EXPERIMENT_HYPERPARAMETER_SEARCH_DIR, config["LEARNING_METHOD"], arguments.model.split(".")[0])
        experiment_utils.run_wl_experiments_hyperparamter_search(
            arguments.model, config, HYPERPARAMETER_SEARCH_TRAIN_SIZES, HYPERPARAMETER_SEARCH_UNLABELED_RATIOS, base_out_dir
        )

    else:
        base_out_dir = os.path.join(EXPERIMENT_RESULTS_DIR, config["LEARNING_METHOD"], arguments.model.split(".")[0])
        experiment_utils.run_wl_experiments(arguments.model, config, TRAIN_SIZES, UNLABELED_RATIOS, base_out_dir,
                                            time_out=10 * 60 * 60)


def _load_args():
    parser = argparse.ArgumentParser(description='Run unsupervised learning experiments.')

    parser.add_argument('--learning-method', dest='learning_method',
                        action='store', type=str, required=True,
                        choices=["bilevel", "energy", "policy_gradient"],
                        help='The learning method to run unsupervised experiments for.')

    parser.add_argument('--model', dest='model',
                        action='store', type=str, required=True,
                        choices=["mnist-add1.json", "mnist-add2.json"],
                        help='The symbolic model to run unsupervised experiments for.')

    parser.add_argument('--run-hyperparameter-search', dest='run_hyperparameter_search',
                        action='store_true', default=False,
                        help='Run hyperparameter search for the given model.')

    arguments = parser.parse_args()

    return arguments


if __name__ == '__main__':
    _load_args()
    main(_load_args())
