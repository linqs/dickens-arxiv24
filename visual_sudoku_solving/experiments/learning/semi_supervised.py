#!/usr/bin/env python3
import argparse
import os
import sys

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.join(THIS_DIR, '..', '..', '..'))

import utils
import visual_sudoku_solving.experiments.experiment_utils as experiment_utils

DATASET_DIR = os.path.join(THIS_DIR, "..", "..")
CLI_DIR = os.path.join(DATASET_DIR, "cli")
SYMBOLIC_MODELS_DIR = os.path.join(DATASET_DIR, "models", "symbolic")
RESULTS_BASE_DIR = os.path.join(DATASET_DIR, "results", "learning")
EXPERIMENT_HYPERPARAMETER_SEARCH_DIR = os.path.join(RESULTS_BASE_DIR, "semi_supervised_hyperparameter_search")
EXPERIMENT_RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, "semi_supervised")

CONFIG_FILE_NAME = "semi_supervised_config.json"

SPLITS = ["0", "1", "2", "3", "4"]
HYPERPARAMETER_SEARCH_TRAIN_SIZES = ["0020"]
HYPERPARAMETER_SEARCH_NUM_CLUES = ["30"]
HYPERPARAMETER_SEARCH_UNLABELED_RATIOS = ["0.00", "0.50", "0.90", "0.95"]
TRAIN_SIZES = ["0020"]
NUM_CLUES = ["30"]
UNLABELED_RATIOS = ["0.00", "0.50", "0.90", "0.95"]


def main(arguments):
    config = utils.load_json_file(os.path.join(THIS_DIR, arguments.learning_method, CONFIG_FILE_NAME))

    if arguments.run_hyperparameter_search:
        assert arguments.model not in ["mnist-9x9-baseline.json"]
        assert arguments.learning_method != "none"

        base_out_dir = os.path.join(EXPERIMENT_HYPERPARAMETER_SEARCH_DIR, config["LEARNING_METHOD"], arguments.model.split(".")[0])
        experiment_utils.run_wl_experiments_hyperparamter_search(
            arguments.model, config, HYPERPARAMETER_SEARCH_TRAIN_SIZES, HYPERPARAMETER_SEARCH_NUM_CLUES,
            HYPERPARAMETER_SEARCH_UNLABELED_RATIOS, base_out_dir
        )

    else:
        base_out_dir = os.path.join(EXPERIMENT_RESULTS_DIR, config["LEARNING_METHOD"], arguments.model.split(".")[0])
        experiment_utils.run_wl_experiments(
            arguments.model, config, TRAIN_SIZES, NUM_CLUES, UNLABELED_RATIOS, base_out_dir, time_out=6 * 60 * 60
        )


def _load_args():
    parser = argparse.ArgumentParser(description='Run semi-supervised learning experiments.')

    parser.add_argument('--learning-method', dest='learning_method',
                        action='store', type=str,
                        choices=["bilevel", "energy", "policy_gradient", "none"],
                        help='The learning method to run semi-supervised experiments for.')

    parser.add_argument('--model', dest='model',
                        action='store', type=str,
                        choices=["mnist-9x9-semi-supervised.json", "mnist-9x9-baseline.json"],
                        help='The symbolic model to run semi-supervised experiments for.')

    parser.add_argument('--run-hyperparameter-search', dest='run_hyperparameter_search',
                        action='store_true', default=False,
                        help='Run hyperparameter search for the given model.')

    arguments = parser.parse_args()

    return arguments


if __name__ == '__main__':
    _load_args()
    main(_load_args())
