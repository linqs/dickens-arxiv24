#!/usr/bin/env python3
import argparse
import os
import sys

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.join(THIS_DIR, '..', '..', '..'))

import utils
import path_finding.experiments.experiment_utils as experiment_utils

DATASET_DIR = os.path.join(THIS_DIR, "..", "..")
CLI_DIR = os.path.join(DATASET_DIR, "cli")
SYMBOLIC_MODELS_DIR = os.path.join(DATASET_DIR, "models", "symbolic")
RESULTS_BASE_DIR = os.path.join(DATASET_DIR, "results", "learning")
EXPERIMENT_HYPERPARAMETER_SEARCH_DIR = os.path.join(RESULTS_BASE_DIR, "semi_supervised_hyperparameter_search")
EXPERIMENT_RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, "semi_supervised")

CONFIG_FILE_NAME = "semi_supervised_config.json"

UNLABELED_RATIOS = ["0.90", "0.50", "0.00"]

def main(arguments):
    config = utils.load_json_file(os.path.join(THIS_DIR, arguments.learning_method, CONFIG_FILE_NAME))

    if arguments.run_hyperparameter_search:
        assert arguments.model not in ["warcraft-map-baseline.json", "warcraft-map-constraint-satisfaction.json"]
        assert arguments.learning_method not in ["none", "constraint_satisfaction"]

        base_out_dir = os.path.join(EXPERIMENT_HYPERPARAMETER_SEARCH_DIR, config["LEARNING_METHOD"], arguments.model.split(".")[0])
        experiment_utils.run_wl_experiments_hyperparamter_search(
            arguments.model, config, UNLABELED_RATIOS, base_out_dir
        )

    base_out_dir = os.path.join(EXPERIMENT_RESULTS_DIR, config["LEARNING_METHOD"], arguments.model.split(".")[0])
    experiment_utils.run_wl_experiments(arguments.model, config, UNLABELED_RATIOS, base_out_dir, time_out=12 * 60 * 60)


def _load_args():
    parser = argparse.ArgumentParser(description='Run semi-supervised learning experiments.')

    parser.add_argument('--learning-method', dest='learning_method',
                        action='store', type=str,
                        choices=["bilevel", "energy", "policy_gradient", "none", "constraint_satisfaction"],
                        required=True,
                        help='The learning method to run semi-supervised experiments for.')

    parser.add_argument('--model', dest='model',
                        action='store', type=str,
                        choices=["warcraft-map.json", "warcraft-map-baseline.json", "warcraft-map-constraint-satisfaction.json"],
                        required=True,
                        help='The symbolic model to run semi-supervised experiments for.')

    parser.add_argument('--run-hyperparameter-search', dest='run_hyperparameter_search',
                        action='store_true', default=False,
                        help='Run hyperparameter search for the given model.')

    arguments = parser.parse_args()

    return arguments


if __name__ == '__main__':
    _load_args()
    main(_load_args())
