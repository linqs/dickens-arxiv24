#!/usr/bin/env python3
import argparse
import os
import sys

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.join(THIS_DIR, '..', '..', '..'))

import utils
import mnist_addition.experiments.experiment_utils as experiment_utils

DATASET_DIR = os.path.join(THIS_DIR, "..", "..")
CLI_DIR = os.path.join(DATASET_DIR, "cli")
SYMBOLIC_MODELS_DIR = os.path.join(DATASET_DIR, "models", "symbolic")
RESULTS_BASE_DIR = os.path.join(DATASET_DIR, "results", "inference")
EXPERIMENT_RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, "constraint_satisfaction")

CONFIG_FILE_NAME = "constraint_satisfaction_config.json"

SPLITS = ["0", "1", "2", "3", "4"]
TRAIN_SIZES = ["00600"]
UNLABELED_RATIOS = ["0.00"]


def main(arguments):
    config = utils.load_json_file(os.path.join(THIS_DIR, CONFIG_FILE_NAME))

    base_out_dir = os.path.join(EXPERIMENT_RESULTS_DIR, arguments.model.split(".")[0])
    experiment_utils.run_inference_experiments(
        arguments.model, config, TRAIN_SIZES, UNLABELED_RATIOS, base_out_dir
    )


def _load_args():
    parser = argparse.ArgumentParser(description='Run constraint satisfaction inference experiments.')

    parser.add_argument('--model', dest='model',
                        action='store', type=str, required=True,
                        choices=["mnist-add1-constraint-satisfaction.json",
                                 "mnist-add2-constraint-satisfaction.json",
                                 "mnist-add4-constraint-satisfaction.json"],
                        help='The symbolic model to run constraint satisfaction experiments for.')

    arguments = parser.parse_args()

    return arguments


if __name__ == '__main__':
    _load_args()
    main(_load_args())
