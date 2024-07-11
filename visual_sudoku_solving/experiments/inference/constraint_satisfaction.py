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
RESULTS_BASE_DIR = os.path.join(DATASET_DIR, "results", "inference")
EXPERIMENT_RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, "constraint_satisfaction")

CONFIG_FILE_NAME = "constraint_satisfaction_config.json"

SPLITS = ["0", "1", "2", "3", "4"]
TRAIN_SIZES = ["0020"]
NUM_CLUES = ["30"]
UNLABELED_RATIOS = ["0.00"]


def main(arguments):
    config = utils.load_json_file(os.path.join(THIS_DIR, CONFIG_FILE_NAME))

    neural_options = {
        "save-path": f"../data/mnist-9x9/split::0/train-size::0020/num-clues::30/unlabeled::0.00/saved-networks/supervision-trained-digit-classifier.pt",
    }

    base_out_dir = os.path.join(EXPERIMENT_RESULTS_DIR, arguments.model.split(".")[0])
    experiment_utils.run_inference_experiments(
        arguments.model, config, neural_options, TRAIN_SIZES, NUM_CLUES, UNLABELED_RATIOS, base_out_dir
    )


def _load_args():
    parser = argparse.ArgumentParser(description='Run constraint satisfaction inference experiments.')

    parser.add_argument('--model', dest='model',
                        action='store', type=str,
                        choices=["mnist-9x9-integer.json"],
                        required=True,
                        help='The symbolic model to run constraint satisfaction experiments for.')

    arguments = parser.parse_args()

    return arguments


if __name__ == '__main__':
    _load_args()
    main(_load_args())
