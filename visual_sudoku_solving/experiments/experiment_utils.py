import json
import os
import re
import sys

import pandas as pd

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(THIS_DIR, '..', '..'))

import utils

DATASET_DIR = os.path.join(THIS_DIR, "..")
CLI_DIR = os.path.join(DATASET_DIR, "cli")
SYMBOLIC_MODELS_DIR = os.path.join(DATASET_DIR, "models", "symbolic")
LEARNING_RESULTS_BASE_DIR = os.path.join(DATASET_DIR, "results", "learning")

SPLITS = ["0", "1", "2", "3", "4"]


def run_wl_experiments_hyperparamter_search(model, config, train_sizes, all_num_clues, unlabeled_ratios, base_out_dir, time_out=3 * 60 * 60):
    os.makedirs(base_out_dir, exist_ok=True)

    # Load the json file with the dataset options.
    dataset_json_path = os.path.join(SYMBOLIC_MODELS_DIR, model)

    dataset_json = None
    with open(dataset_json_path, "r") as file:
        dataset_json = json.load(file)
    original_options = dataset_json["options"]

    standard_experiment_option_ranges = {**config["STANDARD_OPTION_RANGES"]}

    for learning_method in config["MODEL_WL_METHODS"][model]:
        method_out_dir = os.path.join(base_out_dir, "learning_method::{}".format(learning_method))
        os.makedirs(method_out_dir, exist_ok=True)

        for train_size in train_sizes:
            for num_clues in all_num_clues:
                for unlabeled_ratio in unlabeled_ratios:
                    for split in ["0"]:
                        split_out_dir = os.path.join(method_out_dir, "split::{}/train-size::{}/num-clues::{}/unlabeled::{}".format(
                            split, train_size, num_clues, unlabeled_ratio))
                        os.makedirs(split_out_dir, exist_ok=True)

                        # Iterate over every combination options values.
                        method_options_dict = {**standard_experiment_option_ranges,
                                               **config["WL_METHOD_OPTION_RANGES"][learning_method]}
                        for psl_options in utils.enumerate_hyperparameters(method_options_dict):
                            for neural_options in utils.enumerate_hyperparameters(config["NEURAL_NETWORK_OPTION_RANGES"]):
                                experiment_out_dir = split_out_dir
                                for key, value in sorted(psl_options.items()):
                                    experiment_out_dir = os.path.join(experiment_out_dir, "{}::{}".format(key, value))

                                for key, value in sorted(neural_options.items()):
                                    experiment_out_dir = os.path.join(experiment_out_dir, "{}::{}".format(key, value))

                                os.makedirs(experiment_out_dir, exist_ok=True)

                                if os.path.exists(os.path.join(experiment_out_dir, "out.txt")):
                                    print("Skipping experiment: {}.".format(experiment_out_dir))
                                    continue

                                dataset_json.update({"options": {**original_options,
                                                                 **config["STANDARD_EXPERIMENT_OPTIONS"],
                                                                 **psl_options,
                                                                 "runtime.learn.output.model.path": "./visual_sudoku_learned.psl"}})

                                for key, value in neural_options.items():
                                    dataset_json["predicates"]["NeuralClassifier/4"]["options"][key] = value

                                # Set a timeout for weight learning.
                                dataset_json["options"]["weightlearning.timeout"] = time_out

                                # Set the data path.
                                set_data_path(dataset_json, split, train_size, num_clues, unlabeled_ratio, valid_infer=True)

                                # Write the options the json file.
                                with open(os.path.join(CLI_DIR, "visual_sudoku.json"), "w") as file:
                                    json.dump(dataset_json, file, indent=4)

                                utils.run_experiment("visual_sudoku", CLI_DIR, experiment_out_dir,
                                                     trained_model_path=os.path.join(CLI_DIR, dataset_json["predicates"]["NeuralClassifier/4"]["options"]["save-path"]))


def run_wl_experiments(model, config, train_sizes, all_num_clues, unlabeled_ratios, base_out_dir, time_out=3 * 60 * 60):
    os.makedirs(base_out_dir, exist_ok=True)

    # Load the json file with the dataset options.
    dataset_json_path = os.path.join(SYMBOLIC_MODELS_DIR, model)

    dataset_json = None
    with open(dataset_json_path, "r") as file:
        dataset_json = json.load(file)
    original_options = dataset_json["options"]

    for method in config["MODEL_WL_METHODS"][model]:
        method_out_dir = os.path.join(base_out_dir, method)
        os.makedirs(method_out_dir, exist_ok=True)

        for train_size in train_sizes:
            for num_clues in all_num_clues:
                for unlabeled_ratio in unlabeled_ratios:
                    psl_options = config["BEST_HYPERPARAMETERS"][model][method]["TRAIN_SIZE"][train_size]["NUM_CLUES"][num_clues]["UNLABELED_RATIO"][unlabeled_ratio]["PSL_OPTIONS"]
                    neural_options = config["BEST_HYPERPARAMETERS"][model][method]["TRAIN_SIZE"][train_size]["NUM_CLUES"][num_clues]["UNLABELED_RATIO"][unlabeled_ratio]["NEURAL_OPTIONS"]

                    for split in SPLITS:
                        split_out_dir = os.path.join(method_out_dir, "split::{}/train-size::{}/num-clues::{}/unlabeled::{}".format(split, train_size, num_clues, unlabeled_ratio))
                        os.makedirs(split_out_dir, exist_ok=True)

                        experiment_out_dir = split_out_dir
                        for key, value in sorted(psl_options.items()):
                            experiment_out_dir = os.path.join(experiment_out_dir, "{}::{}".format(key, value))

                        for key, value in sorted(neural_options.items()):
                            experiment_out_dir = os.path.join(experiment_out_dir, "{}::{}".format(key, value))

                        os.makedirs(experiment_out_dir, exist_ok=True)

                        if os.path.exists(os.path.join(experiment_out_dir, "out.txt")):
                            print("Skipping experiment: {}.".format(experiment_out_dir))
                            continue

                        dataset_json.update({"options": {**original_options,
                                                         **config["STANDARD_EXPERIMENT_OPTIONS"],
                                                         **psl_options,
                                                         "runtime.learn.output.model.path": "./visual_sudoku_learned.psl"}})

                        for key, value in neural_options.items():
                            dataset_json["predicates"]["NeuralClassifier/4"]["options"][key] = value

                        # Set a timeout for weight learning.
                        dataset_json["options"]["weightlearning.timeout"] = time_out

                        # Set the data path.
                        set_data_path(dataset_json, split, train_size, num_clues, unlabeled_ratio)

                        # Write the options the json file.
                        with open(os.path.join(CLI_DIR, "visual_sudoku.json"), "w") as file:
                            json.dump(dataset_json, file, indent=4)

                        utils.run_experiment("visual_sudoku", CLI_DIR, experiment_out_dir,
                                             trained_model_path=os.path.join(CLI_DIR, dataset_json["predicates"]["NeuralClassifier/4"]["options"]["save-path"]))


def run_inference_experiments(model, config, neural_options, train_sizes, all_num_clues, unlabeled_ratios, base_out_dir):
    os.makedirs(base_out_dir, exist_ok=True)

    # Load the json file with the dataset options.
    dataset_json_path = os.path.join(SYMBOLIC_MODELS_DIR, model)

    dataset_json = None
    with open(dataset_json_path, "r") as file:
        dataset_json = json.load(file)
    original_options = dataset_json["options"]

    for train_size in train_sizes:
        for num_clues in all_num_clues:
            for unlabeled_ratio in unlabeled_ratios:
                for split in SPLITS:
                    experiment_out_dir = os.path.join(
                        base_out_dir, "split::{}/train-size::{}/num-clues::{}/unlabeled::{}".format(
                            split, train_size, num_clues, unlabeled_ratio)
                    )
                    os.makedirs(experiment_out_dir, exist_ok=True)

                    if os.path.exists(os.path.join(experiment_out_dir, "out.txt")):
                        print("Skipping experiment: {}.".format(experiment_out_dir))
                        continue

                    dataset_json.update({"options": {**original_options,
                                                     **config["STANDARD_EXPERIMENT_OPTIONS"]}})

                    for key, value in neural_options.items():
                        dataset_json["predicates"]["NeuralClassifier/4"]["options"][key] = value

                    # Set the data path.
                    set_data_path(dataset_json, split, train_size, num_clues, unlabeled_ratio)

                    # Write the options the json file.
                    with open(os.path.join(CLI_DIR, "visual_sudoku.json"), "w") as file:
                        json.dump(dataset_json, file, indent=4)

                    utils.run_experiment("visual_sudoku", CLI_DIR, experiment_out_dir)


def set_data_path(dataset_json, split, train_size, num_clues, unlabeled_ratio, valid_infer=False):
    # sed dataset paths
    for predicate in dataset_json["predicates"]:
        if "options" in dataset_json["predicates"][predicate]:
            if "entity-data-map-path" in dataset_json["predicates"][predicate]["options"]:
                dataset_json["predicates"][predicate]["options"]["entity-data-map-path"] = \
                    re.sub(r"split::[0-9]+", "split::{}".format(split), dataset_json["predicates"][predicate]["options"]["entity-data-map-path"])
                dataset_json["predicates"][predicate]["options"]["entity-data-map-path"] = \
                    re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), dataset_json["predicates"][predicate]["options"]["entity-data-map-path"])
                dataset_json["predicates"][predicate]["options"]["entity-data-map-path"] = \
                    re.sub(r"num-clues::[0-9]+", "num-clues::{}".format(num_clues), dataset_json["predicates"][predicate]["options"]["entity-data-map-path"])
                dataset_json["predicates"][predicate]["options"]["entity-data-map-path"] = \
                    re.sub(r"unlabeled::[0-9]+\.[0-9]+", "unlabeled::{}".format(unlabeled_ratio), dataset_json["predicates"][predicate]["options"]["entity-data-map-path"])

            if "inference::entity-data-map-path" in dataset_json["predicates"][predicate]["options"]:
                dataset_json["predicates"][predicate]["options"]["inference::entity-data-map-path"] = \
                    re.sub(r"split::[0-9]+", "split::{}".format(split), dataset_json["predicates"][predicate]["options"]["inference::entity-data-map-path"])
                dataset_json["predicates"][predicate]["options"]["inference::entity-data-map-path"] = \
                    re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), dataset_json["predicates"][predicate]["options"]["inference::entity-data-map-path"])
                dataset_json["predicates"][predicate]["options"]["inference::entity-data-map-path"] = \
                    re.sub(r"num-clues::[0-9]+", "num-clues::{}".format(num_clues), dataset_json["predicates"][predicate]["options"]["inference::entity-data-map-path"])
                dataset_json["predicates"][predicate]["options"]["inference::entity-data-map-path"] = \
                    re.sub(r"unlabeled::[0-9]+\.[0-9]+", "unlabeled::{}".format(unlabeled_ratio), dataset_json["predicates"][predicate]["options"]["inference::entity-data-map-path"])

                if valid_infer:
                    dataset_json["predicates"][predicate]["options"]["inference::entity-data-map-path"] = \
                        re.sub("entity-data-map-test", "entity-data-map-valid",
                               dataset_json["predicates"][predicate]["options"]["inference::entity-data-map-path"])

            if "save-path" in dataset_json["predicates"][predicate]["options"]:
                dataset_json["predicates"][predicate]["options"]["save-path"] = \
                    re.sub(r"split::[0-9]+", "split::{}".format(split), dataset_json["predicates"][predicate]["options"]["save-path"])
                dataset_json["predicates"][predicate]["options"]["save-path"] = \
                    re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), dataset_json["predicates"][predicate]["options"]["save-path"])
                dataset_json["predicates"][predicate]["options"]["save-path"] = \
                    re.sub(r"num-clues::[0-9]+", "num-clues::{}".format(num_clues), dataset_json["predicates"][predicate]["options"]["save-path"])
                dataset_json["predicates"][predicate]["options"]["save-path"] = \
                    re.sub(r"unlabeled::[0-9]+\.[0-9]+", "unlabeled::{}".format(unlabeled_ratio), dataset_json["predicates"][predicate]["options"]["save-path"])

            if "pretrain-path" in dataset_json["predicates"][predicate]["options"]:
                dataset_json["predicates"][predicate]["options"]["pretrain-path"] = \
                    re.sub(r"split::[0-9]+", "split::{}".format(split), dataset_json["predicates"][predicate]["options"]["pretrain-path"])
                dataset_json["predicates"][predicate]["options"]["pretrain-path"] = \
                    re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), dataset_json["predicates"][predicate]["options"]["pretrain-path"])
                dataset_json["predicates"][predicate]["options"]["pretrain-path"] = \
                    re.sub(r"num-clues::[0-9]+", "num-clues::{}".format(num_clues), dataset_json["predicates"][predicate]["options"]["pretrain-path"])
                dataset_json["predicates"][predicate]["options"]["pretrain-path"] = \
                    re.sub(r"unlabeled::[0-9]+\.[0-9]+", "unlabeled::{}".format(unlabeled_ratio), dataset_json["predicates"][predicate]["options"]["pretrain-path"])

        if "targets" in dataset_json["predicates"][predicate]:
            if "learn" in dataset_json["predicates"][predicate]["targets"]:
                dataset_json["predicates"][predicate]["targets"]["learn"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), target) for target in dataset_json["predicates"][predicate]["targets"]["learn"]]
                dataset_json["predicates"][predicate]["targets"]["learn"] = \
                    [re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), target) for target in dataset_json["predicates"][predicate]["targets"]["learn"]]
                dataset_json["predicates"][predicate]["targets"]["learn"] = \
                    [re.sub(r"num-clues::[0-9]+", "num-clues::{}".format(num_clues), target) for target in dataset_json["predicates"][predicate]["targets"]["learn"]]
                dataset_json["predicates"][predicate]["targets"]["learn"] = \
                    [re.sub(r"unlabeled::[0-9]+\.[0-9]+", "unlabeled::{}".format(unlabeled_ratio), target) for target in dataset_json["predicates"][predicate]["targets"]["learn"]]

            if "validation" in dataset_json["predicates"][predicate]["targets"]:
                dataset_json["predicates"][predicate]["targets"]["validation"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), target) for target in dataset_json["predicates"][predicate]["targets"]["validation"]]
                dataset_json["predicates"][predicate]["targets"]["validation"] = \
                    [re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), target) for target in dataset_json["predicates"][predicate]["targets"]["validation"]]
                dataset_json["predicates"][predicate]["targets"]["validation"] = \
                    [re.sub(r"num-clues::[0-9]+", "num-clues::{}".format(num_clues), target) for target in dataset_json["predicates"][predicate]["targets"]["validation"]]
                dataset_json["predicates"][predicate]["targets"]["validation"] = \
                    [re.sub(r"unlabeled::[0-9]+\.[0-9]+", "unlabeled::{}".format(unlabeled_ratio), target) for target in dataset_json["predicates"][predicate]["targets"]["validation"]]

            if "infer" in dataset_json["predicates"][predicate]["targets"]:
                dataset_json["predicates"][predicate]["targets"]["infer"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), target) for target in dataset_json["predicates"][predicate]["targets"]["infer"]]
                dataset_json["predicates"][predicate]["targets"]["infer"] = \
                    [re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), target) for target in dataset_json["predicates"][predicate]["targets"]["infer"]]
                dataset_json["predicates"][predicate]["targets"]["infer"] = \
                    [re.sub(r"num-clues::[0-9]+", "num-clues::{}".format(num_clues), target) for target in dataset_json["predicates"][predicate]["targets"]["infer"]]
                dataset_json["predicates"][predicate]["targets"]["infer"] = \
                    [re.sub(r"unlabeled::[0-9]+\.[0-9]+", "unlabeled::{}".format(unlabeled_ratio), target) for target in dataset_json["predicates"][predicate]["targets"]["infer"]]

                if valid_infer:
                    dataset_json["predicates"][predicate]["targets"]["infer"] = \
                        [re.sub("test", "valid", target) for target in dataset_json["predicates"][predicate]["targets"]["infer"]]

        if "truth" in dataset_json["predicates"][predicate]:
            if "learn" in dataset_json["predicates"][predicate]["truth"]:
                dataset_json["predicates"][predicate]["truth"]["learn"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), observation) for observation in
                     dataset_json["predicates"][predicate]["truth"]["learn"]]
                dataset_json["predicates"][predicate]["truth"]["learn"] = \
                    [re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), observation) for observation in
                     dataset_json["predicates"][predicate]["truth"]["learn"]]
                dataset_json["predicates"][predicate]["truth"]["learn"] = \
                    [re.sub(r"num-clues::[0-9]+", "num-clues::{}".format(num_clues), observation) for observation in
                     dataset_json["predicates"][predicate]["truth"]["learn"]]
                dataset_json["predicates"][predicate]["truth"]["learn"] = \
                    [re.sub(r"unlabeled::[0-9]+\.[0-9]+", "unlabeled::{}".format(unlabeled_ratio), observation) for observation in
                     dataset_json["predicates"][predicate]["truth"]["learn"]]

            if "validation" in dataset_json["predicates"][predicate]["truth"]:
                dataset_json["predicates"][predicate]["truth"]["validation"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), observation) for observation in
                     dataset_json["predicates"][predicate]["truth"]["validation"]]
                dataset_json["predicates"][predicate]["truth"]["validation"] = \
                    [re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), observation) for observation in
                     dataset_json["predicates"][predicate]["truth"]["validation"]]
                dataset_json["predicates"][predicate]["truth"]["validation"] = \
                    [re.sub(r"num-clues::[0-9]+", "num-clues::{}".format(num_clues), observation) for observation in
                     dataset_json["predicates"][predicate]["truth"]["validation"]]
                dataset_json["predicates"][predicate]["truth"]["validation"] = \
                    [re.sub(r"unlabeled::[0-9]+\.[0-9]+", "unlabeled::{}".format(unlabeled_ratio), observation) for observation in
                     dataset_json["predicates"][predicate]["truth"]["validation"]]

            if "infer" in dataset_json["predicates"][predicate]["truth"]:
                dataset_json["predicates"][predicate]["truth"]["infer"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), observation) for observation in
                     dataset_json["predicates"][predicate]["truth"]["infer"]]
                dataset_json["predicates"][predicate]["truth"]["infer"] = \
                    [re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), observation) for observation in
                     dataset_json["predicates"][predicate]["truth"]["infer"]]
                dataset_json["predicates"][predicate]["truth"]["infer"] = \
                    [re.sub(r"num-clues::[0-9]+", "num-clues::{}".format(num_clues), observation) for observation in
                     dataset_json["predicates"][predicate]["truth"]["infer"]]
                dataset_json["predicates"][predicate]["truth"]["infer"] = \
                    [re.sub(r"unlabeled::[0-9]+\.[0-9]+", "unlabeled::{}".format(unlabeled_ratio), observation) for observation in
                     dataset_json["predicates"][predicate]["truth"]["infer"]]

                if valid_infer:
                    dataset_json["predicates"][predicate]["truth"]["infer"] = \
                        [re.sub("test", "valid", target) for target in dataset_json["predicates"][predicate]["truth"]["infer"]]

        if "observations" in dataset_json["predicates"][predicate]:
            if "learn" in dataset_json["predicates"][predicate]["observations"]:
                dataset_json["predicates"][predicate]["observations"]["learn"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), observation) for observation in dataset_json["predicates"][predicate]["observations"]["learn"]]
                dataset_json["predicates"][predicate]["observations"]["learn"] = \
                    [re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), observation) for observation in dataset_json["predicates"][predicate]["observations"]["learn"]]
                dataset_json["predicates"][predicate]["observations"]["learn"] = \
                    [re.sub(r"num-clues::[0-9]+", "num-clues::{}".format(num_clues), observation) for observation in dataset_json["predicates"][predicate]["observations"]["learn"]]
                dataset_json["predicates"][predicate]["observations"]["learn"] = \
                    [re.sub(r"unlabeled::[0-9]+\.[0-9]+", "unlabeled::{}".format(unlabeled_ratio), observation) for observation in dataset_json["predicates"][predicate]["observations"]["learn"]]

            if "validation" in dataset_json["predicates"][predicate]["observations"]:
                dataset_json["predicates"][predicate]["observations"]["validation"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), observation) for observation in dataset_json["predicates"][predicate]["observations"]["validation"]]
                dataset_json["predicates"][predicate]["observations"]["validation"] = \
                    [re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), observation) for observation in dataset_json["predicates"][predicate]["observations"]["validation"]]
                dataset_json["predicates"][predicate]["observations"]["validation"] = \
                    [re.sub(r"num-clues::[0-9]+", "num-clues::{}".format(num_clues), observation) for observation in dataset_json["predicates"][predicate]["observations"]["validation"]]
                dataset_json["predicates"][predicate]["observations"]["validation"] = \
                    [re.sub(r"unlabeled::[0-9]+\.[0-9]+", "unlabeled::{}".format(unlabeled_ratio), observation) for observation in dataset_json["predicates"][predicate]["observations"]["validation"]]

            if "infer" in dataset_json["predicates"][predicate]["observations"]:
                dataset_json["predicates"][predicate]["observations"]["infer"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), observation) for observation in dataset_json["predicates"][predicate]["observations"]["infer"]]
                dataset_json["predicates"][predicate]["observations"]["infer"] = \
                    [re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), observation) for observation in dataset_json["predicates"][predicate]["observations"]["infer"]]
                dataset_json["predicates"][predicate]["observations"]["infer"] = \
                    [re.sub(r"num-clues::[0-9]+", "num-clues::{}".format(num_clues), observation) for observation in dataset_json["predicates"][predicate]["observations"]["infer"]]
                dataset_json["predicates"][predicate]["observations"]["infer"] = \
                    [re.sub(r"unlabeled::[0-9]+\.[0-9]+", "unlabeled::{}".format(unlabeled_ratio), observation) for observation in dataset_json["predicates"][predicate]["observations"]["infer"]]

                if valid_infer:
                    dataset_json["predicates"][predicate]["observations"]["infer"] = \
                        [re.sub("test", "valid", target) for target in dataset_json["predicates"][predicate]["observations"]["infer"]]


def get_puzzle_consistency(inferred_predicates_path):
    inferred_predicates = pd.read_csv(inferred_predicates_path, sep="\t", header=None)
    inferred_predicates.columns = ["PuzzleId", "X", "Y", "Num", "Value"]
    inferred_predicates.set_index(["PuzzleId", "X", "Y", "Num"], inplace=True)

    # Set predictions for each cell to the Num with the maximum value.
    inferred_predicates = inferred_predicates["Value"].unstack().idxmax(axis=1)

    # Count the number of valid puzzles.
    valid_puzzles = 0
    for puzzle_id, puzzle in inferred_predicates.groupby("PuzzleId"):
        invalid_puzzle = False
        # Check if there are unique digits in each row.
        for row_id in puzzle.index.get_level_values("X"):
            if len(puzzle.loc[puzzle_id, row_id]) != len(puzzle.loc[puzzle_id, row_id].unique()):
                invalid_puzzle = True
                break

        # Check if there are unique digits in each column.
        for col_id in puzzle.index.get_level_values("Y"):
            if len(puzzle.loc[puzzle_id, :, col_id]) != len(puzzle.loc[puzzle_id, :, col_id].unique()):
                invalid_puzzle = True
                break

        # Check if there are unique digits in each 3x3 square.
        for row_id in range(0, 9, 3):
            for col_id in range(0, 9, 3):
                square = puzzle.loc[puzzle_id, row_id: row_id+2, col_id: col_id+2].values.flatten()
                if len(square) != len(set(square)):
                    invalid_puzzle = True
                    break

        if not invalid_puzzle:
            valid_puzzles += 1

    return valid_puzzles / len(inferred_predicates.groupby("PuzzleId"))
