import json
import os
import re
import subprocess
import sys
from itertools import product

import pandas as pd

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(THIS_DIR, '..', '..'))

import utils

DATASET_DIR = os.path.join(THIS_DIR, "..")
CLI_DIR = os.path.join(DATASET_DIR, "cli")
SYMBOLIC_MODELS_DIR = os.path.join(DATASET_DIR, "models", "symbolic")
LEARNING_RESULTS_BASE_DIR = os.path.join(DATASET_DIR, "results", "learning")

SPLITS = ["0", "1", "2", "3", "4"]


def run_wl_experiments_hyperparamter_search(model, config, unlabeled_ratios, base_out_dir, time_out=6 * 60 * 60):
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

        for unlabeled_ratio in unlabeled_ratios:
            for split in ["0"]:
                split_out_dir = os.path.join(method_out_dir, "split::{}/unlabeled::{}".format(split, unlabeled_ratio))
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
                                                         "runtime.learn.output.model.path": "./path_finding_learned.psl"}})

                        for key, value in neural_options.items():
                            dataset_json["predicates"]["NeuralPathfinder/2"]["options"][key] = value

                        # Set a timeout for weight learning.
                        dataset_json["options"]["weightlearning.timeout"] = time_out

                        # Set the data path.
                        set_data_path(dataset_json, split, unlabeled_ratio)

                        # Write the options the json file.
                        with open(os.path.join(CLI_DIR, "path_finding.json"), "w") as file:
                            json.dump(dataset_json, file, indent=4)

                        utils.run_experiment("path_finding", CLI_DIR, experiment_out_dir,
                                             trained_model_path=os.path.join(CLI_DIR, dataset_json["predicates"]["NeuralPathfinder/2"]["options"]["save-path"]))


def run_wl_experiments(model, config, unlabeled_ratios, base_out_dir, time_out=12 * 60 * 60):
    os.makedirs(base_out_dir, exist_ok=True)

    # Load the json file with the dataset options.
    dataset_json_path = os.path.join(SYMBOLIC_MODELS_DIR, model)

    dataset_json = None
    with open(dataset_json_path, "r") as file:
        dataset_json = json.load(file)
    original_options = dataset_json["options"]

    for learning_method in config["MODEL_WL_METHODS"][model]:
        method_out_dir = os.path.join(base_out_dir, "learning_method::{}".format(learning_method))
        os.makedirs(method_out_dir, exist_ok=True)

        for unlabeled_ratio in unlabeled_ratios:
            psl_options = config["BEST_HYPERPARAMETERS"][model][learning_method]["UNLABELED_RATIO"][unlabeled_ratio]["PSL_OPTIONS"]
            neural_options = config["BEST_HYPERPARAMETERS"][model][learning_method]["UNLABELED_RATIO"][unlabeled_ratio]["NEURAL_OPTIONS"]

            for split in SPLITS:
                split_out_dir = os.path.join(method_out_dir, "split::{}/unlabeled::{}".format(split, unlabeled_ratio))
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
                                                 "runtime.learn.output.model.path": "./path_finding_learned.psl"}})

                for key, value in neural_options.items():
                    dataset_json["predicates"]["NeuralPathfinder/2"]["options"][key] = value

                # Set a timeout for weight learning.
                dataset_json["options"]["weightlearning.timeout"] = time_out

                # Set the data path.
                set_data_path(dataset_json, split, unlabeled_ratio)

                # Write the options the json file.
                with open(os.path.join(CLI_DIR, "path_finding.json"), "w") as file:
                    json.dump(dataset_json, file, indent=4)

                utils.run_experiment("path_finding", CLI_DIR, experiment_out_dir,
                                     trained_model_path=os.path.join(CLI_DIR, dataset_json["predicates"]["NeuralPathfinder/2"]["options"]["save-path"]))


def run_inference_experiments(model, config, neural_options, train_sizes, unlabeled_ratios, base_out_dir):
    os.makedirs(base_out_dir, exist_ok=True)

    # Load the json file with the dataset options.
    dataset_json_path = os.path.join(SYMBOLIC_MODELS_DIR, model)

    dataset_json = None
    with open(dataset_json_path, "r") as file:
        dataset_json = json.load(file)
    original_options = dataset_json["options"]

    for train_size in train_sizes:
        for unlabeled_ratio in unlabeled_ratios:
            for split in SPLITS:
                experiment_out_dir = os.path.join(
                    base_out_dir, "split::{}/train-size::{}/unlabeled::{}".format(
                        split, train_size, unlabeled_ratio)
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
                set_data_path(dataset_json, split, train_size, unlabeled_ratio)

                # Write the options the json file.
                with open(os.path.join(CLI_DIR, "path_finding.json"), "w") as file:
                    json.dump(dataset_json, file, indent=4)

                utils.run_experiment("path_finding", CLI_DIR, experiment_out_dir)


def set_data_path(dataset_json, split, unlabeled_ratio, valid_infer=False):
    # sed dataset paths
    for predicate in dataset_json["predicates"]:
        if "options" in dataset_json["predicates"][predicate]:
            if "entity-data-map-path" in dataset_json["predicates"][predicate]["options"]:
                dataset_json["predicates"][predicate]["options"]["entity-data-map-path"] = \
                    re.sub(r"split::[0-9]+", "split::{}".format(split), dataset_json["predicates"][predicate]["options"]["entity-data-map-path"])
                dataset_json["predicates"][predicate]["options"]["entity-data-map-path"] = \
                    re.sub(r"unlabeled::[0-9]+\.[0-9]+", "unlabeled::{}".format(unlabeled_ratio), dataset_json["predicates"][predicate]["options"]["entity-data-map-path"])

            if "inference::entity-data-map-path" in dataset_json["predicates"][predicate]["options"]:
                dataset_json["predicates"][predicate]["options"]["inference::entity-data-map-path"] = \
                    re.sub(r"split::[0-9]+", "split::{}".format(split), dataset_json["predicates"][predicate]["options"]["inference::entity-data-map-path"])
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
                    re.sub(r"unlabeled::[0-9]+\.[0-9]+", "unlabeled::{}".format(unlabeled_ratio), dataset_json["predicates"][predicate]["options"]["save-path"])

            if "pretrain-path" in dataset_json["predicates"][predicate]["options"]:
                dataset_json["predicates"][predicate]["options"]["pretrain-path"] = \
                    re.sub(r"split::[0-9]+", "split::{}".format(split), dataset_json["predicates"][predicate]["options"]["pretrain-path"])
                dataset_json["predicates"][predicate]["options"]["pretrain-path"] = \
                    re.sub(r"unlabeled::[0-9]+\.[0-9]+", "unlabeled::{}".format(unlabeled_ratio), dataset_json["predicates"][predicate]["options"]["pretrain-path"])

        if "targets" in dataset_json["predicates"][predicate]:
            if "learn" in dataset_json["predicates"][predicate]["targets"]:
                dataset_json["predicates"][predicate]["targets"]["learn"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), target) for target in dataset_json["predicates"][predicate]["targets"]["learn"]]
                dataset_json["predicates"][predicate]["targets"]["learn"] = \
                    [re.sub(r"unlabeled::[0-9]+\.[0-9]+", "unlabeled::{}".format(unlabeled_ratio), target) for target in dataset_json["predicates"][predicate]["targets"]["learn"]]

            if "validation" in dataset_json["predicates"][predicate]["targets"]:
                dataset_json["predicates"][predicate]["targets"]["validation"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), target) for target in dataset_json["predicates"][predicate]["targets"]["validation"]]
                dataset_json["predicates"][predicate]["targets"]["validation"] = \
                    [re.sub(r"unlabeled::[0-9]+\.[0-9]+", "unlabeled::{}".format(unlabeled_ratio), target) for target in dataset_json["predicates"][predicate]["targets"]["validation"]]

            if "infer" in dataset_json["predicates"][predicate]["targets"]:
                dataset_json["predicates"][predicate]["targets"]["infer"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), target) for target in dataset_json["predicates"][predicate]["targets"]["infer"]]
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
                    [re.sub(r"unlabeled::[0-9]+\.[0-9]+", "unlabeled::{}".format(unlabeled_ratio), observation) for observation in
                     dataset_json["predicates"][predicate]["truth"]["learn"]]

            if "validation" in dataset_json["predicates"][predicate]["truth"]:
                dataset_json["predicates"][predicate]["truth"]["validation"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), observation) for observation in
                     dataset_json["predicates"][predicate]["truth"]["validation"]]
                dataset_json["predicates"][predicate]["truth"]["validation"] = \
                    [re.sub(r"unlabeled::[0-9]+\.[0-9]+", "unlabeled::{}".format(unlabeled_ratio), observation) for observation in
                     dataset_json["predicates"][predicate]["truth"]["validation"]]

            if "infer" in dataset_json["predicates"][predicate]["truth"]:
                dataset_json["predicates"][predicate]["truth"]["infer"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), observation) for observation in
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
                    [re.sub(r"unlabeled::[0-9]+\.[0-9]+", "unlabeled::{}".format(unlabeled_ratio), observation) for observation in dataset_json["predicates"][predicate]["observations"]["learn"]]

            if "validation" in dataset_json["predicates"][predicate]["observations"]:
                dataset_json["predicates"][predicate]["observations"]["validation"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), observation) for observation in dataset_json["predicates"][predicate]["observations"]["validation"]]
                dataset_json["predicates"][predicate]["observations"]["validation"] = \
                    [re.sub(r"unlabeled::[0-9]+\.[0-9]+", "unlabeled::{}".format(unlabeled_ratio), observation) for observation in dataset_json["predicates"][predicate]["observations"]["validation"]]

            if "infer" in dataset_json["predicates"][predicate]["observations"]:
                dataset_json["predicates"][predicate]["observations"]["infer"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), observation) for observation in dataset_json["predicates"][predicate]["observations"]["infer"]]
                dataset_json["predicates"][predicate]["observations"]["infer"] = \
                    [re.sub(r"unlabeled::[0-9]+\.[0-9]+", "unlabeled::{}".format(unlabeled_ratio), observation) for observation in dataset_json["predicates"][predicate]["observations"]["infer"]]

                if valid_infer:
                    dataset_json["predicates"][predicate]["observations"]["infer"] = \
                        [re.sub("test", "valid", target) for target in dataset_json["predicates"][predicate]["observations"]["infer"]]


def get_path_consistency(inferred_path, truth_path, map_costs, xy_inferred_format=True, dimension=12, threshold=0.5):
    # Compute the consistency of inferring a path with the minimum cost, i.e., the same cost as the ground truth path.
    # Load the inferred path.
    inferred_path = pd.read_csv(inferred_path, sep="\t", header=None)
    if not xy_inferred_format:
        inferred_path.columns = ["map", "cell_id", "Value"]
        inferred_path.set_index(["map", "cell_id"], inplace=True)
        xy_inferred_paths = []
        # Convert the inferred path to the format (map, X, Y, Value).
        for map_id in inferred_path.index.get_level_values("map").unique():
            cell_id = 0
            for x in range(dimension):
                for y in range(dimension):
                    xy_inferred_paths.append([map_id, x, y, inferred_path.loc[(map_id, cell_id), "Value"]])
                    cell_id += 1

        inferred_path = pd.DataFrame(xy_inferred_paths)

    inferred_path.columns = ["map", "X", "Y", "Value"]

    # Load the ground truth path.
    truth_path = pd.read_csv(truth_path, sep="\t", header=None)
    truth_path.columns = ["map", "X1", "Y1", "X2", "Y2", "Value"]

    # Load the map costs.
    map_costs = pd.read_csv(map_costs, sep="\t", header=None)
    map_costs.columns = ["map", "X", "Y", "Value"]
    map_costs.set_index(["map", "X", "Y"], inplace=True)

    # For each map, compute and compare the costs of the inferred path and the truth path.
    min_cost_consistency = 0
    valid_consistency = 0
    for map_id in inferred_path["map"].unique():
        # Get the inferred and truth paths for the current map.
        inferred_map_path = inferred_path[inferred_path["map"] == map_id]
        multi_indexed_inferred_map_path = inferred_map_path.set_index(["map", "X", "Y"])
        inferred_map_path = inferred_map_path[inferred_map_path["Value"] >= threshold]

        truth_map_path = truth_path[truth_path["map"] == map_id]
        truth_map_path = truth_map_path[truth_map_path["Value"] >= threshold]

        # Perform DFS to check if the inferred path is valid.
        inferred_path_valid = False
        x, y = 0, 0
        visited = set({})
        next_cell = [(0, 0)]
        while next_cell:
            x, y = next_cell.pop()

            if (x, y) == (11, 11):
                inferred_path_valid = True
                break

            visited.add((x, y))

            for dx, dy in product([-1, 0, 1], repeat=2):
                if ((x + dx < 0)
                        or (x + dx >= 12)
                        or (y + dy < 0)
                        or (y + dy >= 12)
                        or (dx == 0 and dy == 0)
                        or (x + dx, y + dy) in visited):
                    continue

                if multi_indexed_inferred_map_path.loc[(map_id, x + dx, y + dy), "Value"] >= threshold:
                    next_cell.append((x + dx, y + dy))

        if not inferred_path_valid:
            continue

        valid_consistency += 1

        # Compute the cost of the inferred and truth paths.
        inferred_cost = 0
        for index, row in inferred_map_path.iterrows():
            if (row["X"] == 12) or (row["Y"] == 12):
                continue

            inferred_cost += map_costs.loc[(map_id, row["X"], row["Y"]), "Value"]

        truth_cost = 0
        for index, row in truth_map_path.iterrows():
            if (row["X1"] == 12) or (row["Y1"] == 12):
                continue

            truth_cost += map_costs.loc[(map_id, row["X1"], row["Y1"]), "Value"]

        min_cost_consistency += (inferred_cost == truth_cost)

    # Normalize the consistencies by the number of maps.
    min_cost_consistency /= len(inferred_path["map"].unique())
    valid_consistency /= len(inferred_path["map"].unique())

    return valid_consistency, min_cost_consistency
