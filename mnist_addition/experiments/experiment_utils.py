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


def run_wl_experiments_hyperparamter_search(model, config, train_sizes, unlabeled_ratios, base_out_dir, time_out=3 * 60 * 60):
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
            for unlabeled_ratio in unlabeled_ratios:
                for split in ["0"]:
                    split_out_dir = os.path.join(method_out_dir, "split::{}/train-size::{}/unlabeled::{}".format(split, train_size, unlabeled_ratio))
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
                                                             "runtime.learn.output.model.path": "./mnist-addition_learned.psl"}})

                            for key, value in neural_options.items():
                                dataset_json["predicates"]["NeuralClassifier/2"]["options"][key] = value

                            # Set a timeout for weight learning for 3 hours.
                            dataset_json["options"]["weightlearning.timeout"] = time_out

                            # Set the data path.
                            set_data_path(dataset_json, split, train_size, unlabeled_ratio, valid_infer=True)

                            # Write the options the json file.
                            with open(os.path.join(CLI_DIR, "mnist-addition.json"), "w") as file:
                                json.dump(dataset_json, file, indent=4)

                            utils.run_experiment("mnist-addition", CLI_DIR, experiment_out_dir,
                                                 trained_model_path=os.path.join(CLI_DIR, dataset_json["predicates"]["NeuralClassifier/2"]["options"]["save-path"]))


def run_wl_experiments(model, config, train_sizes, unlabeled_ratios, base_out_dir, time_out=3 * 60 * 60):
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
            for unlabeled_ratio in unlabeled_ratios:
                psl_options = config["BEST_HYPERPARAMETERS"][model][method]["TRAIN_SIZE"][train_size]["UNLABELED_RATIO"][unlabeled_ratio]["PSL_OPTIONS"]
                neural_options = config["BEST_HYPERPARAMETERS"][model][method]["TRAIN_SIZE"][train_size]["UNLABELED_RATIO"][unlabeled_ratio]["NEURAL_OPTIONS"]

                for split in SPLITS:
                    split_out_dir = os.path.join(method_out_dir, "split::{}/train-size::{}/unlabeled::{}".format(split, train_size, unlabeled_ratio))
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
                                                     "runtime.learn.output.model.path": "./mnist-addition_learned.psl"}})

                    for key, value in neural_options.items():
                        dataset_json["predicates"]["NeuralClassifier/2"]["options"][key] = value

                    # Set a timeout for weight learning.
                    dataset_json["options"]["weightlearning.timeout"] = time_out

                    # Set the data path.
                    set_data_path(dataset_json, split, train_size, unlabeled_ratio)

                    # Write the options the json file.
                    with open(os.path.join(CLI_DIR, "mnist-addition.json"), "w") as file:
                        json.dump(dataset_json, file, indent=4)

                    utils.run_experiment("mnist-addition", CLI_DIR, experiment_out_dir,
                                         trained_model_path=os.path.join(CLI_DIR, dataset_json["predicates"]["NeuralClassifier/2"]["options"]["save-path"]))


def run_inference_experiments(model, config, train_sizes, unlabeled_ratios, base_out_dir):
    os.makedirs(base_out_dir, exist_ok=True)

    # Load the json file with the dataset options.
    model_json_path = os.path.join(SYMBOLIC_MODELS_DIR, model)

    model_json = None
    with open(model_json_path, "r") as file:
        model_json = json.load(file)
    original_options = model_json["options"]

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

                model_json.update({"options": {**original_options,
                                                 **config["STANDARD_EXPERIMENT_OPTIONS"]}})

                # Set the data path.
                set_data_path(model_json, split, train_size, unlabeled_ratio)

                # Write the options the json file.
                with open(os.path.join(CLI_DIR, "mnist-addition.json"), "w") as file:
                    json.dump(model_json, file, indent=4)

                utils.run_experiment("mnist-addition", CLI_DIR, experiment_out_dir)


def set_data_path(dataset_json, split, train_size, unlabeled_ratio, valid_infer=False):
    # sed dataset paths
    for predicate in dataset_json["predicates"]:
        if "options" in dataset_json["predicates"][predicate]:
            if "entity-data-map-path" in dataset_json["predicates"][predicate]["options"]:
                dataset_json["predicates"][predicate]["options"]["entity-data-map-path"] = \
                    re.sub(r"split::[0-9]+", "split::{}".format(split), dataset_json["predicates"][predicate]["options"]["entity-data-map-path"])
                dataset_json["predicates"][predicate]["options"]["entity-data-map-path"] = \
                    re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), dataset_json["predicates"][predicate]["options"]["entity-data-map-path"])
                dataset_json["predicates"][predicate]["options"]["entity-data-map-path"] = \
                    re.sub(r"unlabeled::[0-9]+\.[0-9]+", "unlabeled::{}".format(unlabeled_ratio), dataset_json["predicates"][predicate]["options"]["entity-data-map-path"])

            if "inference::entity-data-map-path" in dataset_json["predicates"][predicate]["options"]:
                dataset_json["predicates"][predicate]["options"]["inference::entity-data-map-path"] = \
                    re.sub(r"split::[0-9]+", "split::{}".format(split), dataset_json["predicates"][predicate]["options"]["inference::entity-data-map-path"])
                dataset_json["predicates"][predicate]["options"]["inference::entity-data-map-path"] = \
                    re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), dataset_json["predicates"][predicate]["options"]["inference::entity-data-map-path"])
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
                    re.sub(r"unlabeled::[0-9]+\.[0-9]+", "unlabeled::{}".format(unlabeled_ratio), dataset_json["predicates"][predicate]["options"]["save-path"])

            if "pretrain-path" in dataset_json["predicates"][predicate]["options"]:
                dataset_json["predicates"][predicate]["options"]["pretrain-path"] = \
                    re.sub(r"split::[0-9]+", "split::{}".format(split), dataset_json["predicates"][predicate]["options"]["pretrain-path"])
                dataset_json["predicates"][predicate]["options"]["pretrain-path"] = \
                    re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), dataset_json["predicates"][predicate]["options"]["pretrain-path"])
                dataset_json["predicates"][predicate]["options"]["pretrain-path"] = \
                    re.sub(r"unlabeled::[0-9]+\.[0-9]+", "unlabeled::{}".format(unlabeled_ratio), dataset_json["predicates"][predicate]["options"]["pretrain-path"])

        if "targets" in dataset_json["predicates"][predicate]:
            if "learn" in dataset_json["predicates"][predicate]["targets"]:
                dataset_json["predicates"][predicate]["targets"]["learn"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), target) for target in dataset_json["predicates"][predicate]["targets"]["learn"]]
                dataset_json["predicates"][predicate]["targets"]["learn"] = \
                    [re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), target) for target in dataset_json["predicates"][predicate]["targets"]["learn"]]
                dataset_json["predicates"][predicate]["targets"]["learn"] = \
                    [re.sub(r"unlabeled::[0-9]+\.[0-9]+", "unlabeled::{}".format(unlabeled_ratio), target) for target in dataset_json["predicates"][predicate]["targets"]["learn"]]

            if "validation" in dataset_json["predicates"][predicate]["targets"]:
                dataset_json["predicates"][predicate]["targets"]["validation"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), target) for target in dataset_json["predicates"][predicate]["targets"]["validation"]]
                dataset_json["predicates"][predicate]["targets"]["validation"] = \
                    [re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), target) for target in dataset_json["predicates"][predicate]["targets"]["validation"]]
                dataset_json["predicates"][predicate]["targets"]["validation"] = \
                    [re.sub(r"unlabeled::[0-9]+\.[0-9]+", "unlabeled::{}".format(unlabeled_ratio), target) for target in dataset_json["predicates"][predicate]["targets"]["validation"]]

            if "infer" in dataset_json["predicates"][predicate]["targets"]:
                dataset_json["predicates"][predicate]["targets"]["infer"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), target) for target in dataset_json["predicates"][predicate]["targets"]["infer"]]
                dataset_json["predicates"][predicate]["targets"]["infer"] = \
                    [re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), target) for target in dataset_json["predicates"][predicate]["targets"]["infer"]]
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
                    [re.sub(r"unlabeled::[0-9]+\.[0-9]+", "unlabeled::{}".format(unlabeled_ratio), observation) for observation in dataset_json["predicates"][predicate]["observations"]["learn"]]

            if "validation" in dataset_json["predicates"][predicate]["observations"]:
                dataset_json["predicates"][predicate]["observations"]["validation"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), observation) for observation in dataset_json["predicates"][predicate]["observations"]["validation"]]
                dataset_json["predicates"][predicate]["observations"]["validation"] = \
                    [re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), observation) for observation in dataset_json["predicates"][predicate]["observations"]["validation"]]
                dataset_json["predicates"][predicate]["observations"]["validation"] = \
                    [re.sub(r"unlabeled::[0-9]+\.[0-9]+", "unlabeled::{}".format(unlabeled_ratio), observation) for observation in dataset_json["predicates"][predicate]["observations"]["validation"]]

            if "infer" in dataset_json["predicates"][predicate]["observations"]:
                dataset_json["predicates"][predicate]["observations"]["infer"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), observation) for observation in dataset_json["predicates"][predicate]["observations"]["infer"]]
                dataset_json["predicates"][predicate]["observations"]["infer"] = \
                    [re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), observation) for observation in dataset_json["predicates"][predicate]["observations"]["infer"]]
                dataset_json["predicates"][predicate]["observations"]["infer"] = \
                    [re.sub(r"unlabeled::[0-9]+\.[0-9]+", "unlabeled::{}".format(unlabeled_ratio), observation) for observation in dataset_json["predicates"][predicate]["observations"]["infer"]]

                if valid_infer:
                    dataset_json["predicates"][predicate]["observations"]["infer"] = \
                        [re.sub("test", "valid", target) for target in dataset_json["predicates"][predicate]["observations"]["infer"]]


def get_dataset_from_model(model):
    if "mnist-add1" in model:
        return "mnist-1"
    elif "mnist-add2" in model:
        return "mnist-2"
    elif "mnist-add4" in model:
        return "mnist-4"
    else:
        raise ValueError(f"Unknown model: {model}")


def get_addition_consistency(inferred_predicates_path, digit_sum_path):
    inferred_digits = pd.read_csv(inferred_predicates_path, sep="\t", header=None)
    inferred_digits.columns = ["ImageId", "Num", "Value"]
    inferred_digits.set_index(["ImageId", "Num"], inplace=True)

    # Set predictions for each cell to the Num with the maximum value.
    inferred_digits = inferred_digits["Value"].unstack().idxmax(axis=1)

    # Load the digit sums.
    digit_sums = pd.read_csv(digit_sum_path, sep="\t", header=None)
    digit_sums.columns = ["ImageId{}".format(i) for i in range(1, digit_sums.shape[1] - 1)] + ["Sum", "Value"]
    digit_sums.set_index(["ImageId{}".format(i) for i in range(1, digit_sums.shape[1] - 1)] + ["Sum"], inplace=True)

    # Set sum for each cell to the Num with the maximum value.
    digit_sums = digit_sums["Value"].unstack().idxmax(axis=1)

    # Reset index for digit_sums.
    digit_sums = digit_sums.reset_index()

    # Count the number of valid additions.
    valid_additions = 0
    for row in digit_sums.iterrows():
        series = row[1]
        num1_image_ids = series.iloc[: (series.shape[0] - 1) // 2].values
        num2_image_ids = series.iloc[(series.shape[0] - 1) // 2: -1].values

        # Reverse the order of the image ids.
        num1_image_ids = num1_image_ids[::-1]
        num2_image_ids = num2_image_ids[::-1]

        place = 0
        predicted_sum = 0
        for num1_image_id, num2_image_id in zip(num1_image_ids, num2_image_ids):
            predicted_sum += inferred_digits.loc[num1_image_id] * (10 ** place) + inferred_digits.loc[num2_image_id] * (10 ** place)

            place += 1

        if predicted_sum == series.iloc[-1]:
            valid_additions += 1

    return valid_additions / digit_sums.shape[0]
