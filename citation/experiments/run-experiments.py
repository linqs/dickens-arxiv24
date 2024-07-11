#!/usr/bin/env python3
import os
import re
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from citation.constants import BASE_CLI_DIR
from citation.constants import BASE_DATA_DIR
from citation.constants import BASE_EXPERIMENTS_DIR
from citation.constants import BASE_MODEL_DIR
from citation.constants import BASE_RESULTS_DIR
from citation.constants import CONFIG_FILENAME
from citation.constants import TRAINED_MODEL_NAME
from utils import enumerate_hyperparameters
from utils import load_json_file
from utils import run_experiment
from utils import write_json_file


BASE_MODEL_NAME = "citation.json"

DATASETS = ["citeseer", "cora"]
EXPERIMENTS = ["modular", "bilevel", "energy"]
SETTINGS = ["low-data", "semi-supervised"]
SIZES = {
    "semi-supervised": ["0.05", "0.10", "0.50", "1.00"],
    "low-data": ["20.00"]
}
SPLITS = ["00", "01", "02", "03", "04"]

RUN_HYPERPARAMETER_SEARCH = True


def parse_log(log_path):
    results = []

    with open(log_path, 'r') as file:
        for line in file:
            match = re.search(r'Categorical Accuracy: (\d+\.\d+)', line)
            if match is not None:
                results.append(float(match.group(1)))

    return results


def main():
    for dataset in DATASETS:
        for setting in SETTINGS:
            for split in SPLITS:
                for size in SIZES[setting]:
                    train_config = load_json_file(os.path.join(BASE_RESULTS_DIR, dataset, setting, split, size, "pretrain", CONFIG_FILENAME))
                    data_config = load_json_file(os.path.join(BASE_DATA_DIR, dataset, setting, split, size, CONFIG_FILENAME))
                    for experiment in EXPERIMENTS:
                        symbolic_model = load_json_file(os.path.join(BASE_MODEL_DIR, "symbolic", setting, dataset + ".json"))

                        if experiment == "modular":
                            if not os.path.exists(os.path.join(BASE_MODEL_DIR, "symbolic", setting, "modular.json")):
                                continue

                            symbolic_model = load_json_file(os.path.join(BASE_MODEL_DIR, "symbolic", setting, "modular.json"))

                        if not os.path.exists(os.path.join(BASE_EXPERIMENTS_DIR, experiment, setting + "-config.json")):
                            continue

                        symbolic_options = load_json_file(os.path.join(BASE_EXPERIMENTS_DIR, experiment, setting + "-config.json"))

                        hyperparameters = [symbolic_options["default"][dataset][size].copy()]
                        if RUN_HYPERPARAMETER_SEARCH and int(split) == 0 and len(symbolic_options["hyperparameters"]) > 0:
                            hyperparameters = enumerate_hyperparameters(symbolic_options["hyperparameters"].copy())
                        elif os.path.exists(os.path.join(BASE_RESULTS_DIR, dataset, setting, "00", size, experiment, CONFIG_FILENAME)):
                            hyperparameters = [load_json_file(os.path.join(BASE_RESULTS_DIR, dataset, setting, "00", size, experiment, CONFIG_FILENAME))["best_hyperparameters"].copy()]

                        best_valid_result = -1
                        if os.path.exists(os.path.join(BASE_RESULTS_DIR, dataset, setting, split, size, experiment, CONFIG_FILENAME)):
                            best_valid_result = load_json_file(os.path.join(BASE_RESULTS_DIR, dataset, setting, split, size, experiment, CONFIG_FILENAME))["results"][0]

                        for index, parameters in enumerate(hyperparameters):
                            output_dir = os.path.join(BASE_RESULTS_DIR, dataset, setting, split, size, experiment)
                            if RUN_HYPERPARAMETER_SEARCH and int(split) == 0:
                                output_dir = os.path.join(output_dir, "hyperparameter-search", "-".join([f"%s::%0.10f" % (k, v) for k, v in parameters.items()]))

                            os.makedirs(output_dir, exist_ok=True)
                            if os.path.exists(os.path.join(output_dir, "out.txt")):
                                print("Skipping experiment: {}.".format(output_dir))
                                continue

                            symbolic_model["options"] = symbolic_options["options"].copy()
                            if experiment == "bilevel":
                                symbolic_model["options"]["minimizer.energylosscoefficient"] = parameters["minimizer.energylosscoefficient"]
                            if experiment == "energy" or experiment == "bilevel":
                                symbolic_model["options"]["gradientdescent.stepsize"] = parameters["gradientdescent.stepsize"]
                                symbolic_model["predicates"]["Neural/2"]["options"]["learning-rate"] = parameters["learning-rate"]
                                symbolic_model["predicates"]["Neural/2"]["options"]["weight-decay"] = parameters["weight-decay"]
                                symbolic_model["predicates"]["Neural/2"]["options"]["hidden-dim"] = train_config["network"]["parameters"]["hidden-dim"]
                                symbolic_model["predicates"]["Neural/2"]["options"]["num-layers"] = train_config["network"]["parameters"]["num-layers"]
                                symbolic_model["predicates"]["Neural/2"]["options"]["dropout"] = train_config["network"]["parameters"]["dropout"]
                                symbolic_model["predicates"]["Neural/2"]["options"]["class-size"] = data_config["class-size"]
                                symbolic_model["predicates"]["Neural/2"]["options"]["input-dim"] = data_config["input-shape"]
                                symbolic_model["predicates"]["Neural/2"]["options"]["save-path"] = os.path.join(BASE_CLI_DIR, TRAINED_MODEL_NAME)
                                symbolic_model["predicates"]["Neural/2"]["options"]["train-path"] = os.path.join(BASE_RESULTS_DIR, dataset, setting, split, size, "pretrain", TRAINED_MODEL_NAME)

                            write_json_file(os.path.join(BASE_CLI_DIR, BASE_MODEL_NAME), symbolic_model)
                            os.makedirs(os.path.join(BASE_CLI_DIR, "data"), exist_ok=True)
                            os.system("cp -r {} {}".format(os.path.join(BASE_DATA_DIR, dataset, setting, split, size, "*"), os.path.join(BASE_CLI_DIR, "data")))
                            os.system("cp -r {} {}".format(os.path.join(BASE_RESULTS_DIR, dataset, setting, split, size, "pretrain", "category-neural-*"), os.path.join(BASE_CLI_DIR, "data")))

                            run_experiment(BASE_MODEL_NAME[:-5], BASE_CLI_DIR, output_dir)

                            results = parse_log(os.path.join(output_dir, "out.txt"))
                            if RUN_HYPERPARAMETER_SEARCH and int(split) == 0:
                                write_json_file(os.path.join(output_dir, CONFIG_FILENAME), {"results": results})
                                if results[0] <= best_valid_result:
                                    continue

                                os.system("cp -r {} {}".format(os.path.join(output_dir, "*"), os.path.join(BASE_RESULTS_DIR, dataset, setting, split, size, experiment)))
                                output_dir = os.path.join(BASE_RESULTS_DIR, dataset, setting, split, size, experiment)

                            write_json_file(os.path.join(output_dir, CONFIG_FILENAME), {"results": results, "best_hyperparameters": parameters})
                            if os.path.exists(os.path.join(BASE_CLI_DIR, TRAINED_MODEL_NAME)):
                                os.system("cp {} {}".format(os.path.join(BASE_CLI_DIR, TRAINED_MODEL_NAME), os.path.join(output_dir, TRAINED_MODEL_NAME)))

                            best_valid_result = results[0]


if __name__ == '__main__':
    main()
