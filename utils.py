import csv
import json
import os
import random
import subprocess

import dgl
import numpy
import torch


def enumerate_hyperparameters(hyperparameters_dict, current_hyperparameters={}):
    for key in sorted(hyperparameters_dict):
        hyperparameters = []
        for value in hyperparameters_dict[key]:
            next_hyperparameters = current_hyperparameters.copy()
            next_hyperparameters[key] = value

            remaining_hyperparameters = hyperparameters_dict.copy()
            remaining_hyperparameters.pop(key)

            if remaining_hyperparameters:
                hyperparameters = hyperparameters + enumerate_hyperparameters(remaining_hyperparameters, next_hyperparameters)
            else:
                hyperparameters.append(next_hyperparameters)
        return hyperparameters


def get_log_paths(path, log_filename):
    log_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if log_filename == file.split("/")[-1]:
                log_paths.append(os.path.join(root, file))

    return sorted(log_paths)


def one_hot_encoding(label, num_labels):
    encoding = [0] * num_labels
    encoding[label] = 1
    return encoding


def write_json_file(path, data, indent=4):
    with open(path, "w") as file:
        json.dump(data, file, indent=indent)


def load_json_file(path):
    with open(path, "r") as file:
        return json.load(file)


def load_fake_json_file(path, size=None):
    with open(path, "r") as file:
        data = []
        current_line = 0
        for line in file:
            if size is not None and current_line > size - 1:
                break
            data.append(json.loads(line))
            current_line += 1

    return data


def write_psl_data_file(path, data):
    with open(path, 'w') as file:
        for row in data:
            file.write('\t'.join([str(item) for item in row]) + "\n")


def load_psl_data_file(path, dtype=str):
    data = []

    with open(path, 'r') as file:
        for line in file:
            line = line.strip()
            if line == '':
                continue

            data.append(list(map(dtype, line.split("\t"))))

    return data


def load_csv_file(path, delimiter=','):
    with open(path, 'r') as file:
        reader = csv.reader(file, delimiter=delimiter)
        return list(reader)


def write_csv_file(path, data, delimiter=','):
    with open(path, 'w') as file:
        writer = csv.writer(file, delimiter=delimiter)
        writer.writerows(data)


def run_experiment(model_name, cli_dir, experiment_out_dir, trained_model_path=None):
    # Run the experiment.
    print("Running experiment: {}.".format(experiment_out_dir))

    try:
        subprocess.run(["cd {} && ./run.sh {} > out.txt 2> out.err".format(cli_dir, experiment_out_dir)],
                       capture_output=True, shell=True, check=True)

    except subprocess.TimeoutExpired as e:
        print("Experiment Timeout: {}.".format(experiment_out_dir))

    save_output_and_json(cli_dir, experiment_out_dir, model_name, learn=trained_model_path is not None)

    if trained_model_path is not None:
        save_nesy_trained_neural_component(experiment_out_dir, trained_model_path)


def save_nesy_trained_neural_component(experiment_out_dir, trained_model_path):
    os.system("cp {} {}".format(trained_model_path, experiment_out_dir))


def save_output_and_json(cli_dir, experiment_out_dir, model_name, learn=False):
    os.system("cp {} {}".format(os.path.join(cli_dir, "out.txt"), experiment_out_dir))
    os.system("cp {} {}".format(os.path.join(cli_dir, "out.err"), experiment_out_dir))
    os.system("cp {} {}".format(os.path.join(cli_dir, f"{model_name}.json"), experiment_out_dir))
    if learn:
        os.system("cp {} {}".format(os.path.join(cli_dir, f"{model_name}_learned.psl"), experiment_out_dir))
    os.system("cp -r {} {}".format(os.path.join(cli_dir, "inferred-predicates"), experiment_out_dir))


def seed_everything(seed=42):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
