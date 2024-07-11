import csv
import json
import os

THIS_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

DATA_DIR = os.path.join(THIS_DIR, "data")
RESULTS_DIR = os.path.join(THIS_DIR, "results")

PROMPTS_DIR = os.path.join(THIS_DIR, "models", "prompts")
SYMBOLIC_REASONER_DIR = os.path.join(THIS_DIR, "models", "symbolic_reasoner")

CONFIG_FILENAME = "config.json"
PROMPT_FILENAME = "prompt.txt"
RESPONSE_FILENAME = "response.json"
SYMBOLIC_RESULTS_FILENAME = "symbolic_results.json"

DEFAULT_LOGGING_LEVEL = "INFO"


def write_psl_file(path, data):
    with open(path, 'w') as file:
        for row in data:
            file.write('\t'.join([str(item) for item in row]) + "\n")


def load_psl_file(path, dtype=str):
    data = []

    with open(path, 'r') as file:
        for line in file:
            line = line.strip()
            if line == '':
                continue

            data.append(list(map(dtype, line.split("\t"))))

    return data


def write_json_file(path, data, indent=4):
    with open(path, "w") as file:
        if indent is None:
            json.dump(data, file)
        else:
            json.dump(data, file, indent=indent)


def load_json_file(path):
    with open(path, "r") as file:
        return json.load(file)


def load_csv_file(path, delimiter=','):
    with open(path, 'r') as file:
        reader = csv.reader(file, delimiter=delimiter)
        return list(reader)

def load_txt_file(path):
    with open(path, 'r') as file:
        return file.read()
