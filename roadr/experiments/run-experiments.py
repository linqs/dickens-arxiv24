#!/usr/bin/env python3
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from roadr.constants import BASE_CLI_DIR
from roadr.constants import BASE_EXPERIMENTS_DIR
from roadr.constants import BASE_MODEL_DIR
from roadr.constants import BASE_RESULTS_DIR
from roadr.constants import BOX_CONFIDENCE_THRESHOLD
from roadr.constants import CONFIG_FILENAME
from roadr.constants import IOU_THRESHOLD
from roadr.constants import IMAGE_HEIGHT
from roadr.constants import IMAGE_WIDTH
from roadr.constants import LABEL_CONFIDENCE_THRESHOLD
from roadr.constants import LABEL_MAPPING
from roadr.constants import RGB_IMAGES_DIR
from roadr.constants import TRAINED_MODEL_NAME
from roadr.utils import confusion_matrix
from roadr.utils import precision_recall_f1
from roadr.utils import save_images
from utils import enumerate_hyperparameters
from utils import load_json_file
from utils import run_experiment
from utils import write_json_file

BASE_MODEL_NAME = "roadr.json"

DATASETS = ["roadr"]
EXPERIMENTS = ["modular", "energy"]
SETTINGS = ["constraint-satisfaction", "semi-supervised"]
SIZES = {
    "constraint-satisfaction": ["15"],
    "semi-supervised": ["02", "04", "08", "15"]
}
SPLITS = ["00"]

RUN_HYPERPARAMETER_SEARCH = False


def calculate_results(path, output_dir):
    predictions = load_json_file(path)

    pred_probabilities = torch.tensor(predictions["pred_probabilities"])
    pred_boxes = torch.tensor(predictions["pred_boxes"])
    pred_boxes_confidence = torch.tensor(predictions["pred_boxes_confidence"])
    truth_labels = torch.tensor(predictions["truth_labels"])
    truth_boxes = torch.tensor(predictions["truth_boxes"])
    truth_boxes_confidence = torch.sum(truth_labels, dim=2)

    indexes = torch.argsort(pred_boxes_confidence[:, :, 0], descending=True)
    pred_probabilities = pred_probabilities[torch.arange(len(pred_probabilities)).unsqueeze(1), indexes]
    pred_boxes = pred_boxes[torch.arange(len(pred_boxes)).unsqueeze(1), indexes]
    pred_boxes_confidence = pred_boxes_confidence[torch.arange(len(pred_boxes_confidence)).unsqueeze(1), indexes]

    tp, fp, fn = confusion_matrix(pred_boxes, pred_probabilities, pred_boxes_confidence, truth_boxes, truth_labels,
                                  IOU_THRESHOLD, LABEL_CONFIDENCE_THRESHOLD, BOX_CONFIDENCE_THRESHOLD)

    total_precision, total_recall, total_f1 = precision_recall_f1(sum(tp), sum(fp), sum(fn))
    agent_precision, agent_recall, agent_f1 = precision_recall_f1(sum(tp[:10]), sum(fp[:10]), sum(fn[:10]))
    action_precision, action_recall, action_f1 = precision_recall_f1(sum(tp[10:29]), sum(fp[10:29]), sum(fn[10:29]))
    loc_precision, loc_recall, loc_f1 = precision_recall_f1(sum(tp[29:]), sum(fp[29:]), sum(fn[29:]))

    save_images(LABEL_MAPPING, IMAGE_HEIGHT, IMAGE_WIDTH, RGB_IMAGES_DIR, os.path.join(output_dir, "images", "predicted"),
                predictions["ids"], pred_boxes, pred_probabilities, pred_boxes_confidence,
                LABEL_CONFIDENCE_THRESHOLD, BOX_CONFIDENCE_THRESHOLD, color="blue")

    save_images(LABEL_MAPPING, IMAGE_HEIGHT, IMAGE_WIDTH, RGB_IMAGES_DIR, os.path.join(output_dir, "images", "truth"),
                predictions["ids"], truth_boxes, truth_labels, truth_boxes_confidence,
                LABEL_CONFIDENCE_THRESHOLD, BOX_CONFIDENCE_THRESHOLD, color="red")

    return [total_precision, total_recall, total_f1, agent_precision, agent_recall, agent_f1, action_precision, action_recall, action_f1, loc_precision, loc_recall, loc_f1]


def main():
    for dataset in DATASETS:
        for setting in SETTINGS:
            for split in SPLITS:
                for size in SIZES[setting]:
                    for experiment in EXPERIMENTS:
                        output_dir = os.path.join(BASE_RESULTS_DIR, dataset, setting, split, size, experiment)
                        os.makedirs(output_dir, exist_ok=True)

                        if os.path.exists(os.path.join(output_dir, "out.txt")):
                            print("Skipping experiment: {}.".format(output_dir))
                            continue

                        symbolic_model = load_json_file(os.path.join(BASE_MODEL_DIR, "symbolic", dataset + ".json"))
                        if not os.path.exists(os.path.join(BASE_EXPERIMENTS_DIR, experiment, setting + "-config.json")):
                            continue
                        symbolic_options = load_json_file(os.path.join(BASE_EXPERIMENTS_DIR, experiment, setting + "-config.json"))

                        hyperparameters = [symbolic_options["default"][dataset][size].copy()]
                        if RUN_HYPERPARAMETER_SEARCH and int(split) == 0 and len(symbolic_options["hyperparameters"]) > 0:
                            hyperparameters = enumerate_hyperparameters(symbolic_options["hyperparameters"].copy())
                        elif os.path.exists(os.path.join(BASE_RESULTS_DIR, dataset, setting, "00", size, experiment, CONFIG_FILENAME)):
                            hyperparameters = load_json_file(os.path.join(BASE_RESULTS_DIR, dataset, setting, "00", size, experiment, CONFIG_FILENAME))["best_hyperparameters"].copy()

                        best_f1 = -1
                        for index, parameters in enumerate(hyperparameters):
                            symbolic_model["options"] = symbolic_options["options"].copy()

                            if os.path.exists(os.path.join(output_dir, "out.txt")):
                                print("Skipping experiment: {}.".format(output_dir))
                                continue

                            symbolic_model["predicates"]["Neural/3"]["options"]["eval"] = "test"
                            symbolic_model["predicates"]["Neural/3"]["options"]["output_dir"] = output_dir
                            symbolic_model["predicates"]["Neural/3"]["options"]["pretrained_dir"] = os.path.join(BASE_RESULTS_DIR, dataset, setting, split, size, "pretrain")

                            write_json_file(os.path.join(BASE_CLI_DIR, BASE_MODEL_NAME), symbolic_model)

                            run_experiment(BASE_MODEL_NAME[:-5], BASE_CLI_DIR, output_dir)

                            results = calculate_results(os.path.join(output_dir, "predictions.json"), output_dir)
                            if (RUN_HYPERPARAMETER_SEARCH and int(split) == 0) and results[2] < best_f1:
                                continue

                            write_json_file(os.path.join(output_dir, CONFIG_FILENAME), {"results": results, "best_hyperparameters": parameters})
                            if os.path.exists(os.path.join(BASE_CLI_DIR, TRAINED_MODEL_NAME)):
                                os.system("cp {} {}".format(os.path.join(BASE_CLI_DIR, TRAINED_MODEL_NAME), os.path.join(output_dir, TRAINED_MODEL_NAME)))


if __name__ == '__main__':
    main()
