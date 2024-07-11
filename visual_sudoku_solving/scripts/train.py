import copy
import json
import os
import sys
import torch
import torchvision

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(THIS_DIR, '..', '..'))

import utils
import neural.utils as neural_utils

from neural.datasets.mnist_digit_dataset import MNISTDigitDataset
from neural.models.mnist_classifier import MNISTClassifier
from neural.models.mlp import MLP
from neural.trainers.supervised import train_supervised

PRETRAINED_MODEL_NAME = "pretrained-digit-classifier.pt"

TRAINED_BACKBONE_NAME = "supervision-trained-backbone.pt"
TRAINED_PREDICTION_HEAD_NAME = "supervision-trained-projection-head.pt"
TRAINED_MODEL_NAME = "supervision-trained-digit-classifier.pt"

HYPERPARAMETER_SEARCH_LEARNING_RATES = [1.0e-3, 1.0e-4, 1.0e-5]
HYPERPARAMETER_SEARCH_WEIGHT_DECAYS = [1.0e-3, 1.0e-4, 1.0e-5]

CONFIG = {
    "Backbone": {
        "Model": "resnet18",
        "Hidden-Size": 128
    },
    "Projection-Head": {
        "Hidden-Size": 64,
        "Num-Layers": 2
    },
    "Optimizer": {
        "Batch-Size": 32,
        "Learning-Rate": 1.0e-3,
        "Weight-Decay": 1.0e-4
    }
}

BACKBONE_MODELS = {
    "resnet18": torchvision.models.resnet18
}


def train_internal(data_dir, config, device):
    train_dataset = MNISTDigitDataset(
        data_dir, [0, 1, 2], class_size=9, partition="labeled"
    )

    validation_dataset = MNISTDigitDataset(
        data_dir, [0, 1, 2], class_size=9, partition="valid"
    )

    backbone = BACKBONE_MODELS[config["Backbone"]["Model"]](num_classes=config["Backbone"]["Hidden-Size"])
    prediction_head = MLP(config["Backbone"]["Hidden-Size"], config["Projection-Head"]["Hidden-Size"], 9,
                          config["Projection-Head"]["Num-Layers"])
    model = MNISTClassifier(backbone, prediction_head, device=device)

    neural_utils.load_model(model, data_dir, PRETRAINED_MODEL_NAME)

    model, validation_accuracy = train_supervised(
        train_dataset, validation_dataset, model,
        learning_rate=config["Optimizer"]["Learning-Rate"],
        weight_decay=config["Optimizer"]["Weight-Decay"],
        batch_size=config["Optimizer"]["Batch-Size"],
        device=device
    )

    return backbone, prediction_head, model, validation_accuracy


def train(experiment, split, train_size, num_clues, unlabeled, config, device, backbone=None, prediction_head=None, model=None):
    data_dir = f"{THIS_DIR}/../data/{experiment}/split::{split}/train-size::{train_size}/num-clues::{num_clues}/unlabeled::{unlabeled}"

    utils.seed_everything()

    if model is None:
        backbone, prediction_head, model, validation_accuracy = train_internal(data_dir, config, device)

    print(f"Saving trained models for {experiment}/split::{split}/train-size::{train_size}/num-clues::{num_clues}/unlabeled::{unlabeled}...")

    neural_utils.save_model(backbone, data_dir, TRAINED_BACKBONE_NAME)
    neural_utils.save_model(prediction_head, data_dir, TRAINED_PREDICTION_HEAD_NAME)
    neural_utils.save_model(model, data_dir, TRAINED_MODEL_NAME)
    json.dump(config, open(f"{data_dir}/config.json", "w"), indent=4)

    return backbone, prediction_head, model


def hyperparameter_search(experiment, split, train_size, num_clues, unlabeled, config, device):
    best_validation_accuracy = 0.0
    best_config = copy.deepcopy(config)

    data_dir = f"{THIS_DIR}/../data/{experiment}/split::{split}/train-size::{train_size}/num-clues::{num_clues}/unlabeled::{unlabeled}"

    print(f"Hyperparameter search for {experiment}/split::{split}/train-size::{train_size}/num-clues::{num_clues}/unlabeled::{unlabeled}...")

    for learning_rate in HYPERPARAMETER_SEARCH_LEARNING_RATES:
        for weight_decay in HYPERPARAMETER_SEARCH_WEIGHT_DECAYS:
            config["Optimizer"]["Learning-Rate"] = learning_rate
            config["Optimizer"]["Weight-Decay"] = weight_decay

            utils.seed_everything()

            _, _, _, validation_accuracy = train_internal(data_dir, config, device)

            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_config = copy.deepcopy(config)

    print(f"Best validation accuracy: {best_validation_accuracy}")

    return best_config


def main():
    device = neural_utils.get_torch_device()

    for dataset in ["mnist-9x9"]:
        for train_size in ["0020"]:
            for num_clues in ["30"]:
                for unlabeled in ["0.00", "0.50", "0.90", "0.95"]:
                    best_config = copy.deepcopy(CONFIG)
                    backbone, prediction_head, model = None, None, None
                    for split in ["0", "1", "2", "3", "4"]:
                        experiment_dir = f"{THIS_DIR}/../data/{dataset}/split::{split}/train-size::{train_size}/num-clues::{num_clues}/unlabeled::{unlabeled}"
                        if os.path.exists(f"{experiment_dir}/saved-networks/{TRAINED_MODEL_NAME}"):
                            print(f"Found trained models for {dataset}/split::{split}/train-size::{train_size}/num-clues::{num_clues}/unlabeled::{unlabeled}. Skipping...")
                            continue

                        print(f"Training {dataset}/split::{split}/train-size::{train_size}/num-clues::{num_clues}/unlabeled::{unlabeled}...")

                        if split == "0":
                            best_config = hyperparameter_search(
                                dataset, split, train_size, num_clues, unlabeled, copy.deepcopy(CONFIG), device
                            )

                            backbone, prediction_head, model = train(
                                dataset, split, train_size, num_clues, unlabeled, best_config, device
                            )
                        else:
                            train(
                                dataset, split, train_size, num_clues, unlabeled, best_config, device,
                                backbone, prediction_head, model
                            )


if __name__ == "__main__":
    main()
