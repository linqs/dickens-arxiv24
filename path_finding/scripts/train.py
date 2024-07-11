import copy
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torchvision

from torch.utils.data import Dataset

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(THIS_DIR, '..', '..'))

import utils
import neural.utils as neural_utils

from neural.trainers.supervised import train_supervised
from path_finding.models.neural.pathfinder import Pathfinder

BACKBONE_MODELS = {
    "resnet18": torchvision.models.resnet18
}

CONFIG = {
    "Backbone": {
        "Model": "resnet18",
        "Hidden-Size": 144
    },
    "Optimizer": {
        "Batch-Size": 128,
        "Learning-Rate": 5.0e-4,
        "Weight-Decay": 1.0e-3
    }
}

TRAINED_BACKBONE_NAME = "supervision-trained-backbone.pt"
TRAINED_MODEL_NAME = "supervision-trained-pathfinder.pt"

HYPERPARAMETER_SEARCH_LEARNING_RATES = [5.0e-4]
HYPERPARAMETER_SEARCH_WEIGHT_DECAYS = [1.0e-3]

WARCRAFT_IMAGE_HEIGHT = 96
WARCRAFT_IMAGE_WIDTH = 96
WARCRAFT_IMAGE_CHANNELS = 3


class WarcraftMapDataset(Dataset):
    def __init__(self, data_dir, partition):
        self.experiment_dir = data_dir

        self.labels = pd.read_csv(f"{data_dir}/neural-path-truth-{partition}.txt",
                                  sep="\t", header=None, dtype=np.int32)
        self.labels.columns = ["MapId", "XY", "Label"]
        self.labels.set_index(["MapId", "XY"], inplace=True)

        self.maps = pd.read_csv(f"{data_dir}/neural-data.txt", sep="\t", header=None, dtype=np.float32)
        self.maps.set_index(0, inplace=True)

        # Filter out maps that are not in the partition.
        self.maps = self.maps[self.maps.index.isin(self.labels.index.get_level_values(0))]

        # Convert to tensors
        self.labels = torch.tensor(self.labels.unstack(level=1).to_numpy(), dtype=torch.float32, device="cpu")
        self.maps = torch.tensor(self.maps.to_numpy(), dtype=torch.float32, device="cpu")
        self.maps = self.maps.reshape(self.maps.shape[0], WARCRAFT_IMAGE_HEIGHT, WARCRAFT_IMAGE_WIDTH, WARCRAFT_IMAGE_CHANNELS).permute(0, 3, 1, 2)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return self.maps[index], self.labels[index]


def train_internal(data_dir, config, device):
    train_dataset = WarcraftMapDataset(
        data_dir, partition="train"
    )

    validation_dataset = WarcraftMapDataset(
        data_dir, partition="valid"
    )

    backbone = BACKBONE_MODELS[config["Backbone"]["Model"]](num_classes=config["Backbone"]["Hidden-Size"])

    model = Pathfinder(backbone, device=device)

    model, validation_accuracy = train_supervised(
        train_dataset, validation_dataset, model,
        learning_rate=config["Optimizer"]["Learning-Rate"],
        weight_decay=config["Optimizer"]["Weight-Decay"],
        batch_size=config["Optimizer"]["Batch-Size"],
        validation_patience=50,
        loss_name="masked_binary_cross_entropy_with_logits",
        evaluator="masked_accuracy",
        device=device
    )

    return backbone, model, validation_accuracy


def train(data_dir, config, device, backbone=None, model=None):
    utils.seed_everything()

    if model is None:
        backbone, model, validation_accuracy = train_internal(data_dir, config, device)

    print(f"Saving trained models for {data_dir}...")

    neural_utils.save_model(backbone, data_dir, TRAINED_BACKBONE_NAME)
    neural_utils.save_model(model, data_dir, TRAINED_MODEL_NAME)
    json.dump(config, open(f"{data_dir}/config.json", "w"), indent=4)

    return backbone, model


def hyperparameter_search(data_dir, config, device):
    best_validation_accuracy = 0.0
    best_config = copy.deepcopy(config)

    print(f"Hyperparameter search for {data_dir}...")

    for learning_rate in HYPERPARAMETER_SEARCH_LEARNING_RATES:
        for weight_decay in HYPERPARAMETER_SEARCH_WEIGHT_DECAYS:
            config["Optimizer"]["Learning-Rate"] = learning_rate
            config["Optimizer"]["Weight-Decay"] = weight_decay

            utils.seed_everything()

            _, _, validation_accuracy = train_internal(data_dir, config, device)

            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_config = copy.deepcopy(config)

    print(f"Best validation accuracy: {best_validation_accuracy}")

    return best_config


def main():
    device = neural_utils.get_torch_device()

    for dataset in ["warcraft-map"]:
        for unlabeled_ratio in ["0.00", "0.50", "0.90", "0.95"]:
            best_config = copy.deepcopy(CONFIG)
            backbone, model = None, None
            for split in ["0", "1", "2", "3", "4"]:
                data_dir = f"{THIS_DIR}/../data/{dataset}/split::{split}/unlabeled::{unlabeled_ratio}"
                if os.path.exists(f"{data_dir}/saved-networks/{TRAINED_MODEL_NAME}"):
                    print(f"Found pretrained models for {dataset}. Skipping...")
                    continue

                print(f"Training {dataset}/split::{split}/unlabeled::{unlabeled_ratio}...")

                if split == "0":
                    best_config = hyperparameter_search(data_dir, copy.deepcopy(CONFIG), device)

                    backbone, model = train(data_dir, best_config, device)
                else:
                    train(data_dir, best_config, device, backbone, model)


if __name__ == "__main__":
    main()
