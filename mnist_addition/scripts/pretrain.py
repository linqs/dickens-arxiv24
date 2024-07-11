import os
import sys
import torchvision

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(THIS_DIR, '..', '..'))

import utils
import neural.utils as neural_utils

from neural.datasets.mnist_digit_dataset import MNISTDigitDataset
from neural.models.mnist_classifier import MNISTClassifier
from neural.models.mlp import MLP
from neural.trainers.simclr import pretrain_simclr

TRAINED_BACKBONE_NAME = "simclr-pretrained-backbone.pt"
TRAINED_PREDICTION_HEAD_NAME = "simclr-pretrained-projection-head.pt"
TRAINED_MODEL_NAME = "pretrained-digit-classifier.pt"

CONFIG = {
    "augmentations": [
        ("RandomAffine", {"degrees": (-30, 30), "translate": (0.1, 0.1), "scale": (0.75, 1.25)}),
        ("ElasticTransform", {"alpha": 125.0}),
    ]
}


def main():
    device = neural_utils.get_torch_device()

    for experiment in ["mnist-1", "mnist-2"]:
        for split in ["0", "1", "2", "3", "4"]:
            backbone, prediction_head, model = None, None, None
            for train_size in ["00600"]:
                for unlabeled in ["0.00", "0.50", "0.90", "0.95", "1.00"]:
                    for overlap in ["0.00"]:
                        experiment_dir = f"{THIS_DIR}/../data/{experiment}/split::{split}/train-size::{train_size}/unlabeled::{unlabeled}/overlap::{overlap}"
                        if os.path.exists(f"{experiment_dir}/saved-networks/{TRAINED_MODEL_NAME}"):
                            print(f"Found pretrained models for {experiment}/split::{split}/train-size::{train_size}/unlabeled::{unlabeled}/overlap::{overlap}. Skipping...")
                            continue

                        utils.seed_everything()

                        if model is None:
                            train_dataset = MNISTDigitDataset(
                                experiment_dir, [0], class_size=10, partition="train", id_labels=True
                            )

                            backbone = torchvision.models.resnet18(num_classes=128)
                            backbone, pretrained_projection_head = pretrain_simclr(
                                train_dataset, experiment_dir, CONFIG, backbone, MNISTClassifier, device=device)

                            # Reset the projection head.
                            model = MNISTClassifier(backbone, MLP(128, 64, 10, 2).to(device), device=device)

                        neural_utils.save_model(backbone, experiment_dir, TRAINED_BACKBONE_NAME)
                        neural_utils.save_model(pretrained_projection_head, experiment_dir, TRAINED_PREDICTION_HEAD_NAME)
                        neural_utils.save_model(model, experiment_dir, TRAINED_MODEL_NAME)


if __name__ == "__main__":
    main()
