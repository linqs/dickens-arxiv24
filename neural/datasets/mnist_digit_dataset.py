import numpy as np
import os
import pandas as pd
import sys
import torch

from torch.utils.data import Dataset

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(THIS_DIR, '..', '..'))

import utils


class MNISTDigitDataset(Dataset):
    def __init__(self, experiment_dir: str, index: list, class_size: int = 10,
                 partition: str = "labeled", id_labels: bool = False):
        self.experiment_dir = experiment_dir

        if partition == "labeled":
            self.data_frame = pd.read_csv(f"{experiment_dir}/neural-data-train.txt", sep="\t", header=None)
            self.data_frame = self.data_frame[self.data_frame.iloc[:, -1] == 0]
        elif partition == "train":
            self.data_frame = pd.read_csv(f"{experiment_dir}/neural-data-train.txt", sep="\t", header=None)
            self.data_frame = self.data_frame[self.data_frame.iloc[:, -1] <= 1]
        elif partition == "valid":
            self.data_frame = pd.read_csv(f"{experiment_dir}/neural-data-valid.txt", sep="\t", header=None)
            self.data_frame = self.data_frame[self.data_frame.iloc[:, -1] == 2]
        elif partition == "test":
            self.data_frame = pd.read_csv(f"{experiment_dir}/neural-data-test.txt", sep="\t", header=None)
            self.data_frame = self.data_frame[self.data_frame.iloc[:, -1] == 3]
        else:
            raise ValueError(f"Invalid partition: {partition}")

        self.data_frame = self.data_frame.set_index(index)
        self.data_frame = self.data_frame.reset_index(drop=True)

        # Only save the digit features and labels.
        self.features = torch.tensor(self.data_frame.iloc[:, :-2].to_numpy(), dtype=torch.float32, device="cpu")
        self.features = self.features.reshape(self.features.shape[0], 1, 28, 28)
        if id_labels:
            self.labels = torch.tensor(np.asarray([label for label in self.data_frame.index], dtype=np.float32), dtype=torch.float32, device="cpu")
        else:
            self.labels = torch.tensor(np.asarray([utils.one_hot_encoding(int(label), class_size) for label in self.data_frame.iloc[:, -2]], dtype=np.float32), dtype=torch.float32, device="cpu")

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
