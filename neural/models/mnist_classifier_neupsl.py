#!/usr/bin/env python3
import numpy as np
import os
import pslpython.deeppsl.model
import sys
import torch
import torchvision

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.join(THIS_DIR, '..', '..'))

import utils

from neural import utils as neural_utils
from neural.models.mnist_classifier import MNISTClassifier
from neural.models.mlp import MLP


class MNISTClassifierNeuPSL(pslpython.deeppsl.model.DeepModel):
    def __init__(self):
        super().__init__()
        self._application = None

        self.model = None
        self.optimizer = None

        self.epoch = 0
        self.initial_temperature = 1.0

        self.features = None
        self.digit_labels = None

        self.labeled_mask = None

        self.predicted_logits = None
        self.predicted_probabilities = None

        self.average_loss = None
        self.num_labeled = None

        self.device = neural_utils.get_torch_device()

    def internal_init(self, application, options={}):
        options = options.copy()
        self.model = self._create_model(options=options).to(self.device)

        if application == 'learning':
            self.epoch = 0

            self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                               lr=float(options["neural_learning_rate"]),
                                               weight_decay=float(options["weight_decay"]))

            self.model.load_state_dict(torch.load(options["pretrain-path"], map_location=self.device))

        elif application == "inference":
            self.model.load_state_dict(torch.load(options["save-path"], map_location=self.device))

        return {}

    def internal_fit(self, data, gradients, options={}):
        self._prepare_data(data, options=options)

        if self.predicted_probabilities is None:
            return {}

        structured_gradients = float(options["alpha"]) * torch.tensor(gradients.astype(np.float32),
                                                                      dtype=torch.float32, device=self.device,
                                                                      requires_grad=False)

        self.predicted_probabilities.backward(structured_gradients, retain_graph=True)

        if self.labeled_mask.sum() > 0:
            loss = (1 - float(options["alpha"])) * torch.nn.functional.cross_entropy(
                self.predicted_logits[self.labeled_mask],
                self.digit_labels[self.labeled_mask],
                reduction="sum"
            )

            loss.backward()

            self.average_loss += loss.item()
            self.num_labeled += 1

        # Clip the gradients.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)

        self.optimizer.step()
        self.optimizer.zero_grad()

        return {}

    def internal_predict(self, data, options={}):
        self._prepare_data(data, options=options)

        if self.features is None:
            self.predicted_logits = None
            self.predicted_probabilities = None
            return np.array([]), {}

        results = {}

        if self._train:
            self.model.train()
            self.predicted_logits = self.model(self.features)
            self.predicted_probabilities = neural_utils.gumbel_softmax(
                self.predicted_logits,
                temperature=neural_utils.get_temperature(self.initial_temperature, 0.5, self.epoch),
                hard=False)
        else:
            self.model.eval()
            with torch.no_grad():
                self.predicted_logits = self.model(self.features)
                self.predicted_probabilities = torch.nn.functional.softmax(self.predicted_logits, dim=1)

        return self.predicted_probabilities.cpu().detach(), results

    def internal_epoch_start(self, options={}):
        self.average_loss = 0.0
        self.num_labeled = 0

        if self._train:
            self.optimizer.zero_grad()

    def internal_epoch_end(self, options={}):
        if self._train:
            self.epoch += 1

            self.optimizer.zero_grad()

            if self.num_labeled > 0:
                self.average_loss /= self.num_labeled

            return {"average_loss": self.average_loss}

        return {}

    def internal_save(self, options={}):
        os.makedirs(os.path.dirname(options["save-path"]), exist_ok=True)
        torch.save(self.model.state_dict(), options["save-path"])

        return {}

    def _create_model(self, options={}):
        return MNISTClassifier(
            torchvision.models.resnet18(num_classes=128),
            MLP(128, 64, int(options["class-size"]), 2, float(options["dropout"])),
            device=self.device)

    def _prepare_data(self, data, options={}):
        if data.shape[0] == 0:
            self.features = None
            self.digit_labels = None
            self.labeled_mask = None
            return

        self.features = torch.tensor(np.asarray(data[:, :-2], dtype=np.float32), dtype=torch.float32, device=self.device)
        self.features = self.features.reshape(self.features.shape[0], 1, 28, 28)
        self.digit_labels = torch.tensor(
            np.asarray([utils.one_hot_encoding(int(float(label)), int(options["class-size"])) for label in data[:, -2]], dtype=np.float32),
            dtype=torch.float32, device=self.device
        )

        self.labeled_mask = data[:, -1] == 0.0
