#!/usr/bin/env python3
import os
import sys

import numpy as np
import pslpython.deeppsl.model
import torch.nn

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from neural.models.mlp import MLP
from neural.utils import get_torch_device
from citation.constants import BASE_CLI_DIR
from utils import one_hot_encoding
from utils import seed_everything


SYMBOLIC_LABELS_PATH = os.path.join(BASE_CLI_DIR, "inferred-predicates", "LABEL.txt")


class CitationNeuPSL(pslpython.deeppsl.model.DeepModel):
    def __init__(self):
        super().__init__()
        self.output_dir = None

        self.model = None
        self.optimizer = None

        self.features = None
        self.labels = None
        self.logits = None
        self.probabilities = None

        self.epoch = 0
        self.epoch_loss = 0.0

        self.device = get_torch_device()
        seed_everything()

    def internal_init(self, application, options={}):
        self.model = MLP(int(options["input-dim"]), int(options["hidden-dim"]), int(options["class-size"]), int(options["num-layers"]), float(options["dropout"])).to(self.device)

        if application == 'learning':
            self.epoch = 0

            self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                               lr=float(options["learning-rate"]),
                                               weight_decay=float(options["weight-decay"]))

            self.model.load_state_dict(torch.load(options["train-path"], map_location=self.device))

        elif application == "inference":
            self.model.load_state_dict(torch.load(options["save-path"], map_location=self.device))

        return {}

    def internal_fit(self, data, gradients, options={}):
        self._prepare_data(data, options=options)

        structured_gradients = torch.tensor(gradients.astype(np.float32), dtype=torch.float32, device=self.device, requires_grad=False)

        self.probabilities.backward(structured_gradients, retain_graph=True)

        # Clip the gradients.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)

        self.optimizer.step()
        self.optimizer.zero_grad()

        return {}

    def internal_predict(self, data, options={}):
        self._prepare_data(data, options=options)

        if self._train:
            self.model.train()
            self.logits = self.model(self.features)
            self.probabilities = torch.nn.functional.softmax(self.logits, dim=1)
        else:
            self.model.eval()
            with torch.no_grad():
                self.logits = self.model(self.features)
                self.probabilities = torch.nn.functional.softmax(self.logits, dim=1)

        return self.probabilities.cpu().detach(), {}

    def internal_epoch_start(self, options={}):
        self.epoch_loss = 0.0

        if self._train:
            self.optimizer.zero_grad()

    def internal_epoch_end(self, options={}):
        if self._train:
            self.epoch += 1

            self.optimizer.zero_grad()
            self.epoch_loss /= len(self.labels)

            return {"average-loss": self.epoch_loss}

        return {}

    def internal_save(self, options={}):
        torch.save(self.model.state_dict(), options["save-path"])

    def _prepare_data(self, data, options={}):
        if self.features is not None and self.labels is not None:
            return

        numpy_features = data[:, :-1].astype(np.float32)
        self.features = torch.tensor(numpy_features, dtype=torch.float32, device=self.device)
        self.labels = torch.tensor([one_hot_encoding(int(label), int(options['class-size'])) for label in data[:, -1]], dtype=torch.float32, device=self.device)