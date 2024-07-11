#!/usr/bin/env python3
import numpy as np
import os
import pslpython.deeppsl.model
import sys
import torch
import torchvision

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.join(THIS_DIR, '..', '..', '..'))

from neural import utils as neural_utils

from path_finding.models.neural.pathfinder import Pathfinder


WARCRAFT_IMAGE_HEIGHT = 96
WARCRAFT_IMAGE_WIDTH = 96
WARCRAFT_IMAGE_CHANNELS = 3


class WarcraftPathfinderNeuPSL(pslpython.deeppsl.model.DeepModel):
    def __init__(self):
        super().__init__()
        self._application = None

        self.model = None
        self.optimizer = None

        self.epoch = 0
        self.initial_temperature = 1.0

        self.features = None

        self.predicted_logits = None
        self.predicted_probabilities = None

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

        structured_gradients = torch.tensor(gradients.astype(np.float32),
                                            dtype=torch.float32, device=self.device,
                                            requires_grad=False)

        self.predicted_probabilities.backward(structured_gradients)

        # Clip the gradients.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)

        self.optimizer.step()
        self.optimizer.zero_grad()

        return {}

    def internal_predict(self, data, options={}):
        self._prepare_data(data, options=options)

        if self.features is None:
            return np.array([]), {}

        results = {}

        if self._train:
            self.model.train()
            self.predicted_logits = self.model(self.features)
            self.predicted_probabilities = neural_utils.gumbel_sigmoid(
                self.predicted_logits,
                temperature=neural_utils.get_temperature(self.initial_temperature, 0.5, self.epoch),
                hard=False
            )
        else:
            self.model.eval()
            with torch.no_grad():
                self.predicted_logits = self.model(self.features)
                self.predicted_probabilities = torch.nn.functional.sigmoid(self.predicted_logits)

        return self.predicted_probabilities.cpu().detach(), results

    def internal_epoch_start(self, options={}):
        if self._train:
            self.optimizer.zero_grad()

    def internal_epoch_end(self, options={}):
        if self._train:
            self.epoch += 1

            self.optimizer.zero_grad()

        return {}

    def internal_save(self, options={}):
        os.makedirs(os.path.dirname(options["save-path"]), exist_ok=True)
        torch.save(self.model.state_dict(), options["save-path"])

        return {}

    def _create_model(self, options={}):
        return Pathfinder(
            torchvision.models.resnet18(num_classes=int(options["map-dimension"]) ** 2),
            device=self.device)

    def _prepare_data(self, data, options={}):
        self.features = torch.tensor(np.asarray(data, dtype=np.float32), dtype=torch.float32, device=self.device)
        self.features = self.features.reshape(self.features.shape[0], WARCRAFT_IMAGE_HEIGHT, WARCRAFT_IMAGE_WIDTH, WARCRAFT_IMAGE_CHANNELS).permute(0, 3, 1, 2)
