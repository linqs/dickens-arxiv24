#!/usr/bin/env python3
import os
import sys

import numpy as np
import pslpython.deeppsl.model
import torch.nn

from transformers import DetrForObjectDetection
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from neural.utils import get_torch_device
from roadr.constants import ANNOTATIONS_PATH
from roadr.constants import BASE_CLI_DIR
from roadr.constants import BATCH_SIZE
from roadr.constants import CLASS_SIZE
from roadr.constants import QUERY_SIZE
from roadr.constants import TRAINED_MODEL_NAME
from roadr.constants import VIDEO_PARTITIONS
from roadr.scripts.roadr_dataset import RoadRDataset
from roadr.utils import box_cxcywh_to_xyxy
from roadr.utils import detr_loss
from utils import seed_everything
from utils import write_json_file


SYMBOLIC_LABELS_PATH = os.path.join(BASE_CLI_DIR, "inferred-predicates", "LABEL.txt")


class RoadRDETRNeuPSL(pslpython.deeppsl.model.DeepModel):
    def __init__(self):
        super().__init__()
        self.output_dir = None

        self.model = None
        self.dataset = None
        self.dataloader = None
        self.optimizer = None
        self.scheduler = None

        self.batch_index = -1
        self.iterator = None
        self.batch = None

        self.batch_logits = None
        self.batch_probabilities = None
        self.batch_boxes = None
        self.batch_boxes_confidence = None
        self.batch_symbolic_gradients = None

        self.ids = None
        self.boxes = None
        self.probabilities = None
        self.boxes_confidence = None
        self.truth_boxes = None
        self.truth_labels = None

        self.device = get_torch_device()
        seed_everything()

    def internal_init(self, application, options={}):
        partition = "train" if application == "learning" else options["eval"]
        shuffle = True if application == "learning" else False

        self.dataset = RoadRDataset(
            VIDEO_PARTITIONS[partition],
            VIDEO_PARTITIONS[partition],
            ANNOTATIONS_PATH,
            float(options["image-resize"]),
            int(options["max-frames"])
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=int(os.cpu_count()) - 2,
            prefetch_factor=4,
            persistent_workers=True
        )

        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=CLASS_SIZE,
            revision="no_timm",
            ignore_mismatched_sizes=True
        ).to(self.device)

        print("Number of batches: ", len(self.dataloader))
        print("Device: ", self.device)

        self.output_dir = options["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

        pretrained_path = os.path.join(options["pretrained_dir"], TRAINED_MODEL_NAME)
        if os.path.isfile(pretrained_path):
            print("Trained model found: ", pretrained_path)
            self.model.load_state_dict(torch.load(pretrained_path, map_location=self.device))

        if application == "learning":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(options["learning-rate"]), weight_decay=float(options["weight-decay"]))
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(options["step-size"]), gamma=float(options["gamma"]))

        self.batch_symbolic_gradients = torch.zeros(size=(BATCH_SIZE, QUERY_SIZE, CLASS_SIZE + 1),
                                                    dtype=torch.float32, device=self.device, requires_grad=False)

        self.batch_probabilities = torch.zeros(size=(BATCH_SIZE, QUERY_SIZE, CLASS_SIZE),
                                               dtype=torch.float32, device=self.device, requires_grad=False)
        self.batch_boxes = torch.zeros(size=(BATCH_SIZE, QUERY_SIZE, 4),
                                       dtype=torch.float32, device=self.device, requires_grad=False)
        self.batch_boxes_confidence = torch.zeros(size=(BATCH_SIZE, QUERY_SIZE, 1),
                                                  dtype=torch.float32, device=self.device, requires_grad=False)

        extra_batch = 0 if len(self.dataset) % BATCH_SIZE == 0 else 1
        dataset_size = (len(self.dataset) // BATCH_SIZE + extra_batch) * BATCH_SIZE

        self.ids = []
        self.probabilities = np.zeros(shape=(dataset_size, QUERY_SIZE, CLASS_SIZE), dtype=np.float32)
        self.boxes = np.zeros(shape=(dataset_size, QUERY_SIZE, 4),  dtype=np.float32)
        self.boxes_confidence = np.zeros(shape=(dataset_size, QUERY_SIZE, 1), dtype=np.float32)
        self.truth_boxes = np.zeros(shape=(dataset_size, QUERY_SIZE, 4), dtype=np.float32)
        self.truth_labels = np.zeros(shape=(dataset_size, QUERY_SIZE, CLASS_SIZE), dtype=np.int32)

        return {}

    def internal_fit(self, data, gradients, options={}):
        self.batch_symbolic_gradients = float(options["alpha"]) * gradients[:, :CLASS_SIZE + 1].reshape(BATCH_SIZE, QUERY_SIZE, CLASS_SIZE + 1)

        _, _, _, labels, boxes, masked_labels = self.batch

        masked_gradients = self.batch_symbolic_gradients * (1 - masked_labels)
        self.batch_logits.backward(masked_gradients, retain_graph=True)

        masked_logits = self.batch_logits * masked_labels
        masked_labels = labels * masked_labels
        loss, results = (1 - float(options["alpha"])) * detr_loss(self.batch_boxes, masked_logits, boxes, masked_labels)
        loss.backward(retain_graph=True)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)

        self.optimizer.step()
        self.optimizer.zero_grad()

        return results

    def internal_predict(self, data, options={}):
        indexes, pixel_values, pixel_mask, labels, boxes, _ = self.batch

        if self._train:
            self.model.train()
            batch_predictions = self.model(**{"pixel_values": pixel_values, "pixel_mask": pixel_mask})
        else:
            for index in range(len(indexes)):
                self.ids.append(self.dataset.frame_ids[indexes[index]])
                self.truth_boxes[self.batch_index * BATCH_SIZE + index] = boxes[index].detach().cpu().numpy()
                self.truth_labels[self.batch_index * BATCH_SIZE + index] = labels[index][:, :-1].detach().cpu().numpy()

            self.model.eval()
            with torch.no_grad():
                batch_predictions = self.model(**{"pixel_values": pixel_values, "pixel_mask": pixel_mask})

        self.batch_logits = batch_predictions["logits"]

        self.batch_probabilities[:batch_predictions["logits"].shape[0]] = torch.sigmoid(batch_predictions["logits"][:, :, :-1])
        for index in range(batch_predictions["logits"].shape[0]):
            self.batch_boxes[index] = box_cxcywh_to_xyxy(batch_predictions["pred_boxes"][index])
        self.batch_boxes_confidence[:batch_predictions["logits"].shape[0]] = torch.sigmoid(batch_predictions["logits"][:, :, -1].reshape(batch_predictions["logits"].shape[0], QUERY_SIZE, 1))

        if batch_predictions["logits"].shape[0] < BATCH_SIZE:
            self.batch_probabilities[batch_predictions["logits"].shape[0]:] = 0.0
            self.batch_boxes[batch_predictions["logits"].shape[0]:] = 0.0
            self.batch_boxes_confidence[batch_predictions["logits"].shape[0]:] = 0.0

        return torch.cat((self.batch_probabilities, self.batch_boxes_confidence, self.batch_boxes), dim=-1).flatten(start_dim=0, end_dim=1).cpu().detach().numpy().tolist(), {}

    def internal_eval(self, data, options={}):
        if not self._train:
            self.store_batch_results()

        return 0.0

    def internal_epoch_start(self, options={}):
        self.batch_index = -1
        self.iterator = iter(self.dataloader)
        self.next_batch()

        if self._train:
            self.optimizer.zero_grad()

    def internal_is_epoch_complete(self, options={}):
        if self.batch is None:
            return True
        return False

    def internal_epoch_end(self, options={}):
        if self._train:
            self.scheduler.step()
            self.save()
            return

        predictions = {
            "ids": self.ids,
            "pred_boxes": self.boxes.tolist(),
            "pred_probabilities": self.probabilities.tolist(),
            "pred_boxes_confidence": self.boxes_confidence.tolist(),
            "truth_boxes": self.truth_boxes.tolist(),
            "truth_labels": self.truth_labels.tolist()
        }

        write_json_file(os.path.join(self.output_dir, "predictions.json"), predictions, indent=None)

    def internal_next_batch(self, options={}):
        self.batch = next(self.iterator, None)
        if self.batch is not None:
            self.batch_index += 1
            self.batch = [b.to(self.device) for b in self.batch]

    def internal_save(self, options={}):
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, TRAINED_MODEL_NAME))

    def store_batch_results(self):
        predictions_index = 0
        with open(SYMBOLIC_LABELS_PATH, "r") as file:
            for line in file.readlines():
                batch_index = predictions_index // ((CLASS_SIZE + 5) * QUERY_SIZE)
                query_index = predictions_index % ((CLASS_SIZE + 5) * QUERY_SIZE) // (CLASS_SIZE + 5)
                class_index = predictions_index % (CLASS_SIZE + 5)

                if class_index > CLASS_SIZE:
                    self.boxes[self.batch_index * BATCH_SIZE + batch_index][query_index][class_index - CLASS_SIZE - 1] = self.batch_boxes[batch_index][query_index][class_index - CLASS_SIZE - 1]
                elif class_index == CLASS_SIZE:
                    self.boxes_confidence[self.batch_index * BATCH_SIZE + batch_index][query_index][0] = self.batch_boxes_confidence[batch_index][query_index][0]
                else:
                    self.probabilities[self.batch_index * BATCH_SIZE + batch_index][query_index][class_index] = float(line.strip().split("\t")[-1])

                predictions_index += 1
