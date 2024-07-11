import os
import sys

import torch
import tqdm

from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

from roadr.constants import ANNOTATIONS_PATH
from roadr.constants import BASE_RESULTS_DIR
from roadr.constants import BOX_CONFIDENCE_THRESHOLD
from roadr.constants import CLASS_SIZE
from roadr.constants import CONSTRAINTS_PATH
from roadr.constants import IOU_THRESHOLD
from roadr.constants import IMAGE_HEIGHT
from roadr.constants import IMAGE_WIDTH
from roadr.constants import LABEL_CONFIDENCE_THRESHOLD
from roadr.constants import LABEL_MAPPING
from roadr.constants import RGB_IMAGES_DIR
from roadr.constants import TRAINED_MODEL_NAME
from roadr.constants import VIDEO_PARTITIONS
from roadr.utils import box_cxcywh_to_xyxy
from roadr.utils import confusion_matrix
from roadr.utils import count_violated_constraints
from roadr.utils import load_constraint_file
from roadr.utils import precision_recall_f1
from roadr.utils import save_images
from roadr.scripts.roadr_dataset import RoadRDataset
from neural.utils import get_torch_device
from utils import load_json_file
from utils import write_json_file


BASE_MODEL_NAME = "roadr.json"

DATASETS = ["roadr"]
EXPERIMENTS = ["pretrain"]
SETTINGS = ["constraint-satisfaction"]
SIZES = {
    "constraint-satisfaction": ["15"],
    "semi-supervised": ["02", "04", "08", "15"]
}
SPLITS = ["00"]

RESULTS_FILENAME = "results.json"

IMAGE_RESIZE = 1.0
MAX_FRAMES = 0
BATCH_SIZE = 4


def calculate_results(dataset, predictions, output_dir):
    frame_ids = predictions["ids"]
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

    num_constraint_violations, num_frames_with_violation, constraint_violation_dict = count_violated_constraints(
        load_constraint_file(CONSTRAINTS_PATH), dataset, frame_ids, pred_probabilities, pred_boxes_confidence,
        LABEL_CONFIDENCE_THRESHOLD, BOX_CONFIDENCE_THRESHOLD, CLASS_SIZE
    )

    percent_frames_with_violation = num_frames_with_violation / len(frame_ids)

    save_images(LABEL_MAPPING, IMAGE_HEIGHT, IMAGE_WIDTH, RGB_IMAGES_DIR, os.path.join(output_dir, "images", "predicted"),
                predictions["ids"], pred_boxes, pred_probabilities, pred_boxes_confidence,
                LABEL_CONFIDENCE_THRESHOLD, BOX_CONFIDENCE_THRESHOLD, color="blue")

    save_images(LABEL_MAPPING, IMAGE_HEIGHT, IMAGE_WIDTH, RGB_IMAGES_DIR, os.path.join(output_dir, "images", "truth"),
                predictions["ids"], truth_boxes, truth_labels, truth_boxes_confidence,
                LABEL_CONFIDENCE_THRESHOLD, BOX_CONFIDENCE_THRESHOLD, color="red")

    return [total_precision, total_recall, total_f1, agent_precision, agent_recall, agent_f1, action_precision, action_recall, action_f1, loc_precision, loc_recall, loc_f1, num_frames_with_violation, percent_frames_with_violation]


def evaluate(dataset, loader, model, device):
    predictions = {
        "ids": [],
        "pred_probabilities": [],
        "pred_boxes": [],
        "pred_boxes_confidence": [],
        "truth_labels": [],
        "truth_boxes": [],
    }

    with tqdm.tqdm(loader) as tq:
        tq.set_description("Evaluating")
        for step, batch in enumerate(tq):
            batch = [b.to(device) for b in batch]
            batch_indexes, batch_pixel_values, batch_pixel_mask, batch_labels, batch_boxes, batch_labels_mask = batch

            with torch.no_grad():
                batch_predictions = model(**{"pixel_values": batch_pixel_values, "pixel_mask": batch_pixel_mask})
                batch_probabilities = torch.sigmoid(batch_predictions["logits"][:, :, :-1])
                batch_confidences = torch.sigmoid(batch_predictions["logits"][:, :, -1].reshape(batch_probabilities.shape[0], batch_probabilities[0].shape[0], 1))
                batch_pred_boxes = [box_cxcywh_to_xyxy(box) for box in batch_predictions["pred_boxes"]]

                for frame_index in range(len(batch_indexes)):
                    predictions["ids"].append(dataset.frame_ids[batch_indexes[frame_index].item()])
                    predictions["pred_probabilities"].append(batch_probabilities[frame_index].cpu().numpy().tolist())
                    predictions["pred_boxes"].append(batch_pred_boxes[frame_index].cpu().numpy().tolist())
                    predictions["pred_boxes_confidence"].append(batch_confidences[frame_index].cpu().numpy().tolist())
                    predictions["truth_labels"].append(batch_labels[frame_index][:, :-1].cpu().numpy().tolist())
                    predictions["truth_boxes"].append(batch_boxes[frame_index].cpu().numpy().tolist())

    return predictions


def main():
    device = get_torch_device()

    dataset = RoadRDataset(
        VIDEO_PARTITIONS["test"],
        VIDEO_PARTITIONS["test"],
        ANNOTATIONS_PATH,
        IMAGE_RESIZE,
        MAX_FRAMES,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=int(os.cpu_count()) - 2,
        prefetch_factor=4
    )

    for dataset_name in DATASETS:
        for setting in SETTINGS:
            for split in SPLITS:
                for size in SIZES[setting]:
                    for experiment in EXPERIMENTS:
                        output_dir = os.path.join(BASE_RESULTS_DIR, dataset_name, setting, split, size, experiment)
                        os.makedirs(output_dir, exist_ok=True)

                        if os.path.exists(os.path.join(output_dir, RESULTS_FILENAME)):
                            print("Results already calculated: ", output_dir)
                            continue

                        if os.path.exists(os.path.join(output_dir, "predictions.json")):
                            predictions = load_json_file(os.path.join(output_dir, "predictions.json"))
                        else:
                            model = DetrForObjectDetection.from_pretrained(
                                "facebook/detr-resnet-50",
                                num_labels=CLASS_SIZE,
                                revision="no_timm",
                                ignore_mismatched_sizes=True
                            ).to(device)

                            model.load_state_dict(torch.load(os.path.join(output_dir, TRAINED_MODEL_NAME)))
                            model.eval()
                            predictions = evaluate(dataset, loader, model, device)
                            write_json_file(os.path.join(output_dir, "predictions.json"), predictions)

                        results = calculate_results(dataset, predictions, output_dir)
                        write_json_file(os.path.join(output_dir, RESULTS_FILENAME), {"results": results})


if __name__ == "__main__":
    main()