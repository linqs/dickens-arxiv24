import os
import random
import sys
import traceback

import torch
import tqdm

from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from neural.utils import get_torch_device
from roadr.constants import ANNOTATIONS_PATH
from roadr.constants import BASE_RESULTS_DIR
from roadr.constants import BOX_CONFIDENCE_THRESHOLD
from roadr.constants import CLASS_SIZE
from roadr.constants import CONFIG_FILENAME
from roadr.constants import CURRENT_MODEL_NAME
from roadr.constants import IOU_THRESHOLD
from roadr.constants import LABEL_CONFIDENCE_THRESHOLD
from roadr.constants import SUPERVISED_VIDEOS
from roadr.constants import TRAINED_MODEL_NAME
from roadr.constants import TRAINING_HISTORY_FILENAME
from roadr.constants import VIDEO_PARTITIONS
from roadr.utils import box_cxcywh_to_xyxy
from roadr.utils import confusion_matrix
from roadr.utils import detr_loss
from roadr.utils import precision_recall_f1
from roadr.scripts.roadr_dataset import RoadRDataset
from utils import enumerate_hyperparameters
from utils import load_json_file
from utils import seed_everything
from utils import write_csv_file
from utils import write_json_file

HYPERPARAMETERS = {
    'learning-rate-backbone': [1.0e-5, 1.0e-6, 1.0e-7],
    'weight-decay-backbone': [1.0e-4, 1.0e-5, 1.0e-6],
    'learning-rate-transformer': [1.0e-5, 1.0e-6, 1.0e-7],
    'weight-decay-transformer': [1.0e-4, 1.0e-5, 1.0e-6],
}

DATASETS = ["roadr"]

DEFAULT_PARAMETERS = {
    'roadr': {
        'learning-rate-backbone': 1.0e-7,
        'weight-decay-backbone': 1.0e-5,
        'learning-rate-transformer': 1.0e-6,
        'weight-decay-transformer': 1.0e-5,
    }
}

EXPERIMENTS = ["constraint-satisfaction", "semi-supervised"]

SIZES = {
    "constraint-satisfaction": ["15"],
    "semi-supervised": ["02", "04", "08", "15"]
}

SPLITS = ["00"]

RANDOM_SEEDS = 1
SEED = 42
EPOCHS = 25
BATCH_SIZE = 2
MINI_BATCH_SIZE = 2
IMAGE_RESIZE = 1.0
MAX_FRAMES = 0

RUN_HYPERPARAMETER_SEARCH = False
LOAD_CHECKPOINT = True
COMPUTE_PERIOD = 1
STEP_SIZE = 10
GAMMA = 0.5


def train(training_history_path: str,
          epochs: int,
          learning_rate_backbone: float,
          learning_rate_transformer: float,
          weight_decay_backbone: float,
          weight_decay_transformer: float,
          batch_size: int,
          mini_batch_size: int,
          image_resize: float,
          max_frames: int,
          annotations_path: str,
          video_partitions: dict,
          supervised_videos: list,
          load_checkpoint: bool,
          load_model_path: str = None):
    device = get_torch_device()

    train_dataset = RoadRDataset(
        video_partitions["train"],
        supervised_videos,
        annotations_path,
        image_resize,
        max_frames
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(os.cpu_count()) - 2,
        prefetch_factor=4,
        persistent_workers=True
    )

    test_dataset = RoadRDataset(
        video_partitions["test"],
        video_partitions["test"],
        annotations_path,
        image_resize,
        max_frames,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(os.cpu_count()) - 2,
        prefetch_factor=4,
        persistent_workers=True
    )

    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=CLASS_SIZE,
        revision="no_timm",
        ignore_mismatched_sizes=True
    ).to(device)

    if load_checkpoint and os.path.exists(load_model_path):
        model.load_state_dict(torch.load(load_model_path))

    transformer_parameters = []
    for parameter in list(model.parameters()):
        is_in_backbone = False
        for backbone_parameter in list(model.model.backbone.parameters()):
            if parameter is backbone_parameter:
                is_in_backbone = True
                break
        if not is_in_backbone:
            transformer_parameters.append(parameter)

    model.model.unfreeze_backbone()

    optimizer_backbone = torch.optim.AdamW(list(model.model.backbone.parameters()), lr=learning_rate_backbone, weight_decay=weight_decay_backbone)
    optimizer_transformer = torch.optim.AdamW(transformer_parameters, lr=learning_rate_transformer, weight_decay=weight_decay_transformer)

    scheduler_backbone = torch.optim.lr_scheduler.StepLR(optimizer_backbone, step_size=STEP_SIZE, gamma=GAMMA)
    scheduler_transformer = torch.optim.lr_scheduler.StepLR(optimizer_transformer, step_size=STEP_SIZE, gamma=GAMMA)

    max_f1 = -1
    max_model = None
    training_history = []
    for epoch in tqdm.tqdm(range(epochs), "Training Model", leave=True):
        epoch_loss = 0
        total_precision, total_recall, total_f1 = None, None, None
        agent_precision, agent_recall, agent_f1 = None, None, None
        action_precision, action_recall, action_f1 = None, None, None
        loc_precision, loc_recall, loc_f1 = None, None, None

        with tqdm.tqdm(train_loader) as tq:
            tq.set_description("Epoch:{}".format(epoch))

            for step, batch in enumerate(tq):
                optimizer_backbone.zero_grad(set_to_none=True)
                optimizer_transformer.zero_grad(set_to_none=True)

                for mini_batch_index in range(0, int(len(batch[0]) / mini_batch_size)):
                    mini_batch = [b[mini_batch_index * mini_batch_size:(mini_batch_index + 1) * mini_batch_size].to(device) for b in batch]
                    _, mini_batch_pixel_values, mini_batch_pixel_mask, mini_batch_labels, mini_batch_boxes, mini_batch_labels_mask = mini_batch

                    batch_predictions = model(**{"pixel_values": mini_batch_pixel_values, "pixel_mask": mini_batch_pixel_mask})

                    formatted_boxes = torch.zeros(size=batch_predictions["pred_boxes"].shape, device=device)
                    for i in range(len(batch_predictions["pred_boxes"])):
                        formatted_boxes[i] += box_cxcywh_to_xyxy(batch_predictions["pred_boxes"][i])

                    masked_labels = mini_batch_labels * mini_batch_labels_mask
                    masked_predictions = batch_predictions["logits"] * mini_batch_labels_mask

                    loss = detr_loss(formatted_boxes, masked_predictions, mini_batch_boxes, masked_labels)
                    epoch_loss += loss.item()

                    loss.backward()
                optimizer_backbone.step()
                optimizer_transformer.step()

                tq.set_postfix(loss=epoch_loss / ((step + 1) * train_loader.batch_size))
            scheduler_backbone.step()
            scheduler_transformer.step()

        if epoch % COMPUTE_PERIOD == 0:
            tp = [0] * CLASS_SIZE
            fp = [0] * CLASS_SIZE
            fn = [0] * CLASS_SIZE

            with torch.no_grad():
                for batch in test_loader:
                    batch = [b.to(device) for b in batch]

                    _, pixel_values, pixel_mask, labels, boxes, _ = batch
                    for mini_batch_index in range(0, len(pixel_values), mini_batch_size):

                        mini_batch_pixel_values = pixel_values[mini_batch_index:mini_batch_index + mini_batch_size]
                        mini_batch_pixel_mask = pixel_mask[mini_batch_index:mini_batch_index + mini_batch_size]
                        mini_batch_labels = labels[mini_batch_index:mini_batch_index + mini_batch_size]
                        mini_batch_boxes = boxes[mini_batch_index:mini_batch_index + mini_batch_size]

                        batch_predictions = model(**{"pixel_values": mini_batch_pixel_values, "pixel_mask": mini_batch_pixel_mask})

                        indexes = torch.argsort(batch_predictions["logits"][:, :, -1], descending=True)
                        batch_predictions["logits"] = batch_predictions["logits"][torch.arange(len(batch_predictions["logits"])).unsqueeze(1), indexes]
                        batch_predictions["pred_boxes"] = batch_predictions["pred_boxes"][torch.arange(len(batch_predictions["pred_boxes"])).unsqueeze(1), indexes]

                        probabilities = torch.sigmoid(batch_predictions["logits"][:, :, :-1])
                        box_confidences = torch.sigmoid(batch_predictions["logits"][:, :, -1])

                        formatted_boxes = torch.zeros(size=batch_predictions["pred_boxes"].shape, device=device)
                        for i in range(len(batch_predictions["pred_boxes"])):
                            formatted_boxes[i] += box_cxcywh_to_xyxy(batch_predictions["pred_boxes"][i])

                        batch_tp, batch_fp, batch_fn = confusion_matrix(formatted_boxes, probabilities, box_confidences, mini_batch_boxes, mini_batch_labels,
                                                                        IOU_THRESHOLD, LABEL_CONFIDENCE_THRESHOLD, BOX_CONFIDENCE_THRESHOLD)

                        tp = [tp[i] + batch_tp[i] for i in range(CLASS_SIZE)]
                        fp = [fp[i] + batch_fp[i] for i in range(CLASS_SIZE)]
                        fn = [fn[i] + batch_fn[i] for i in range(CLASS_SIZE)]

            total_precision, total_recall, total_f1 = precision_recall_f1(sum(tp), sum(fp), sum(fn))
            agent_precision, agent_recall, agent_f1 = precision_recall_f1(sum(tp[:10]), sum(fp[:10]), sum(fn[:10]))
            action_precision, action_recall, action_f1 = precision_recall_f1(sum(tp[10:29]), sum(fp[10:29]), sum(fn[10:29]))
            loc_precision, loc_recall, loc_f1 = precision_recall_f1(sum(tp[29:]), sum(fp[29:]), sum(fn[29:]))
            print(
                f"Epoch: {epoch}, Loss: {epoch_loss / len(train_dataset)}, \
                Total Precision: {total_precision}, Total Recall: {total_recall}, Total F1: {total_f1}, \
                Agent Precision: {agent_precision}, Agent Recall: {agent_recall}, Agent F1: {agent_f1}, \
                Action Precision: {action_precision}, Action Recall: {action_recall}, Action F1: {action_f1}, \
                Location Precision: {loc_precision}, Location Recall: {loc_recall}, Location F1: {loc_f1}"
            )
            if total_f1 > max_f1:
                max_f1 = total_f1
                max_model = model

        training_history.append([
            epoch_loss / len(train_dataset),
            total_precision, total_recall, total_f1,
            agent_precision, agent_recall, agent_f1,
            action_precision, action_recall, action_f1,
            loc_precision, loc_recall, loc_f1
        ])

        write_csv_file(training_history_path, training_history)

    return {'f1': max_f1, 'training-history': training_history}, max_model


def main():
    for dataset_id in DATASETS:
        hyperparameters = [DEFAULT_PARAMETERS[dataset_id]]
        for experiment_id in EXPERIMENTS:
            for split_id in SPLITS:
                for size_id in SIZES[experiment_id]:
                    out_dir = os.path.join(BASE_RESULTS_DIR, dataset_id, experiment_id, split_id, size_id, "pretrain")
                    if RUN_HYPERPARAMETER_SEARCH and int(split_id) == 0:
                        hyperparameters = enumerate_hyperparameters(HYPERPARAMETERS)
                    elif RUN_HYPERPARAMETER_SEARCH and int(split_id) != 0:
                        hyperparameters = [load_json_file(os.path.join(BASE_RESULTS_DIR, dataset_id, experiment_id, "00", size_id, "pretrain", CONFIG_FILENAME))['network']['parameters']]
                    out_path = os.path.join(out_dir, TRAINED_MODEL_NAME)
                    os.makedirs(out_dir, exist_ok=True)

                    max_f1 = -1
                    if os.path.exists(os.path.join(out_dir, CONFIG_FILENAME)):
                        max_f1 = load_json_file(os.path.join(out_dir, CONFIG_FILENAME))['network']['results']['f1']

                    seed_everything(SEED)
                    for parameters in hyperparameters:
                        for seed_index in range(RANDOM_SEEDS):
                            if RUN_HYPERPARAMETER_SEARCH and int(split_id) == 0:
                                os.makedirs(os.path.join(out_dir, "hyperparameter-search"), exist_ok=True)
                                out_path = os.path.join(out_dir, "hyperparameter-search", "-".join([f"%s::%0.10f" % (k, v) for k, v in parameters.items()]) + f"-seed::{seed_index}.pt")

                            checkpoint_history = []
                            if os.path.exists(out_path):
                                if LOAD_CHECKPOINT:
                                    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

                                    checkpoint_dir = os.path.join(out_dir, "checkpoints", f"{len(os.listdir(os.path.join(out_dir, 'checkpoints')))}")
                                    os.makedirs(checkpoint_dir, exist_ok=True)

                                    os.system("cp %s %s" % (os.path.join(out_dir, CURRENT_MODEL_NAME), checkpoint_dir))
                                    os.system("cp %s %s" % (os.path.join(out_dir, TRAINED_MODEL_NAME), checkpoint_dir))
                                    os.system("cp %s %s" % (os.path.join(out_dir, TRAINING_HISTORY_FILENAME), checkpoint_dir))
                                    os.system("cp %s %s" % (os.path.join(out_dir, CONFIG_FILENAME), checkpoint_dir))

                                    config = load_json_file(os.path.join(out_dir, CONFIG_FILENAME))
                                    config_checkpoint_history = config.pop('checkpoint_history')

                                    checkpoint_history.append(config)
                                    checkpoint_history.extend(config_checkpoint_history)
                                else:
                                    print(f"Found pretrained models for {out_path}. Skipping...")
                                    continue

                            restart_budget = 3
                            while restart_budget > 0:
                                seed = random.randrange(2 ** 64)
                                torch.manual_seed(seed)

                                try:
                                    results, model = train(os.path.join(out_dir, TRAINING_HISTORY_FILENAME), EPOCHS,
                                                           parameters['learning-rate-backbone'], parameters['learning-rate-transformer'],
                                                           parameters['weight-decay-backbone'], parameters['weight-decay-transformer'],
                                                           BATCH_SIZE, MINI_BATCH_SIZE, IMAGE_RESIZE, MAX_FRAMES, ANNOTATIONS_PATH,
                                                           VIDEO_PARTITIONS, SUPERVISED_VIDEOS[size_id], LOAD_CHECKPOINT,
                                                           os.path.join(out_dir, CURRENT_MODEL_NAME))
                                except Exception:
                                    print(f"Error training model with seed {seed}. Restarting...")
                                    print(traceback.format_exc())
                                    restart_budget -= 1
                                    continue

                                if RUN_HYPERPARAMETER_SEARCH:
                                    torch.save(model.state_dict(), out_path)
                                else:
                                    torch.save(model.state_dict(), os.path.join(out_dir, CURRENT_MODEL_NAME))

                                if results['f1'] <= max_f1:
                                    break

                                max_f1 = results['f1']

                                if not RUN_HYPERPARAMETER_SEARCH:
                                    torch.save(model.state_dict(), os.path.join(out_dir, TRAINED_MODEL_NAME))

                                total_epochs = EPOCHS
                                if len(checkpoint_history) > 0:
                                    total_epochs += checkpoint_history[0]['network']['total-epochs']

                                results_config = {
                                    'seed': seed,
                                    'num-random-seeds': RANDOM_SEEDS,
                                    'network': {
                                        'total-epochs': total_epochs,
                                        'run-epochs': EPOCHS,
                                        'batch-size': BATCH_SIZE,
                                        'image-resize': IMAGE_RESIZE,
                                        'max-frames': MAX_FRAMES,
                                        **parameters,
                                        'results': results,
                                    },
                                    'checkpoint_history': checkpoint_history,
                                }

                                write_json_file(os.path.join(out_dir, CONFIG_FILENAME), results_config)
                                break


if __name__ == "__main__":
    main()
