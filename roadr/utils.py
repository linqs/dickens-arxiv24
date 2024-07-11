import os
import sys

import torch
import torchvision

from scipy.optimize import linear_sum_assignment
from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

from utils import load_csv_file


def load_constraint_file(path):
    raw_constraints = load_csv_file(path)

    constraints = []
    for index_i in range(len(raw_constraints) - 1):
        for index_j in range(len(raw_constraints[index_i]) - 1):
            constraints.append([index_i, index_j, int(raw_constraints[index_i + 1][index_j + 1])])

    return constraints


def box_cxcywh_to_xyxy(x):
    """
    Convert bounding box format from [x_c, y_c, w, h] to [x0, y0, x1, y1]
    where x_c, y_c are the center coordinates, w, h are the width and height of the box.
    :param x: The bounding box in [x_c, y_c, w, h] format.
    :return: The bounding box in [x0, y0, x1, y1] format.
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def count_violated_constraints(constraints, dataset, frame_ids, pred_probabilities, pred_boxes_confidence,
                               label_confidence_threshold, box_confidence_threshold, class_size):
    """
    Counts the number of violated pairwise constraints.
    :param constraints: List of pairwise constraints.
    :param dataset: Dataset for which the predictions were computed.
    :param frame_ids: List of frame ids (video name, frame name).
    :param pred_probabilities: List containing the predictions.
    :param pred_boxes_confidence: List containing the box confidence scores.
    :param label_confidence_threshold: Threshold for the label confidence score.
    :param box_confidence_threshold: Threshold for the box confidence score.
    :param class_size: Number of classes.
    :return: Number of violated constraints, Number of frames with a violation.
    """
    num_constraint_violations = 0
    num_frames_with_violation = 0

    constraint_violation_dict = {
        "simplex": {
            "no-agent": 0,
            "no-location": 0
        }
    }

    for frame_id in frame_ids:
        frame_index = dataset.frame_indexes[(frame_id[0], frame_id[1])]

        frame_violation = False

        for pred_box_index in range(len(pred_probabilities[frame_index])):
            if pred_boxes_confidence[frame_index][pred_box_index] < box_confidence_threshold:
                break

            detected_label = [label for label in range(class_size) if pred_probabilities[frame_index][pred_box_index][label] > label_confidence_threshold]

            has_agent = False
            has_location = False

            for index_i, class_i in enumerate(detected_label):
                if class_i < 10:
                    has_agent = True
                if class_i == 8 or class_i == 9:
                    has_location = True
                if class_i > 28:
                    has_location = True

                for index_j, class_j in enumerate(detected_label[index_i + 1:]):
                    if constraints[class_i * class_size + class_j][2] == 1:
                        continue

                    if class_i not in constraint_violation_dict:
                        constraint_violation_dict[class_i] = {}
                    if class_j not in constraint_violation_dict[class_i]:
                        constraint_violation_dict[class_i][class_j] = 0

                    constraint_violation_dict[class_i][class_j] += 1
                    num_constraint_violations += 1
                    frame_violation = True

            if not has_agent:
                constraint_violation_dict["simplex"]["no-agent"] += 1
                num_constraint_violations += 1
                frame_violation = True
            if not has_location:
                constraint_violation_dict["simplex"]["no-location"] += 1
                num_constraint_violations += 1
                frame_violation = True

        if frame_violation:
            num_frames_with_violation += 1

    return num_constraint_violations, num_frames_with_violation, constraint_violation_dict


def precision_recall_f1(tp, fp, fn):
    """
    Computes the precision, recall, and f1 score for the given true positives, false positives, and false negatives.
    :param tp: The true positives.
    :param fp: The false positives.
    :param fn: The false negatives.
    :return: The precision, recall, and f1 score.
    """
    precision = 0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


def confusion_matrix(pred_boxes, pred_probabilities, pred_boxes_confidence, truth_boxes, truth_labels,
                     iou_threshold, label_confidence_threshold, box_confidence_threshold):
    """
    Computes the confusion matrix for the given predictions and targets.
    :param pred_boxes: The predicted boxes. tensor of shape [batch_size, num_queries, 4]
    :param pred_probabilities: The predicted class probabilities. tensor of shape [batch_size, num_queries, num_classes]
    :param pred_boxes_confidence: The predicted box confidences. tensor of shape [batch_size, num_queries]
    :param truth_boxes: The ground truth boxes. tensor of shape [batch_size, num_target_boxes, 4]
    :param truth_labels: The ground truth labels. tensor of shape [batch_size, num_target_boxes, num_classes]
    :param iou_threshold: Threshold for the IoU.
    :param label_confidence_threshold: Threshold for the class confidence.
    :param box_confidence_threshold: Threshold for the box confidence.
    """
    tp = [0] * (len(pred_probabilities[0]) - 1)
    fp = [0] * (len(pred_probabilities[0]) - 1)
    fn = [0] * (len(pred_probabilities[0]) - 1)

    for pred_frame_boxes, pred_frame_probabilities, pred_frame_boxes_confidence, truth_frame_boxes, truth_frame_labels in zip(pred_boxes, pred_probabilities, pred_boxes_confidence, truth_boxes, truth_labels):
        matched_detection_indexes = set()
        for pred_detection_box, pred_detection_probabilities, pred_detected_box_confidence in zip(pred_frame_boxes, pred_frame_probabilities, pred_frame_boxes_confidence):
            if pred_detected_box_confidence < box_confidence_threshold:
                continue

            truth_box_index = match_box(pred_detection_box, truth_frame_boxes, matched_detection_indexes, iou_threshold)

            truth_detection_label = [0] * (len(pred_detection_probabilities) - 1)
            if truth_box_index is not None:
                matched_detection_indexes.add(truth_box_index)
                truth_detection_label = truth_frame_labels[truth_box_index]

            pred_detected_label = pred_detection_probabilities.gt(label_confidence_threshold).int()
            for class_index in range(len(pred_detection_probabilities) - 1):
                if truth_detection_label[class_index] == 1 and pred_detected_label[class_index] == 1:
                    tp[class_index] += 1
                elif truth_detection_label[class_index] == 1 and pred_detected_label[class_index] == 0:
                    fn[class_index] += 1
                elif truth_detection_label[class_index] == 0 and pred_detected_label[class_index] == 1:
                    fp[class_index] += 1

        for detection_index in range(len(truth_frame_labels)):
            if detection_index in matched_detection_indexes:
                continue
            if truth_frame_labels[detection_index].sum() == 0:
                break
            for class_index in range(len(pred_frame_probabilities[0]) - 1):
                if truth_frame_labels[detection_index][class_index] == 1:
                    fn[class_index] += 1

    return tp, fp, fn


def match_box(pred_box, truth_boxes, skip_indexes, iou_threshold, epsilon=1e-6):
    """
    Matches a predicted box to a ground truth box.
    :param pred_box: A single box of shape (1, 4).
    :param truth_boxes: A tensor of shape (N, 4) containing the ground truth boxes.
    :param skip_indexes: A set of indexes of boxes that have already been matched.
    :param iou_threshold: Threshold for the IoU.
    :param epsilon: Small value to avoid division by zero.
    :return: The index of the matched ground truth box or None if no match was found.
    """
    max_box_iou = 0
    max_truth_box_index = None
    for truth_box_index in range(len(truth_boxes)):
        if truth_box_index in skip_indexes:
            continue

        truth_box = truth_boxes[truth_box_index].reshape(1, 4)
        pred_box = pred_box.reshape(1, 4)
        if truth_box.sum() < epsilon:
            break

        box_iou = single_box_iou(truth_box, pred_box)
        if box_iou > iou_threshold:
            if box_iou > max_box_iou:
                max_box_iou = box_iou
                max_truth_box_index = truth_box_index

    return max_truth_box_index


def detr_loss(pred_boxes, pred_logits, truth_boxes, truth_labels, bce_weight: int = 1, giou_weight: int = 5, l1_weight: int = 2):
    """
    Computes the detr loss for the given outputs and targets.
    First compute the matching between the predictions and the ground truth using hungarian sort.
    Then compute the classification loss, pairwise generalized box iou loss, and pairwise l1 loss using this matching.
    Finally, compute the total loss as a weighted sum of the three losses.
    :param pred_boxes: Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
    :param pred_logits: Tensor of dim [batch_size, num_queries, num_classes] with predicted class logits
    :param truth_boxes: This is a list of targets bounding boxes (len(targets) = batch_size), where each entry is
            a tensor of dim [num_target_boxes, 4] containing the target box coordinates
    :param truth_labels: Tensor of dim [batch_size, num_queries, num_classes] with target class labels
    :param bce_weight: Weight for the binary cross entropy loss
    :param giou_weight: Weight for the generalized box iou loss
    :param l1_weight: Weight for the l1 loss
    :return: Detr loss and a dict containing the computed losses
    """
    # First compute the matching between the predictions and the ground truth.
    matching = _hungarian_match(pred_boxes, truth_boxes, l1_weight, giou_weight)

    # Compute the classification loss using the matching.
    bce_loss = binary_cross_entropy_with_logits(pred_logits, truth_labels, matching)

    # Compute the bounding box loss using the matching.
    giou_loss = pairwise_generalized_box_iou(pred_boxes, truth_boxes, matching)

    # Compute the bounding box l2 loss using the matching.
    l1_loss = pairwise_l1_loss(pred_boxes, truth_boxes, matching)

    return bce_weight * bce_loss + giou_weight * giou_loss + l1_weight * l1_loss


def binary_cross_entropy_with_logits(outputs, truth, indices) -> torch.Tensor:
    """
    Computes the binary cross entropy loss for the given outputs and targets.
    The targets are aligned with the outputs using the given indices.
    :param outputs: the outputs of the model. dict of tensors of shape [batch_size, num_queries, num_classes]
    :param truth: the ground truth labels. tensor of shape [batch_size, num_queries, num_classes]
    :param indices: A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
    :return: the computed binary cross entropy loss
    """
    # Align the outputs and targets using the given indices and flatten.
    aligned_outputs = torch.stack([outputs[i, indices[i, 0, :], :] for i in range(outputs.shape[0])]).flatten(0, 1)
    aligned_truth = torch.stack([truth[i, indices[i, 1, :], :] for i in range(truth.shape[0])]).flatten(0, 1).to(torch.float32)

    return torch.nn.functional.binary_cross_entropy_with_logits(aligned_outputs, aligned_truth)


def pairwise_l1_loss(boxes1, boxes2, indices) -> torch.Tensor:
    """
    Computes the pairwise l1 loss for the given outputs and targets aligned using the given indices.
    :param boxes1: The first set of boxes. tensor of shape [batch_size, num_queries, 4]
    :param boxes2: The second set of boxes. tensor of shape [batch_size, num_queries, 4]
    :param indices: The indices used to align the boxes. A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
    :return: The computed pairwise l1 loss.
    """
    # Align the boxes using the given indices.
    aligned_boxes1 = torch.stack([boxes1[i, indices[i, 0, :], :] for i in range(boxes1.shape[0])]).flatten(0, 1)
    aligned_boxes2 = torch.stack([boxes2[i, indices[i, 1, :], :] for i in range(boxes2.shape[0])]).flatten(0, 1)

    return torch.nn.functional.l1_loss(aligned_boxes1, aligned_boxes2)


def pairwise_generalized_box_iou(boxes1, boxes2, indices) -> torch.Tensor:
    """
    Computes the pairwise generalized box iou loss for the given outputs and targets aligned using the given indices.
    :param boxes1: The first set of boxes. tensor of shape [batch_size, num_queries, 4]
    :param boxes2: The second set of boxes. tensor of shape [batch_size, num_queries, 4]
    :param indices: The indices used to align the boxes. A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
    :return: The computed pairwise generalized box iou loss.
    """
    # Align the boxes using the given indices.
    aligned_boxes1 = torch.stack([boxes1[i, indices[i, 0, :], :] for i in range(boxes1.shape[0])]).flatten(0, 1)
    aligned_boxes2 = torch.stack([boxes2[i, indices[i, 1, :], :] for i in range(boxes2.shape[0])]).flatten(0, 1)

    return (1 - torch.diag(generalized_box_iou(aligned_boxes1, aligned_boxes2)).sum() / aligned_boxes1.shape[0])


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    :param boxes1: Tensor of shape [N, 4]
    :param boxes2: Tensor of shape [M, 4]
    :return: A [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def box_iou(boxes1, boxes2):
    """
    Compute the intersection over union of two set of boxes, each box is [x0, y0, x1, y1].
    This is modified from torchvision to also return the union.
    :param boxes1: Tensor of shape [N, 4]
    :param boxes2: Tensor of shape [M, 4]
    :return: A [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    area1 = torchvision.ops.boxes.box_area(boxes1)
    area2 = torchvision.ops.boxes.box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def single_box_iou(box1, box2):
    """
    Compute the intersection over union of two boxes, each box is [x0, y0, x1, y1].
    :param box1: Tensor of shape [4]
    :param box2: Tensor of shape [4]
    :return: A scalar representing the iou of the two boxes.
    """
    area1 = torchvision.ops.boxes.box_area(box1)
    area2 = torchvision.ops.boxes.box_area(box2)

    lt = torch.max(box1[:, :2], box2[:, :2])  # [2]
    rb = torch.min(box1[:, 2:], box2[:, 2:])  # [2]

    wh = (rb - lt).clamp(min=0)  # [2]
    inter = wh[:, 0] * wh[:, 1]  # [1]

    union = area1 + area2 - inter

    iou = inter / union
    return iou


def _hungarian_match(pred_boxes, truth_boxes, l1_weight, giou_weight):
    """
    Computes an assignment between the predictions and the truth boxes.
    :param pred_boxes: Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates in [x0, y0, x1, y1] format.
    :parm truth_boxes: This is a list of targets bounding boxes (len(targets) = batch_size), where each entry is
            a tensor of dim [num_target_boxes, 4] containing the target box coordinates in [x0, y0, x1, y1] format.
    :return: A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
    """
    batch_size, num_queries = pred_boxes.shape[:2]

    # We flatten to compute the cost matrices in a batch
    out_bbox = pred_boxes.flatten(0, 1)  # [batch_size * num_queries, 4]

    # Also concat the target boxes
    tgt_bbox = truth_boxes.flatten(0, 1)  # [batch_size * num_target_boxes, 4]

    # Compute the L1 cost between boxes
    cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

    # Compute the giou cost between boxes
    cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)

    # Final cost matrix
    C = l1_weight * cost_bbox + giou_weight * cost_giou
    C = C.view(batch_size, num_queries, -1).cpu()

    sizes = [len(v) for v in truth_boxes]
    indices = [linear_sum_assignment(c[i].detach().numpy()) for i, c in enumerate(C.split(sizes, -1))]

    return torch.stack([torch.stack((torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))) for i, j in indices])


def save_images(label_map, height, width, image_dir, out_dir, frame_ids, boxes, labels, boxes_confidence,
                label_confidence_threshold, box_confidence_threshold, images=10, color="red"):
    """
    Saves images with bounding boxes.
    :param label_map: The label map.
    :param height: The height of the images.
    :param width: The width of the images.
    :param image_dir: The image directory.
    :param out_dir: The output directory.
    :param frame_ids: The frame ids.
    :param boxes: The boxes.
    :param labels: The labels.
    :param boxes_confidence: The box confidence.
    :param label_confidence_threshold: Threshold for the class confidence.
    :param box_confidence_threshold: Threshold for the box confidence.
    :param images: Maximum number of images to save.
    :param color: Color of the bounding boxes.
    """
    saved_image_dict = {}

    for frame_id, frame_boxes, frame_labels, frame_boxes_confidence in zip(frame_ids, boxes, labels, boxes_confidence):
        if frame_id[0] not in saved_image_dict:
            saved_image_dict[frame_id[0]] = 0
            os.makedirs(os.path.join(out_dir, frame_id[0]), exist_ok=True)

        if saved_image_dict[frame_id[0]] >= images:
            continue
        saved_image_dict[frame_id[0]] += 1

        detected_boxes = []
        detected_labels = []
        for index, box_confidence in enumerate(frame_boxes_confidence):
            if box_confidence < box_confidence_threshold:
                break
            detected_boxes.append([frame_boxes[index][0] * width, frame_boxes[index][1] * height, frame_boxes[index][2] * width, frame_boxes[index][3] * height])
            detected_labels.append(frame_labels[index].gt(label_confidence_threshold).float())

        frame = torchvision.io.read_image(os.path.join(image_dir, frame_id[0], frame_id[1]))

        fig, ax = plt.subplots()
        ax.imshow(frame.permute(1, 2, 0))

        for box, label in zip(detected_boxes, detected_labels):
            rectangle = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor=color, facecolor="none")
            ax.add_patch(rectangle)

            text = [label_map[index] for index in range(len(label)) if label[index] == 1]
            ax.text(box[0], box[3], "\n".join(text), fontsize=7, color="blue", bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=-0.1))

        plt.savefig(os.path.join(out_dir, frame_id[0], frame_id[1]))
        plt.close()
