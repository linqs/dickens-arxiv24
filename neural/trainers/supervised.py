import copy
import os
import sys
import torch
import tqdm

from torch.utils.data import Dataset

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(THIS_DIR, '..', '..'))


def masked_binary_cross_entropy_with_logits(logits, labels):
    masked_labels = labels.clone()
    masked_labels[labels == -1] = 0
    masked_logits = logits.clone()
    masked_logits[labels == -1] = 0
    return torch.nn.functional.binary_cross_entropy_with_logits(masked_logits, masked_labels)


LOSSES = {
    "cross_entropy": torch.nn.functional.cross_entropy,
    "binary_cross_entropy": torch.nn.functional.binary_cross_entropy_with_logits,
    "masked_binary_cross_entropy_with_logits": masked_binary_cross_entropy_with_logits
}


def categorical_accuracy(logits, labels):
    return torch.sum(torch.argmax(logits, dim=1) == torch.argmax(labels, dim=1)).item() / len(logits)


def accuracy(logits, labels, threshold=0.5):
    return torch.sum((logits > threshold) == (labels > threshold)).item() / (torch.sum((labels > threshold) == (labels > threshold)).item())


def masked_accuracy(logits, labels, threshold=0.5):
    masked_labels = labels.clone()
    masked_labels = masked_labels[labels != -1]
    return accuracy(logits[labels != -1], masked_labels, threshold=threshold)


EVALUATORS = {
    "categorical_accuracy": categorical_accuracy,
    "accuracy": accuracy,
    "masked_accuracy": masked_accuracy
}


def evaluate_supervised(model: torch.nn.Module, dataset: Dataset, evaluator: str):
    model.eval()

    with torch.no_grad():
        logits = model(dataset[:][0].to(model.device))

    model.train()

    return EVALUATORS[evaluator](logits, dataset[:][1].to(model.device))


def train_supervised(train_dataset: Dataset, validation_dataset: Dataset, model: torch.nn.Module,
                     epochs: int = 250, batch_size: int = 32,
                     compute_period: int = 5, validation_patience: int = 25,
                     learning_rate: float = 0.001, weight_decay: float = 1.0e-4,
                     loss_name: str = "cross_entropy", evaluator: str = "categorical_accuracy",
                     device: torch.device = torch.device("cpu")):

    best_validation_accuracy = 0.0
    best_validation_model = copy.deepcopy(model.state_dict())
    improvement_epoch = 0

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Train the model.
    validation_accuracy = 0.0
    for epoch in range(epochs):
        # Evaluate the model on the validation set.
        if epoch % compute_period == 0:
            validation_accuracy = evaluate_supervised(model, validation_dataset, evaluator)

            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_validation_model = copy.deepcopy(model.state_dict())

                improvement_epoch = epoch

            if epoch - improvement_epoch > validation_patience:
                break

        # Perform a training epoch.
        with tqdm.tqdm(train_loader) as tq:
            tq.set_description("Epoch:{}".format(epoch))
            moving_average_loss = 0.0
            moving_average_training_eval = 0.0
            batch_count = 0
            for features, labels in tq:
                # Move to device.
                features = features.to(device)
                labels = labels.to(device)

                # Forward pass.
                logits = model(features)

                # Compute loss.
                loss = LOSSES[loss_name](logits, labels)
                loss_value = loss.item()

                # Backward pass.
                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                if batch_count == 0:
                    moving_average_loss = loss_value
                    moving_average_training_eval = EVALUATORS[evaluator](logits, labels)
                else:
                    moving_average_loss = (moving_average_loss * 0.9) + (loss_value * 0.1)
                    moving_average_training_eval = ((moving_average_training_eval * 0.9) + EVALUATORS[evaluator](logits, labels) * 0.1)

                tq.set_postfix(loss=moving_average_loss, training_accuracy=moving_average_training_eval, validation_accuracy=validation_accuracy)

                batch_count += 1

    # Evaluate the model on the validation set.
    validation_accuracy = evaluate_supervised(model, validation_dataset, evaluator)

    if validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = validation_accuracy
        best_validation_model = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_validation_model)

    return model, best_validation_accuracy
