#!/usr/bin/env python3
import copy
import os
import random
import sys
import traceback

import torch
import tqdm

from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from citation.constants import BASE_DATA_DIR
from citation.constants import BASE_RESULTS_DIR
from citation.constants import CONFIG_FILENAME
from citation.constants import CURRENT_MODEL_NAME
from citation.constants import TRAINED_MODEL_NAME
from citation.scripts.citation_dataset import CitationDataset
from neural.models.mlp import MLP
from neural.utils import get_torch_device
from utils import enumerate_hyperparameters
from utils import load_json_file
from utils import seed_everything
from utils import write_json_file
from utils import write_psl_data_file

# The number of layers is set to 1 therefore there is no hidden layer and the hidden dimension is set to 0.
HYPERPARAMETERS = {
    'hidden-dim': [0],
    'num-layers': [1],
    'learning-rate': [1.0e-2, 1.0e-1, 1.0e-0],
    'weight-decay': [1.0e-6],
    'dropout': [0.0],
}

DATASETS = ['citeseer', 'cora']

DEFAULT_PARAMETERS = {
    'citeseer': {
        'hidden-dim': 0,
        'num-layers': 1,
        'learning-rate': 1.5e-0,
        'weight-decay': 1.0e-6,
        'dropout': 0.0
    },
    'cora': {
        'hidden-dim': 0,
        'num-layers': 1,
        'learning-rate': 1.5e-0,
        'weight-decay': 5.0e-7,
        'dropout': 0.0
    }
}

RANDOM_SEEDS = 10
EPOCHS = 250
BATCH_SIZE = 32

RUN_HYPERPARAMETER_SEARCH = True


def predict(data: dict, out_dir: str, model: torch.nn.Module):
    device = get_torch_device()

    for partition in ['train', 'test', 'valid', 'latent']:
        dataset = CitationDataset(
            data[partition]['features'],
            data[partition]['labels'],
            data[partition]['entity-ids']
        )

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        model.eval()
        predictions = []

        with torch.no_grad():
            for batch in loader:
                batch = [b.to(device) for b in batch]

                features, labels, ids = batch
                batch_predictions = model(features)
                batch_predictions = torch.nn.functional.sigmoid(batch_predictions)

                for prediction_id, prediction in zip(ids, batch_predictions):
                    for index, class_prediction in enumerate(prediction):
                        predictions.append([prediction_id.item(), index, class_prediction.item()])

        write_psl_data_file(os.path.join(out_dir, f"category-neural-{partition}.txt"), predictions)


def pretrain(data: dict,
             epochs: int,
             learning_rate: float,
             weight_decay: float,
             batch_size: int,
             input_dim: int,
             hidden_dim: int,
             output_dim: int,
             num_layers: int,
             dropout_rate: float):
    device = get_torch_device()

    train_dataset = CitationDataset(
        data['train']['features'],
        data['train']['labels'],
        data['train']['entity-ids']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    valid_dataset = CitationDataset(
        data['valid']['features'],
        data['valid']['labels'],
        data['valid']['entity-ids']
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_dataset = CitationDataset(
        data['test']['features'],
        data['test']['labels'],
        data['test']['entity-ids']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model = MLP(input_dim, hidden_dim, output_dim, num_layers, dropout_rate).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    max_validation_accuracy = -1
    max_model = None
    training_history = []
    for epoch in tqdm.tqdm(range(epochs), "Training Model", leave=True):
        model.train()
        epoch_loss = 0
        with tqdm.tqdm(train_loader) as tq:
            tq.set_description("Epoch:{}".format(epoch))

            for step, batch in enumerate(tq):
                optimizer.zero_grad(set_to_none=True)

                batch = [b.to(device) for b in batch]

                features, labels, _ = batch
                batch_predictions = model(features)

                loss = criterion(batch_predictions, labels)
                epoch_loss += loss.item()

                loss.backward()
                optimizer.step()

                tq.set_postfix(loss=epoch_loss / ((step + 1) * train_loader.batch_size))

        model.eval()
        validation_accuracy = 0
        with torch.no_grad():
            for batch in valid_loader:
                batch = [b.to(device) for b in batch]

                features, labels, _ = batch
                batch_predictions = model(features)

                validation_accuracy += torch.sum(torch.argmax(batch_predictions, dim=1) == torch.argmax(labels, dim=1)).item()

        validation_accuracy /= len(valid_dataset)

        if validation_accuracy > max_validation_accuracy:
            max_validation_accuracy = validation_accuracy
            max_model = MLP(input_dim, hidden_dim, output_dim, num_layers, dropout_rate).to(device)
            max_model.load_state_dict(copy.deepcopy(model.state_dict()))

        training_history.append([epoch_loss / len(train_dataset), validation_accuracy])

    model.eval()
    test_accuracy = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = [b.to(device) for b in batch]

            features, labels, _ = batch
            batch_predictions = max_model(features)

            test_accuracy += torch.sum(torch.argmax(batch_predictions, dim=1) == torch.argmax(labels, dim=1)).item()

    test_accuracy /= len(test_dataset)

    return {'test_accuracy': test_accuracy, 'validation_accuracy': max_validation_accuracy, 'training-history': training_history}, max_model


def main():
    for dataset_id in DATASETS:
        hyperparameters = [DEFAULT_PARAMETERS[dataset_id]]
        for experiment_id in sorted(os.listdir(os.path.join(BASE_DATA_DIR, dataset_id))):
            for split_id in sorted(os.listdir(os.path.join(BASE_DATA_DIR, dataset_id, experiment_id))):
                for size_id in sorted(os.listdir(os.path.join(BASE_DATA_DIR, dataset_id, experiment_id, split_id))):
                    out_dir = os.path.join(BASE_RESULTS_DIR, dataset_id, experiment_id, split_id, size_id, "pretrain")
                    if RUN_HYPERPARAMETER_SEARCH and int(split_id) == 0:
                        hyperparameters = enumerate_hyperparameters(HYPERPARAMETERS)
                    elif RUN_HYPERPARAMETER_SEARCH and int(split_id) != 0:
                        hyperparameters = [load_json_file(os.path.join(BASE_RESULTS_DIR, dataset_id, experiment_id, "00", size_id, "pretrain", CONFIG_FILENAME))['network']['parameters']]
                    out_path = os.path.join(out_dir, TRAINED_MODEL_NAME)
                    os.makedirs(out_dir, exist_ok=True)

                    max_accuracy = -1
                    if os.path.exists(os.path.join(out_dir, CONFIG_FILENAME)):
                        max_accuracy = load_json_file(os.path.join(out_dir, CONFIG_FILENAME))['network']['results']['validation_accuracy']

                    data_dir = os.path.join(BASE_DATA_DIR, dataset_id, experiment_id, split_id, size_id)

                    data = load_json_file(os.path.join(data_dir, "deep-data.json"))
                    config = load_json_file(os.path.join(data_dir, CONFIG_FILENAME))

                    seed_everything(config['seed'])
                    for parameters in hyperparameters:
                        for seed_index in range(RANDOM_SEEDS):
                            if RUN_HYPERPARAMETER_SEARCH and int(split_id) == 0:
                                os.makedirs(os.path.join(out_dir, "hyperparameter-search"), exist_ok=True)
                                out_path = os.path.join(out_dir, "hyperparameter-search", "-".join([f"%s::%0.10f" % (k, v) for k, v in parameters.items()]) + f"-seed::{seed_index}.pt")

                            restart_budget = 3
                            while restart_budget > 0:
                                seed = random.randrange(2 ** 64)
                                torch.manual_seed(seed)

                                try:
                                    results, model = pretrain(data, EPOCHS, parameters['learning-rate'], parameters['weight-decay'],
                                                              BATCH_SIZE, config['input-shape'], parameters['hidden-dim'],
                                                              config['class-size'], parameters['num-layers'], parameters['dropout'])
                                except Exception:
                                    print(f"Error training model with seed {seed}. Restarting...")
                                    print(traceback.format_exc())
                                    restart_budget -= 1
                                    continue

                                if RUN_HYPERPARAMETER_SEARCH and int(split_id) == 0:
                                    torch.save(model.state_dict(), out_path)
                                else:
                                    torch.save(model.state_dict(), os.path.join(out_dir, CURRENT_MODEL_NAME))

                                if results['validation_accuracy'] <= max_accuracy:
                                    break

                                max_accuracy = results['validation_accuracy']
                                torch.save(model.state_dict(), os.path.join(out_dir, TRAINED_MODEL_NAME))

                                results_config = {
                                    'seed': seed,
                                    'num-random-seeds': RANDOM_SEEDS,
                                    'network': {
                                        'epochs': EPOCHS,
                                        'batch-size': BATCH_SIZE,
                                        'parameters': {**parameters},
                                        'results': results,
                                    },
                                }

                                predict(data, out_dir, model)
                                write_json_file(os.path.join(out_dir, CONFIG_FILENAME), results_config)
                                break


if __name__ == '__main__':
    main()
