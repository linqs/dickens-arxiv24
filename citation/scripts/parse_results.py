#!/usr/bin/env python3

import os
import sys
import torch
import numpy

from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from citation.constants import BASE_RESULTS_DIR, BASE_DATA_DIR
from citation.scripts.citation_dataset import CitationDataset
from neural.models.mlp import MLP
from utils import load_json_file

DATASETS = ['citeseer', 'cora']
EXPERIMENT = ['low-data', 'semi-supervised']
SPLITS = ['00', '01', '02', '03', '04']
SETTING = ['20.00', '0.05', '0.10', '0.50', '1.00']
LEARNING = ['bilevel', 'energy', 'modular', 'pretrain']


def main():
    for dataset in DATASETS:
        for experiment in EXPERIMENT:
            for setting in SETTING:
                for learning in LEARNING:
                    results = []
                    for split in SPLITS:
                        results_config = os.path.join(BASE_RESULTS_DIR, dataset, experiment, split, setting, learning, 'config.json')
                        if not os.path.exists(results_config):
                            continue

                        result_dict = load_json_file(results_config)
                        if learning == 'pretrain':
                            results.append(result_dict['network']['results']['test_accuracy'])
                        else:
                            results.append(result_dict['results'][0])
                    if len(results) > 0:
                        print(dataset, experiment, setting, learning, numpy.mean(results), numpy.std(results))

                    if learning not in ['bilevel', 'energy']:
                        continue

                    results = []
                    for split in SPLITS:
                        results_dir = os.path.join(BASE_RESULTS_DIR, dataset, experiment, split, setting, learning)
                        results_config = os.path.join(results_dir, 'config.json')
                        if not os.path.exists(results_config):
                            continue

                        # Load the NeSy trained neural component.
                        citation_json = load_json_file(os.path.join(results_dir, 'citation.json'))
                        model = MLP(int(citation_json["predicates"]["Neural/2"]["options"]["input-dim"]),
                                    int(citation_json["predicates"]["Neural/2"]["options"]["hidden-dim"]),
                                    int(citation_json["predicates"]["Neural/2"]["options"]["class-size"]),
                                    int(citation_json["predicates"]["Neural/2"]["options"]["num-layers"]))
                        model.load_state_dict(torch.load(os.path.join(results_dir, 'trained-model.pt')))

                        # Load the data.
                        data_dir = os.path.join(BASE_DATA_DIR, dataset, experiment, split, setting)
                        data = load_json_file(os.path.join(data_dir, "deep-data.json"))

                        test_dataset = CitationDataset(
                            data['test']['features'],
                            data['test']['labels'],
                            data['test']['entity-ids']
                        )

                        test_loader = DataLoader(
                            test_dataset,
                            batch_size=32,
                            shuffle=False
                        )

                        # Evaluate the model.
                        model.eval()
                        test_accuracy = 0
                        with torch.no_grad():
                            for batch in test_loader:
                                features, labels, _ = batch
                                batch_predictions = model(features)

                                test_accuracy += torch.sum(
                                    torch.argmax(batch_predictions, dim=1) == torch.argmax(labels, dim=1)).item()

                        test_accuracy /= len(test_dataset)

                        results.append(test_accuracy)

                    if len(results) > 0:
                        print(dataset, experiment, setting, f"{learning}_neural", numpy.mean(results), numpy.std(results))


if __name__ == '__main__':
    main()
