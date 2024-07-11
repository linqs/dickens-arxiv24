#!/usr/bin/env python3

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import dgl
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from citation.constants import BASE_DATA_DIR
from citation.constants import CONFIG_FILENAME
from neural.utils import get_torch_device
from utils import one_hot_encoding
from utils import seed_everything
from utils import write_json_file
from utils import write_psl_data_file

DATASETS = {
    'citeseer': {
        "class-size": 6,
        "num-splits": 5,
    },
    'cora': {
        "class-size": 7,
        "num-splits": 5,
    },
}


def generate_random_label_partitions(graph, device, class_size, train_count, test_count, valid_count):
    """
    Generate train, test, and valid partition masks. Guarantee at least one node per class for each partition.
    """
    found_sample = False
    while not found_sample:
        graph.ndata["train-mask"] = torch.zeros(graph.num_nodes(), dtype=torch.bool, device=device)
        graph.ndata["test-mask"] = torch.zeros(graph.num_nodes(), dtype=torch.bool, device=device)
        graph.ndata["valid-mask"] = torch.zeros(graph.num_nodes(), dtype=torch.bool, device=device)
        graph.ndata["latent-mask"] = torch.zeros(graph.num_nodes(), dtype=torch.bool, device=device)

        permutation = torch.randperm(graph.num_nodes(), device=device)

        graph.ndata["train-mask"][permutation[:train_count]] = True
        graph.ndata["test-mask"][permutation[train_count:train_count + test_count]] = True
        graph.ndata["valid-mask"][permutation[train_count + test_count:train_count + test_count + valid_count]] = True
        graph.ndata["latent-mask"][permutation[train_count + test_count + valid_count:]] = True

        for mask_name in ["train-mask", "test-mask", "valid-mask"]:
            found_sample = found_sample or len(torch.unique(graph.ndata["label"][graph.ndata[mask_name]])) == class_size

    return graph


def generate_equal_label_partitions(graph, device, class_size, train_count, test_count, valid_count):
    """
    Generate train, test, and valid partition masks. Guarantee equal number of nodes per class for train partition.
    """
    found_sample = False
    while not found_sample:
        graph.ndata["train-mask"] = torch.zeros(graph.num_nodes(), dtype=torch.bool, device=device)
        graph.ndata["test-mask"] = torch.zeros(graph.num_nodes(), dtype=torch.bool, device=device)
        graph.ndata["valid-mask"] = torch.zeros(graph.num_nodes(), dtype=torch.bool, device=device)
        graph.ndata["latent-mask"] = torch.zeros(graph.num_nodes(), dtype=torch.bool, device=device)

        for label in range(class_size):
            label_nodes = (graph.ndata["label"] == label).nonzero().flatten()
            permutation = torch.randperm(label_nodes.shape[0], device=device)
            graph.ndata["train-mask"][label_nodes[permutation[:train_count]]] = True

        remianing_nodes = (graph.ndata["train-mask"] == False).nonzero().flatten()
        permutation = torch.randperm(remianing_nodes.shape[0], device=device)
        graph.ndata["test-mask"][remianing_nodes[permutation[:test_count]]] = True
        graph.ndata["valid-mask"][remianing_nodes[permutation[test_count:test_count + valid_count]]] = True
        graph.ndata["latent-mask"][remianing_nodes[permutation[test_count + valid_count:]]] = True

        for mask_name in ["train-mask", "test-mask", "valid-mask"]:
            found_sample = found_sample or len(torch.unique(graph.ndata["label"][graph.ndata[mask_name]])) == class_size

    return graph


def generate_smoothed_features(graph, device):
    """
    Generate smoothed features for each node in the graph.
    """
    symmetric_graph = dgl.to_simple(dgl.add_reverse_edges(graph)).to(device)

    sgconv_layer = dgl.nn.pytorch.conv.SGConv(symmetric_graph.ndata['feat'].shape[1],
                                              symmetric_graph.ndata['feat'].shape[1],
                                              k=3).to(device)
    torch.nn.init.eye_(sgconv_layer.fc.weight)
    features = sgconv_layer(symmetric_graph, symmetric_graph.ndata['feat'])

    return features


def fetch_data(config):
    seed_everything(config['seed'])
    device = get_torch_device()

    if config['dataset'] == 'citeseer':
        graph = dgl.data.CiteseerGraphDataset()[0].to(device)
    elif config['dataset'] == 'cora':
        graph = dgl.data.CoraGraphDataset()[0].to(device)
    else:
        raise ValueError("Unknown dataset: '%s'." % (config['dataset'],))

    graph = dgl.add_self_loop(dgl.remove_self_loop(graph))

    if config['experiment'] == 'low-data':
        train_size = config['train-size']
        graph = generate_equal_label_partitions(graph, device, config['class-size'], train_size, config['test-size'], config['valid-size'])
    elif config['experiment'] == 'semi-supervised':
        train_size = int((graph.num_nodes() - config['test-size'] - config['valid-size']) * config['train-size'])
        graph = generate_random_label_partitions(graph, device, config['class-size'], train_size, config['test-size'], config['valid-size'])
    else:
        raise ValueError("Unknown experiment: '%s'." % (config['experiment'],))

    features = generate_smoothed_features(graph, device)

    graph = dgl.remove_self_loop(graph)
    data = {}

    for partition in ['train', 'test', 'valid', 'latent']:
        indexes = graph.nodes()[graph.ndata[partition + "-mask"]]
        data[partition] = {
            'entity-ids': indexes.detach().cpu().numpy().tolist(),
            'labels': [one_hot_encoding(int(label), config['class-size']) for label in graph.ndata['label'][indexes].detach().cpu().numpy().tolist()],
            'features': features[indexes].detach().cpu().numpy().tolist(),
        }

    edges = torch.stack(graph.edges()).T.detach().cpu().numpy()

    return data, edges, graph


def write_data(config, out_dir, graph, data, edges):
    entity_data_map = []

    for key in data:
        category_targets = []
        category_truth = []

        for entity_index in range(len(data[key]['entity-ids'])):
            entity = data[key]['entity-ids'][entity_index]

            entity_data_map.append([entity] + data[key]['features'][entity_index] + [data[key]['labels'][entity_index].index(max(data[key]['labels'][entity_index]))])

            for label_index in range(config['class-size']):
                category_targets.append([entity, str(label_index)])
                category_truth.append([entity, str(label_index), data[key]['labels'][entity_index][label_index]])

        write_psl_data_file(os.path.join(out_dir, "category-target-%s.txt" % key), category_targets)
        write_psl_data_file(os.path.join(out_dir, "category-truth-%s.txt" % key), category_truth)

    write_psl_data_file(os.path.join(out_dir, "edges.txt"), edges)

    write_psl_data_file(os.path.join(out_dir, "entity-data-map.txt"), entity_data_map)
    write_json_file(os.path.join(out_dir, "deep-data.json"), data, indent=None)

    dgl.save_graphs(os.path.join(out_dir, 'dgl-graph.bin'), [graph])

    write_json_file(os.path.join(out_dir, CONFIG_FILENAME), config)


def main():
    for dataset_id in DATASETS:
        config = DATASETS[dataset_id]
        config['dataset'] = dataset_id
        for experiment in ['low-data', 'semi-supervised']:
            config['experiment'] = experiment
            config['test-size'] = 1000
            config['valid-size'] = 200
            for split in range(config['num-splits']):
                config['seed'] = split
                if experiment == 'low-data':
                    train_sizes = [20]
                elif experiment == 'semi-supervised':
                    train_sizes = [0.05, 0.10, 0.50, 1.00]
                else:
                    raise ValueError("Unknown experiment: '%s'." % (experiment,))

                for train_size in train_sizes:
                    config['train-size'] = train_size

                    out_dir = os.path.join(BASE_DATA_DIR, dataset_id, experiment, "{:02d}".format(split), "{:0.2f}".format(train_size))
                    os.makedirs(out_dir, exist_ok=True)

                    if os.path.isfile(os.path.join(out_dir, CONFIG_FILENAME)):
                        print("Data already exists for %s. Skipping generation." % out_dir)
                        continue

                    data, edges, graph = fetch_data(config)
                    config['input-shape'] = len(data['train']['features'][0])
                    write_data(config, out_dir, graph, data, edges)


if __name__ == '__main__':
    main()
