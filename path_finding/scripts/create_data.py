#!/usr/bin/env python3

# Construct the data and neural model for this experiment.
# Before a directory is generated, the existence of a config file for that directory will be checked,
# if it exists generation is skipped.

import numpy as np
import os
import sys

from itertools import product

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.join(THIS_DIR, '..', '..'))

import utils

DATASET_WARCRAFT_MAP = 'warcraft-map'
DATASETS = [DATASET_WARCRAFT_MAP]

CONFIGS = {
    DATASET_WARCRAFT_MAP: {
        "dimension": 12,
        "name": DATASET_WARCRAFT_MAP,
        "num-splits": 5,
        "test-size": 1000,
        "train-size": 10000,
        "unlabeled-ratios": [0.00, 0.50, 0.90, 0.95],
        "val-size": 1000,
    }
}

CONFIG_FILENAME = "config.json"


def normalize_images(images):
    # Normalize the pixel intensity to [0,1].
    images = images / 255.0

    # Round so that the output is significantly smaller.
    images = images.round(4)

    return images


def create_entity_data_map(train_maps, val_maps, test_maps, config, map_start_id=0):
    all_maps = np.concatenate((train_maps, val_maps, test_maps), axis=0)
    all_normalized_maps = normalize_images(all_maps)

    # Create the entity data map.
    entity_data_map = []
    for map_id in range(len(all_maps)):
        entity_data_map.append([map_id + map_start_id, all_normalized_maps[map_id + map_start_id].flatten().tolist()])

    return [[int(row[0])] + row[1] for row in entity_data_map]


def create_neural_path_targets(maps, config, map_start_id=0):
    neural_path_targets = []
    for map_id in range(len(maps)):
        for x in range(config["dimension"]):
            for y in range(config["dimension"]):
                neural_path_targets.append([map_id + map_start_id, x * config["dimension"] + y])

    return neural_path_targets


def create_on_path_targets(maps, config, map_start_id=0):
    on_path_targets = []
    for map_id in range(len(maps)):
        for x in range(config["dimension"]):
            for y in range(config["dimension"]):
                on_path_targets.append([map_id + map_start_id, x, y])

        on_path_targets.append([map_id + map_start_id, config["dimension"], config["dimension"]])

    return on_path_targets


def create_path_targets(maps, config, map_start_id=0):
    path_targets = []
    for map_id in range(len(maps)):
        for x in range(config["dimension"]):
            for y in range(config["dimension"]):
                for dx, dy in product([-1, 0, 1], repeat=2):
                    new_x = x + dx
                    new_y = y + dy
                    if ((new_x < 0)
                            or (new_x >= config["dimension"])
                            or (new_y < 0)
                            or (new_y >= config["dimension"])
                            or (dx == 0 and dy == 0)):
                        continue
                    path_targets.append([map_id + map_start_id, x, y, new_x, new_y])

        path_targets.append([map_id + map_start_id, config["dimension"], config["dimension"], 0, 0])
        path_targets.append([map_id + map_start_id, 0, 0, config["dimension"], config["dimension"]])
        path_targets.append([map_id + map_start_id, config["dimension"], config["dimension"], config["dimension"] - 1, config["dimension"] - 1])
        path_targets.append([map_id + map_start_id, config["dimension"] - 1, config["dimension"] - 1, config["dimension"], config["dimension"]])

    return path_targets


def create_path_truths(map_path_truths, unlabeled_ratio, config, map_start_id=0):
    # Sample unlabeled vertices for each map.
    all_maps_and_vertices = list(product(range(len(map_path_truths)), product(range(config["dimension"]), repeat=2)))
    unlabeled_maps_and_vertices_indexes = np.random.choice(list(range(len(all_maps_and_vertices))), int(unlabeled_ratio * len(all_maps_and_vertices)), replace=False)
    unlabeled_maps_and_vertices = set([all_maps_and_vertices[i] for i in unlabeled_maps_and_vertices_indexes])

    on_path_truths = []
    neural_path_truths = []
    for map_id in range(len(map_path_truths)):
        for x in range(config["dimension"]):
            for y in range(config["dimension"]):
                if (map_id, (x, y)) in unlabeled_maps_and_vertices:
                    neural_path_truths.append([map_id + map_start_id, x * config["dimension"] + y, -1])
                else:
                    on_path_truths.append([map_id + map_start_id, x, y, map_path_truths[map_id][x][y]])
                    neural_path_truths.append([map_id + map_start_id, x * config["dimension"] + y, map_path_truths[map_id][x][y]])

        on_path_truths.append([map_id + map_start_id, config["dimension"], config["dimension"], 1])

    # Transform the path truths into a list of edges.
    path_truths = []
    for map_id in range(len(map_path_truths)):
        path_truths.append(set({}))
        visited = set({})
        x, y = 0, 0
        while (x, y) != (config["dimension"] - 1, config["dimension"] - 1):
            visited.add((x, y))
            for dx, dy in product([-1, 0, 1], repeat=2):
                new_x = x + dx
                new_y = y + dy

                if ((new_x < 0)
                        or (new_x >= config["dimension"])
                        or (new_y < 0)
                        or (new_y >= config["dimension"])
                        or (dx == 0 and dy == 0)
                        or (new_x, new_y) in visited):
                    continue

                if map_path_truths[map_id][new_x][new_y] == 1:
                    path_truths[map_id].add((map_id + map_start_id, x, y, new_x, new_y))
                    x, y = new_x, new_y
                    break

    filled_path_truths = []
    for map_id in range(len(map_path_truths)):
        for x in range(config["dimension"]):
            for y in range(config["dimension"]):
                for dx, dy in product([-1, 0, 1], repeat=2):
                    new_x = x + dx
                    new_y = y + dy
                    if ((new_x < 0)
                            or (new_x >= config["dimension"])
                            or (new_y < 0)
                            or (new_y >= config["dimension"])
                            or (dx == 0 and dy == 0)):
                        continue

                    if ((map_id, (x, y)) in unlabeled_maps_and_vertices) and ((map_id, (new_x, new_y)) in unlabeled_maps_and_vertices):
                        continue

                    if (map_id + map_start_id, x, y, new_x, new_y) in path_truths[map_id]:
                        # Both vertices of the edge must be labeled for the edge to be labeled.
                        if ((map_id, (x, y)) in unlabeled_maps_and_vertices) or ((map_id, (new_x, new_y)) in unlabeled_maps_and_vertices):
                            continue

                        filled_path_truths.append([map_id + map_start_id, x, y, new_x, new_y, 1])
                    else:
                        # At least one of the vertices of the edge must be labeled and 0 for the edge to be unlabeled.
                        if not ((((map_id, (x, y)) not in unlabeled_maps_and_vertices) and (map_path_truths[map_id][x][y] == 0))
                                or ((((map_id, (new_x, new_y)) not in unlabeled_maps_and_vertices) and (map_path_truths[map_id][new_x][new_y] == 0)))):
                            continue

                        filled_path_truths.append([map_id + map_start_id, x, y, new_x, new_y, 0])

        filled_path_truths.append([map_id + map_start_id, config["dimension"], config["dimension"], 0, 0, 1])
        filled_path_truths.append([map_id + map_start_id, 0, 0, config["dimension"], config["dimension"], 0])
        filled_path_truths.append([map_id + map_start_id, config["dimension"] - 1, config["dimension"] - 1, config["dimension"], config["dimension"], 1])
        filled_path_truths.append([map_id + map_start_id, config["dimension"], config["dimension"], config["dimension"] - 1, config["dimension"] - 1, 0])

    return neural_path_truths, on_path_truths, filled_path_truths


def create_map_costs_truths(map_vertex_weights, config, map_start_id=0):
    map_costs_truths = []
    for map_id in range(len(map_vertex_weights)):
        for x in range(config["dimension"]):
            for y in range(config["dimension"]):
                map_costs_truths.append([map_id + map_start_id, x, y, map_vertex_weights[map_id][x][y]])

    return map_costs_truths


def write_specific_data(config, out_dir, unlabeled_ratio, train_maps, train_maps_vertex_weights, train_paths,
                        val_maps, val_maps_vertex_weights, val_paths,
                        test_maps, test_maps_vertex_weights, test_paths):
    partition_maps = {
        'train': train_maps,
        'valid': val_maps,
        'test': test_maps
    }

    partition_maps_vertex_weights = {
        'train': train_maps_vertex_weights,
        'valid': val_maps_vertex_weights,
        'test': test_maps_vertex_weights
    }

    partition_paths = {
        'train': train_paths,
        'valid': val_paths,
        'test': test_paths
    }

    map_start_id = 0
    for partition in ['train', 'valid', 'test']:
        print(f"Writing data for {partition}.")
        neural_path_targets = create_neural_path_targets(partition_maps[partition], config, map_start_id=map_start_id)
        on_path_targets = create_on_path_targets(partition_maps[partition], config, map_start_id=map_start_id)
        path_targets = create_path_targets(partition_maps[partition], config, map_start_id=map_start_id)
        if partition == 'train':
            neural_path_truths, on_path_truths, path_truths = create_path_truths(partition_paths[partition], unlabeled_ratio, config, map_start_id=map_start_id)
        else:
            neural_path_truths, on_path_truths, path_truths = create_path_truths(partition_paths[partition], 0.0, config, map_start_id=map_start_id)
        map_costs_truths = create_map_costs_truths(partition_maps_vertex_weights[partition], config, map_start_id=map_start_id)

        utils.write_psl_data_file(os.path.join(out_dir, f'neural-path-target-{partition}.txt'), neural_path_targets)
        utils.write_psl_data_file(os.path.join(out_dir, f'on-path-target-{partition}.txt'), on_path_targets)
        utils.write_psl_data_file(os.path.join(out_dir, f'path-target-{partition}.txt'), path_targets)
        utils.write_psl_data_file(os.path.join(out_dir, f'neural-path-truth-{partition}.txt'), neural_path_truths)
        utils.write_psl_data_file(os.path.join(out_dir, f'on-path-truth-{partition}.txt'), on_path_truths)
        utils.write_psl_data_file(os.path.join(out_dir, f'path-truth-{partition}.txt'), path_truths)
        utils.write_psl_data_file(os.path.join(out_dir, f'map-costs-truth-{partition}.txt'), map_costs_truths)

        map_start_id += len(partition_maps[partition])

    entity_data_map = create_entity_data_map(
        train_maps, val_maps, test_maps, config, map_start_id=0
    )
    utils.write_psl_data_file(os.path.join(out_dir, f'neural-data.txt'), entity_data_map)

    utils.write_json_file(os.path.join(out_dir, CONFIG_FILENAME), config)


def create_flattened_to_coordinate_block_observations(config):
    flattened_to_coordinate_block_observations = []
    for x, y in product(range(config["dimension"]), repeat=2):
        flattened_to_coordinate_block_observations.append([x * config["dimension"] + y, x, y])

    return flattened_to_coordinate_block_observations


def write_shared_data(config, out_dir):
    flattened_to_coordinate_block_observations = create_flattened_to_coordinate_block_observations(config)
    utils.write_psl_data_file(os.path.join(out_dir, 'flattened-to-coordinate-block-obs.txt'), flattened_to_coordinate_block_observations)


def fetch_data(config):
    os.makedirs(os.path.join(THIS_DIR, "..", "data", "warcraft_raw_data"), exist_ok=True)

    if not os.path.exists(os.path.join(THIS_DIR, "..", "data", "warcraft_raw_data", "warcraft_maps.tar.gz")):
        # Download the warcraft map data.
        os.system("wget https://linqs-data.soe.ucsc.edu/public/nesy-ebm-jmlr24/warcraft_maps.tar.gz -O {}/warcraft_maps.tar.gz".format(os.path.join(THIS_DIR, "..", "data", "warcraft_raw_data")))
        os.system("tar -xvzf {} -C {}".format(os.path.join(THIS_DIR, "..", "data", "warcraft_raw_data", "warcraft_maps.tar.gz"), os.path.join(THIS_DIR, "..", "data", "warcraft_raw_data")))
    else:
        print("Warcraft map data already exists. Skipping download.")

    # Load the warcraft map data.
    train_maps = np.load("{}/warcraft_shortest_path_oneskin/12x12/train_maps.npy".format(os.path.join(THIS_DIR, "..", "data", "warcraft_raw_data")))
    train_maps_vertex_weights = np.load("{}/warcraft_shortest_path_oneskin/12x12/train_vertex_weights.npy".format(os.path.join(THIS_DIR, "..", "data", "warcraft_raw_data")))
    train_paths = np.load("{}/warcraft_shortest_path_oneskin/12x12/train_shortest_paths.npy".format(os.path.join(THIS_DIR, "..", "data", "warcraft_raw_data")))

    val_maps = np.load("{}/warcraft_shortest_path_oneskin/12x12/val_maps.npy".format(os.path.join(THIS_DIR, "..", "data", "warcraft_raw_data")))
    val_maps_vertex_weights = np.load("{}/warcraft_shortest_path_oneskin/12x12/val_vertex_weights.npy".format(os.path.join(THIS_DIR, "..", "data", "warcraft_raw_data")))
    val_paths = np.load("{}/warcraft_shortest_path_oneskin/12x12/val_shortest_paths.npy".format(os.path.join(THIS_DIR, "..", "data", "warcraft_raw_data")))

    test_maps = np.load("{}/warcraft_shortest_path_oneskin/12x12/test_maps.npy".format(os.path.join(THIS_DIR, "..", "data", "warcraft_raw_data")))
    test_maps_vertex_weights = np.load("{}/warcraft_shortest_path_oneskin/12x12/test_vertex_weights.npy".format(os.path.join(THIS_DIR, "..", "data", "warcraft_raw_data")))
    test_paths = np.load("{}/warcraft_shortest_path_oneskin/12x12/test_shortest_paths.npy".format(os.path.join(THIS_DIR, "..", "data", "warcraft_raw_data")))

    # Aggregate the train, val, and test maps and paths.
    maps = np.concatenate((train_maps, val_maps, test_maps), axis=0)
    maps_vertex_weights = np.concatenate((train_maps_vertex_weights, val_maps_vertex_weights, test_maps_vertex_weights), axis=0)
    paths = np.concatenate((train_paths, val_paths, test_paths), axis=0)

    return maps, maps_vertex_weights, paths


def split_data(maps, maps_vertex_weights, paths, config):
    # Shuffle data.
    indices = np.arange(len(maps))
    np.random.shuffle(indices)
    maps = maps[indices]
    maps_vertex_weights = maps_vertex_weights[indices]
    paths = paths[indices]

    # Split data.
    train_maps = maps[:config['train-size']]
    train_maps_vertex_weights = maps_vertex_weights[:config['train-size']]
    train_paths = paths[:config['train-size']]

    val_maps = maps[config['train-size']:config['train-size'] + config['val-size']]
    val_maps_vertex_weights = maps_vertex_weights[config['train-size']:config['train-size'] + config['val-size']]
    val_paths = paths[config['train-size']:config['train-size'] + config['val-size']]

    test_maps = maps[config['train-size'] + config['val-size']:config['train-size'] + config['val-size'] + config['test-size']]
    test_maps_vertex_weights = maps_vertex_weights[config['train-size'] + config['val-size']:config['train-size'] + config['val-size'] + config['test-size']]
    test_paths = paths[config['train-size'] + config['val-size']:config['train-size'] + config['val-size'] + config['test-size']]

    return (train_maps, train_maps_vertex_weights, train_paths,
            val_maps, val_maps_vertex_weights, val_paths,
            test_maps, test_maps_vertex_weights, test_paths)


def main():
    for dataset_id in DATASETS:
        config = CONFIGS[dataset_id]

        shared_out_dir = os.path.join(THIS_DIR, "..", "data", dataset_id)
        os.makedirs(shared_out_dir, exist_ok=True)
        if os.path.isfile(os.path.join(shared_out_dir, CONFIG_FILENAME)):
            print("Shared data already exists for %s. Skipping generation." % dataset_id)
        else:
            print("Generating shared data for %s." % dataset_id)
            write_shared_data(config, shared_out_dir)

        for split in range(config['num-splits']):
            for unlabeled in config['unlabeled-ratios']:
                config['seed'] = 10 * split
                utils.seed_everything(config['seed'])

                out_dir = os.path.join(shared_out_dir, "split::%d" % split, "unlabeled::%.2f" % unlabeled)
                os.makedirs(out_dir, exist_ok=True)

                if os.path.isfile(os.path.join(out_dir, CONFIG_FILENAME)):
                    print("Data already exists for %s. Skipping generation." % out_dir)
                    continue

                print("Generating data for %s." % out_dir)
                maps, maps_vertex_weights, paths = fetch_data(config)
                (train_maps, train_maps_vertex_weights, train_paths,
                 val_maps, val_maps_vertex_weights, val_paths,
                 test_maps, test_maps_vertex_weights, test_paths) = split_data(maps, maps_vertex_weights, paths, config)
                write_specific_data(config, out_dir, unlabeled, train_maps, train_maps_vertex_weights, train_paths,
                                    val_maps, val_maps_vertex_weights, val_paths,
                                    test_maps, test_maps_vertex_weights, test_paths)


if __name__ == '__main__':
    main()
