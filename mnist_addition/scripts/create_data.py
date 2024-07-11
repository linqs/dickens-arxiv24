#!/usr/bin/env python3
import copy
# Construct the data and neural model for this experiment.
# Before a directory is generated, the existence of a config file for that directory will be checked,
# if it exists generation is skipped.

import os
import sys

import numpy as np
import torchvision

from itertools import product

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.join(THIS_DIR, '..', '..'))

import utils

DATASET_MNIST_1 = 'mnist-1'
DATASET_MNIST_2 = 'mnist-2'
DATASET_MNIST_4 = 'mnist-4'
DATASETS = [DATASET_MNIST_1, DATASET_MNIST_2, DATASET_MNIST_4]

DATASET_CONFIG = {
    DATASET_MNIST_1: {
        "name": DATASET_MNIST_1,
        "class-size": 10,
        "train-sizes": [600],
        "valid-size": 1000,
        "test-size": 1000,
        "num-splits": 5,
        "num-digits": 1,
        "max-sum": 9 + 9,
        "overlaps": [0.0],
        "unlabeled-ratios": [0.00, 0.50, 0.90, 0.95, 1.00]
    },
    DATASET_MNIST_2: {
        "name": DATASET_MNIST_2,
        "class-size": 10,
        "train-sizes": [600],
        "valid-size": 1000,
        "test-size": 1000,
        "num-splits": 5,
        "num-digits": 2,
        "max-sum": 99 + 99,
        "overlaps": [0.0],
        "unlabeled-ratios": [0.00, 0.50, 0.90, 0.95, 1.00]
    },
    DATASET_MNIST_4: {
        "name": DATASET_MNIST_4,
        "class-size": 10,
        "train-sizes": [600],
        "valid-size": 1000,
        "test-size": 1000,
        "num-splits": 5,
        "num-digits": 4,
        "max-sum": 9999 + 9999,
        "overlaps": [0.0],
        "unlabeled-ratios": [0.00]
    }
}

CONFIG_FILENAME = "config.json"


def normalize_images(images):
    (numImages, width, height) = images.shape

    # Flatten out the images into a 1d array.
    images = images.reshape(numImages, width * height)

    # Normalize the greyscale intensity to [0,1].
    images = images / 255.0

    # Round so that the output is significantly smaller.
    images = images.round(4)

    return images


def digits_to_number(digits):
    number = 0
    for digit in digits:
        number *= 10
        number += digit
    return number


def digits_to_sum(digits, n_digits):
    return digits_to_number(digits[:n_digits]) + digits_to_number(digits[n_digits:])


def generate_split(config, labels, indexes, shuffle=True):
    original_length = len(indexes)
    sum_indexes = copy.deepcopy(indexes)
    for _ in range(int(len(indexes) * config['overlap'])):
        sum_indexes = np.append(sum_indexes, indexes[np.random.randint(0, original_length)])

    if shuffle:
        np.random.shuffle(sum_indexes)

    sum_indexes = sum_indexes[:len(sum_indexes) - (len(sum_indexes) % (2 * config['num-digits']))]
    sum_indexes = np.unique(sum_indexes.reshape(-1, 2 * config['num-digits']), axis=0)

    sum_labels = np.array([digits_to_sum(digits, config['num-digits']) for digits in labels[sum_indexes]])

    return sum_indexes, sum_labels


def create_image_digit_truths(config, labels, labeled_ids):
    digit_truths = []
    for index in labeled_ids:
        for digit_label in range(config["class-size"]):
            digit_truths.append([index[0], digit_label, int(digit_label == labels[index])])

    return digit_truths


def create_image_digit_targets(config, image_ids):
    image_target = []
    for index in image_ids:
        for digit_label in range(config['class-size']):
            image_target.append(list(index) + [digit_label])

    return image_target


def create_entity_data_map(all_features, all_labels, all_ids, labeled_ids, train_ids, valid_ids, test_ids):
    all_normalized_features = normalize_images(all_features)[all_ids]
    all_labels = all_labels[all_ids].reshape(-1, 1)
    all_ids = all_ids.reshape(-1, 1)

    # Add label_ratio% of the digits to digit_truths.
    # The second to last entry of the feature vector is the digit label.
    all_normalized_features = all_normalized_features.reshape(all_normalized_features.shape[0], all_normalized_features.shape[2])
    all_normalized_features = np.append(all_normalized_features, all_labels, axis=1)
    # The final entry of the feature vector indicates the partition of the data.
    # 0 is labeled training, 1 is training, 2 is validation, 3 is test.
    all_normalized_features = np.append(all_normalized_features,
                                        np.zeros((len(all_normalized_features), 1), dtype=float) + 3.0, axis=1)
    all_normalized_features[test_ids, -1] = 3
    all_normalized_features[valid_ids, -1] = 2
    all_normalized_features[train_ids, -1] = 1
    if labeled_ids.size > 0:
        all_normalized_features[labeled_ids.flatten(), -1] = 0

    entity_data_map = np.concatenate((all_ids, all_normalized_features), axis=1).tolist()

    return [[int(row[0])] + row[1:] for row in entity_data_map]


def create_image_digit_sum_data(sum_entities):
    image_digit_sum_targets = []
    for example_indices in sum_entities:
        image_digit_sum_targets += [[example_indices[0], example_indices[2], k] for k in range(19)]
        image_digit_sum_targets += [[example_indices[1], example_indices[3], k] for k in range(19)]
    image_digit_sum_targets = np.unique(image_digit_sum_targets, axis=0).tolist()

    return image_digit_sum_targets


def create_image_sum_data(config, sum_entities, sum_labels):
    image_sum_target = []
    image_sum_truth = []
    for index_i in range(len(sum_entities)):
        for index_j in range(config['max-sum'] + 1):
            image_sum_target.append(list(sum_entities[index_i]) + [index_j])
            image_sum_truth.append(list(sum_entities[index_i]) + [index_j] + [1 if index_j == sum_labels[index_i] else 0])

    sum_place_target = []
    sum_place_truth = []
    for entity_index in range(len(sum_entities)):
        places = [int(10 ** i) for i in range(config['num-digits'] + 1)]

        for place in places:
            for z in range(config['class-size']):
                if place == 1:
                    sum_place_target.append(list(sum_entities[entity_index]) + [place] + [z])
                    sum_place_truth.append(list(sum_entities[entity_index]) + [place] + [z] + [1 if z == int((str(sum_labels[entity_index]).zfill(config['num-digits'] + 1))[-1]) else 0])
                if place == 10:
                    sum_place_target.append(list(sum_entities[entity_index]) + [place] + [z])
                    sum_place_truth.append(list(sum_entities[entity_index]) + [place] + [z] + [1 if z == int((str(sum_labels[entity_index]).zfill(config['num-digits'] + 1))[-2]) else 0])
                if place == 100:
                    sum_place_target.append(list(sum_entities[entity_index]) + [place] + [z])
                    sum_place_truth.append(list(sum_entities[entity_index]) + [place] + [z] + [1 if z == int((str(sum_labels[entity_index]).zfill(config['num-digits'] + 1))[-3]) else 0])
                if place == 1000:
                    sum_place_target.append(list(sum_entities[entity_index]) + [place] + [z])
                    sum_place_truth.append(list(sum_entities[entity_index]) + [place] + [z] + [1 if z == int((str(sum_labels[entity_index]).zfill(config['num-digits'] + 1))[-4]) else 0])
                if place == 10000:
                    sum_place_target.append(list(sum_entities[entity_index]) + [place] + [z])
                    sum_place_truth.append(list(sum_entities[entity_index]) + [place] + [z] + [1 if z == int((str(sum_labels[entity_index]).zfill(config['num-digits'] + 1))[-5]) else 0])

    carry_target = []
    for entity_index in range(len(sum_entities)):
        if config['num-digits'] == 1:
            carry_target.append([sum_entities[entity_index][0], sum_entities[entity_index][1], 0])
            carry_target.append([sum_entities[entity_index][0], sum_entities[entity_index][1], 1])
        elif config['num-digits'] == 2:
            carry_target.append([sum_entities[entity_index][1], sum_entities[entity_index][3], 0])
            carry_target.append([sum_entities[entity_index][1], sum_entities[entity_index][3], 1])
            carry_target.append([sum_entities[entity_index][0], sum_entities[entity_index][2], 0])
            carry_target.append([sum_entities[entity_index][0], sum_entities[entity_index][2], 1])
        elif config['num-digits'] == 4:
            carry_target.append([sum_entities[entity_index][3], sum_entities[entity_index][7], 0])
            carry_target.append([sum_entities[entity_index][3], sum_entities[entity_index][7], 1])
            carry_target.append([sum_entities[entity_index][2], sum_entities[entity_index][6], 0])
            carry_target.append([sum_entities[entity_index][2], sum_entities[entity_index][6], 1])
            carry_target.append([sum_entities[entity_index][1], sum_entities[entity_index][5], 0])
            carry_target.append([sum_entities[entity_index][1], sum_entities[entity_index][5], 1])
            carry_target.append([sum_entities[entity_index][0], sum_entities[entity_index][4], 0])
            carry_target.append([sum_entities[entity_index][0], sum_entities[entity_index][4], 1])

    carry_target = np.unique(np.array(carry_target), axis=0).tolist()

    return image_sum_target, image_sum_truth, sum_place_target, sum_place_truth, carry_target


def write_specific_data(config, out_dir, features, labels):
    all_image_ids = np.array(range(len(features)))
    np.random.shuffle(all_image_ids)

    partition_indexes = {
        'train': all_image_ids[0: config['train-size']],
        'valid': all_image_ids[config['train-size']: config['train-size'] + config['valid-size']],
        'test': all_image_ids[config['train-size'] + config['valid-size']: config['train-size'] + config['valid-size'] + config['test-size']]
    }

    for partition in ['train', 'valid', 'test']:
        image_sum_ids, image_sum_labels = generate_split(config, labels, partition_indexes[partition])
        image_sum_target, image_sum_truth, sum_place_target, sum_place_truth, carry_target = create_image_sum_data(config, image_sum_ids, image_sum_labels)

        image_ids = np.unique(image_sum_ids.reshape(-1)).reshape(-1, 1)
        image_targets = create_image_digit_targets(config, image_ids)

        utils.write_psl_data_file(os.path.join(out_dir, f'image-sum-block-{partition}.txt'), image_sum_ids)
        utils.write_psl_data_file(os.path.join(out_dir, f'image-sum-place-target-{partition}.txt'), sum_place_target)
        utils.write_psl_data_file(os.path.join(out_dir, f'image-sum-place-truth-{partition}.txt'), sum_place_truth)
        utils.write_psl_data_file(os.path.join(out_dir, f'image-sum-target-{partition}.txt'), image_sum_target)
        utils.write_psl_data_file(os.path.join(out_dir, f'image-sum-truth-{partition}.txt'), image_sum_truth)
        utils.write_psl_data_file(os.path.join(out_dir, f'image-target-{partition}.txt'), image_targets)
        utils.write_psl_data_file(os.path.join(out_dir, f'carry-target-{partition}.txt'), carry_target)

        if partition == 'train':
            labeled_training_ids = image_ids[np.random.choice(len(image_ids), size=int(len(image_ids) * (1 - config["unlabeled"])), replace=False)]
            entity_data_map = create_entity_data_map(
                features, labels, all_image_ids.reshape(-1, 1),
                labeled_training_ids, partition_indexes['train'], partition_indexes['valid'], partition_indexes['test']
            )

        else:
            labeled_training_ids = image_ids
            entity_data_map = create_entity_data_map(
                features, labels, all_image_ids.reshape(-1, 1),
                np.array([]), partition_indexes['train'], partition_indexes['valid'], partition_indexes['test']
            )

        digit_truths = create_image_digit_truths(config, labels, labeled_training_ids)

        utils.write_psl_data_file(os.path.join(out_dir, f'neural-data-{partition}.txt'), entity_data_map)

        utils.write_psl_data_file(os.path.join(out_dir, f'image-digit-labels-{partition}.txt'), digit_truths)

    utils.write_json_file(os.path.join(out_dir, CONFIG_FILENAME), config)


def create_sum_data_add1(config):
    possible_digits = []
    for index_i in range(config['class-size']):
        for index_j in range(config['class-size']):
            possible_digits.append([index_i, index_i + index_j])

    digit_sum_ones_place_obs = []
    digit_sum_tens_place_obs = []
    for index_i in range(2):
        for index_j in range(config['class-size']):
            for index_k in range(config['class-size']):
                digit_sum_ones_place_obs.append([index_i, index_j, index_k, (index_i + index_j + index_k) % 10])
                digit_sum_tens_place_obs.append([index_i, index_j, index_k, (index_i + index_j + index_k) // 10])

    placed_representation = []
    for i in range(0, config['max-sum'] + 1):
        representation = "%02d" % i
        placed_representation += [[int(representation[0]), int(representation[1]), i]]

    return possible_digits, digit_sum_ones_place_obs, digit_sum_tens_place_obs, placed_representation


def create_sum_data_add2(config):
    # Possible ones place digits.
    digits_sums = product(range(10), repeat=4)
    possible_ones_digits_dict = {}
    for digits_sum in digits_sums:
        if digits_sum[1] in possible_ones_digits_dict:
            possible_ones_digits_dict[digits_sum[1]].add(
                10 * digits_sum[0] + digits_sum[1] + 10 * digits_sum[2] + digits_sum[3])
        else:
            possible_ones_digits_dict[digits_sum[1]] = {
                10 * digits_sum[0] + digits_sum[1] + 10 * digits_sum[2] + digits_sum[3]}

    possible_ones_digits = []
    for key in possible_ones_digits_dict:
        for value in possible_ones_digits_dict[key]:
            possible_ones_digits.append([key, value])

    # Possible tens place digits.
    digits_sums = product(range(10), repeat=4)
    possible_tens_digits_dict = {}
    for digits_sum in digits_sums:
        if digits_sum[0] in possible_tens_digits_dict:
            possible_tens_digits_dict[digits_sum[0]].add(
                10 * digits_sum[0] + digits_sum[1] + 10 * digits_sum[2] + digits_sum[3])
        else:
            possible_tens_digits_dict[digits_sum[0]] = {
                10 * digits_sum[0] + digits_sum[1] + 10 * digits_sum[2] + digits_sum[3]}

    possible_tens_digits = []
    for key in possible_tens_digits_dict:
        for value in possible_tens_digits_dict[key]:
            possible_tens_digits.append([key, value])

    # Digit sum place observations.
    digit_sum_ones_place_obs = []
    digit_sum_tens_place_obs = []
    for index_i in range(2):
        for index_j in range(config['class-size']):
            for index_k in range(config['class-size']):
                digit_sum_ones_place_obs.append([index_i, index_j, index_k, (index_i + index_j + index_k) % 10])
                digit_sum_tens_place_obs.append([index_i, index_j, index_k, (index_i + index_j + index_k) // 10])

    # Placed Representation.
    placed_representation = []
    for i in range(0, config['max-sum'] + 1):
        representation = "%03d" % i
        placed_representation += [[int(representation[0]), int(representation[1]), int(representation[2]), i]]

    return possible_ones_digits, possible_tens_digits, digit_sum_ones_place_obs, digit_sum_tens_place_obs, placed_representation


def create_sum_data_add4(config):
    # Digit sum place observations.
    digit_sum_ones_place_obs = []
    digit_sum_tens_place_obs = []
    for index_i in range(2):
        for index_j in range(config['class-size']):
            for index_k in range(config['class-size']):
                digit_sum_ones_place_obs.append([index_i, index_j, index_k, (index_i + index_j + index_k) % 10])
                digit_sum_tens_place_obs.append([index_i, index_j, index_k, (index_i + index_j + index_k) // 10])

    # Placed Representation.
    placed_representation = []
    for i in range(0, config['max-sum'] + 1):
        representation = "%05d" % i
        placed_representation += [[
            int(representation[0]), int(representation[1]), int(representation[2]),
            int(representation[3]), int(representation[4]), i
        ]]

    return digit_sum_ones_place_obs, digit_sum_tens_place_obs, placed_representation


def write_shared_data(config, out_dir):
    if config['num-digits'] == 1:
        (possible_digits, digit_sum_ones_place_obs, digit_sum_tens_place_obs,
         placed_representation) = create_sum_data_add1(config)
        utils.write_psl_data_file(os.path.join(out_dir, 'possible-digit-obs.txt'), possible_digits)
        utils.write_psl_data_file(os.path.join(out_dir, 'digit-sum-ones-place-obs.txt'), digit_sum_ones_place_obs)
        utils.write_psl_data_file(os.path.join(out_dir, 'digit-sum-tens-place-obs.txt'), digit_sum_tens_place_obs)
        utils.write_psl_data_file(os.path.join(out_dir, 'placed-representation.txt'), placed_representation)

    elif config['num-digits'] == 2:
        (possible_ones_digits, possible_tens_digits,
         digit_sum_ones_place_obs, digit_sum_tens_place_obs, placed_representation) = create_sum_data_add2(config)
        utils.write_psl_data_file(os.path.join(out_dir, 'possible-ones-digit-obs.txt'), possible_ones_digits)
        utils.write_psl_data_file(os.path.join(out_dir, 'possible-tens-digit-obs.txt'), possible_tens_digits)
        utils.write_psl_data_file(os.path.join(out_dir, 'digit-sum-ones-place-obs.txt'), digit_sum_ones_place_obs)
        utils.write_psl_data_file(os.path.join(out_dir, 'digit-sum-tens-place-obs.txt'), digit_sum_tens_place_obs)
        utils.write_psl_data_file(os.path.join(out_dir, 'placed-representation.txt'), placed_representation)

    elif config['num-digits'] == 4:
        digit_sum_ones_place_obs, digit_sum_tens_place_obs, placed_representation = create_sum_data_add4(config)
        utils.write_psl_data_file(os.path.join(out_dir, 'digit-sum-ones-place-obs.txt'), digit_sum_ones_place_obs)
        utils.write_psl_data_file(os.path.join(out_dir, 'digit-sum-tens-place-obs.txt'), digit_sum_tens_place_obs)
        utils.write_psl_data_file(os.path.join(out_dir, 'placed-representation.txt'), placed_representation)

    else:
        raise Exception(f"Unsupported num-digits: {config['num-digits']}")

    utils.write_json_file(os.path.join(out_dir, CONFIG_FILENAME), config)


def fetch_data():
    os.makedirs(os.path.join(THIS_DIR, "..", "data", "mnist_raw_data"), exist_ok=True)
    mnist_dataset = torchvision.datasets.MNIST(os.path.join(THIS_DIR, "..", "data", "mnist_raw_data"), download=True)
    return mnist_dataset.data.numpy(), mnist_dataset.targets.numpy().astype(int)


def main():
    for dataset_id in DATASETS:
        config = DATASET_CONFIG[dataset_id]

        shared_out_dir = os.path.join(THIS_DIR, "..", "data", dataset_id)
        os.makedirs(shared_out_dir, exist_ok=True)
        if os.path.isfile(os.path.join(shared_out_dir, CONFIG_FILENAME)):
            print("Shared data already exists for %s. Skipping generation." % dataset_id)
        else:
            print("Generating shared data for %s." % dataset_id)
            write_shared_data(config, shared_out_dir)

        for split in range(config['num-splits']):
            for train_size in config['train-sizes']:
                config['train-size'] = train_size
                for unlabeled in config['unlabeled-ratios']:
                    config['unlabeled'] = unlabeled
                    for overlap in config['overlaps']:
                        config['overlap'] = overlap
                        out_dir = os.path.join(shared_out_dir, "split::%01d" % split, "train-size::%05d" % train_size, "unlabeled::%.2f" % unlabeled, "overlap::%.2f" % overlap)
                        os.makedirs(out_dir, exist_ok=True)

                        config['seed'] = 10 * (10 * train_size + split)
                        utils.seed_everything(config['seed'])

                        if os.path.isfile(os.path.join(out_dir, CONFIG_FILENAME)):
                            print("Data already exists for %s. Skipping generation." % out_dir)
                            continue

                        print("Generating data for %s." % out_dir)
                        features, labels = fetch_data()
                        write_specific_data(config, out_dir, features, labels)


if __name__ == '__main__':
    main()
