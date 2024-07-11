#!/usr/bin/env python3

# A setup script similar to setup.py,
# except this script will not rely on existing puzzles.
# Instead, fully new puzzles will be generated.

import datetime
import json
import math
import os
import random
import sys

import numpy
import numpy as np
import torchvision

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'data', '{}'))

sys.path.append(os.path.join(THIS_DIR, '..', '..'))

import utils

SUBPATH_FORMAT = os.path.join('mnist-{:01d}x{:01d}', 'split::{:01d}', 'train-size::{:04d}', 'num-clues::{:02d}', 'unlabeled::{:1.2f}')
CONFIG_PATH = os.path.join(DATA_DIR, 'config.json')

ENTITY_DATA_MAP_PATH = os.path.join(DATA_DIR, 'neural-data-{}.txt')
DIGIT_TARGETS_PATH = os.path.join(DATA_DIR, 'digit-targets')
DIGIT_TRUTH_PATH = os.path.join(DATA_DIR, 'digit-truth')

EMPTY_BLOCK_OBS_PATH = os.path.join(DATA_DIR, 'empty-block-obs')
EMPTY_BLOCK_TARGETS_PATH = os.path.join(DATA_DIR, 'empty-block-targets')
BOX_OBS_PATH = os.path.join(DATA_DIR, 'box-obs')

LABELS = list(range(0, 10))

# MNIST images are 28 x 28 = 784.
MNIST_DIMENSION = 28

CONFIG = {
    "num-splits": 5,
    "dimension": 9,
    "num-clues": 30,
    "num-train-puzzles": [20],
    "num-valid-puzzles": 100,
    "num-test-puzzles": 100,
    "unlabeled-ratios": [0.00, 0.50, 0.90, 0.95, 1.00]
}

FEW_SHOT_SAMPLES = 5

SIGNIFICANT_DIGITS = 4


class DigitChooser(object):
    # digits: {label: [image, ...], ...}
    def __init__(self, digits):
        self.digits = digits
        self.nextIndexes = {label: 0 for label in digits}

    # Takes the next image for a digit,
    def takeDigit(self, label):
        assert(self.nextIndexes[label] < len(self.digits[label]))

        image = self.digits[label][self.nextIndexes[label]]
        self.nextIndexes[label] += 1
        return image

    # Get a digit randomly from anywhere in the sequence.
    def getDigit(self, label):
        return random.choice(self.digits[label])


def create_digit_chooser(labels, num_train_puzzles, num_valid_puzzles, num_test_puzzles):
    digit_images = loadMNIST()

    unique_digit_count = (num_train_puzzles + num_test_puzzles + num_valid_puzzles) * len(labels)

    unique_digits = {label: digit_images[label][0:unique_digit_count] for label in labels}
    digits = {label: digits for (label, digits) in unique_digits.items()}

    for label in labels:
        digits[label].extend(random.choices(unique_digits[label], k=int(unique_digit_count)))
        random.shuffle(digits[label])

    return DigitChooser(digits)


def normalizeMNISTImages(images):
    (numImages, width, height) = images.shape

    # Flatten out the images into a 1d array.
    images = images.reshape(numImages, width * height)

    # Normalize the greyscale intensity to [0,1].
    images = images / 255.0

    # Round so that the output is significantly smaller.
    images = images.round(decimals=SIGNIFICANT_DIGITS)

    return images


# Returns: {digit: [image, ...], ...}
def loadMNIST(shuffle = True):
    os.makedirs(os.path.join(THIS_DIR, "..", "data", "mnist_raw_data"), exist_ok=True)
    mnist_dataset = torchvision.datasets.MNIST(os.path.join(THIS_DIR, "..", "data", "mnist_raw_data"), download=True)
    (images, labels) = (mnist_dataset.data, mnist_dataset.targets)

    images = normalizeMNISTImages(images.numpy())
    mnistLabels = LABELS

    # {digit: [image, ...], ...}
    digits = {label: [] for label in mnistLabels}
    for i in range(len(images)):
        digits[int(labels[i])].append(images[i])

    if (shuffle):
        for label in mnistLabels:
            random.shuffle(digits[label])
            random.shuffle(digits[label])

    return digits


def generatePuzzle(digitChooser, labels, num_clues):
    puzzleImages = [[None] * len(labels) for i in range(len(labels))]
    puzzleLabels = [[None] * len(labels) for i in range(len(labels))]

    # Keep track of the possible options for each location.
    # [row][col][label].
    # Remove options as we add to the puzzle.
    options = [[list(labels) for j in range(len(labels))] for i in range(len(labels))]

    blockSize = int(math.sqrt(len(labels)))

    for row in range(len(labels)):
        for col in range(len(labels)):
            if (len(options[row][col]) == 0):
                # Failed to create a puzzle, try again.
                return None, None

            label = random.choice(options[row][col])
            options[row][col].clear()

            puzzleLabels[row][col] = label

            blockRow = row // blockSize
            blockCol = col // blockSize

            # Remove the chosen digit from row/col/grid options.
            for i in range(len(labels)):
                if label in options[i][col]:
                    options[i][col].remove(label)

                if label in options[row][i]:
                    options[row][i].remove(label)

                for j in range(len(labels)):
                    if (i // blockSize == blockRow and j // blockSize == blockCol):
                        if label in options[i][j]:
                            options[i][j].remove(label)

    # Once we have a complete puzzle, choose the digits.
    for row in range(len(labels)):
        for col in range(len(labels)):
            puzzleImages[row][col] = digitChooser.takeDigit(puzzleLabels[row][col])

    # Only keep num_clues clues by randomly removing images and labels from the puzzle.
    for i in range(len(labels) ** 2 - num_clues):
        while True:
            row = random.randrange(len(labels))
            col = random.randrange(len(labels))

            if (puzzleLabels[row][col] is not None):
                break

        puzzleImages[row][col] = np.zeros_like(puzzleImages[row][col], dtype=np.float32)
        puzzleLabels[row][col] = None

    return puzzleImages, puzzleLabels


def generatePuzzles(digit_chooser, labels, num_puzzles, num_clues):
    # [puzzleIndex][row][col]
    allPuzzleImages = []
    allDigitLabels = []

    count = 0

    while (count < num_puzzles):
        puzzleImages, digitLabels = generatePuzzle(digit_chooser, labels, num_clues)
        if (puzzleImages is None):
            continue

        allPuzzleImages.append(puzzleImages)
        allDigitLabels.append(digitLabels)

        count += 1

    return numpy.stack(allPuzzleImages), numpy.stack(allDigitLabels)


def get_puzzle_digits(puzzles, puzzle_digits, labels, start_index, label_ratio, partition):
    digit_targets = []
    digit_truths = []
    digit_features = []
    digit_labels = []

    empty_block_obs = []
    empty_block_targets = []
    box_obs = []

    for index in range(len(puzzles)):
        puzzleId = start_index + index

        for row in range(len(puzzles[index])):
            for col in range(len(puzzles[index][row])):
                if puzzle_digits[index][row][col] is not None:
                    # The second to last entry of the feature vector is the digit label.
                    # The final entry of the feature vector indicates the partition of the data.
                    # 0 is labeled training, 1 is training, 2 is validation, 3 is test.
                    if partition == 'train':
                        digit_features.append([puzzleId, row, col] + puzzles[index][row][col].tolist() + [int(puzzle_digits[index][row][col]), 1])
                    elif partition == 'valid':
                        digit_features.append([puzzleId, row, col] + puzzles[index][row][col].tolist() + [int(puzzle_digits[index][row][col]), 2])
                    elif partition == 'test':
                        digit_features.append([puzzleId, row, col] + puzzles[index][row][col].tolist() + [int(puzzle_digits[index][row][col]), 3])
                    else:
                        raise ValueError("Invalid partition: " + partition)

                    digit_labels.append(puzzle_digits[index][row][col])

                    for digit in labels:
                        digit_targets.append([puzzleId, row, col, digit])

                else:
                    empty_block_obs.append([puzzleId, row, col])
                    for digit in labels:
                        empty_block_targets.append([puzzleId, row, col, digit])

    digit_labels = np.array(digit_labels)

    if partition == 'train':
        # Add label_ratio% of the digits to digit_truths and set the final entry of the feature vector to 0.
        truth_indices = get_truth_indices(digit_labels, label_ratio)
        for digit_index in truth_indices:
            puzzleId, row, col = digit_features[digit_index][0:3]
            true_digit = digit_features[digit_index][-2]
            digit_features[digit_index][-1] = 0
            for digit_label in labels:
                digit_truths.append([puzzleId, row, col, digit_label, int(digit_label == true_digit)])
    else:
        # Add all digits to digit_truths.
        for digit_index in range(len(digit_features)):
            puzzleId, row, col = digit_features[digit_index][0:3]
            true_digit = digit_features[digit_index][-2]
            for digit_label in labels:
                digit_truths.append([puzzleId, row, col, digit_label, int(digit_label == true_digit)])

    dim = len(puzzles[0])
    for row in range(dim):
        for column in range(dim):
            row_modulo = math.floor((row % dim) / math.sqrt(dim))
            column_modulo = math.floor((column % dim) / math.sqrt(dim))
            box_obs.append([math.floor(row_modulo * dim / math.sqrt(dim)) + column_modulo, row, column])

    return digit_targets, digit_truths, digit_features, empty_block_obs, empty_block_targets, box_obs


def get_truth_indices(digit_labels, label_ratio):
    if label_ratio > 0:
        return np.random.choice(len(digit_labels), size=int(len(digit_labels) * label_ratio), replace=False)

    else:
        # Sample FEW_SHOT_SAMPLES of each class.
        truth_indices = np.array([], dtype=int)
        for _class in np.unique(digit_labels):
            indices = np.where(digit_labels == _class)[0]
            if len(indices) > FEW_SHOT_SAMPLES:
                truth_indices = np.concatenate((truth_indices, np.random.choice(indices, size=FEW_SHOT_SAMPLES, replace=False)))
            else:
                truth_indices = np.concatenate((truth_indices, indices))

        return truth_indices


def write_data(subpath, labels, unlabeled,
               train_puzzles, train_puzzle_digits,
               valid_puzzles, valid_puzzle_digits,
               test_puzzles, test_puzzle_digits,):
    partition_data = {
        'train': [train_puzzles, train_puzzle_digits, labels, 0],
        'valid': [valid_puzzles, valid_puzzle_digits, labels, len(train_puzzles)],
        'test': [test_puzzles, test_puzzle_digits, labels, len(train_puzzles) + len(test_puzzles)],
    }

    all_digit_features = []

    for partition in partition_data:
        suffix = "-" + partition + ".txt"

        puzzles, puzzle_digits, labels, offset = partition_data[partition]
        if partition == 'train':
            digit_targets, digit_truths, digit_features, empty_block_obs, empty_block_targets, box_obs = get_puzzle_digits(puzzles, puzzle_digits, labels, offset, 1 - unlabeled, partition)
        else:
            digit_targets, digit_truths, digit_features, empty_block_obs, empty_block_targets, box_obs = get_puzzle_digits(puzzles, puzzle_digits, labels, offset, 0, partition)

        all_digit_features.extend(digit_features)

        utils.write_psl_data_file(DIGIT_TARGETS_PATH.format(subpath) + suffix, digit_targets)
        utils.write_psl_data_file(DIGIT_TRUTH_PATH.format(subpath) + suffix, digit_truths)

        utils.write_psl_data_file(EMPTY_BLOCK_OBS_PATH.format(subpath) + suffix, empty_block_obs)
        utils.write_psl_data_file(EMPTY_BLOCK_TARGETS_PATH.format(subpath) + suffix, empty_block_targets)

        utils.write_psl_data_file(BOX_OBS_PATH.format(subpath) + suffix, box_obs)

    for partition in partition_data:
        utils.write_psl_data_file(ENTITY_DATA_MAP_PATH.format(subpath, partition), all_digit_features)


def build_dataset(digit_chooser, labels, split, num_clues, num_train_puzzles, num_valid_puzzles, num_test_puzzles, unlabeled, seed):
    subpath = SUBPATH_FORMAT.format(len(labels), len(labels), split, num_train_puzzles, num_clues, unlabeled)

    config_path = CONFIG_PATH.format(subpath)
    if os.path.isfile(config_path):
        print("Found existing config file, skipping generation. " + config_path)
        return
    print("Generating data for %s." % config_path)

    train_puzzles, train_puzzle_digits = generatePuzzles(digit_chooser, labels, num_train_puzzles, num_clues)
    valid_puzzles, valid_puzzle_digits = generatePuzzles(digit_chooser, labels, num_valid_puzzles, num_clues)
    test_puzzles, test_puzzle_digits = generatePuzzles(digit_chooser, labels, num_test_puzzles, num_clues)

    os.makedirs(DATA_DIR.format(subpath), exist_ok=True)

    write_data(subpath, labels, unlabeled,
               train_puzzles, train_puzzle_digits,
               valid_puzzles, valid_puzzle_digits,
               test_puzzles, test_puzzle_digits)

    config = {
        'labels': labels,
        'num_train_puzzles': num_train_puzzles,
        'num_test_puzzles': num_test_puzzles,
        'num_valid_puzzles': num_valid_puzzles,
        'seed': seed,
        'timestamp': str(datetime.datetime.now()),
        'generator': os.path.basename(os.path.realpath(__file__)),
    }

    with open(config_path, 'w') as file:
        json.dump(config, file, indent=4)


def main():
    config = CONFIG

    for train_size in config["num-train-puzzles"]:
        labels = list(LABELS[0: config["dimension"]])

        for split in range(config["num-splits"]):
            for unlabeled in config['unlabeled-ratios']:

                config['seed'] = 10 * (10 * train_size + split)
                utils.seed_everything(config['seed'])

                digits = create_digit_chooser(labels, train_size, config["num-valid-puzzles"], config["num-test-puzzles"])
                build_dataset(digits, labels, split, config["num-clues"], train_size, config["num-valid-puzzles"],
                              config["num-test-puzzles"], unlabeled, config['seed'])


if __name__ == '__main__':
    main()
