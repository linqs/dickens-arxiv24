import os

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
BASE_CLI_DIR = os.path.join(THIS_DIR, "cli")
BASE_DATA_DIR = os.path.join(THIS_DIR, "data")
BASE_EXPERIMENTS_DIR = os.path.join(THIS_DIR, "experiments")
BASE_MODEL_DIR = os.path.join(THIS_DIR, "models")
BASE_RESULTS_DIR = os.path.join(THIS_DIR, "results")

CONFIG_FILENAME = "config.json"
TRAINED_MODEL_NAME = "trained-model.pt"
CURRENT_MODEL_NAME = "current-model.pt"

BATCH_SIZE = 4