import os

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
BASE_CLI_DIR = os.path.join(THIS_DIR, "cli")
BASE_DATA_DIR = os.path.join(THIS_DIR, "data")
BASE_EXPERIMENTS_DIR = os.path.join(THIS_DIR, "experiments")
BASE_MODEL_DIR = os.path.join(THIS_DIR, "models")
BASE_RESULTS_DIR = os.path.join(THIS_DIR, "results")

RGB_IMAGES_DIR = os.path.join(BASE_DATA_DIR, "rgb-images")
SYMBOLIC_DATA_DIR = os.path.join(BASE_DATA_DIR, "symbolic-data")

ANNOTATIONS_PATH = os.path.join(BASE_DATA_DIR, "raw-data", "road_trainval_v1.0.json")
CONSTRAINTS_PATH = os.path.join(BASE_DATA_DIR, "raw-data", "co-occurrence.csv")
VIDEOS_DIR = os.path.join(BASE_DATA_DIR, "raw-data", "videos")

CONFIG_FILENAME = "config.json"
TRAINED_MODEL_NAME = "trained-model.pt"
CURRENT_MODEL_NAME = "current-model.pt"
TRAINING_HISTORY_FILENAME = "training-history.csv"

BATCH_SIZE = 4

IMAGE_HEIGHT = 960
IMAGE_WIDTH = 1280

CLASS_SIZE = 41
QUERY_SIZE = 100

BOX_CONFIDENCE_THRESHOLD = 0.7
IOU_THRESHOLD = 0.5
LABEL_CONFIDENCE_THRESHOLD = 0.2

AGENT_CLASSES = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
ACTION_CLASSES = [[10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28]]
LOCATION_CLASSES = [[29], [30], [31], [32], [33], [34], [35], [36], [37], [38], [39], [40]]
BOX_CONFIDENCE_CLASS = [[41]]
BOUNDING_BOX_CLASSES = [[42], [43], [44], [45]]

LABEL_TYPES = ['agent', 'action', 'loc']

VIDEO_PARTITIONS = {
    "train": ["2014-06-25-16-45-34_stereo_centre_02",
              "2014-07-14-14-49-50_stereo_centre_01",
              "2014-08-08-13-15-11_stereo_centre_01",
              "2014-08-11-10-59-18_stereo_centre_02",
              "2014-11-14-16-34-33_stereo_centre_06",
              "2014-11-18-13-20-12_stereo_centre_05",
              "2014-11-21-16-07-03_stereo_centre_01",
              "2014-12-09-13-21-02_stereo_centre_01",
              "2015-02-03-08-45-10_stereo_centre_02",
              "2015-02-06-13-57-16_stereo_centre_02",
              "2015-02-13-09-16-26_stereo_centre_05",
              "2015-02-24-12-32-19_stereo_centre_04",
              "2015-03-03-11-31-36_stereo_centre_01",
              "2014-07-14-15-42-55_stereo_centre_03",
              "2015-02-03-19-43-11_stereo_centre_04"],
    "test": ["2014-06-26-09-53-12_stereo_centre_02",
             "2014-11-25-09-18-32_stereo_centre_04",
             "2015-02-13-09-16-26_stereo_centre_02"]
}

SUPERVISED_VIDEOS = {
    "02": ["2014-06-25-16-45-34_stereo_centre_02",
           "2015-02-03-08-45-10_stereo_centre_02"],
    "04": ["2014-06-25-16-45-34_stereo_centre_02",
           "2014-11-14-16-34-33_stereo_centre_06",
           "2015-02-03-08-45-10_stereo_centre_02",
           "2015-03-03-11-31-36_stereo_centre_01"],
    "08": ["2014-06-25-16-45-34_stereo_centre_02",
           "2014-08-08-13-15-11_stereo_centre_01",
           "2014-11-14-16-34-33_stereo_centre_06",
           "2014-11-21-16-07-03_stereo_centre_01",
           "2015-02-03-08-45-10_stereo_centre_02",
           "2015-02-13-09-16-26_stereo_centre_05",
           "2015-03-03-11-31-36_stereo_centre_01",
           "2015-02-03-19-43-11_stereo_centre_04"],
    "15": ["2014-06-25-16-45-34_stereo_centre_02",
           "2014-07-14-14-49-50_stereo_centre_01",
           "2014-08-08-13-15-11_stereo_centre_01",
           "2014-08-11-10-59-18_stereo_centre_02",
           "2014-11-14-16-34-33_stereo_centre_06",
           "2014-11-18-13-20-12_stereo_centre_05",
           "2014-11-21-16-07-03_stereo_centre_01",
           "2014-12-09-13-21-02_stereo_centre_01",
           "2015-02-03-08-45-10_stereo_centre_02",
           "2015-02-06-13-57-16_stereo_centre_02",
           "2015-02-13-09-16-26_stereo_centre_05",
           "2015-02-24-12-32-19_stereo_centre_04",
           "2015-03-03-11-31-36_stereo_centre_01",
           "2014-07-14-15-42-55_stereo_centre_03",
           "2015-02-03-19-43-11_stereo_centre_04"],
}

LABEL_MAPPING = {
    0: "Ped",
    1: "Car",
    2: "Cyc",
    3: "Mobike",
    4: "MedVeh",
    5: "LarVeh",
    6: "Bus",
    7: "EmVeh",
    8: "TL",
    9: "OthTL",
    10: "Red",
    11: "Amber",
    12: "Green",
    13: "MovAway",
    14: "MovTow",
    15: "Mov",
    16: "Brake",
    17: "Stop",
    18: "IncatLft",
    19: "IncatRgt",
    20: "HazLit",
    21: "TurLft",
    22: "TurRht",
    23: "Ovtak",
    24: "Wait2X",
    25: "XingFmLft",
    26: "XingFmRht",
    27: "Xing",
    28: "PushObj",
    29: "VehLane",
    30: "OutgoLane",
    31: "OutgoCycLane",
    32: "IncomLane",
    33: "IncomCycLane",
    34: "Pav",
    35: "LftPav",
    36: "RhtPav",
    37: "Jun",
    38: "xing",
    39: "BusStop",
    40: "parking"
}

ORIGINAL_LABEL_MAPPING = {
    "agent": {
        0: [0, "Ped"],
        1: [1, "Car"],
        2: [2, "Cyc"],
        3: [3, "Mobike"],
        5: [4, "MedVeh"],
        6: [5, "LarVeh"],
        7: [6, "Bus"],
        8: [7, "EmVeh"],
        9: [8, "TL"],
        10: [9, "OthTL"]
    },
    "action": {
        0: [10, "Red"],
        1: [11, "Amber"],
        2: [12, "Green"],
        3: [13, "MovAway"],
        4: [14, "MovTow"],
        5: [15, "Mov"],
        7: [16, "Brake"],
        8: [17, "Stop"],
        9: [18, "IncatLft"],
        10: [19, "IncatRht"],
        11: [20, "HazLit"],
        12: [21, "TurLft"],
        13: [22, "TurRht"],
        16: [23, "Ovtak"],
        17: [24, "Wait2X"],
        18: [25, "XingFmLft"],
        19: [26, "XingFmRht"],
        20: [27, "Xing"],
        21: [28, "PushObj"]
    },
    "loc": {
        0: [29, "VehLane"],
        1: [30, "OutgoLane"],
        2: [31, "OutgoCycLane"],
        3: [32, "IncomLane"],
        4: [33, "IncomCycLane"],
        5: [34, "Pav"],
        6: [35, "LftPav"],
        7: [36, "RhtPav"],
        8: [37, "Jun"],
        9: [38, "xing"],
        10: [39, "BusStop"],
        11: [40, "parking"]
    }
}