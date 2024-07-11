import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from roadr.constants import ACTION_CLASSES
from roadr.constants import AGENT_CLASSES
from roadr.constants import BASE_DATA_DIR
from roadr.constants import BATCH_SIZE
from roadr.constants import BOUNDING_BOX_CLASSES
from roadr.constants import BOX_CONFIDENCE_CLASS
from roadr.constants import CONSTRAINTS_PATH
from roadr.constants import CLASS_SIZE
from roadr.constants import LOCATION_CLASSES
from roadr.constants import QUERY_SIZE
from roadr.constants import RGB_IMAGES_DIR
from roadr.constants import SYMBOLIC_DATA_DIR
from roadr.constants import VIDEOS_DIR
from utils import load_csv_file
from utils import write_psl_data_file


def fetch_data():
    os.system("cd {0} && ./fetch_data.sh".format(BASE_DATA_DIR))


def video_to_jpgs(video_path, rgb_images_dir):
    video_filename = os.path.basename(video_path)[:-4]
    os.makedirs(os.path.join(rgb_images_dir, video_filename), exist_ok=True)

    if len(os.listdir(os.path.join(rgb_images_dir, video_filename))) > 0:
        print("Video {0} already processed: Skipping".format(video_filename))
        return

    command = 'ffmpeg  -i {} -q:v 1 {}/%05d.jpg'.format(video_path, os.path.join(rgb_images_dir, video_filename))
    os.system(command)


def load_constraint_file(path):
    raw_constraints = load_csv_file(path)

    constraints = []
    for index_i in range(len(raw_constraints) - 1):
        for index_j in range(len(raw_constraints[index_i]) - 1):
            constraints.append([index_i, index_j, int(raw_constraints[index_i + 1][index_j + 1])])

    return constraints


def generate_symbolic_data(experiment_dir, batch_size):
    os.makedirs(experiment_dir, exist_ok=True)

    entity_data_map = []
    entity_targets = []
    for batch_index in range(batch_size):
        for bounding_box_index in range(QUERY_SIZE):
            entity_data_map.append([batch_index, bounding_box_index])
            for class_index in range(CLASS_SIZE + len(BOX_CONFIDENCE_CLASS) + len(BOUNDING_BOX_CLASSES)):
                entity_targets.append([batch_index, bounding_box_index, class_index])

    co_occurrence = load_constraint_file(CONSTRAINTS_PATH)

    write_psl_data_file(os.path.join(experiment_dir, "classes-agent.txt"), AGENT_CLASSES)
    write_psl_data_file(os.path.join(experiment_dir, "classes-action.txt"), ACTION_CLASSES)
    write_psl_data_file(os.path.join(experiment_dir, "classes-location.txt"), LOCATION_CLASSES)
    write_psl_data_file(os.path.join(experiment_dir, "classes-box-confidence.txt"), BOX_CONFIDENCE_CLASS)
    write_psl_data_file(os.path.join(experiment_dir, "classes-bounding-box.txt"), BOUNDING_BOX_CLASSES)

    write_psl_data_file(os.path.join(experiment_dir, "entity-data-map.txt"), entity_data_map)
    write_psl_data_file(os.path.join(experiment_dir, "entity-targets.txt"), entity_targets)

    write_psl_data_file(os.path.join(experiment_dir, "co-occurrence.txt"), co_occurrence)


def main():
    print("Fetching data.")
    fetch_data()

    print("Converting videos to jpgs.")
    for video_filename in os.listdir(VIDEOS_DIR):
        if not video_filename.endswith(".mp4"):
            continue
        video_to_jpgs(os.path.join(VIDEOS_DIR, video_filename), RGB_IMAGES_DIR)

    print("Generating symbolic data.")
    experiment_dir = os.path.join(SYMBOLIC_DATA_DIR, "experiment::batch-size-" + str(BATCH_SIZE))
    generate_symbolic_data(experiment_dir, BATCH_SIZE)


if __name__ == '__main__':
    main()
