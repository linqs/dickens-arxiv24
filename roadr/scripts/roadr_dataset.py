import os
import sys

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import DetrImageProcessor

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils import load_json_file
from roadr.constants import CLASS_SIZE
from roadr.constants import IMAGE_HEIGHT
from roadr.constants import IMAGE_WIDTH
from roadr.constants import LABEL_TYPES
from roadr.constants import ORIGINAL_LABEL_MAPPING
from roadr.constants import QUERY_SIZE
from roadr.constants import RGB_IMAGES_DIR


class RoadRDataset(Dataset):
    def __init__(self,
                 videos: list,
                 supervised_videos: list,
                 annotations_path: str,
                 image_resize: float,
                 max_frames: int,
                 load_image: bool = True):
        self.videos = videos
        self.supervised_videos = supervised_videos
        self.annotations_path = annotations_path
        self.image_resize = image_resize
        self.max_frames = max_frames
        self.load_image = load_image

        self.base_rgb_images_dir = RGB_IMAGES_DIR
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", size={"shortest_edge": self.image_height(), "longest_edge": self.image_width()})

        self.annotations = load_json_file(self.annotations_path)["db"]

        self.frame_indexes = {}
        self.frame_ids = []
        self.load_labels = False
        self.load_frame_ids()

        self.labels_mask = torch.ones(size=(len(self.frame_ids), QUERY_SIZE, CLASS_SIZE + 1), dtype=torch.int8)
        for frame_id in self.frame_ids:
            if frame_id[0] not in self.supervised_videos:
                self.labels_mask[self.frame_indexes[(frame_id[0], frame_id[1])], :, 10:-1] = 0

    def load_frame_ids(self):
        for video in self.videos:
            num_video_frames = 0

            rgb_images_dir = os.path.join(self.base_rgb_images_dir, video)

            for frame_file_name in sorted(os.listdir(rgb_images_dir)):
                if self.max_frames != 0 and num_video_frames >= self.max_frames:
                    break

                if "annos" not in self.annotations[video]['frames'][str(int(frame_file_name.strip(".jpg")))]:
                    continue

                self.frame_indexes[(video, frame_file_name)] = len(self.frame_ids)
                self.frame_ids.append([video, frame_file_name])
                num_video_frames += 1

    def load_frame(self, frame_index: int):
        video, framename = self.frame_ids[frame_index]

        image = {'pixel_values': [[]], 'pixel_mask': [[]]}
        if self.load_image:
            image = self.processor(Image.open(os.path.join(self.base_rgb_images_dir, video, framename)))

        annotations = self.annotations[video]['frames'][str(int(framename[:-4]))]

        frame_labels = torch.zeros(size=(QUERY_SIZE, CLASS_SIZE + 1), dtype=torch.int8)
        frame_boxes = torch.zeros(size=(QUERY_SIZE, 4), dtype=torch.float32)

        for bounding_box_index, bounding_box in enumerate(annotations['annos']):
            frame_boxes[bounding_box_index] = torch.Tensor(annotations['annos'][bounding_box]['box'])
            frame_labels[bounding_box_index, -1] = 1

            for label_type in LABEL_TYPES:
                for label_id in annotations['annos'][bounding_box][label_type + '_ids']:
                    if int(label_id) not in ORIGINAL_LABEL_MAPPING[label_type]:
                        continue
                    frame_labels[bounding_box_index][ORIGINAL_LABEL_MAPPING[label_type][int(label_id)][0]] = 1

        return image['pixel_values'][0], image['pixel_mask'][0], frame_labels, frame_boxes, self.labels_mask[frame_index]

    def image_height(self):
        return int(IMAGE_HEIGHT * self.image_resize)

    def image_width(self):
        return int(IMAGE_WIDTH * self.image_resize)

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, frame_index: int):
        pixel_values, pixel_mask, labels, boxes, labels_mask = self.load_frame(frame_index)
        return frame_index, pixel_values, pixel_mask, labels, boxes, labels_mask
