import albumentations as A
import argparse
import cv2
import json, os
import shutil

from tqdm import tqdm
from common import transform, convert_8coords_to_4coords, CheckExistsDir
from pycocotools.coco import COCO


def valid_data(coco_path):
    with open(coco_path) as json_file:
        json_data = json.load(json_file)

    over_point = 0
    under_point = 0
    same_point = 0

    for idx, ann in enumerate(json_data['annotations']):
        width = json_data['images'][0]['width']
        height = json_data['images'][0]['height']

        x1 = (ann['segmentation'][0][0], ann['segmentation'][0][1])
        x2 = (ann['segmentation'][0][2], ann['segmentation'][0][3])
        x3 = (ann['segmentation'][0][4], ann['segmentation'][0][5])
        x4 = (ann['segmentation'][0][6], ann['segmentation'][0][7])
        if x1 == x2 or x1 == x3 or x1 == x4 or x2 == x3 or x2 == x4 or x3 == x4:
            same_point += 1
            del json_data['annotations'][idx]
            continue

        for p_idx, point in enumerate(ann['segmentation'][0]):
            if point >= width:
                json_data['annotations'][idx]['segmentation'][0][p_idx] = width - 1
                over_point += 1
            if point < 0:
                json_data['annotations'][idx]['segmentation'][0][p_idx] = 0
                under_point += 1

    print('modify same_point :: ', same_point)
    print('modify over_point :: ', over_point)
    print('modify under_point :: ', under_point)

    json_str = json.dumps(json_data, indent=4)
    save_path = os.path.join(coco_path)
    with open(save_path, 'w') as f:
        f.write(json_str)

def main():
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--coco-path', type=str, help='')
    args = parser.parse_args()

    valid_data(args.coco_path)