#!./dataset_env python(python 3.7)
import albumentations as A
import os
import shutil

nia_categories = ['crane', 'ship', 'individual container', 'grouped container']

NIA_DET_CLASS_MAPPING = [{"id": 0, "name": "crane", "supercategory": "crane"},
                         {"id": 1, "name": "container", "supercategory": "container"},
                         {"id": 2, "name": "small_ship", "supercategory": "ship"},
                         {"id": 3, "name": "middle_ship", "supercategory": "ship"},
                         {"id": 4, "name": "large_ship", "supercategory": "ship"}
                         ]

transform = A.Compose([# Geometric transform
                       A.VerticalFlip(p=0.7),
                       A.HorizontalFlip(p=0.7),
                       A.RandomRotate90(p=0.7),
                       # Light transform
                       A.ChannelShuffle(p=0.7),
                       A.RandomBrightnessContrast(brightness_limit=(0.0,0.2), p=0.8),
                       # Noise transform
                       A.OneOf([
                           A.Blur(p=1),
                           A.GaussianBlur(p=1),
                           A.MedianBlur(p=1),
                           A.RGBShift(p=1)], p=0.5)],
                      keypoint_params=A.KeypointParams(format='xy'))

def convert_8coords_to_4coords(coords):
    x_coords = coords[0::2]
    y_coords = coords[1::2]

    xmin = min(x_coords)
    ymin = min(y_coords)

    xmax = max(x_coords)
    ymax = max(y_coords)

    w = xmax - xmin
    h = ymax - ymin

    return [xmin, ymin, w, h]

def DefaultJson_labelme():
    export_data = {}

    export_data["version"] = "4.5.7"
    export_data["flags"] = {}
    export_data['info'] = {"description": "CONTEC Dataset",
                           "url": "https://www.contec.kr",
                           "version": 2021,
                           "contributor": "Contec",
                           "data_created": 2021}
    export_data['licenses'] = {"url": "https://www.contec.kr",
                               "name": "contec"}

    export_data['shapes'] = []

    return export_data

def DefaultJson(ignore_cat):
    export_data = {}

    export_data["type"] = "instances"

    export_data['info'] = {"description": "CONTEC Dataset",
                           "url": "https://www.contec.kr",
                           "version": 2020,
                           "contributor": "Contec",
                           "data_created": 2020}

    export_data['licenses'] = {"url": "https://www.contec.kr",
                               "name": "contec"}

    export_data['categories'] = [ele for ele in DET_CLASS_MAPPING if ele['name'] not in ignore_cat]

    return export_data


def make_nia_json(points, ingest_time, type_id, image_id, type_name):
    feature = {}

    feature['geometry'] = {}
    feature['geometry']['coordinates'] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    feature['geometry']['type'] = "Polygon"

    feature['properties'] = {}
    feature['properties']['object_imcoords'] = ','.join(str(e) for e in points)
    feature['properties']['object_angle'] = 0
    feature['properties']['building_imcoords'] = "EMPTY"
    feature['properties']['road_imcoords'] = "EMPTY"
    feature['properties']['image_id'] = image_id
    feature['properties']['ingest_time'] = ingest_time
    feature['properties']['type_id'] = type_id
    feature['properties']['type_name'] = type_name

    feature['type'] = "Feature"

    return feature

def CheckExistsDir(check_dir, is_remove_sub_dir=True):
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)
    elif os.path.exists(check_dir) and is_remove_sub_dir:
        for file in os.scandir(check_dir):
            if os.path.isdir(file.path):
                shutil.rmtree(file.path)
            elif os.path.isfile(file.path):
                os.remove(file.path)

