import json, os
import datetime
import xml.etree.ElementTree as ET

from common import DefaultJson_labelme, DefaultJson, CheckExistsDir, nia_categories, make_nia_json, NIA_DET_CLASS_MAPPING
from tqdm import tqdm
from shutil import copyfile

class Transform():
    def __init__(self, data_dir, transform_type):
        self.data_dir = data_dir
        self.transform_type = transform_type
        self.xml_dir = os.path.join(self.data_dir, 'xml')
        self.labeling_dir = os.path.join(self.data_dir, 'objects_labeling')
        self.objects_data_dir = os.path.join(self.data_dir, 'objects_data')
        self.labelme_data_dir = os.path.join(self.data_dir, 'objects_labeling_labelme')
        self.from_labelme_dir = os.path.join(self.data_dir, 'objects_labeling_from_labelme')
        self.coco_dir = os.path.join(self.data_dir, 'objects_labeling_coco')
        self.coco_images_dir = os.path.join(self.coco_dir, 'Images')
        self.coco_annotations_dir = os.path.join(self.coco_dir, 'Annotations')

    def LabelImg2Labelme(self, ignore_cat_list):
        CheckExistsDir(self.labeling_dir)

        xml_list = os.listdir(self.xml_dir)

        for xml_file in tqdm(xml_list):
            file_path = os.path.join(self.xml_dir, xml_file)
            doc = ET.parse(file_path)
            root = doc.getroot()

            file_name = root.find('filename').text
            image_id = file_name+'.png'
            object_tag = root.findall("object")

            export_data = {}
            export_data['features'] = []
            for object in object_tag:
                points = []
                type_name = object.find('name').text

                points.append(object.find('robndbox').find('x1').text)
                points.append(object.find('robndbox').find('y1').text)
                points.append(object.find('robndbox').find('x2').text)
                points.append(object.find('robndbox').find('y2').text)
                points.append(object.find('robndbox').find('x3').text)
                points.append(object.find('robndbox').find('y3').text)
                points.append(object.find('robndbox').find('x4').text)
                points.append(object.find('robndbox').find('y4').text)

                print(points)

                ingest_time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
                type_id = str(nia_categories.index(type_name))

                feature = make_nia_json(points, ingest_time, type_id, image_id, type_name)
                export_data['features'].append(feature)
            export_data['type'] = "FeatureCollection"

            json_str = json.dumps(export_data, indent=4)
            save_path = os.path.join(self.labeling_dir, '{}.json'.format(file_name))
            with open(save_path, 'w') as f:
                f.write(json_str)


    def Labelme2coco(self):
        CheckExistsDir(self.coco_dir)
        CheckExistsDir(self.coco_images_dir)
        CheckExistsDir(self.coco_annotations_dir)

        labelme_list = os.listdir(self.labelme_data_dir)
        png_list = [file for file in labelme_list if file.endswith(".png")]
        json_list = [file for file in labelme_list if file.endswith(".json")]

        export_data = DefaultJson(ignore_cat=[], data_type=NIA_DET_CLASS_MAPPING)
        export_data['images'] = []
        export_data['annotations'] = []
        ann_id = 0

        for idx, json_file in tqdm(enumerate(json_list)):
            json_file_path = os.path.join(self.labelme_data_dir, json_file)
            with open(json_file_path) as json_read:
                json_data = json.load(json_read)

            image = {}
            image['file_name'] = json_data['imagePath']
            image['id'] = idx
            image['width'] = 1024
            image['height'] = 1024
            export_data['images'].append(image)

            for shape in json_data['shapes']:
                points = []
                points.extend(shape['points'][0])
                points.extend(shape['points'][1])
                points.extend(shape['points'][2])
                points.extend(shape['points'][3])
                x1 = (((points[0] - points[2]) ** 2) + ((points[1] - points[3]) ** 2)) ** 0.5
                x2 = (((points[2] - points[4]) ** 2) + ((points[3] - points[5]) ** 2)) ** 0.5
                area = x1 * x2

                annotation = {}
                annotation['difficult'] = "-1"
                annotation['area'] = area
                annotation['category_id'] = nia_categories.index(shape['label'])
                annotation['segmentation'] = [points]
                annotation['iscrowd'] = 0
                annotation['bbox'] = [min(points[0::2]), min(points[1::2]), max(points[0::2]) - min(points[0::2]),
                                      max(points[1::2]) - min(points[1::2])]
                annotation['image_id'] = idx
                annotation['id'] = ann_id

                ann_id += 1
                export_data['annotations'].append(annotation)
        json_str = json.dumps(export_data, indent=4)
        save_path = os.path.join(self.coco_annotations_dir, 'annotations.json')
        with open(save_path, 'w') as f:
            f.write(json_str)

        for image in png_list:
            src = os.path.join(self.labelme_data_dir, image)
            dst = os.path.join(self.coco_images_dir, image)
            copyfile(src, dst)

    def Nia2labelme(self):
        CheckExistsDir(self.labelme_data_dir)

        NIA_list = os.listdir(self.labeling_dir)

        for json_file in tqdm(NIA_list[:]):
            export_data = DefaultJson_labelme()

            save_bool = False
            json_file_path = os.path.join(self.labeling_dir, json_file)
            with open(json_file_path) as json_read:
                json_data = json.load(json_read)

            for idx in range(len(json_data['features'])):
                class_id = json_data['features'][idx]['properties']['type_id']
                class_name = json_data['features'][idx]['properties']['type_name']
                bbox_coord = json_data['features'][idx]['properties']['object_imcoords']

                if class_name in nia_categories:
                    save_bool = True
                    x1 = [float(bbox_coord.split(',')[0]), float(bbox_coord.split(',')[1])]
                    x2 = [float(bbox_coord.split(',')[2]), float(bbox_coord.split(',')[3])]
                    x3 = [float(bbox_coord.split(',')[4]), float(bbox_coord.split(',')[5])]
                    x4 = [float(bbox_coord.split(',')[6]), float(bbox_coord.split(',')[7])]

                    shape = {}
                    shape['label'] = class_name
                    shape['points'] = [x1, x2, x3, x4]
                    shape['group_id'] = None
                    shape['shape_type'] = "polygon"
                    shape['flags'] = {}
                    export_data['shapes'].append(shape)

            if save_bool:
                export_data['imagePath'] = json_data['features'][idx]['properties']['image_id']
                export_data['imageData'] = None
                export_data['imageHeight'] = 1024
                export_data['imageWidth'] = 1024
                export_data['geometry_coordinates'] = json_data['features'][idx]['geometry']['coordinates']

                json_str = json.dumps(export_data, indent=4)
                save_path = os.path.join(self.labelme_data_dir, json_file)
                with open(save_path, 'w') as f:
                    f.write(json_str)
                img_srt = os.path.join(self.objects_data_dir, json_data['features'][idx]['properties']['image_id'])
                img_dst = os.path.join(self.labelme_data_dir, json_data['features'][idx]['properties']['image_id'])
                copyfile(img_srt, img_dst)

    def Labelme2Nia(self, ignore_cat_list):
        if ignore_cat_list[0] == 'container':
            ignore_cat_list = ['grouped container', 'individual container']
        object_data = os.path.join(self.from_labelme_dir, 'objects_data')
        object_labeling = os.path.join(self.from_labelme_dir, 'objects_labeling')
        CheckExistsDir(self.from_labelme_dir)
        CheckExistsDir(object_data)
        CheckExistsDir(object_labeling)

        labelme_list = os.listdir(self.labelme_data_dir)
        json_list = [file for file in labelme_list if file.endswith(".json")]

        for json_file in tqdm(json_list[:]):
            json_file_path = os.path.join(self.labelme_data_dir, json_file)
            with open(json_file_path) as json_read:
                json_data = json.load(json_read)

            export_data = {}
            export_data['features'] = []

            for shape in json_data['shapes']:
                if shape['label'] not in ignore_cat_list:
                    points = []
                    points.extend(shape['points'][0])
                    points.extend(shape['points'][1])
                    points.extend(shape['points'][2])
                    points.extend(shape['points'][3])

                    feature = {}

                    feature['geometry'] = {}
                    feature['geometry']['coordinates'] = json_data['geometry_coordinates']
                    feature['geometry']['type'] = "Polygon"

                    feature['properties'] = {}
                    feature['properties']['object_imcoords'] = ','.join(str(e) for e in points)
                    feature['properties']['object_angle'] = 0
                    feature['properties']['building_imcoords'] = "EMPTY"
                    feature['properties']['road_imcoords'] = "EMPTY"
                    feature['properties']['image_id'] = json_data['imagePath']
                    feature['properties']['ingest_time'] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
                    feature['properties']['type_id'] = str(nia_categories.index(shape['label']))
                    feature['properties']['type_name'] = shape['label']

                    feature['type'] = "Feature"

                    export_data['features'].append(feature)
                    export_data['type'] = "FeatureCollection"

            if len(export_data['features']) > 0:
                json_str = json.dumps(export_data, indent=4)
                save_path = os.path.join(object_labeling, json_file)
                with open(save_path, 'w') as f:
                    f.write(json_str)

        json_list = os.listdir(object_labeling)

        for json_file in json_list:
            png_file = os.path.splitext(json_file)[0] + '.png'
            src = os.path.join(self.labelme_data_dir, png_file)
            dst = os.path.join(object_data, png_file)

            copyfile(src, dst)