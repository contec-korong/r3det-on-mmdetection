import albumentations as A
import argparse
import cv2
import json, os
import shutil

from tqdm import tqdm
from common import transform, convert_8coords_to_4coords, CheckExistsDir
from coco_validation import valid_data
from pycocotools.coco import COCO

crane_color = (255, 255, 0)
container_color = (255, 0, 0)
ss_color = (255, 0, 255)
ms_color = (0, 0, 255)
ls_color = (0, 255, 255)

def main():
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--data-dir', type=str, default='', help='')
    parser.add_argument('--multiple', type=int, default=3, help='crane whole data multiple num')
    args = parser.parse_args()

    aug_dir = os.path.join(args.data_dir, '{}_train768_2x'.format(args.multiple))
    ori_dir = os.path.join(args.data_dir, 'train768_2x')
    CheckExistsDir(aug_dir)

    ori_json = os.path.join(ori_dir, 'coco.json')
    ori_images = os.path.join(ori_dir, 'images')
    shutil.copy(ori_json, aug_dir)
    shutil.copytree(ori_images, aug_dir+'/images')

    coco_dir = aug_dir
    coco_file = os.path.join(coco_dir, 'coco.json')
    coco = COCO(coco_file)
    with open(os.path.join(coco_dir, 'coco.json')) as json_file:
        json_data = json.load(json_file)

    print('create augmentation...')
    for img_ids in tqdm(coco.getImgIds()):
        keypoints = []
        ann_class = []
        img_path = os.path.join(coco_dir, 'images', coco.loadImgs(ids=img_ids)[0]['file_name'])
        image = cv2.imread(img_path)

        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_ids)):
            for idx, an in enumerate(ann['segmentation'][0]):
                if an >= image.shape[0]:
                    ann['segmentation'][0][idx] = image.shape[0]-1

            keypoints.append((ann['segmentation'][0][0], ann['segmentation'][0][1]))
            keypoints.append((ann['segmentation'][0][2], ann['segmentation'][0][3]))
            keypoints.append((ann['segmentation'][0][4], ann['segmentation'][0][5]))
            keypoints.append((ann['segmentation'][0][6], ann['segmentation'][0][7]))
            ann_class.append(ann['category_id'])

        file_name = coco.loadImgs(ids=img_ids)[0]['file_name']
        if coco.getCatIds(catNms='crane')[0] in ann_class:
            json_data = acc_transform(image, keypoints, ann_class, json_data, coco_dir, 'crane', args.multiple, file_name)
        if coco.getCatIds(catNms='middle_ship')[0] in ann_class:
            json_data = acc_transform(image, keypoints, ann_class, json_data, coco_dir, 'middle_ship', args.multiple, file_name)
        if coco.getCatIds(catNms='large_ship')[0] in ann_class:
            json_data = acc_transform(image, keypoints, ann_class, json_data, coco_dir, 'large_ship', args.multiple, file_name)
        if coco.getCatIds(catNms='container')[0] in ann_class:
            json_data = acc_transform(image, keypoints, ann_class, json_data, coco_dir, 'container', args.multiple,
                                      file_name)

    json_str = json.dumps(json_data, indent=4)
    save_path = os.path.join(coco_dir, 'coco.json')
    with open(save_path, 'w') as f:
        f.write(json_str)

    print('visualization annotations...')
    vis_ann(coco_dir)

    print('validation coco...')
    valid_data(save_path)

def acc_transform(image, keypoints, ann_class, json_data, coco_dir, catNms,  itera, file_name):
    ann_id_last = json_data['annotations'][-1]['id'] + 1
    image_id_last = json_data['images'][-1]['id']

    for idx in range(1, itera + 1):
        transformed = transform(image=image, keypoints=keypoints)
        save_path = os.path.join(coco_dir, 'images', 'aug_{}_{}_'.format(catNms, idx) + file_name)
        cv2.imwrite(save_path, transformed['image'])

        trans_images = {'id': image_id_last + idx,
                        'file_name': 'aug_{}_{}_'.format(catNms, idx) + file_name,
                        'height': transformed['image'].shape[0],
                        'width': transformed['image'].shape[1]}

        for point in range(int(len(transformed['keypoints']) / 4)):
            segmentation = []
            cat_id = ann_class[point]
            for key_idx in range(0, 4):
                segmentation.extend(transformed['keypoints'][point * 4 + key_idx])

            xmin, ymin, w, h = convert_8coords_to_4coords(segmentation)
            trans_annotations = {'difficult': '-1',
                                 'area': w * h,
                                 'category_id': cat_id,
                                 'segmentation': [segmentation],
                                 'iscrowd': 0,
                                 'bbox': (xmin, ymin, w, h),
                                 'image_id': image_id_last + idx,
                                 'id': ann_id_last}
            ann_id_last += 1
            json_data['annotations'].append(trans_annotations)
        json_data['images'].append(trans_images)

    return json_data

def vis_ann(aug_dir):
    coco_path = os.path.join(aug_dir, 'coco.json')
    img_dir = os.path.join(aug_dir, 'images')
    vis_dir = os.path.join(aug_dir, 'vis_ann')
    CheckExistsDir(vis_dir)

    coco = COCO(coco_path)

    img_ids = coco.getImgIds()

    for img_id in tqdm(img_ids):
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        img_file = coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)

        for ann in anns:
            category_id = ann['category_id']
            x1 = (ann['segmentation'][0][0], ann['segmentation'][0][1])
            x2 = (ann['segmentation'][0][2], ann['segmentation'][0][3])
            x3 = (ann['segmentation'][0][4], ann['segmentation'][0][5])
            x4 = (ann['segmentation'][0][6], ann['segmentation'][0][7])

            if category_id == 1:
                color = crane_color
            elif category_id == 2:
                color = container_color
            elif category_id == 3:
                color = ss_color
            elif category_id == 4:
                color = ms_color
            elif category_id == 5:
                color = ls_color

            cv2.line(img, x1, x2, color)
            cv2.line(img, x2, x3, color)
            cv2.line(img, x3, x4, color)
            cv2.line(img, x4, x1, color)
        save_file = os.path.join(vis_dir, img_file)
        cv2.imwrite(save_file, img)

if __name__ == '__main__':
    main()