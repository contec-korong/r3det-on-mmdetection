import argparse
import imgsplit_multiprocess
import os

from coco_validation import valid_data
from geojson2coco import geojson2coco
from utils import ship_Division

def prepare(srcpath, dstpath):
    """
    :param srcpath: train, val, test
          train --> train512, test --> test512
    :return:
    """
    rate = 2
    train_dst_name = 'train768_2x'
    test_dst_name = 'test1024_2x'

    if not os.path.exists(os.path.join(dstpath, test_dst_name)):
        os.makedirs(os.path.join(dstpath, test_dst_name))
    if not os.path.exists(os.path.join(dstpath, train_dst_name)):
        os.makedirs(os.path.join(dstpath, train_dst_name))

    split_train = imgsplit_multiprocess.splitbase(os.path.join(srcpath, 'train_Busan_container_add'),
                       os.path.join(dstpath, train_dst_name),
                      gap=256,
                      subsize=768,
                      num_process=32
                      )
    split_train.splitdata(rate)

    rate = 1
    split_test = imgsplit_multiprocess.splitbase(os.path.join(srcpath, 'test_Busan_container_add'),
                       os.path.join(dstpath, test_dst_name),
                      gap=0,
                      subsize=1024,
                      num_process=32
                      )
    split_test.splitdata(rate)

    # Ship Class Division
    print('ship Division ...')
    ship_Division(geojsonpath=os.path.join(dstpath, train_dst_name, 'json'))
    ship_Division(geojsonpath=os.path.join(dstpath, test_dst_name, 'json'))
    print('Done ...')

    # GeoJson 2 Coco Format Transform
    geojson2coco(os.path.join(dstpath, train_dst_name, 'images'), os.path.join(dstpath, train_dst_name, 'json'), os.path.join(dstpath, train_dst_name, 'coco.json'), classes=6)
    geojson2coco(os.path.join(dstpath, test_dst_name, 'images'), os.path.join(dstpath, test_dst_name, 'json'), os.path.join(dstpath, test_dst_name, 'coco.json'), classes=6)

    # Validation Coco Dataset
    valid_data(os.path.join(dstpath, train_dst_name, 'coco.json'))
    valid_data(os.path.join(dstpath, test_dst_name, 'coco.json'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare roas_for_split')
    parser.add_argument('--srcpath', default='../dataset/K3A_NIA/05_roas')
    parser.add_argument('--dstpath', default='../dataset/K3A_NIA/06_roas4-split', help='prepare data')
    args = parser.parse_args()

    srcpath = args.srcpath
    dstpath = args.dstpath
    prepare(srcpath, dstpath)
