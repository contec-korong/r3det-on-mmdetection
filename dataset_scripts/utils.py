import os
import math
import json

coco_class = {'crane':'0', 'container': '1', 'small_ship':'2', 'middle_ship':'3', 'large_ship':'4'}
ship_size = {'large_ship':100, 'middle_ship':50}

def ship_Division(geojsonpath: str):
    cnt_large = cnt_middle = cnt_small = 0

    for file in os.listdir(geojsonpath):
        json_file_path = os.path.join(geojsonpath, file)
        with open(json_file_path, 'rt', encoding='UTF8') as json_file:
            json_data = json.load(json_file)

        for idx, feature in enumerate(json_data['features']):
            if feature['properties']['type_name'] == 'ship':
                dist = calculate_distance(feature['properties']['object_imcoords'].split(','))

                if dist > ship_size['large_ship']:
                    type_name = 'large_ship'
                    type_id = coco_class['large_ship']
                    cnt_large += 1
                elif dist >= ship_size['middle_ship']:
                    type_name = 'middle_ship'
                    type_id = coco_class['middle_ship']
                    cnt_middle += 1
                else:
                    type_name = 'small_ship'
                    type_id = coco_class['small_ship']
                    cnt_small += 1

                feature['properties']['type_name'] = type_name
                feature['properties']['type_id'] = type_id
                json_data['features'][idx] = feature

        json_str = json.dumps(json_data, indent=4)
        with open(json_file_path, 'w') as f:
            f.write(json_str)

    print('large ship :', cnt_large)
    print('middle ship :', cnt_middle)
    print('small ship :', cnt_small)

def calculate_distance(obj_coord):
    x1 = float(obj_coord[0])
    y1 = float(obj_coord[1])
    x4 = float(obj_coord[6])
    y4 = float(obj_coord[7])

    return math.hypot(x1 - x4, y1 - y4) * 0.55

def GetFileFromThisRootDir(dir,ext = None):
    allfiles = []
    needExtFilter = (ext != None)
    for root,dirs,files in os.walk(dir):
        for filespath in files:
            filepath = os.path.join(root, filespath)
            extension = os.path.splitext(filepath)[1][1:]
            if needExtFilter and extension in ext:
                allfiles.append(filepath)
            elif not needExtFilter:
                allfiles.append(filepath)
    return allfiles

def get_basename(fullname):
    return os.path.basename(os.path.splitext(fullname)[0])

def get_extent(fullname):
    _, ext = os.path.splitext(fullname)
    return ext

def single_copy(src_dst_tuple):
    shutil.copyfile(*src_dst_tuple)

def filecopy(srcpath, dstpath, num_process=32):
    pool = Pool(num_process)
    filelist = util.GetFileFromThisRootDir(srcpath)

    name_pairs = []
    for file in filelist:
        basename = os.path.basename(file.strip())
        dstname = os.path.join(dstpath, basename)
        name_tuple = (file, dstname)
        name_pairs.append(name_tuple)

    pool.map(single_copy, name_pairs)

def singel_move(src_dst_tuple):
    shutil.move(*src_dst_tuple)

def filemove(srcpath, dstpath, num_process=32):
    pool = Pool(num_process)
    filelist = util.GetFileFromThisRootDir(srcpath)

    name_pairs = []
    for file in filelist:
        basename = os.path.basename(file.strip())
        dstname = os.path.join(dstpath, basename)
        name_tuple = (file, dstname)
        name_pairs.append(name_tuple)

    pool.map(filemove, name_pairs)

def getnamelist(srcpath, dstfile):
    filelist = util.GetFileFromThisRootDir(srcpath)
    with open(dstfile, 'w') as f_out:
        for file in filelist:
            basename = get_basename(file)
            f_out.write(basename + '\n')