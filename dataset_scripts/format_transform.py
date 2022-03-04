import argparse

from transform import Transform

def main():
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--data-dir', type=str, default='', help='')
    parser.add_argument('--transform-type', type=str, default='nia2labelme', choices=['nia2labelme', 'labelme2nia', 'labelme2coco', 'labelimg2labelme'],
                        help='transform dataset format, default: nia2labelme')
    parser.add_argument('-l', '--ignore-cat-list', nargs='*', help='ignore categories in labelme2nia', required=False)
    args = parser.parse_args()

    transform = Transform(args.data_dir, args.transform_type)

    if args.transform_type == 'nia2labelme':
        print('\n\n* Nia 2 Labelme Transform...')
        transform.Nia2labelme()
        print('Done...')
    elif args.transform_type == 'labelme2nia':
        print('\n\n* Labelme 2 Nia Transform...')
        transform.Labelme2Nia(args.ignore_cat_list)
        print('Done...')
    elif args.transform_type == 'labelimg2labelme':
        print('\n\n* LabelImg 2 Labelme Transform...')
        transform.LabelImg2Labelme(args.ignore_cat_list)
        print('Done...')
    else:
        print('\n\n* Labelme 2 Coco Transform...')
        transform.Labelme2coco()
        print('Done...')

if __name__ == '__main__':
    main()