# dataset settings
dataset_type = 'DOTADatasetV1'
# dataset root path:
data_root = '/home/dhkim/work/r3det-on-mmdetection/data/base'
trainsplit_ann_folder = 'train768_2x/labelTxt'
trainsplit_img_folder = 'train768_2x/images'

valsplit_ann_folder = 'test1024_2x/labelTxt'
valsplit_img_folder = 'test1024_2x/images'

val_ann_folder = 'test1024_2x/labelTxt'
val_img_folder = 'test1024_2x/images'

test_img_folder = 'test1024_2x/images'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    #dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    #dict(type='Pad', size=(800, 800)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='CroppedTilesFlipAug',
        tile_scale=(1024, 1024),
        tile_shape=(1024, 1024),
        tile_overlap=(150, 150),
        flip=False,
        transforms=[
            #dict(type='RResize', img_scale=(1024, 1024)),
            dict(type='RRandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            #dict(type='Pad', size=(800, 800)),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=2,
    train=[
        dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=trainsplit_ann_folder,
            img_prefix=trainsplit_img_folder,
            pipeline=train_pipeline),
        dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=valsplit_ann_folder,
            img_prefix=valsplit_img_folder,
            pipeline=train_pipeline),
    ],
    val=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=val_ann_folder,
            difficulty_thresh=1,
            img_prefix=val_img_folder,
            pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=test_img_folder,
        difficulty_thresh=1,
        img_prefix=test_img_folder,
        pipeline=test_pipeline))
