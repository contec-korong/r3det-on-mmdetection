_base_ = [
    'models/r3det_r50_fpn.py',
    'datasets/768_rotational_detection.py',
    'schedules/schedule_1x_768.py'
]

# runtime settings
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = './work_dirs/r3det_r50_fpn_2x_768'
evaluation = dict(interval=1, metric='mAP')
