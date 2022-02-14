# optimizer
optimizer = dict(type='SGD',
                 lr=4e-3,
                 momentum=0.9,
                 weight_decay=0.0001,
                 )  # paramwise_options=dict(bias_lr_mult=2, bias_decay_mult=0))
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=0.1,
    step=[12, 24, 36, 48, 60, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196])
total_epochs = 200
