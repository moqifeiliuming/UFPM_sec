# Copyright (c) 2021-2022 Alibaba Group Holding Limited.
# maedet_l_6x_lr0.02：表示maedet的L模型，训练6x表示72轮，学习率为0.02

_base_ = 'gfocal_r50_fpn_ms6x.py'
# model settings
model = dict(
    backbone=dict(
        type='MadNas',
        # 默认为net_str，改为structure_txt，与MadNas相对应
        structure_txt="configs/gfocal_maedet/maedet_l.txt",
        out_indices=(1,2,3,4), 
        init_cfg=None), # if load pretrained model, fill the path of the pth file.
    neck=dict(
        type='FPN',
        in_channels=[144, 608, 1912, 2400],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5),
        )
# training and testing settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 480), (1333, 960)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)

# learning policy
lr_config = dict(
    warmup='linear',
    warmup_iters=7330,
    warmup_ratio=0.1,
    step=[65, 71])
total_epochs = 73
use_syncBN_torch = True