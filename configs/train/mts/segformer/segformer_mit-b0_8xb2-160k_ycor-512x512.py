_base_ = [
    '../../_base_/models/segformer_mit-b0.py',
    '../../_base_/datasets/ycor-lm-3cls.py',
    '../runtime.py',
    '../../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    decode_head=dict(num_classes=3))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(1024, 544),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=16000,
        save_best='val/mIoU',
        rule='greater',
        max_keep_ckpts=5))

# custom_hooks = [
#     dict(
#         type='EarlyStoppingHook',
#         monitor='val/mIoU',
#         rule='greater',
#         min_delta=0.001,
#         patience=3)
# ]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='MLflowVisBackend',
        tracking_uri='sqlite:///work_dirs/mlflow.db',
        save_dir='work_dirs/ycor-lm-3cls-exps',
        artifact_suffix=('.pth', '.jpg', '.png', '.json', '.log', 'yaml', '.txt'),
        exp_name='deeplabv3plus_r18_ycor-lm-3cls_512x512',
        tags=dict(
            model='DeepLabV3Plus',
            backbone='ResNet18',
            dataset='ycor-lm-3cls',
            resolution='512x512',
            num_classes='3',
        ),
    ),
]

visualizer = dict(
    type='CustomSegLocalVisualizer',
    vis_backends=vis_backends,
    name='custom_seg_local_visualizer',
    save_interval=10,
    max_images_per_iter=5,
)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=4000,
        save_best='val/mIoU',
        rule='greater',
        max_keep_ckpts=5),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1)
)