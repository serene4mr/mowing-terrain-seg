# DeepLabV3 Plus with ResNet-50 backbone on YCOR dataset
# 4 GPUs x 2 samples per GPU, trained for 40K iterations

_base_ = [
    '../../../_base_/models/deeplabv3plus_r50-d8.py', 
    '../../../_base_/datasets/ycor-lm-3cls.py',
    '../../../_base_/default_runtime.py', 
    '../../../_base_/schedules/schedule_40k.py'
]

# YCOR-specific crop size
crop_size = (640, 480)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,  
    size=crop_size
)

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

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

# Model configuration - override num_classes for YCOR
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=3,
        ignore_index=255,
        loss_decode=dict(
            type='FixedCrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=1.0,
            avg_non_ignore=True,
        )
    ),  
    auxiliary_head=dict(
        num_classes=3,
        ignore_index=255,
        loss_decode=dict(
            type='FixedCrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=0.4,
            avg_non_ignore=True,
        )
    )
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='MLflowVisBackend',
        tracking_uri='sqlite:///work_dirs/mlflow.db',
        save_dir='work_dirs/ycor-lm-3cls-exps',
        artifact_suffix=('.pth', '.jpg', '.png', '.json', '.log', 'yaml', '.txt'),
        exp_name='deeplabv3plus_r50_ycor-lm-3cls_640x480',
        tags=dict(
            model='DeepLabV3Plus',
            backbone='ResNet50',
            dataset='ycor-lm-3cls',
            resolution='640x480',
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
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1)
)