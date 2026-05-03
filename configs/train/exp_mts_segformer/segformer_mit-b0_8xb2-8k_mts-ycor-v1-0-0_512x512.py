# ==============================================================================
# 1. Base Configuration Inheritance
# ==============================================================================
_base_ = [
    './common.py',
]

# ==============================================================================
# 2. Model & Preprocessing Overrides
# ==============================================================================
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', 
            checkpoint=checkpoint
        )
    ),
    decode_head=dict(num_classes=3)
)

# ==============================================================================
# 3. Data Pipelines & Dataloaders
# ==============================================================================
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

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(pipeline=train_pipeline)
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(pipeline=test_pipeline)
)
test_dataloader = val_dataloader

# ==============================================================================
# 4. Optimization & Schedule
# ==============================================================================
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', 
        lr=0.00006, 
        betas=(0.9, 0.999), 
        weight_decay=0.01
    ),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }
    )
)

param_scheduler = [
    dict(
        type='LinearLR', 
        start_factor=1e-6, 
        by_epoch=False, 
        begin=0, 
        end=1500
    ),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=8000,
        by_epoch=False,
    )
]

# ==============================================================================
# 5. Training Loop
# ==============================================================================
train_cfg = dict(type='IterBasedTrainLoop', 
    max_iters=8000, 
    val_interval=4000
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')




default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', 
        interval=50, 
        log_metric_by_epoch=False
    ),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=4000,
        save_best='val/mIoU',
        rule='greater',
        max_keep_ckpts=5
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='SegVisualizationHook', 
        draw=True, 
        interval=1
    )
)
