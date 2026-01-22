# Training schedule for 40k iterations
# Based on MMSegmentation standard schedule configuration

# Optimizer configuration
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

# Learning rate policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=40000,
        by_epoch=False
    )
]

# Training schedule for 40k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Default hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=4000,
        max_keep_ckpts=0,
        save_best=[
            'val/mIoU',
            'val/mDice',
            'val/mFscore',
        ],
        rule='greater',
        save_last=True
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1)
)
