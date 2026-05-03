# ==============================================================================
# 1. Base Configuration Inheritance
# ==============================================================================
_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/mts-ycor-v1.0.0.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

# ==============================================================================
# 6. Hooks & Visualizers
# ==============================================================================

# custom_hooks = [
#     dict(
#         type='EarlyStoppingHook',
#         monitor='val/mIoU',
#         rule='greater',
#         min_delta=0.001,
#         patience=3)
# ]

vis_backends = [
    # dict(type='LocalVisBackend'),
    dict(
        type='MLflowVisBackend',
        # 1. Store the global DB in a hidden folder to keep work_dirs clean
        # tracking_uri='sqlite:///work_dirs/.mlflow/mlflow.db',
        tracking_uri='http://172.17.0.1:5000',
        
        # 2. Automatically log .pth, .log, and config files to MLflow
        artifact_suffix=('.pth', '.json', '.log', '.py', '.yaml', '.txt'),
        
        # 3. Dynamic setup (OMIT save_dir and exp_name)
        
        # 4. Tags (Hardcode these based on the specific config file)
        tags=dict(
            model='SegFormer',
            backbone='mit-b0',
            dataset='ycor-lm-3cls',
            resolution='512x512'
        )
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