
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='MLflowVisBackend', 
        tracking_uri='sqlite:///work_dirs/mlflow.db',
        save_dir='work_dirs/ycor-lm-3cls-exps-0',
        artifact_suffix=('.pth', '.jpg', '.png', '.json', '.log', 'yaml', '.txt'),
        exp_name='ycor-lm-3cls-exps-0',
        tags={
            'dataset': 'ycor-lm-3cls',
            'num_classes': '3'
        },
    ),
]

visualizer = dict(
    type='CustomSegLocalVisualizer',
    vis_backends=vis_backends,       
    name='custom_seg_local_visualizer',       
    save_interval=5,                 
    max_images_per_iter=5            
)

log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

tta_model = dict(type='SegTTAModel')
