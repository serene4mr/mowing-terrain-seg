# Default runtime configuration
# Based on MMSegmentation standard runtime settings

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
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
