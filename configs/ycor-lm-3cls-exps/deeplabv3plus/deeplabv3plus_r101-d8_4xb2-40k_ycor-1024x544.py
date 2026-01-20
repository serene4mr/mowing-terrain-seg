# DeepLabV3 Plus with ResNet-50 backbone on YCOR dataset
# 4 GPUs x 2 samples per GPU, trained for 40K iterations

_base_ = [
    '../../_base_/models/deeplabv3plus_r50-d8.py', 
    '../../_base_/datasets/ycor-lm-3cls.py',
    '../../_base_/default_runtime.py', 
    '../../_base_/schedules/schedule_40k.py'
]

inverse_frequency_weights_train = [1.946165, 0.956220, 0.694259]
sqrt_inverse_weights_train = [2.416298,1.693711,1.443183]
log_inverse_weights_train = [1.764473, 1.053845, 0.733702]

# YCOR-specific crop size (full size for maximum coverage)
crop_size = (1024, 544)  # Full YCOR size - no information loss

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,  
    size=crop_size
)

# Model configuration - override num_classes for YCOR
model = dict(
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101),
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=3,
        ignore_index=255,
        loss_decode=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=1.0,
            class_weight=log_inverse_weights_train,
            avg_non_ignore=True,
        )
    ),  
    auxiliary_head=dict(
        num_classes=3,
        ignore_index=255,
        loss_decode=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=0.4,
            class_weight=log_inverse_weights_train,
            avg_non_ignore=True,
        )
    )
)
