# DeepLabV3 with ResNet-50 backbone on YCOR dataset
# 4 GPUs x 2 samples per GPU, trained for 40K iterations

_base_ = [
    '../../_base_/models/deeplabv3_r50-d8.py', 
    '../../_base_/datasets/ycor.py',
    '../../_base_/default_runtime.py', 
    '../../_base_/schedules/schedule_40k.py'
]

frequency_inverse_weights_train = [0.672440,1.028959,0.752052,70.239090,15.458582,2.498081,0.404913,1.012719]
sqrt_inverse_weights_train = [2.319379, 2.869089, 2.452838, 23.704698, 11.120641, 4.470420, 1.799807, 2.846358]
log_inverse_weights_train = [1.682599, 2.107989, 1.794491, 6.331347, 4.817606, 2.994965, 1.175359, 2.092080]


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
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=8,
        ignore_index=255,
        loss_decode=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=1.0,
            class_weight=frequency_inverse_weights_train,
            avg_non_ignore=True,
        )
    ),  
    auxiliary_head=dict(
        num_classes=8,
        ignore_index=255,
        loss_decode=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=0.4,
            class_weight=frequency_inverse_weights_train,
            avg_non_ignore=True,
        )
    )
)