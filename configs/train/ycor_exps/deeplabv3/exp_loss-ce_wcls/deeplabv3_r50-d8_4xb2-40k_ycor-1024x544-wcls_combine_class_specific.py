# DeepLabV3 with ResNet-50 backbone on YCOR dataset - Class-Specific Strategy
# Strategy: Use best performing method for each individual class

_base_ = [
    '../../_base_/models/deeplabv3_r50-d8.py', 
    '../../_base_/datasets/ycor.py',
    '../../_base_/default_runtime.py', 
    '../../_base_/schedules/schedule_40k.py'
]

# Original individual weighting methods
frequency_inverse_weights_train = [0.672440,1.028959,0.752052,70.239090,15.458582,2.498081,0.404913,1.012719]
sqrt_inverse_weights_train = [2.319379, 2.869089, 2.452838, 23.704698, 11.120641, 4.470420, 1.799807, 2.846358]
log_inverse_weights_train = [1.682599, 2.107989, 1.794491, 6.331347, 4.817606, 2.994965, 1.175359, 2.092080]

# Class-Specific Selection: Best performing method per class
class_specific_weights = [
    sqrt_inverse_weights_train[0],  # smooth_trail: sqrt (2.32) - best IoU performance
    log_inverse_weights_train[1],   # traversable_grass: log (2.11) - best IoU performance
    sqrt_inverse_weights_train[2],  # rough_trail: sqrt (2.45) - best IoU performance
    3.0,                           # puddle: moderate weight (3.0) - avoid extreme 70.24 weight
    log_inverse_weights_train[4],   # obstacle: log (4.82) - best IoU performance
    frequency_inverse_weights_train[5], # non_traversable_vegetation: direct (2.50) - best IoU performance
    log_inverse_weights_train[6],   # high_vegetation: log (1.18) - best IoU performance
    log_inverse_weights_train[7],   # sky: log (2.09) - best IoU performance
]

# YCOR-specific crop size
crop_size = (1024, 544)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,  
    size=crop_size
)

# Model configuration with class-specific strategy
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=8,
        ignore_index=255,
        loss_decode=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=1.0,
            class_weight=class_specific_weights,
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
            class_weight=class_specific_weights,
            avg_non_ignore=True,
        )
    )
)
