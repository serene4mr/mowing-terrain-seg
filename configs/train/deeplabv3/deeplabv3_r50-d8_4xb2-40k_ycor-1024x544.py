# DeepLabV3 with ResNet-50 backbone on YCOR dataset
# 4 GPUs x 2 samples per GPU, trained for 40K iterations

_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py', 
    '../_base_/datasets/ycor.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_40k.py'
]

# YCOR-specific crop size (full size for maximum coverage)
crop_size = (1024, 544)  # Full YCOR size - no information loss
data_preprocessor = dict(size=crop_size)

# Model configuration - override num_classes for YCOR
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=8),  # YCOR has 8 classes (background reduced by reduce_zero_label=True)
    auxiliary_head=dict(num_classes=8)  # YCOR has 8 classes (background reduced by reduce_zero_label=True)
)
