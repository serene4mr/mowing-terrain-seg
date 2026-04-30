_base_ = [
    '../mmseg/segmentation_static.py', 
    '../_base_/backends/onnxruntime.py'
]

codebase_config = dict(
    type='mmseg',
    task='Segmentation',
    with_argmax=False
)

onnx_config = dict(
    opset_version=17,
    input_shape=[512, 512]
)