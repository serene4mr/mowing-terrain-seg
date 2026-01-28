_base_ = ['../mmseg/segmentation_static.py', '../_base_/backends/onnxruntime.py']
codebase_config = dict(type='mmseg', task='Segmentation', with_argmax=False)
# onnx_config = dict(input_shape=[1024, 544])

onnx_config = dict(
    type='onnx',
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=17,
    save_file='end2end.onnx',
    input_names=['input'],
    output_names=['output'],
    input_shape=[1024, 544],
    optimize=True)
