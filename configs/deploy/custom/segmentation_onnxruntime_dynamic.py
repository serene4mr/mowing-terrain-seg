_base_ = ['../mmseg/segmentation_dynamic.py', '../_base_/backends/onnxruntime.py']
codebase_config = dict(type='mmseg', task='Segmentation', with_argmax=False)

onnx_config = dict(
    opset_version=17,
    dynamic_axes={
        'input': {0: 'batch', 2: 'height', 3: 'width'},
        'output': {0: 'batch', 2: 'height', 3: 'width'},
    },
)