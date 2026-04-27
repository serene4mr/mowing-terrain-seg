"""
Deploy config for Mask2Former → ONNX Runtime (dynamic shapes).

Extends the generic dynamic config with a custom_imports entry that loads
the Mask2FormerHead.predict rewriter before torch.onnx.export runs.
The rewriter avoids the ``SegDataSample(metainfo=...) → deepcopy`` path that
crashes the tracer with:
    RuntimeError: NYI: Named tensors are not supported with the tracer

See src/mowing_terrain_seg/deploy/mask2former_rewriter.py for details.
"""

_base_ = ['segmentation_onnxruntime_dynamic.py']

# Loaded by mmengine.Config.fromfile in the export subprocess so that the
# FUNCTION_REWRITER is registered before torch.onnx.export is called.
custom_imports = dict(
    imports=['mowing_terrain_seg.deploy.mask2former_rewriter'],
    allow_failed_imports=False,
)
