"""
mmdeploy FUNCTION_REWRITER for Mask2Former ONNX export compatibility.

Root cause: ``Mask2FormerHead.predict`` creates ``SegDataSample(metainfo=...)``
which calls ``set_metainfo`` → ``copy.deepcopy(metainfo_dict)``.  During
``torch.onnx.export`` tracing, ``metainfo`` can hold traced Tensor values
(e.g. ``img_shape`` is a Tensor under dynamic-shape export).  Deepcopying a
Tensor via ``storage.clone()`` inside the tracer raises::

    RuntimeError: NYI: Named tensors are not supported with the tracer

mmdeploy's own ``copy__default`` rewriter only intercepts a top-level
``deepcopy(tensor)``; tensors *nested inside a dict* fall through to the
normal path and crash.

Fix: replace the SegDataSample construction with ``set_field`` calls.
``set_field`` uses ``object.__setattr__`` (no deepcopy), which is tracer-safe.
This mirrors the pattern used by mmdeploy's own mmdet MaskFormer rewriter
(mmdeploy/codebase/mmdet/models/detectors/maskformer.py) which carries an
explicit comment about this exact issue.
"""

import torch.nn.functional as F
import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name=(
        'mmseg.models.decode_heads.mask2former_head.Mask2FormerHead.predict'
    )
)
def mask2former_head__predict(self, x, batch_img_metas, test_cfg):
    """Rewrite ``Mask2FormerHead.predict`` to avoid deepcopy during ONNX export.

    ``SegDataSample(metainfo=...)`` triggers ``set_metainfo`` →
    ``copy.deepcopy``, which crashes the tracer when ``metainfo`` contains
    Tensors.  Build the samples via ``set_field`` instead (direct
    ``object.__setattr__``, no copy).
    """
    from mmseg.structures import SegDataSample

    # Build SegDataSample objects without set_metainfo / deepcopy.
    # Note: set_field uses object.__setattr__ directly — no deepcopy.
    batch_data_samples = []
    for metainfo in batch_img_metas:
        sample = SegDataSample()
        for key, value in metainfo.items():
            sample.set_field(name=key, value=value, field_type='metainfo')
        batch_data_samples.append(sample)

    all_cls_scores, all_mask_preds = self(x, batch_data_samples)
    mask_cls_results = all_cls_scores[-1]
    mask_pred_results = all_mask_preds[-1]

    if 'pad_shape' in batch_img_metas[0]:
        size = batch_img_metas[0]['pad_shape']
    else:
        size = batch_img_metas[0]['img_shape']

    mask_pred_results = F.interpolate(
        mask_pred_results, size=size, mode='bilinear', align_corners=False)
    cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
    mask_pred = mask_pred_results.sigmoid()
    seg_logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)
    return seg_logits
