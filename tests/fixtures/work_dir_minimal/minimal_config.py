# Root-level training config copy (for sha256 in summary)
default_scope = 'mmseg'
metainfo = dict(
    classes=('Alpha', 'Beta', 'Gamma'),
    palette=[[0, 0, 0], [1, 1, 1], [2, 2, 2]],
)
classes = metainfo['classes']
dataset_type = 'DummyYcorDataset'
data_root = 'data/'
