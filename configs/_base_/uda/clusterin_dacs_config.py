_base_ = ['dacs_a999_fdthings.py']
uda = dict(
    type='ClusteringDACS',
    source_ds='synthia',
    target_ds='cityspaces',
    n_clusters=16,
    acc_amount=1,
    freeze_decoder=True
)