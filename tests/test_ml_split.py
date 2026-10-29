import numpy as np

from archcut3d.ml.dataset import make_dataset
from tests.fakes import FakeExample, straight_trace


def _examples():
    return [
        FakeExample("B.wrl", [straight_trace(100)]),
        FakeExample("A.wrl", [straight_trace(60)]),
        FakeExample("A.wrl", [straight_trace(140)]),   # second label file, same scan as above
    ]


def test_groups_index_distinct_source_files():
    ds = make_dataset(_examples(), n_jitter=0, end_overshoot=0, rng=0)
    assert ds.group_names == ["A.wrl", "B.wrl"]
    assert sorted(set(ds.groups.tolist())) == [0, 1]
    for g, meta in zip(ds.groups, ds.meta):
        assert ds.group_names[g] == meta["source_file"]


def test_two_label_files_of_one_scan_share_a_group():
    ds = make_dataset(_examples(), n_jitter=0, end_overshoot=0, rng=0)
    a_groups = {g for g, m in zip(ds.groups, ds.meta) if m["source_file"] == "A.wrl"}
    assert a_groups == {0}


def test_group_ids_do_not_depend_on_input_order():
    ds1 = make_dataset(_examples(), n_jitter=0, end_overshoot=0, rng=0)
    ds2 = make_dataset(list(reversed(_examples())), n_jitter=0, end_overshoot=0, rng=0)
    assert ds1.group_names == ds2.group_names
    for ds in (ds1, ds2):
        assert all(ds.group_names[g] == m["source_file"] for g, m in zip(ds.groups, ds.meta))
