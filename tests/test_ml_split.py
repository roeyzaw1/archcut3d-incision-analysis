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
    ds = make_dataset(_examples(), n_jitter=0, end_overshoot=0, seed=0)
    assert ds.group_names == ["A.wrl", "B.wrl"]
    assert sorted(set(ds.groups.tolist())) == [0, 1]
    for g, meta in zip(ds.groups, ds.meta):
        assert ds.group_names[g] == meta["source_file"]


def test_two_label_files_of_one_scan_share_a_group():
    ds = make_dataset(_examples(), n_jitter=0, end_overshoot=0, seed=0)
    a_groups = {g for g, m in zip(ds.groups, ds.meta) if m["source_file"] == "A.wrl"}
    assert a_groups == {0}


def test_group_ids_do_not_depend_on_input_order():
    ds1 = make_dataset(_examples(), n_jitter=0, end_overshoot=0, seed=0)
    ds2 = make_dataset(list(reversed(_examples())), n_jitter=0, end_overshoot=0, seed=0)
    assert ds1.group_names == ds2.group_names
    for ds in (ds1, ds2):
        assert all(ds.group_names[g] == m["source_file"] for g, m in zip(ds.groups, ds.meta))


def _three_model_dataset():
    # different trace lengths -> different sample counts per model, so a
    # loader's size identifies which model it holds
    exs = [FakeExample("A.wrl", [straight_trace(60, 40, 100)]),
           FakeExample("B.wrl", [straight_trace(100, 40, 140)]),
           FakeExample("C.wrl", [straight_trace(140, 40, 180)])]
    return make_dataset(exs, n_jitter=1, end_overshoot=0, seed=0)


def test_nested_split_is_disjoint_and_cyclic():
    from archcut3d.ml.torch_data import nested_split
    ds = _three_model_dataset()
    for test_group in range(3):
        sp = nested_split(ds, test_group)
        assert sp.val_group == (test_group + 1) % 3
        train_g = set(ds.groups[sp.train_idx].tolist())
        val_g = set(ds.groups[sp.val_idx].tolist())
        test_g = set(ds.groups[sp.test_idx].tolist())
        assert test_g == {test_group} and val_g == {sp.val_group}
        assert train_g.isdisjoint(val_g | test_g) and len(train_g) == 1
        assert len(sp.train_idx) + len(sp.val_idx) + len(sp.test_idx) == len(ds)


def test_nested_split_needs_three_models():
    import pytest
    from archcut3d.ml.torch_data import nested_split
    exs = [FakeExample(f"{n}.wrl", [straight_trace(r)]) for n, r in (("A", 60), ("B", 100))]
    ds = make_dataset(exs, n_jitter=0, end_overshoot=0, seed=0)
    with pytest.raises(ValueError):
        nested_split(ds, 0)


def test_same_source_file_never_on_both_sides():
    from archcut3d.ml.torch_data import nested_split
    exs = [FakeExample("A.wrl", [straight_trace(60)]), FakeExample("A.wrl", [straight_trace(80)]),
           FakeExample("B.wrl", [straight_trace(100)]), FakeExample("C.wrl", [straight_trace(140)])]
    ds = make_dataset(exs, n_jitter=0, end_overshoot=0, seed=0)
    for test_group in range(3):
        sp = nested_split(ds, test_group)
        names = lambda idx: {ds.meta[i]["source_file"] for i in idx}
        assert names(sp.train_idx).isdisjoint(names(sp.val_idx))
        assert names(sp.train_idx).isdisjoint(names(sp.test_idx))
        assert names(sp.val_idx).isdisjoint(names(sp.test_idx))


def test_test_model_is_evaluated_only_once_after_training(monkeypatch):
    from archcut3d.ml import train as train_mod
    ds = _three_model_dataset()
    calls = []
    real = train_mod.evaluate

    def recording(net, loader, device):
        calls.append(("train_mode_after_eval", len(loader.dataset)))
        return real(net, loader, device)

    monkeypatch.setattr(train_mod, "evaluate", recording)
    res = train_mod.train_one_split(ds, test_group=0, epochs=3, patience=99, batch_size=16, verbose=False)
    n_test = int((ds.groups == 0).sum())
    n_val = int((ds.groups == 1).sum())
    sizes = [c[1] for c in calls]
    assert sizes.count(n_test) == 1 and sizes[-1] == n_test      # test model: once, last
    assert sizes.count(n_val) == 3 + 1                           # validation: per epoch + final
    assert res.test_group == 0 and res.val_group == 1 and res.train_groups == [2]
