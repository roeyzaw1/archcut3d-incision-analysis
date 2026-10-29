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


def test_anchor_kinds_and_pose_are_recorded():
    ds = make_dataset([FakeExample("A.wrl", [straight_trace(100, 40, 140)])], n_jitter=2, seed=0)
    kinds = ds.kinds
    assert set(kinds.tolist()) == {0, 1, 2}
    n_clean = int((kinds == 0).sum())
    assert int((kinds == 1).sum()) == 2 * n_clean
    for m, k in zip(ds.meta, kinds):
        assert m["center"].shape == (2,) and abs(np.linalg.norm(m["heading"]) - 1.0) < 1e-9
        if k == 0:
            assert abs(m["center"][1] - 100.0) < 1e-9       # clean anchors lie on the hand line


def test_evaluate_splits_angle_error_by_anchor_kind():
    from archcut3d.ml.train import train_one_split
    ds = _three_model_dataset()
    res = train_one_split(ds, test_group=0, epochs=2, patience=99, batch_size=16, verbose=False)
    assert np.isfinite(res.test.angle_err_clean_deg) and np.isfinite(res.test.angle_err_jittered_deg)


def test_keep_channels_zeroes_excluded_channels_but_keeps_shape_and_mask():
    from archcut3d.ml.torch_data import NormStats, normalize_patches
    rng = np.random.default_rng(0)
    patches = rng.normal(size=(5, 12, 12, 3)).astype(np.float32) + 2.0
    patches[..., 2] = np.abs(patches[..., 2])
    masks = np.ones((5, 12, 12), bool)
    stats = NormStats(1.0, 1.0, 1.0)
    full = normalize_patches(patches, masks, stats)
    only_relief = normalize_patches(patches, masks, stats, keep_channels=(1,))
    assert only_relief.shape == full.shape == (5, 12, 12, 4)
    assert np.all(only_relief[..., 0] == 0) and np.all(only_relief[..., 2] == 0)
    assert np.array_equal(only_relief[..., 1], full[..., 1])
    assert np.array_equal(only_relief[..., 3], full[..., 3]) and only_relief[..., 3].all()


def test_frozen_encoder_does_not_change_but_heads_do():
    import copy
    import torch
    from archcut3d.ml.model import TracerNet
    from archcut3d.ml.train import train_one_split
    ds = _three_model_dataset()
    torch.manual_seed(5)
    init = copy.deepcopy(TracerNet().state_dict())            # same init train_one_split draws for seed=5
    res = train_one_split(ds, test_group=0, epochs=3, patience=99, batch_size=16, verbose=False,
                          seed=5, freeze_encoder=True)
    sd = res.net.state_dict()
    for k in init:
        if k.startswith("encoder.") and k.endswith(("weight", "bias")) and "num_batches" not in k:
            assert torch.equal(init[k], sd[k].cpu()), k
    assert any(not torch.equal(init[k], sd[k].cpu()) for k in init if k.startswith(("step_head", "stop_head")))
    assert res.freeze_encoder


def test_fold_checkpoint_round_trips(tmp_path):
    import torch
    from archcut3d.ml.experiments import fold_path, load_fold, save_fold
    from archcut3d.ml.train import train_one_split
    ds = _three_model_dataset()
    fold = train_one_split(ds, test_group=0, epochs=2, patience=99, batch_size=16, verbose=False, seed=3)
    path = fold_path(str(tmp_path), "tracernet", "A model.wrl", 3)
    save_fold(path, fold, "tracernet", "A model.wrl")
    net, stats, meta = load_fold(path)
    assert stats == fold.stats and meta["seed"] == 3 and meta["test_model"] == "A model.wrl"
    x = torch.randn(2, 4, 48, 48)
    fold.net.eval()
    with torch.no_grad():
        a, b = fold.net(x.to(next(fold.net.parameters()).device)), net(x)
    assert torch.allclose(a[0].cpu(), b[0], atol=1e-6)


def test_subsample_groups_caps_every_group_except_the_exempt_one():
    from archcut3d.ml.dataset import subsample_groups
    ds = _three_model_dataset()
    counts = np.bincount(ds.groups)
    cap = int(counts.min()) // 2
    sub = subsample_groups(ds, cap, seed=0, exempt=(1,))
    sc = np.bincount(sub.groups)
    assert sc[0] == cap and sc[2] == cap and sc[1] == counts[1]
    assert sub.group_names == ds.group_names and len(sub.meta) == len(sub)
    again = subsample_groups(ds, cap, seed=0, exempt=(1,))
    assert np.array_equal(sub.steps, again.steps)
