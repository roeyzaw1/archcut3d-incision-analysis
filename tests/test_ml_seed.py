import numpy as np
import torch

from archcut3d.ml.dataset import make_dataset
from archcut3d.ml.train import train_one_split
from archcut3d.ml.torch_data import group_balanced_sampler
from tests.fakes import FakeExample, straight_trace


def _examples():
    return [FakeExample("A.wrl", [straight_trace(60, 40, 100)]),
            FakeExample("B.wrl", [straight_trace(100, 40, 140)]),
            FakeExample("C.wrl", [straight_trace(140, 40, 180)])]


def test_dataset_is_reproducible_and_seed_is_recorded():
    a = make_dataset(_examples(), n_jitter=2, seed=3)
    b = make_dataset(_examples(), n_jitter=2, seed=3)
    c = make_dataset(_examples(), n_jitter=2, seed=4)
    assert a.seed == 3 and c.seed == 4
    assert np.array_equal(a.steps, b.steps) and np.array_equal(a.patches, b.patches)
    assert not np.array_equal(a.steps, c.steps)


def test_sampler_is_reproducible():
    groups = np.array([0] * 30 + [1] * 10)
    idx = np.arange(40)
    s1 = list(group_balanced_sampler(groups, idx, seed=5))
    s2 = list(group_balanced_sampler(groups, idx, seed=5))
    s3 = list(group_balanced_sampler(groups, idx, seed=6))
    assert s1 == s2 and s1 != s3


def test_training_is_reproducible_for_a_seed():
    ds = make_dataset(_examples(), n_jitter=1, seed=0)
    kw = dict(test_group=0, epochs=3, patience=99, batch_size=16, verbose=False)
    r1 = train_one_split(ds, seed=7, **kw)
    r2 = train_one_split(ds, seed=7, **kw)
    r3 = train_one_split(ds, seed=8, **kw)
    assert r1.seed == 7
    assert [h["train_loss"] for h in r1.history] == [h["train_loss"] for h in r2.history]
    assert [h["train_loss"] for h in r1.history] != [h["train_loss"] for h in r3.history]
    for p1, p2 in zip(r1.net.parameters(), r2.net.parameters()):
        assert torch.equal(p1, p2)
