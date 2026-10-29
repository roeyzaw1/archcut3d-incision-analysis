"""Train TracerNet with a nested leave-one-model-out protocol.

For each fold one model is the test model; among the remaining models one
validation model (see torch_data.nested_split) is used only for early
stopping; the rest train. Metrics are reported on the test model, which is
evaluated once, after training, with the best-validation weights.

Normalization statistics are fit on the training models only.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .dataset import TracerSamples
from .model import TracerNet, tracer_loss
from .torch_data import NormStats, compute_norm_stats, make_nested_dataloaders, nested_split


def _subset(samples: TracerSamples, idx: np.ndarray) -> TracerSamples:
    idx = np.asarray(idx)
    return TracerSamples(
        patches=samples.patches[idx], masks=samples.masks[idx],
        steps=samples.steps[idx], stops=samples.stops[idx],
        groups=samples.groups[idx], meta=[samples.meta[i] for i in idx],
        group_names=samples.group_names, seed=samples.seed,
    )


def angular_error_deg(step_pred: np.ndarray, step_target: np.ndarray) -> np.ndarray:
    """Angle in degrees between predicted and true step vectors, per sample.
    NaN where either vector is ~zero (undefined direction)."""
    pn = np.linalg.norm(step_pred, axis=1)
    tn = np.linalg.norm(step_target, axis=1)
    ok = (pn > 1e-6) & (tn > 1e-6)
    cos = np.full(len(step_pred), np.nan)
    dot = (step_pred * step_target).sum(axis=1)
    cos[ok] = np.clip(dot[ok] / (pn[ok] * tn[ok]), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def stop_metrics(prob: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> dict:
    """Stop-detection metrics for stop=1 as the positive class.

    Accuracy alone is misleading here (stop=1 is a few percent of samples,
    so predicting "never stop" scores well), hence AUROC, precision, recall
    and the base rate. Quantities that are undefined (no positives, or no
    predicted positives) are NaN."""
    target = np.asarray(target) > 0.5
    pred = np.asarray(prob) > threshold
    n_pos = int(target.sum())
    tp = int((pred & target).sum())
    fp = int((pred & ~target).sum())
    both_classes = 0 < n_pos < len(target)
    return {
        "accuracy": float((pred == target).mean()) if len(target) else float("nan"),
        "auroc": float(roc_auc_score(target, prob)) if both_classes else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "recall": tp / n_pos if n_pos else float("nan"),
        "base_rate": float(target.mean()) if len(target) else float("nan"),
    }


@dataclass
class EvalResult:
    loss: float
    step_loss: float
    stop_loss: float
    mean_angle_err_deg: float
    stop_accuracy: float
    n: int
    stop_auroc: float = float("nan")
    stop_precision: float = float("nan")
    stop_recall: float = float("nan")
    stop_base_rate: float = float("nan")


@torch.no_grad()
def evaluate(net: TracerNet, loader, device: str) -> EvalResult:
    net.eval()
    losses, step_losses, stop_losses = [], [], []
    all_pred_step, all_tgt_step, all_pred_stop, all_tgt_stop = [], [], [], []
    for xb, yb, zb in loader:
        xb, yb, zb = xb.to(device), yb.to(device), zb.to(device)
        sp, lg = net(xb)
        loss, parts = tracer_loss(sp, lg, yb, zb)
        losses.append(loss.item())
        step_losses.append(parts["step_loss"].item())
        stop_losses.append(parts["stop_loss"].item())
        all_pred_step.append(sp.cpu().numpy())
        all_tgt_step.append(yb.cpu().numpy())
        all_pred_stop.append(torch.sigmoid(lg).cpu().numpy())
        all_tgt_stop.append(zb.cpu().numpy())

    pred_step = np.concatenate(all_pred_step)
    tgt_step = np.concatenate(all_tgt_step)
    stop_prob = np.concatenate(all_pred_stop)
    tgt_stop = np.concatenate(all_tgt_stop)

    ang = angular_error_deg(pred_step, tgt_step)
    # angle is only meaningful where the model is meant to be moving forward
    interior = tgt_stop < 0.5
    mean_ang = float(np.nanmean(ang[interior])) if interior.any() else float("nan")
    sm = stop_metrics(stop_prob, tgt_stop)

    return EvalResult(loss=float(np.mean(losses)), step_loss=float(np.mean(step_losses)),
                      stop_loss=float(np.mean(stop_losses)), mean_angle_err_deg=mean_ang,
                      stop_accuracy=sm["accuracy"], n=len(tgt_stop),
                      stop_auroc=sm["auroc"], stop_precision=sm["precision"],
                      stop_recall=sm["recall"], stop_base_rate=sm["base_rate"])


@dataclass
class FoldResult:
    net: TracerNet
    stats: NormStats
    history: list
    best_val_loss: float
    val: EvalResult           # validation model, best-validation weights
    test: EvalResult          # test model, evaluated once after training
    test_group: int
    val_group: int
    train_groups: list
    seed: int


def train_one_split(samples: TracerSamples, test_group: int, epochs: int = 150, patience: int = 20,
                    lr: float = 1e-3, batch_size: int = 64, device: str = "cpu", seed: int = 0,
                    verbose: bool = True) -> FoldResult:
    """Train on the training models, early-stop on the validation model, and
    report metrics on the held-out test model (see module docstring)."""
    split = nested_split(samples, test_group)
    stats = compute_norm_stats(_subset(samples, split.train_idx))   # fit on train only
    train_loader, val_loader, test_loader = make_nested_dataloaders(
        samples, stats, split, batch_size=batch_size, seed=seed)

    torch.manual_seed(seed)          # weight init and dropout masks
    net = TracerNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history = []

    for epoch in range(epochs):
        net.train()
        train_losses = []
        for xb, yb, zb in train_loader:
            xb, yb, zb = xb.to(device), yb.to(device), zb.to(device)
            opt.zero_grad()
            sp, lg = net(xb)
            loss, _ = tracer_loss(sp, lg, yb, zb)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        val_eval = evaluate(net, val_loader, device)
        tr = float(np.mean(train_losses))
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": val_eval.loss,
                        "val_angle_err": val_eval.mean_angle_err_deg})

        if val_eval.loss < best_val - 1e-5:
            best_val = val_eval.loss
            best_state = copy.deepcopy(net.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if verbose and (epoch % 10 == 0 or bad_epochs == 0):
            print(f"  epoch {epoch:3d}  train {tr:.4f}  val {val_eval.loss:.4f}  "
                  f"val_angle_err {val_eval.mean_angle_err_deg:5.1f} deg"
                  f"{'  *' if bad_epochs == 0 else ''}")

        if bad_epochs >= patience:
            if verbose:
                print(f"  early stop at epoch {epoch} (no val improvement for {patience} epochs)")
            break

    net.load_state_dict(best_state)
    val_final = evaluate(net, val_loader, device)
    test_final = evaluate(net, test_loader, device)   # the only time the test model is evaluated
    train_groups = sorted(set(samples.groups[split.train_idx].tolist()))
    return FoldResult(net=net, stats=stats, history=history, best_val_loss=best_val,
                      val=val_final, test=test_final, test_group=split.test_group,
                      val_group=split.val_group, train_groups=train_groups, seed=seed)
