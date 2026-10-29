"""Experiments on what TracerNet learns.

    python -m archcut3d.ml.experiments --data <dir with .label.npz> --out results/ \
        [--experiments E0,E0b,E1,E2,E3,E5] [--seeds 3] [--quick]

Question: does TracerNet follow the local geometric minimum of the surface,
or something closer to the hand-drawn line where the two differ?

  E0   TracerNet vs two baselines without learning: straight ahead, and a
       geometric tracer that follows the local-relief minimum.
  E0b  Departure analysis: hand-trace points that are far from the local
       geometric minimum ("departures"), and how each tracer behaves there.
  E1   Random-encoder baseline: encoder frozen at its random initialisation,
       only the heads are trained.
  E2   Input-channel ablation: height / local_relief / curvature alone.

Protocol: leave-one-scanned-model-out, with a separate validation model for
early stopping (train.py). Hand traces drawn by machine (crest-line output,
detected by their point spacing) are excluded. Results are written as CSVs
plus figures under --out; RESULTS.md is generated from them.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from .anchors import (DEPARTURE_DEPTH_PERCENTILE, DEPARTURE_OFFSET_MM, build_anchor_table,
                      depth_threshold, label_departures)
from .dataset import (DEFAULT_PATCH_MM, DEFAULT_PATCH_RES, DEFAULT_STEP_MM, load_dataset,
                      make_dataset)
from .evaluation import evaluate_policy
from .figures import departure_example_figure, pick_departure_examples
from .policies import GeometricMinimumPolicy, StraightAheadPolicy, TracerPolicy
from .train import train_one_split

MACHINE_SPACING_PX = 2.0     # median spacing between consecutive points below this => machine-generated
ALL_EXPERIMENTS = ["E0", "E0b", "E1", "E2"]
CHANNELS = {"height": (0,), "local_relief": (1,), "curvature": (2,)}


# ----------------------------------------------------------------- data

def click_spacing_px(example) -> float:
    d = [np.linalg.norm(np.diff(t, axis=0), axis=1) for t in example.traces if len(t) > 1]
    return float(np.median(np.concatenate(d))) if d else float("nan")


def load_models(data_dir: str, exclude=(), include_machine: bool = False):
    """Load label files; split into usable hand-traced examples and excluded
    ones. Returns (examples, summary_rows)."""
    all_ex = load_dataset(data_dir)
    rows, used = [], []
    for e in all_ex:
        spacing = click_spacing_px(e)
        machine = spacing < MACHINE_SPACING_PX
        excluded = os.path.basename(e.path) in exclude or e.source_file in exclude
        keep = (not excluded) and (include_machine or not machine)
        lengths = [e.trace_length_mm(t) for t in e.traces]
        rows.append({
            "label_file": os.path.basename(e.path), "source_file": e.source_file,
            "n_traces": len(e.traces), "n_points": int(sum(len(t) for t in e.traces)),
            "traced_mm": round(float(sum(lengths)), 2), "pixel_size_mm": round(e.pixel_size, 5),
            "median_point_spacing_px": round(spacing, 2),
            "median_point_spacing_mm": round(spacing * e.pixel_size, 3),
            "machine_generated": machine,
            "used": keep,
            "reason_excluded": "" if keep else ("explicitly excluded" if excluded else "machine-generated (point spacing < %.1f px)" % MACHINE_SPACING_PX),
        })
        if keep:
            used.append(e)
    return used, rows


def save_fold(path: str, fold, variant: str, test_model: str):
    """Checkpoint of a trained fold: weights plus what is needed to rebuild
    the policy (normalisation statistics, channels, groups)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in fold.net.state_dict().items()},
                "stats": fold.stats.to_dict(), "keep_channels": tuple(fold.keep_channels),
                "freeze_encoder": fold.freeze_encoder, "seed": fold.seed, "variant": variant,
                "test_model": test_model, "val_group": fold.val_group, "train_groups": fold.train_groups},
               path)


def load_fold(path: str, device: str = "cpu"):
    """Inverse of save_fold: returns (net in eval mode, NormStats, metadata dict)."""
    from .model import TracerNet
    from .torch_data import NormStats
    ck = torch.load(path, map_location=device, weights_only=False)
    net = TracerNet()
    net.load_state_dict(ck["state_dict"])
    return net.to(device).eval(), NormStats.from_dict(ck["stats"]), ck


def fold_path(out: str, variant: str, test_model: str, seed: int) -> str:
    safe = os.path.splitext(test_model)[0].replace(" ", "_")
    return os.path.join(out, "models", f"{variant}__{safe}__seed{seed}.pt")


def pick_device(arg: str) -> str:
    if arg != "auto":
        return arg
    return "mps" if torch.backends.mps.is_available() else "cpu"


def git_info() -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", *a], capture_output=True, text=True, cwd=os.path.dirname(__file__)).stdout.strip()
        except OSError:
            return ""
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--", "archcut3d/ml"))}


# ----------------------------------------------------------------- output

def write_csv(path: str, rows: list):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()})


class Context:
    def __init__(self, args):
        self.args = args
        self.out = args.out
        os.makedirs(os.path.join(self.out, "figures"), exist_ok=True)
        self.device = pick_device(args.device)
        self.seeds = list(range(args.seeds))
        self.epochs = 3 if args.quick else args.epochs
        self.patience = 2 if args.quick else args.patience

        self.examples, self.file_rows = load_models(args.data, exclude=set(args.exclude), include_machine=args.include_machine)
        if len(self.examples) < 3:
            raise SystemExit(f"need >= 3 usable models for the nested split, found {len(self.examples)}")
        self.by_name = {e.source_file: e for e in self.examples}
        self.names = sorted(self.by_name)
        self.test_groups = [0] if args.quick else list(range(len(self.names)))
        if args.quick:
            self.seeds = [0]

        self.tables = {n: build_anchor_table(self.by_name[n]) for n in self.names}
        self.depth_thr = label_departures(list(self.tables.values()), offset_mm=args.departure_offset_mm)
        self.rows = {}            # experiment -> list of row dicts
        self.paths = {}           # (variant, model) -> predicted paths (first seed)
        self._datasets = {}

    def dataset(self, seed):
        if seed not in self._datasets:
            self._datasets[seed] = make_dataset(self.examples, seed=seed)
        return self._datasets[seed]

    def save(self, experiment):
        write_csv(os.path.join(self.out, f"{experiment.lower()}.csv"), self.rows.get(experiment, []))


def segment_rows(experiment, variant, test_model, seed, segments, extra=None):
    out = []
    for name, d in segments.items():
        r = {"experiment": experiment, "variant": variant, "test_model": test_model,
             "seed": "" if seed is None else seed, "segment": name}
        r.update(d)
        if name == "all" and extra:
            r.update(extra)
        out.append(r)
    return out


def fold_extra(fold) -> dict:
    t = fold.test
    return {"test_angle_err_clean_deg": t.angle_err_clean_deg, "test_angle_err_jittered_deg": t.angle_err_jittered_deg,
            "stop_auroc": t.stop_auroc, "stop_precision": t.stop_precision, "stop_recall": t.stop_recall,
            "stop_base_rate": t.stop_base_rate, "stop_accuracy": t.stop_accuracy,
            "best_val_loss": fold.best_val_loss, "epochs_run": len(fold.history), "n_test_samples": t.n}


# ----------------------------------------------------------------- experiments

def run_baselines(ctx: Context, experiment: str, variants: dict):
    """Evaluate policies that need no training, once per model."""
    for variant, policy in variants.items():
        for g in ctx.test_groups:
            name = ctx.names[g]
            seg, traces, paths = evaluate_policy(policy, ctx.by_name[name], ctx.tables[name], can_stop=False)
            ctx.rows.setdefault(experiment, []).extend(segment_rows(experiment, variant, name, None, seg))
            ctx.paths[(variant, name)] = paths
    ctx.save(experiment)


def run_trained(ctx: Context, experiment: str, variant: str, **train_kw):
    keep = tuple(train_kw.get("keep_channels", (0, 1, 2)))
    for seed in ctx.seeds:
        ds = ctx.dataset(seed)
        for g in ctx.test_groups:
            name = ds.group_names[g]
            t0 = time.time()
            fold = train_one_split(ds, test_group=g, epochs=ctx.epochs, patience=ctx.patience,
                                   device=ctx.device, seed=seed, verbose=False, **train_kw)
            policy = TracerPolicy(fold.net, fold.stats, ctx.device, keep_channels=keep)
            seg, traces, paths = evaluate_policy(policy, ctx.by_name[name], ctx.tables[name], can_stop=True)
            extra = fold_extra(fold)
            extra["val_model"] = ds.group_names[fold.val_group]
            extra["train_models"] = ";".join(ds.group_names[x] for x in fold.train_groups)
            ctx.rows.setdefault(experiment, []).extend(segment_rows(experiment, variant, name, seed, seg, extra))
            save_fold(fold_path(ctx.out, variant, name, seed), fold, variant, name)
            if seed == ctx.seeds[0]:
                ctx.paths[(variant, name)] = paths
            print(f"  [{experiment}/{variant}] test={name[:20]:20s} seed={seed}  epochs={len(fold.history):3d}  "
                  f"chamfer={seg['all'].get('chamfer_mm', float('nan')):.3f} mm  ({time.time() - t0:.0f}s)", flush=True)
            ctx.save(experiment)


def departure_outputs(ctx: Context):
    """Per-point CSV, per-model summary (with threshold sensitivity), and figures."""
    pts, summ = [], []
    for name in ctx.names:
        t = ctx.tables[name]
        for k in range(len(t)):
            pts.append({"model": name, "trace": int(t.trace[k]), "point": int(t.point[k]),
                        "col": round(float(t.pos[k, 0]), 2), "row": round(float(t.pos[k, 1]), 2),
                        "geo_offset_mm": float(t.geo_offset_mm[k]), "geo_depth": float(t.geo_depth[k]),
                        "departure": bool(t.departure[k])})
        off, dep = np.abs(t.geo_offset_mm), t.geo_depth
        row = {"model": name, "n_points": len(t), "pixel_size_mm": t.pixel_size,
               "departure_fraction": float(t.departure.mean()),
               "median_abs_offset_mm": float(np.nanmedian(off)), "p90_abs_offset_mm": float(np.nanpercentile(off, 90)),
               "median_depth": float(np.nanmedian(dep)),
               "frac_min_at_band_edge": float((off >= 0.9 * t.half_width_mm).mean()),
               "depth_threshold": ctx.depth_thr}
        for thr in (0.1, 0.2, 0.3):
            row[f"departure_fraction_offset_{thr:g}mm"] = float(np.mean(~((off <= thr) & (dep >= ctx.depth_thr))))
        summ.append(row)
    write_csv(os.path.join(ctx.out, "e0b_departure_points.csv"), pts)
    write_csv(os.path.join(ctx.out, "e0b_departure_summary.csv"), summ)

    chosen = pick_departure_examples([ctx.tables[n] for n in ctx.names])
    for i, (ti, k) in enumerate(chosen):
        name = ctx.names[ti]
        paths = {}
        for variant, label in (("tracernet", "TracerNet"), ("geometric", "geometric tracer")):
            if (variant, name) in ctx.paths:
                paths[label] = ctx.paths[(variant, name)]
        departure_example_figure(ctx.by_name[name], ctx.tables[name], k, paths,
                                 os.path.join(ctx.out, "figures", f"e0b_example_{i + 1}.png"))


def write_config(ctx: Context, started: float, experiments):
    a = ctx.args
    cfg = {
        "command": " ".join(sys.argv), "experiments": experiments, "seeds": ctx.seeds, "quick": a.quick,
        "device": ctx.device, "epochs": ctx.epochs, "patience": ctx.patience,
        "optimizer": {"name": "Adam", "lr": 1e-3, "batch_size": 64}, "loss": {"stop_weight": 0.2},
        "patch": {"patch_mm": DEFAULT_PATCH_MM, "out_res": DEFAULT_PATCH_RES, "step_mm": DEFAULT_STEP_MM},
        "samples": {"n_jitter": 4, "jitter_pos_mm": 0.25, "jitter_deg": 15.0, "end_overshoot": 2},
        "geometric_tracer": {"half_width_mm": 0.5},
        "departure": {"offset_mm": a.departure_offset_mm, "depth_percentile": DEPARTURE_DEPTH_PERCENTILE,
                      "depth_threshold": ctx.depth_thr},
        "models": ctx.names, "test_groups": ctx.test_groups,
        "excluded_machine_generated": [r["label_file"] for r in ctx.file_rows if not r["used"]],
        "git": git_info(), "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
        "platform": platform.platform(), "wall_time_s": round(time.time() - started, 1),
    }
    with open(os.path.join(ctx.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="directory with .label.npz files")
    p.add_argument("--out", default="results")
    p.add_argument("--experiments", default=",".join(ALL_EXPERIMENTS))
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--quick", action="store_true", help="1 fold, 1 seed, few epochs: smoke test")
    p.add_argument("--device", default="auto", help="auto, cpu or mps")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--exclude", nargs="*", default=[], help="label files / source files to leave out")
    p.add_argument("--include-machine", action="store_true", help="keep machine-generated (crest-line) label files")
    p.add_argument("--departure-offset-mm", type=float, default=DEPARTURE_OFFSET_MM)
    args = p.parse_args(argv)
    args.experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
    unknown = [e for e in args.experiments if e not in ALL_EXPERIMENTS]
    if unknown:
        p.error(f"unknown experiment(s): {unknown}; available: {ALL_EXPERIMENTS}")
    return args


def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    ctx = Context(args)
    print(f"device={ctx.device} models={ctx.names} seeds={ctx.seeds} folds={ctx.test_groups} "
          f"epochs<={ctx.epochs}; depth threshold {ctx.depth_thr:.5f}", flush=True)
    write_csv(os.path.join(ctx.out, "dataset_files.csv"), ctx.file_rows)

    ex = set(args.experiments)
    if ex & {"E0", "E0b"}:
        print("E0: baselines and TracerNet", flush=True)
        run_baselines(ctx, "E0", {"straight": StraightAheadPolicy(), "geometric": GeometricMinimumPolicy(),
                                  "geometric_w0.25": GeometricMinimumPolicy(half_width_mm=0.25)})
        run_trained(ctx, "E0", "tracernet")
    if "E0b" in ex:
        departure_outputs(ctx)
    if "E1" in ex:
        print("E1: random encoder", flush=True)
        run_trained(ctx, "E1", "random_encoder", freeze_encoder=True)
    if "E2" in ex:
        print("E2: channel ablation", flush=True)
        for cname, keep in CHANNELS.items():
            run_trained(ctx, "E2", f"only_{cname}", keep_channels=keep)
    write_config(ctx, started, args.experiments)
    print(f"done in {time.time() - started:.0f}s -> {ctx.out}", flush=True)


if __name__ == "__main__":
    main()
