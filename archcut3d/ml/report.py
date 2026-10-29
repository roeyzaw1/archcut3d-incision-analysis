"""Build RESULTS.md from the experiment CSVs.

    python -m archcut3d.ml.report results/

Numbers are read from the CSVs; the plain-language findings are read from
results/findings.md, which is written by hand after looking at the tables.
Every "mean +- std" is the mean over test models (folds) of the per-model
mean over seeds; the std is over test models (ddof = 1).
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if v == "":
                r[k] = float("nan")
                continue
            if v in ("True", "False"):
                r[k] = v == "True"
                continue
            try:
                r[k] = float(v)
            except ValueError:
                pass
    return rows


def per_model(rows, variant, segment, metric):
    """{test_model: mean over seeds} for finite values."""
    by = defaultdict(list)
    for r in rows:
        if r["variant"] == variant and r["segment"] == segment and isinstance(r.get(metric), float) and r[metric] == r[metric]:
            by[r["test_model"]].append(r[metric])
    return {m: float(np.mean(v)) for m, v in by.items()}


def agg(rows, variant, segment, metric, digits=3):
    vals = list(per_model(rows, variant, segment, metric).values())
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.{digits}f}"
    return f"{np.mean(vals):.{digits}f} ± {np.std(vals, ddof=1):.{digits}f}"


def md_table(header, body):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    return "\n".join(lines)


VARIANT_LABEL = {
    "tracernet": "TracerNet (hand labels)", "geo_labels": "TracerNet trained on crest-line labels (E5)",
    "random_encoder": "random encoder (E1)", "geometric": "geometric tracer (W = 0.5 mm)",
    "geometric_w0.25": "geometric tracer (W = 0.25 mm)", "straight": "straight ahead",
    "only_height": "height only (E2)", "only_local_relief": "local_relief only (E2)",
    "only_curvature": "curvature only (E2)",
}


def dataset_section(out):
    files = read_csv(os.path.join(out, "dataset_files.csv"))
    with open(os.path.join(out, "config.json")) as f:
        cfg = json.load(f)
    body = [[r["label_file"], int(r["n_traces"]), int(r["n_points"]), r["traced_mm"], r["pixel_size_mm"],
             r["median_point_spacing_mm"], "yes" if r["used"] else f"no: {r['reason_excluded']}"] for r in files]
    used = [r for r in files if r["used"]]
    px = [r["pixel_size_mm"] for r in used]
    txt = ["## Dataset", "",
           md_table(["label file", "traces", "points", "traced mm", "mm/px", "median point spacing (mm)", "used"], body), "",
           f"{len(files)} label files, {len({r['source_file'] for r in files})} distinct scanned models; "
           f"{len(used)} used. Pixel size of the used models: {min(px):.4f}–{max(px):.4f} mm/px "
           f"({max(px) / min(px):.1f}x). Across all 7 scans of the collection the range is 0.011–0.359 mm/px (about 33x). "
           f"Models: {', '.join(cfg['models'])}."]
    return "\n".join(txt)


def departure_section(out):
    summ = read_csv(os.path.join(out, "e0b_departure_summary.csv"))
    if not summ:
        return ""
    cfg = json.load(open(os.path.join(out, "config.json")))
    body = [[r["model"], int(r["n_points"]), f"{r['pixel_size_mm']:.3f}", f"{r['departure_fraction'] * 100:.0f}%",
             f"{r['frac_min_at_band_edge'] * 100:.0f}%", f"{r['median_abs_offset_mm']:.2f}", f"{r['median_depth'] * 1000:.1f}",
             f"{r['departure_fraction_offset_0.2mm'] * 100:.0f}%", f"{r['departure_fraction_offset_0.3mm'] * 100:.0f}%"]
            for r in summ]
    txt = ["### Where the hand line departs from the geometric minimum", "",
           "A hand-trace point is a *departure* if the local-relief minimum on the perpendicular through it "
           f"(±{cfg['geometric_tracer']['half_width_mm']} mm, tilt corrected) is more than {cfg['departure']['offset_mm']} mm away, "
           f"or the dip there is shallower than {cfg['departure']['depth_threshold'] * 1000:.1f} µm "
           f"(the {cfg['departure']['depth_percentile']:.0f}th percentile of dip depths with an interior minimum).", "",
           md_table(["model", "points", "mm/px", "departure", "minimum at band edge", "median |offset| mm", "median dip depth µm",
                     "departure if offset > 0.2 mm", "if > 0.3 mm"], body), "",
           "The offset threshold (0.1 mm) is below the pixel size of the coarsest model, so for that model it is below what a hand click can resolve; "
           "the last two columns show how the fractions change with a looser threshold."]
    return "\n".join(txt)


def key_metric_table(rows, variants, segments, metrics, title, note=""):
    header = ["variant"] + [f"{seg}: {name}" for seg in segments for name, _ in metrics]
    body = []
    for v in variants:
        if not any(r["variant"] == v for r in rows):
            continue
        body.append([VARIANT_LABEL.get(v, v)] +
                    [agg(rows, v, seg, key) for seg in segments for _, key in metrics])
    if not body:
        return ""
    return f"### {title}\n\n{note}\n\n" + md_table(header, body)


def per_model_table(rows, variants, segment, metric, label):
    models = sorted({r["test_model"] for r in rows})
    body = []
    for v in variants:
        pm = per_model(rows, v, segment, metric)
        if pm:
            body.append([VARIANT_LABEL.get(v, v)] + [f"{pm[m]:.3f}" if m in pm else "n/a" for m in models])
    if not body:
        return ""
    return f"**{label}, per test model ({segment} points)**\n\n" + md_table(["variant"] + [m.replace(".wrl", "") for m in models], body)


def build(out):
    e0 = read_csv(os.path.join(out, "e0.csv"))
    e1 = read_csv(os.path.join(out, "e1.csv"))
    e2 = read_csv(os.path.join(out, "e2.csv"))
    e5 = read_csv(os.path.join(out, "e5.csv"))
    cfg = json.load(open(os.path.join(out, "config.json")))
    hand = e0 + e1 + e2 + e5

    one_step = [("step dist mm", "step_dist_mm"), ("lateral corr with geometric step", "lat_corr_geo"),
                ("lateral slope on geometric step", "lat_slope_geo"),
                ("displaced: pred→hand line mm", "jit_pred_to_line_mm"),
                ("displaced: closer to hand", "jit_frac_closer_to_hand")]
    closed = [("hand→path mm", "cl_hand_to_pred_mm")]
    variants = ["tracernet", "random_encoder", "geo_labels", "only_height", "only_local_relief", "only_curvature",
                "geometric", "geometric_w0.25", "straight"]

    parts = [f"# What is TracerNet learning?\n",
             "Does TracerNet follow the geometric minimum of the surface, or the hand-drawn line where the two differ? "
             "Method after Svatko et al., *Deep Learning for BioImaging: What Are We Really Learning?* (arXiv 2603.13377): "
             "simple baselines, untrained-network baselines, layer-wise probing and PCA dimensionality. "
             f"Seeds: {cfg['seeds']}; test models: {len(cfg['test_groups'])}; commit `{cfg['git']['commit'][:10]}`; device {cfg['device']}. "
             "Generated by `python -m archcut3d.ml.report`; raw numbers are in the CSVs next to this file.\n",
             dataset_section(out), "", departure_section(out), "",
             "## Departure vs normal points (key result)\n",
             "One-step columns compare each tracer's step from a hand pose with the hand's next point and with the geometric tracer's step; "
             "*closer to hand* is the fraction of points where the prediction is nearer the hand's next point than the geometric step. "
             "Closed-loop *hand→path* is the mean distance from hand points to the path the tracer draws from the start of the line. "
             "Baselines that cannot stop (straight, geometric) are given exactly the hand-trace length as step budget; this is an oracle "
             "and flatters them. Learned tracers decide their own end (budget 1.5x the hand length).\n",
             key_metric_table(hand, variants, ["departure", "normal"], one_step + closed, "Departure and normal points"), "",
             per_model_table(hand, ["tracernet", "geometric", "straight", "random_encoder", "geo_labels"], "departure", "step_dist_mm", "One-step distance to the hand's next point (mm)"), "",
             per_model_table(hand, ["tracernet", "geometric", "straight", "random_encoder", "geo_labels"], "departure", "frac_step_closer_to_hand", "Fraction of one-step predictions closer to the hand than to the geometric step"), "",
             per_model_table(hand, ["tracernet", "geometric", "straight"], "departure", "cl_hand_to_pred_mm", "Closed-loop distance from hand points to the path (mm)"), "",
             "## E0: TracerNet vs baselines (all points)\n"]
    path_cols = [("chamfer mm", "chamfer_mm"), ("pred→hand mm", "pred_to_hand_mm"), ("hand→pred mm", "hand_to_pred_mm"),
                 ("coverage 0.1 mm", "coverage_0.1"), ("length ratio", "length_ratio"), ("endpoint err mm", "endpoint_error_mm")]
    parts.append(key_metric_table(e0, ["tracernet", "geometric", "geometric_w0.25", "straight"], ["all"], path_cols, "Closed-loop path metrics"))
    one_cols = [("angle err clean °", "test_angle_err_clean_deg"), ("angle err jittered °", "test_angle_err_jittered_deg"),
                ("stop AUROC", "stop_auroc"), ("stop precision", "stop_precision"), ("stop recall", "stop_recall"), ("stop base rate", "stop_base_rate")]
    parts.append(key_metric_table(e0, ["tracernet"], ["all"], one_cols, "TracerNet one-step metrics on the test model"))
    parts.append("## E1 and E2: random encoder and input channels\n")
    parts.append(key_metric_table(e0 + e1 + e2, ["tracernet", "random_encoder", "only_height", "only_local_relief", "only_curvature"],
                                  ["all"], one_cols[:3] + path_cols[:1] + path_cols[3:4], "Metrics, all points"))
    if e5:
        parts.append("## E5: trained on geometry-only (crest-line) labels\n")
        parts.append(key_metric_table(e0 + e5, ["tracernet", "geo_labels"], ["all"], one_cols[:3] + path_cols[:1] + path_cols[3:4], "Metrics, all points"))

    pca = read_csv(os.path.join(out, "e3_pca.csv"))
    if pca:
        parts.append("## E3: representation analysis\n")
        body = []
        for enc in ("trained", "random"):
            sel = [r for r in pca if r["encoder"] == enc]
            row = [enc]
            for k in ("n_components_80", "n_components_90", "n_components_95", "participation_ratio"):
                v = [r[k] for r in sel]
                row.append(f"{np.mean(v):.1f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.1f}")
            body.append(row)
        parts.append("PCA of the pooled 128-d features on the test model's clean anchors (mean ± std over folds and seeds):\n\n" +
                     md_table(["encoder", "components for 80%", "90%", "95%", "participation ratio"], body))
        parts.append("\n![cumulative variance](figures/e3_pca.png)\n")
        probes = read_csv(os.path.join(out, "e3_probes.csv"))
        if probes:
            targets = sorted({r["target"] for r in probes})
            body = []
            for t in targets:
                metrics = sorted({r["metric"] for r in probes if r["target"] == t})
                for metric in metrics:
                    for enc in ("trained", "random"):
                        cells = []
                        for b in (1, 3, 6):
                            v = [r["score"] for r in probes if r["target"] == t and r["metric"] == metric and r["encoder"] == enc
                                 and r["block"] == b and r["score"] == r["score"]]
                            cells.append(f"{np.mean(v):.2f}" if v else "n/a")
                        pr = [r["pearson_r"] for r in probes if r["target"] == t and r["metric"] == metric and r["encoder"] == enc
                              and r["block"] == 6 and r["pearson_r"] == r["pearson_r"]]
                        body.append([t.replace("_", " "), metric, enc] + cells + [f"{np.mean(pr):.2f}" if pr else "–"])
            parts.append("Linear probes on the global-average-pooled output of encoder blocks, trained on the other models' anchors and "
                         "scored on the test model (mean over folds and seeds). `R2` is in the target's own units and goes strongly "
                         "negative when a probe trained on other scans meets a different scale (dip depth differs by orders of magnitude "
                         "between scans); `R2_z` standardises each target within each model first; Pearson r (block 6) ignores scale "
                         "and offset altogether. `AUROC` is for the departure label (0.5 = chance).\n\n" +
                         md_table(["target", "metric", "encoder", "block 1", "block 3", "block 6", "block 6 Pearson r"], body))
            parts.append("\n" + "\n".join(f"![{t}](figures/e3_probe_{t}.png)" for t in targets) + "\n")

    figs = sorted(f for f in os.listdir(os.path.join(out, "figures")) if f.startswith("e0b_example"))
    if figs:
        parts.append("## Departure examples\n\nChosen automatically: largest geometric offsets, at most two per model.\n")
        parts.append("\n".join(f"![{f}](figures/{f})" for f in figs) + "\n")

    findings = os.path.join(out, "findings.md")
    parts.append("## Findings\n")
    parts.append(open(findings).read() if os.path.exists(findings) else "_(findings.md not written yet)_\n")
    return "\n".join(p for p in parts if p is not None) + "\n"


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    out = argv[0] if argv else "results"
    text = build(out)
    with open(os.path.join(out, "RESULTS.md"), "w") as f:
        f.write(text)
    print(f"wrote {os.path.join(out, 'RESULTS.md')}")


if __name__ == "__main__":
    main()
