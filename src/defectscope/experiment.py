import csv
import json
import platform
import time
from pathlib import Path
import numpy as np
from PIL import Image
from .data import splits, provenance
from .metrics import auroc, binary_metrics
from .model import Detector, MODES
from .reporting import gallery

def evaluate(data, output, seed=7, extractor="texture", size=96, patch=6, memory=128, alpha=.05, illumination=False):
    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"metrics.json").exists():
        raise ValueError("results already exist; choose a new output directory")
    train, tune, cal, tests, shifts = splits(data, seed)
    manifest = provenance({"train": train, "tune": tune, "calibration": cal, "test": [r["path"] for r in tests], "shift": shifts}, data)
    (output/"splits.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    start = time.perf_counter()
    model = Detector(extractor, size, patch, memory, seed, alpha, illumination).fit(train, tune, cal)
    fit_seconds = time.perf_counter()-start
    model.save(output/"model.npz")
    rows, results = [], {}
    selected = []
    for mode in MODES:
        ys, scores, ps, pixel_y, pixel_s, latencies = [], [], [], [], [], []
        records = []
        intersection = union = 0
        for item in tests:
            start = time.perf_counter()
            pred = model.predict(item["path"], mode)
            latencies.append((time.perf_counter()-start)*1000)
            truth = np.zeros((size, size), dtype=bool)
            if item["mask"]:
                with Image.open(item["mask"]) as im:
                    truth = np.asarray(im.convert("L").resize((size,size), Image.Resampling.NEAREST)) > 0
            pixel_map = np.asarray(Image.fromarray(pred["heatmap"].astype("float32")).resize((size,size), Image.Resampling.BILINEAR))
            predicted_mask = np.asarray(Image.fromarray(np.uint8(pred["heatmap"] > pred["pixel_threshold"])*255).resize((size,size), Image.Resampling.NEAREST)) > 0
            intersection += int((truth & predicted_mask).sum())
            union += int((truth | predicted_mask).sum())
            ys.append(item["label"]); scores.append(pred["score"]); ps.append(pred["is_anomaly"])
            pixel_y.append(truth.ravel()); pixel_s.append(pixel_map.ravel())
            records.append({"path": item["path"].relative_to(data).as_posix(), "kind": item["kind"], "label": int(item["label"]), "score": pred["score"], "p_value": pred["p_value"], "prediction": int(pred["is_anomaly"])})
            if mode == "gated" and item["path"].stem == "000":
                selected.append((item["path"], item["mask"], pred))
        shift_predictions = [model.predict(p, mode)["is_anomaly"] for p in shifts]
        detail = binary_metrics(ys, ps)
        detail.update(image_auroc=auroc(ys, scores), pixel_auroc=auroc(np.concatenate(pixel_y), np.concatenate(pixel_s)),
                      pixel_iou=intersection/union if union else None,
                      shift_fpr=float(np.mean(shift_predictions)) if shifts else None,
                      latency_ms_median=float(np.median(latencies)), latency_ms_p95=float(np.quantile(latencies,.95)),
                      per_defect_recall={kind: float(np.mean([r["prediction"] for r in records if r["kind"]==kind])) for kind in sorted({r["kind"] for r in records if r["label"]})})
        results[mode] = detail
        with (output/(mode+"_predictions.csv")).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
        rows.append({"mode":mode, **{k:v for k,v in detail.items() if not isinstance(v,dict)}})
    manifest_source = data.parent/"dataset.json"
    source = json.loads(manifest_source.read_text(encoding="utf-8")) if manifest_source.exists() else {"source":"user-supplied MVTec-compatible directory"}
    metadata = {"category": data.name, "source": source, "seed":seed, "config":model.config,
                "split_counts": model.split_counts, "test_count":len(tests), "shift_count":len(shifts),
                "fit_seconds":fit_seconds, "memory_bytes":model.bank.nbytes, "model_file_bytes":(output/"model.npz").stat().st_size,
                "environment":{"python":platform.python_version(), "platform":platform.platform(), "numpy":np.__version__},
                "results":results}
    (output/"metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with (output/"metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if selected:
        gallery(selected, output/"gallery.png", size)
    lines = ["# Experiment report", "", f"Category: {data.name}; extractor: {extractor}; seed: {seed}.", "",
             "Thresholds use normal tuning/calibration images only. Test labels are used only for evaluation.",
             "Pixel metrics are computed after resizing ground truth to model resolution; they are not full-resolution benchmark metrics.", "",
             "| Mode | Image AUROC | Pixel AUROC | F1 | Normal FPR | Shift FPR | Median ms |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for mode, d in results.items():
        fmt=lambda v: "N/A" if v is None else f"{v:.4f}"
        lines.append("| "+" | ".join([mode]+[fmt(d[k]) for k in ("image_auroc","pixel_auroc","f1","fpr","shift_fpr","latency_ms_median")])+" |")
    lines += ["", "## Limits", "", "This is a single-category, single-seed run. A stronger score on synthetic data does not establish improvement on real industrial data. Gating is a design hypothesis; compare all ablations, including failure cases.", "", "![Examples](gallery.png)"]
    (output/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return metadata
