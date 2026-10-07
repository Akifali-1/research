"""Aggregate validation-only Day 2 ablation artifacts; never reads final-test data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


VARIANTS = ("temporal_only", "spatial_only", "fixed_fusion", "stgt", "adaptive_stgt")
SEEDS = (42, 123, 2026)
METRICS = ("MAE", "RMSE", "WAPE", "sMAPE", "R2", "MAPE_masked")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path, rows):
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def manifest_ok(directory, manifest):
    for entry in manifest["files"]:
        path = directory / entry["name"]
        raw = path.read_bytes()
        if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            return False
    return True


def svg_bar(path, title, labels, values, ylabel):
    width, height = 1100, 620
    left, right, top, bottom = 90, 30, 70, 95
    plot_w, plot_h = width - left - right, height - top - bottom
    low, high = min(0.0, min(values)), max(0.0, max(values))
    if high == low:
        high = low + 1.0
    padding = (high - low) * 0.08
    low -= padding
    high += padding
    bar_w = plot_w / max(1, len(values)) * 0.65

    def y(value):
        return top + plot_h * (high - value) / (high - low)

    zero = y(0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="32" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="bold">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{zero:.1f}" x2="{left+plot_w}" y2="{zero:.1f}" stroke="#333"/>',
        f'<text x="20" y="{top+plot_h/2:.1f}" transform="rotate(-90 20 {top+plot_h/2:.1f})" text-anchor="middle" font-family="sans-serif">{ylabel}</text>',
    ]
    for index, (label, value) in enumerate(zip(labels, values)):
        x = left + (index + 0.5) * plot_w / len(values)
        y_value = y(value)
        top_y = min(zero, y_value)
        bar_h = abs(zero - y_value)
        color = "#2563eb" if value >= 0 else "#dc2626"
        parts.append(f'<rect x="{x-bar_w/2:.1f}" y="{top_y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x:.1f}" y="{top+plot_h+24}" text-anchor="middle" font-family="sans-serif" font-size="12">{label}</text>')
        parts.append(f'<text x="{x:.1f}" y="{top_y-5 if value >= 0 else top_y+bar_h+15:.1f}" text-anchor="middle" font-family="sans-serif" font-size="11">{value:.2f}</text>')
    parts.append(f'<text x="{left+plot_w/2:.1f}" y="{height-25}" text-anchor="middle" font-family="sans-serif">Variant</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    registry_path = args.registry.resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("campaign") != "day2_ablation" or registry["budget"]["max_runs"] != 15:
        raise ValueError("This is not the independent 15-run Day 2 registry")
    records = registry["experiments"]
    expected = {(variant, seed) for variant in VARIANTS for seed in SEEDS}
    actual = {(record["model"], int(record["seed"])) for record in records}
    if actual != expected or len(records) != 15 or any(record["status"] != "completed" for record in records):
        raise ValueError("Day 2 report requires exactly 15 completed validation runs")
    if any("final_test" in record["artifact_dir"] for record in records):
        raise ValueError("Day 2 artifacts must not be under final_test")

    rows = []
    for record in records:
        artifact = Path(record["artifact_dir"])
        if not manifest_ok(artifact, record["artifact_manifest"]):
            raise ValueError(f"Artifact manifest mismatch: {artifact}")
        report = json.loads((artifact / "metrics/validation.json").read_text(encoding="utf-8"))
        row = {
            "experiment_id": record["experiment_id"],
            "variant": record["model"],
            "seed": int(record["seed"]),
            "git_commit": record["git_commit"],
            "selected_epoch": report["selected_epoch"],
            "parameter_count": report["parameter_count"],
            "training_seconds": report["training_seconds"],
            "inference_seconds": report["inference_seconds"],
            **report["validation"],
        }
        rows.append(row)
    rows.sort(key=lambda row: (VARIANTS.index(row["variant"]), row["seed"]))
    write_csv(output / "per_seed_results.csv", rows)
    write_csv(output / "ablation_results.csv", rows)

    summary = []
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        item = {"variant": variant, "seed_count": len(selected)}
        for field in (*METRICS, "parameter_count", "selected_epoch", "training_seconds", "inference_seconds"):
            values = [float(row[field]) for row in selected]
            item[f"{field}_mean"] = statistics.fmean(values)
            item[f"{field}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(item)
    write_csv(output / "ablation_summary.csv", summary)
    write_csv(output / "parameter_counts.csv", [{"variant": row["variant"], "seed": row["seed"], "parameter_count": row["parameter_count"]} for row in rows])

    adaptive = {(row["seed"]): row for row in rows if row["variant"] == "adaptive_stgt"}
    comparisons = []
    for baseline in ("stgt", "fixed_fusion", "temporal_only", "spatial_only"):
        for seed in SEEDS:
            proposed, control = adaptive[seed], next(row for row in rows if row["variant"] == baseline and row["seed"] == seed)
            item = {"comparison": f"adaptive_stgt_vs_{baseline}", "baseline_variant": baseline, "seed": seed}
            for metric in METRICS:
                delta = float(proposed[metric]) - float(control[metric])
                item[f"{metric}_delta_adaptive_minus_baseline"] = delta
                item[f"{metric}_improvement_pct"] = ((float(control[metric]) - float(proposed[metric])) / abs(float(control[metric])) * 100) if metric != "R2" else ((float(proposed[metric]) - float(control[metric])) / abs(float(control[metric])) * 100)
            item["parameter_delta_adaptive_minus_baseline"] = int(proposed["parameter_count"]) - int(control["parameter_count"])
            comparisons.append(item)
    write_csv(output / "pairwise_comparisons.csv", comparisons)

    mean_improvements = {}
    for baseline in ("stgt", "fixed_fusion", "temporal_only", "spatial_only"):
        selected = [row for row in comparisons if row["baseline_variant"] == baseline]
        mean_improvements[baseline] = {metric: statistics.fmean(float(row[f"{metric}_improvement_pct"]) for row in selected) for metric in METRICS}
    labels = list(VARIANTS)
    mae_means = [next(item["MAE_mean"] for item in summary if item["variant"] == variant) for variant in labels]
    svg_bar(output / "validation_mae_by_variant.svg", "Day 2 validation MAE", labels, mae_means, "Validation MAE")
    adaptive_mae_improvement = [0.0] + [mean_improvements[baseline]["MAE"] for baseline in ("stgt", "fixed_fusion", "temporal_only", "spatial_only")]
    svg_bar(output / "adaptive_pairwise_mae_improvement.svg", "Adaptive STGT validation MAE improvement", ["self", "STGT", "fixed", "temporal", "spatial"], adaptive_mae_improvement, "Improvement (%)")

    adaptive_wins = {}
    for baseline in ("stgt", "fixed_fusion", "temporal_only", "spatial_only"):
        selected = [row for row in comparisons if row["baseline_variant"] == baseline]
        adaptive_wins[baseline] = {
            "MAE": sum(float(row["MAE_delta_adaptive_minus_baseline"]) < 0 for row in selected),
            "RMSE": sum(float(row["RMSE_delta_adaptive_minus_baseline"]) < 0 for row in selected),
            "WAPE": sum(float(row["WAPE_delta_adaptive_minus_baseline"]) < 0 for row in selected),
            "sMAPE": sum(float(row["sMAPE_delta_adaptive_minus_baseline"]) < 0 for row in selected),
            "R2": sum(float(row["R2_delta_adaptive_minus_baseline"]) > 0 for row in selected),
        }

    parameter_summary = {item["variant"]: item["parameter_count_mean"] for item in summary}
    findings = [
        "# Day 2 validation ablation findings",
        "",
        "This report uses validation artifacts only. No frozen final-test artifact was read or modified.",
        "",
        "## Campaign",
        "",
        f"- Completed runs: {len(rows)} ({len(VARIANTS)} variants × {len(SEEDS)} seeds)",
        f"- Registry: `{registry_path}`",
        "- Checkpoint selection: validation loss only, using the shared training protocol.",
        "",
        "## Results",
        "",
    ]
    for variant in VARIANTS:
        item = next(entry for entry in summary if entry["variant"] == variant)
        findings.append(f"- **{variant}**: MAE {item['MAE_mean']:.3f} ± {item['MAE_std']:.3f}; RMSE {item['RMSE_mean']:.3f} ± {item['RMSE_std']:.3f}; WAPE {item['WAPE_mean']:.5f} ± {item['WAPE_std']:.5f}; R² {item['R2_mean']:.5f} ± {item['R2_std']:.5f}.")
    findings.extend(["", "## Adaptive STGT pairwise evidence", ""])
    for baseline, values in mean_improvements.items():
        findings.append(f"- Versus **{baseline}**, Adaptive STGT mean improvement: MAE {values['MAE']:.2f}%, RMSE {values['RMSE']:.2f}%, WAPE {values['WAPE']:.2f}%, sMAPE {values['sMAPE']:.2f}%, R² {values['R2']:.2f}%; wins across seeds: {adaptive_wins[baseline]}.")
    findings.extend([
        "",
        "## Interpretation rules",
        "",
        "- Adaptive STGT is supported as an adaptive-fusion explanation only if it consistently beats fixed fusion and the non-adaptive controls on validation metrics.",
        "- Temporal-only versus spatial-only versus fused variants distinguish whether both information paths are useful.",
        "- This five-variant campaign does not include a parameter-matched STGT control; parameter counts are reported, but the parameter-count hypothesis cannot be isolated conclusively here.",
        "- No Day 3 experiment is authorized by this report.",
    ])
    (output / "day2_findings.md").write_text("\n".join(findings) + "\n", encoding="utf-8")
    write_json(output / "analysis_manifest.json", {
        "analysis": "Day 2 validation-only ablation report",
        "status": "completed",
        "registry": str(registry_path),
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "run_count": len(rows),
        "final_test_read": False,
        "parameter_matched_control_included": False,
        "outputs": sorted(path.name for path in output.iterdir() if path.is_file()),
    })


if __name__ == "__main__":
    main()
