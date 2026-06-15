#!/usr/bin/env python3
"""Run Q2, Q4, Q5, Q6 and Q7 experiments and generate LaTeX macros.

Usage:
    python3 run_all.py              # run all experiments
    python3 run_all.py --skip-q6    # quick run without MNIST training
    python3 run_all.py --q6-only    # run only the MNIST experiment

All logs are stored in generated/logs. A generated/report_values.tex file is
created for the LaTeX report. Run this script from any working directory.
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent
GENERATED = ROOT / "generated"
LOG_DIR = GENERATED / "logs"
VALUES_TEX = GENERATED / "report_values.tex"

TASKS = [
    ("q2", "q2_renewal_moments_validate.py"),
    ("q4", "q4_direction_decoder_template.py"),
    ("q5", "q5_bss_compare_template.py"),
    ("q6", "q6_mnist_sgd_ng_template.py"),
    ("q7", "q7_maze_rl_template.py"),
]


def run_script(tag: str, filename: str) -> str:
    script = ROOT / filename
    if not script.exists():
        raise FileNotFoundError(f"Missing script: {script}")
    print(f"\n{'=' * 72}\nRunning {tag}: {filename}\n{'=' * 72}")
    process = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = process.stdout
    print(output, end="")
    (LOG_DIR / f"{tag}.log").write_text(output, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(f"{tag} failed with exit code {process.returncode}. See generated/logs/{tag}.log")
    return output


def extract(pattern: str, text: str, default: str = "N/A", flags: int = 0) -> str:
    match = re.search(pattern, text, flags)
    return match.group(1).strip() if match else default


def tex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def format_float(value: str, digits: int = 5) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return value


def q6_summary_from_csv() -> Dict[str, str]:
    csv_path = ROOT / "q6_outputs" / "q6_multiseed_results.csv"
    if not csv_path.exists():
        return {}
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    result: Dict[str, str] = {}
    for method, prefix in [("SGD", "QSixSGD"), ("diag-Fisher", "QSixNG")]:
        selected = [row for row in rows if row["method"] == method]
        if not selected:
            continue
        epoch_names = {1: "One", 2: "Two", 3: "Three"}
        for epoch in (1, 2, 3):
            vals = [float(row[f"epoch_{epoch}_accuracy"]) for row in selected]
            mean = sum(vals) / len(vals)
            variance = sum((x - mean) ** 2 for x in vals) / max(len(vals) - 1, 1)
            epoch_name = epoch_names[epoch]
            result[f"{prefix}Epoch{epoch_name}Mean"] = f"{mean:.5f}"
            result[f"{prefix}Epoch{epoch_name}Std"] = f"{variance ** 0.5:.5f}"
        vals = [float(row["seconds"]) for row in selected]
        mean = sum(vals) / len(vals)
        variance = sum((x - mean) ** 2 for x in vals) / max(len(vals) - 1, 1)
        result[f"{prefix}SecondsMean"] = f"{mean:.3f}"
        result[f"{prefix}SecondsStd"] = f"{variance ** 0.5:.3f}"
    result["QSixSeeds"] = str(len({row["seed"] for row in rows}))
    return result


def build_values(outputs: Dict[str, str]) -> Dict[str, str]:
    values: Dict[str, str] = {}

    q2 = outputs.get("q2", "")
    values.update({
        "QTwoRate": extract(r"rate lambda = ([^\n]+)", q2),
        "QTwoHorizon": extract(r"time horizon t = ([^\n]+)", q2),
        "QTwoTrials": extract(r"number of trials = ([^\n]+)", q2),
        "QTwoTheoryMean": format_float(extract(r"theoretical Poisson mean = ([^\n]+)", q2)),
        "QTwoEmpMean": format_float(extract(r"empirical E\[N\(t\)\] = ([^\n]+)", q2)),
        "QTwoEmpVar": format_float(extract(r"empirical Var\[N\(t\)\] = ([^\n]+)", q2)),
    })

    q4 = outputs.get("q4", "")
    values.update({
        "QFourTuning": tex_escape(extract(r"tuning function = ([^\n]+)", q4)),
        "QFourNeurons": extract(r"number of neurons N = ([^\n]+)", q4),
        "QFourWindow": extract(r"observation window T = ([^\n]+)", q4),
        "QFourTrueDeg": extract(r"true direction \(deg\) = ([^\n]+)", q4),
        "QFourSigmaDeg": extract(r"sigma_a \(deg\) = ([^\n]+)", q4),
        "QFourTrials": extract(r"number of trials = ([^\n]+)", q4),
        "QFourBiasRad": format_float(extract(r"empirical bias \(rad\) = ([^\n]+)", q4), 6),
        "QFourBiasDeg": format_float(extract(r"empirical bias \(deg\) = ([^\n]+)", q4), 4),
        "QFourMSE": format_float(extract(r"empirical MSE \(rad\^2\) = ([^\n]+)", q4), 6),
        "QFourRMSEDeg": format_float(extract(r"empirical RMSE \(deg\) = ([^\n]+)", q4), 4),
        "QFourFisher": format_float(extract(r"Fisher information = ([^\n]+)", q4), 4),
        "QFourCRLB": format_float(extract(r"Cramer-Rao lower bound \(rad\^2\) = ([^\n]+)", q4), 6),
        "QFourRatio": format_float(extract(r"MSE / CRLB = ([^\n]+)", q4), 4),
    })
    try:
        values["QFourExcessPct"] = f"{100.0 * (float(values['QFourRatio']) - 1.0):.2f}"
    except ValueError:
        values["QFourExcessPct"] = "N/A"

    q5 = outputs.get("q5", "")
    fastica_block = extract(r"FastICA\n(.*?)(?:\n\n|$)", q5, flags=re.S)
    pca_block = extract(r"PCA baseline\n(.*?)(?:\n\n|$)", q5, flags=re.S)
    values.update({
        "QFiveFastICACorr": extract(r"correlation = ([^\n]+)", fastica_block),
        "QFiveFastICAMI": format_float(extract(r"mutual information\s+= ([^\n]+)", fastica_block), 5),
        "QFiveFastICAKurt": tex_escape(extract(r"components\s+= ([^\n]+)", fastica_block)),
        "QFivePCACorr": extract(r"correlation = ([^\n]+)", pca_block),
        "QFivePCAMI": format_float(extract(r"mutual information\s+= ([^\n]+)", pca_block), 5),
        "QFivePCAKurt": tex_escape(extract(r"components\s+= ([^\n]+)", pca_block)),
    })
    try:
        mi_reduction = 100.0 * (1.0 - float(values["QFiveFastICAMI"]) / float(values["QFivePCAMI"]))
        values["QFiveMIReduction"] = f"{mi_reduction:.1f}"
    except ValueError:
        values["QFiveMIReduction"] = "N/A"

    q7 = outputs.get("q7", "")
    values.update({
        "QSevenGrid": tex_escape(extract(r"maze grid shape: ([^\n]+)", q7)),
        "QSevenStart": tex_escape(extract(r"start cell: ([^\n]+)", q7)),
        "QSevenGoal": tex_escape(extract(r"goal cell: ([^\n]+)", q7)),
        "QSevenBFSLength": extract(r"BFS shortest-path length: ([^\n]+)", q7),
        "QSevenQLength": extract(r"Q-learning path length: ([^\n]+)", q7),
        "QSevenReached": extract(r"Q-learning reached goal: ([^\n]+)", q7),
    })

    values.update(q6_summary_from_csv())
    return values


def write_values_tex(values: Dict[str, str]) -> None:
    GENERATED.mkdir(exist_ok=True)
    lines = ["% Automatically generated by run_all.py. Do not edit manually."]
    for key in sorted(values):
        lines.append(rf"\providecommand{{\{key}}}{{{values[key]}}}")
    VALUES_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nGenerated LaTeX values: {VALUES_TEX}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-q6", action="store_true", help="run all except MNIST training")
    parser.add_argument("--q6-only", action="store_true", help="run only MNIST training and refresh LaTeX values")
    parser.add_argument("--refresh-values-only", action="store_true", help="do not rerun experiments; rebuild LaTeX values from existing logs and CSV files")
    args = parser.parse_args()

    GENERATED.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    selected: List[tuple[str, str]]
    if args.refresh_values_only:
        selected = []
    elif args.q6_only:
        selected = [("q6", "q6_mnist_sgd_ng_template.py")]
    else:
        selected = [item for item in TASKS if not (args.skip_q6 and item[0] == "q6")]

    outputs: Dict[str, str] = {}
    for tag, filename in selected:
        outputs[tag] = run_script(tag, filename)

    # Reuse old logs for tasks not run in this invocation.
    for tag, _ in TASKS:
        log_path = LOG_DIR / f"{tag}.log"
        if tag not in outputs and log_path.exists():
            outputs[tag] = log_path.read_text(encoding="utf-8")

    values = build_values(outputs)
    write_values_tex(values)
    print("\nAll selected experiments completed successfully.")
    print("The LaTeX macros in generated/report_values.tex have been refreshed automatically.")
    print("Compile the report with: xelatex report.tex")


if __name__ == "__main__":
    main()
