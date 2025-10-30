"""Minimal SDV workflow for generating synthetic tabular data.

Usage (PowerShell):

    python sdv_tabular_example.py --csv data/input.csv --output data/synthetic.csv

The script detects metadata automatically if no JSON schema is supplied.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from sdv.single_table import (
    CTGANSynthesizer,
    GaussianCopulaSynthesizer,
    TVAESynthesizer,
)

from evaluation_utils import (
    error_suggestions,
    interpret_utility_score,
    run_quality_report,
)
from synthesis_runner import (
    estimate_duration_minutes,
    format_duration_label,
    format_elapsed_time,
    load_dataset,
    run_training,
)


SYNTHESIZER_REGISTRY = {
    "ctgan": CTGANSynthesizer,
    "gaussiancopula": GaussianCopulaSynthesizer,
    "tvae": TVAESynthesizer,
}
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic tabular data with SDV")
    parser.add_argument("--csv", required=True, type=Path, help="Pfad zur Quelldatei (CSV)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("synthetic_output.csv"),
        help="Zielpfad für das synthetische Dataset",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Anzahl synthetischer Zeilen (Standard = Originalgröße)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optionaler Pfad zu einer SDV-Metadatei (*.json)",
    )
    parser.add_argument(
        "--model",
        choices=tuple(SYNTHESIZER_REGISTRY.keys()),
        default="ctgan",
        help="Zu verwendender Synthesizer",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed für reproduzierbare Ergebnisse",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optionaler JSON-Report mit Evaluationskennzahlen",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional: Trainings-Epochen (nur Synthesizer mit entsprechender Option)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional: Trainings-Batchgröße (nur Synthesizer mit entsprechender Option)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        df_real, metadata = load_dataset(args.csv, args.metadata)

        num_rows = args.rows or len(df_real)
        additional_rows = max(0, num_rows - len(df_real))
        est_minutes = estimate_duration_minutes(
            rows=len(df_real),
            columns=len(df_real.columns),
            model_name=args.model,
            additional_rows=additional_rows,
        )
        est_seconds = max(5.0, est_minutes * 60)
        print(
            "Estimated duration: "
            f"{format_duration_label(est_minutes)} (heuristic based on dataset size/model)"
        )

        sys.stderr.write("Training gestartet...\n")
        sys.stderr.flush()

        def progress_callback(fraction: float, status_text: str) -> None:
            sys.stderr.write(f"\r{status_text}")
            sys.stderr.flush()

        init_kwargs = {}
        if args.epochs is not None:
            init_kwargs["epochs"] = args.epochs
        if args.batch_size is not None:
            init_kwargs["batch_size"] = args.batch_size

        runner_result = run_training(
            df_real=df_real,
            metadata=metadata,
            model_name=args.model,
            random_seed=args.random_seed,
            registry=SYNTHESIZER_REGISTRY,
            total_rows=num_rows,
            estimated_minutes=est_minutes,
            progress_callback=progress_callback,
            poll_interval=0.5,
            init_kwargs=init_kwargs,
        )
        sys.stderr.write("\n")
        sys.stderr.flush()

        df_synth = runner_result.df_synth
        actual_seconds = runner_result.actual_seconds
        actual_duration_label = format_elapsed_time(actual_seconds)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        df_synth.to_csv(args.output, index=False)

        evaluation_score, report_details, report_warnings = run_quality_report(
            df_real,
            df_synth,
            metadata,
        )
        print(f"Synthetic dataset saved to {args.output}")
        print(f"Utility score (0-1): {evaluation_score:.3f}")
        print(f"Interpretation: {interpret_utility_score(evaluation_score)}")
        print(f"Actual duration: {actual_duration_label}")
        if init_kwargs:
            print("Used training parameters: " + ", ".join(f"{k}={v}" for k, v in init_kwargs.items()))
        if runner_result.used_global_seed:
            print(
                "Hint: selected model lacks random_state; global RNGs were seeded with the provided value.",
            )
        for warning in report_warnings:
            print(f"Warning: {warning}")

        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            report_payload = {
                "input_csv": str(args.csv),
                "output_csv": str(args.output),
                "rows": num_rows,
                "model": args.model,
                "random_seed": args.random_seed,
                "utility_score": evaluation_score,
                "estimated_duration_minutes": est_minutes,
                "actual_duration_seconds": actual_seconds,
                "training_parameters": init_kwargs,
                "quality_report": report_details.to_dict(orient="records"),
            }
            args.report.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
            print(f"Report written to {args.report}")
    except Exception as exc:  # pragma: no cover - CLI diagnostics
        print(f"Error: {exc}", file=sys.stderr)
        for tip in error_suggestions(str(exc)):
            print(f"Suggestion: {tip}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

