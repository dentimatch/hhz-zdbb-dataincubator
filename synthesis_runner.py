"""Shared utilities for running SDV synthesizers with progress feedback."""

from __future__ import annotations

import inspect
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Tuple

import pandas as pd
from sdv.metadata import SingleTableMetadata

LOGGER = logging.getLogger(__name__)

BASE_MINUTES_PER_CELL = 1.0 / 25000  # 25k Zellen ≈ 1 Minute Baseline
MODEL_DURATION_FACTORS = {
    "gaussiancopula": 0.8,
    "ctgan": 4.5,
    "tvae": 2.8,
}


@dataclass
class TrainingResult:
    df_synth: pd.DataFrame
    used_global_seed: bool
    actual_seconds: float


def load_dataset(path: Path | str, metadata_path: Optional[Path] = None) -> Tuple[pd.DataFrame, SingleTableMetadata]:
    dataframe = pd.read_csv(path)
    metadata = load_metadata(metadata_path, dataframe)
    return dataframe, metadata


def load_metadata(path: Optional[Path], dataframe: pd.DataFrame) -> SingleTableMetadata:
    metadata = SingleTableMetadata()
    if path and path.exists():
        metadata = SingleTableMetadata.load_from_json(path)
    else:
        metadata.detect_from_dataframe(dataframe)
    return metadata


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - optional dependency
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover - optional dependency
        pass


def build_synthesizer(
    name: str,
    metadata: SingleTableMetadata,
    registry: Mapping[str, type],
    random_state: Optional[int],
    init_kwargs: Optional[dict] = None,
) -> Tuple[object, bool]:
    normalized = {key.lower(): value for key, value in registry.items()}
    try:
        SynthClass = normalized[name.lower()]
    except KeyError as exc:  # pragma: no cover - defensive guard
        available = ", ".join(sorted(normalized))
        raise ValueError(f"Unknown synthesizer '{name}'. Options: {available}") from exc

    kwargs = {}
    used_global_seed = False
    params = inspect.signature(SynthClass.__init__).parameters
    if random_state is not None:
        if "random_state" in params:
            kwargs["random_state"] = random_state
        else:
            seed_everything(random_state)
            used_global_seed = True
            LOGGER.info(
                "Synthesizer '%s' supports no random_state; seeded global RNGs.",
                name,
            )

    if init_kwargs:
        for key, value in init_kwargs.items():
            if key in params:
                kwargs[key] = value
            else:
                LOGGER.debug("Ignoriere unbekannte Initialisierungsoption '%s' für %s", key, name)

    synthesizer = SynthClass(metadata, **kwargs)
    return synthesizer, used_global_seed


def estimate_duration_minutes(rows: int, columns: int, model_name: str, additional_rows: int) -> float:
    total_rows = rows + additional_rows
    cols = max(columns, 1)
    workload = total_rows * cols

    volume_minutes = workload * BASE_MINUTES_PER_CELL
    sparsity_factor = 1.0 + max(cols - 10, 0) * 0.1
    factor = MODEL_DURATION_FACTORS.get(model_name.lower(), 3.0)
    minutes = volume_minutes * sparsity_factor * factor

    if total_rows > 50000:
        minutes *= 1.3

    return max(minutes, 0.2)


def format_duration_label(minutes: float) -> str:
    if minutes < 1:
        return "< 1 Minute"
    if minutes < 5:
        return "ca. 1–5 Minuten"
    if minutes < 15:
        return "ca. 5–15 Minuten"
    if minutes < 30:
        return "ca. 15–30 Minuten"
    if minutes < 60:
        return "ca. 30–60 Minuten"
    hours = minutes / 60
    return f"> {hours:.1f} Stunden"


def format_remaining_seconds(seconds: float) -> str:
    if seconds <= 0:
        return "< 10s"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"


def format_elapsed_time(seconds: float) -> str:
    if seconds < 1:
        return "<1s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"


ProgressCallback = Callable[[float, str], None]


def run_training(
    df_real: pd.DataFrame,
    metadata: SingleTableMetadata,
    model_name: str,
    random_seed: int,
    registry: Mapping[str, type],
    total_rows: int,
    estimated_minutes: float,
    progress_callback: Optional[ProgressCallback] = None,
    init_kwargs: Optional[dict] = None,
    poll_interval: float = 0.5,
) -> TrainingResult:
    synthesizer, used_global_seed = build_synthesizer(
        model_name,
        metadata,
        registry,
        random_seed,
        init_kwargs=init_kwargs,
    )

    est_seconds = max(5.0, estimated_minutes * 60)
    if progress_callback:
        progress_callback(0.0, f"Training gestartet (≈ {format_duration_label(estimated_minutes)})")

    with ThreadPoolExecutor(max_workers=1) as executor:
        start_time = time.perf_counter()
        future = executor.submit(synthesizer.fit, df_real)
        while not future.done():
            elapsed = time.perf_counter() - start_time
            ratio = min(elapsed / est_seconds, 0.98)
            remaining_label = format_remaining_seconds(max(est_seconds - elapsed, 0))
            if progress_callback:
                progress_callback(ratio, f"Training läuft (≈ {remaining_label} verbleibend)")
            time.sleep(poll_interval)
        future.result()
        actual_seconds = time.perf_counter() - start_time

    if progress_callback:
        progress_callback(1.0, f"Training abgeschlossen in {format_elapsed_time(actual_seconds)}")

    df_synth = synthesizer.sample(num_rows=total_rows)
    return TrainingResult(df_synth=df_synth, used_global_seed=used_global_seed, actual_seconds=actual_seconds)

