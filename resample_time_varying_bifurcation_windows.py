"""Resample training windows from already simulated time-varying bifurcation data.

This script does not integrate any ODEs. It loads full trajectories from an
existing generator output directory and writes new training/test arrays with a
different window-sampling strategy.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

Array = np.ndarray


def path_for_io(path: Path) -> str:
    resolved = str(path.resolve())
    if not resolved.startswith("\\\\?\\") and len(resolved) >= 240:
        if resolved.startswith("\\\\"):
            return "\\\\?\\UNC\\" + resolved[2:]
        return "\\\\?\\" + resolved
    return resolved


def save_npy(path: Path, value: Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path_for_io(path), value)


def open_for_write_text(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path_for_io(path), "w", encoding="utf-8")


def first_phi_crossing(phi_series: Array, reference_phi: float) -> int | None:
    phi_flat = phi_series[:, 0] if phi_series.ndim == 2 else phi_series
    signs = np.signbit(phi_flat - reference_phi)
    candidates = np.flatnonzero(signs[:-1] != signs[1:])
    if candidates.size:
        return int(candidates[0] + 1)
    return None


def choose_sampling_mode(
    rng: np.random.Generator,
    window_sampling: str,
    crossing_fraction: float,
    pre_fraction: float,
) -> str:
    if window_sampling != "mixed":
        return window_sampling
    u = float(rng.random())
    if u < crossing_fraction:
        return "crossing"
    if u < crossing_fraction + pre_fraction:
        return "pre"
    return "post"


def sample_start(
    phi_series: Array,
    n_steps: int,
    context_len: int,
    seq_len: int,
    data_len: int,
    rng: np.random.Generator,
    mode: str,
    reference_phi: float,
) -> tuple[int, int, str]:
    crossing_idx = first_phi_crossing(phi_series, reference_phi)

    if mode == "random":
        return int(rng.integers(0, n_steps - seq_len + 1)), -1, mode

    if crossing_idx is None:
        raise ValueError(
            f"Sampling mode '{mode}' requires phi to cross reference_phi={reference_phi}."
        )

    if mode == "crossing":
        data_offset = seq_len - data_len
        lower = max(0, crossing_idx - seq_len + 1)
        upper = min(
            n_steps - seq_len,
            crossing_idx - context_len,
            crossing_idx - data_offset,
        )
    elif mode == "pre":
        lower = 0
        upper = min(n_steps - seq_len, crossing_idx - seq_len)
    elif mode == "post":
        lower = max(0, crossing_idx)
        upper = n_steps - seq_len
    else:
        raise ValueError(f"Unknown sampling mode: {mode}")

    if lower > upper:
        raise ValueError(
            f"No valid '{mode}' window exists. Try shorter lengths or another sampling mix."
        )
    return int(rng.integers(lower, upper + 1)), crossing_idx, mode


def build_training_windows(
    trajectories: Array,
    phi: Array,
    phi_raw: Array,
    regime_names: Array,
    windows_per_trajectory: int,
    context_len: int,
    seq_len: int,
    overlap_len: int,
    rng: np.random.Generator,
    window_sampling: str,
    crossing_fraction: float,
    pre_fraction: float,
    reference_phi: float,
) -> dict[str, Array]:
    if seq_len <= context_len:
        raise ValueError("seq_len must be greater than context_len")
    if overlap_len < 0:
        raise ValueError("overlap_len must be non-negative")
    if overlap_len > context_len:
        raise ValueError("overlap_len must be <= context_len")

    prediction_window_len = seq_len - context_len
    data_len = overlap_len + prediction_window_len
    if data_len <= 0:
        raise ValueError("data_len must be positive")

    n_series, n_steps, n_dims = trajectories.shape
    if n_steps < seq_len:
        raise ValueError(f"Need n_steps >= seq_len, got {n_steps} < {seq_len}")

    total_windows = n_series * windows_per_trajectory
    data = np.empty((data_len, total_windows, n_dims), dtype=np.float32)
    context = np.empty((context_len, total_windows, n_dims), dtype=np.float32)
    phi_data = np.empty((data_len, total_windows, 1), dtype=np.float32)
    phi_context = np.empty((context_len, total_windows, 1), dtype=np.float32)
    phi_raw_data = np.empty((data_len, total_windows, 1), dtype=np.float32)
    phi_raw_context = np.empty((context_len, total_windows, 1), dtype=np.float32)
    start_indices = np.empty(total_windows, dtype=np.int64)
    crossing_indices = np.empty(total_windows, dtype=np.int64)
    sampled_modes: list[str] = []
    window_regime: list[str] = []

    out_idx = 0
    for series_idx in range(n_series):
        for _ in range(windows_per_trajectory):
            requested_mode = choose_sampling_mode(
                rng,
                window_sampling,
                crossing_fraction,
                pre_fraction,
            )
            start, crossing_idx, actual_mode = sample_start(
                phi[series_idx],
                n_steps,
                context_len,
                seq_len,
                data_len,
                rng,
                requested_mode,
                reference_phi,
            )
            end = start + seq_len
            data_start = start + context_len - overlap_len

            context[:, out_idx, :] = trajectories[
                series_idx, start : start + context_len, :
            ]
            data[:, out_idx, :] = trajectories[series_idx, data_start:end, :]
            phi_context[:, out_idx, :] = phi[series_idx, start : start + context_len, :]
            phi_data[:, out_idx, :] = phi[series_idx, data_start:end, :]
            phi_raw_context[:, out_idx, :] = phi_raw[
                series_idx, start : start + context_len, :
            ]
            phi_raw_data[:, out_idx, :] = phi_raw[series_idx, data_start:end, :]
            start_indices[out_idx] = start
            crossing_indices[out_idx] = crossing_idx
            sampled_modes.append(actual_mode)
            window_regime.append(str(regime_names[series_idx]))
            out_idx += 1

    return {
        "data": data,
        "context": context,
        "phi": phi_data,
        "context_phi": phi_context,
        "phi_raw": phi_raw_data,
        "context_phi_raw": phi_raw_context,
        "start_indices": start_indices,
        "crossing_indices": crossing_indices,
        "window_regime_names": np.array(window_regime),
        "window_sampling_modes": np.array(sampled_modes),
    }


def build_test_windows(
    trajectories: Array,
    phi: Array,
    phi_raw: Array,
    test_len: int,
) -> dict[str, Array]:
    if trajectories.shape[1] < test_len:
        raise ValueError("n_steps must be at least context_len + prediction_len")
    return {
        "test": trajectories[:, -test_len:, :].transpose(1, 0, 2).astype(np.float32),
        "test_phi": phi[:, -test_len:, :].transpose(1, 0, 2).astype(np.float32),
        "test_phi_raw": phi_raw[:, -test_len:, :].transpose(1, 0, 2).astype(np.float32),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--windows-per-trajectory", type=int, default=16)
    parser.add_argument("--context-len", type=int, default=500)
    parser.add_argument("--seq-len", type=int, default=550)
    parser.add_argument(
        "--overlap-len",
        type=int,
        default=None,
        help=(
            "Number of final context steps also included at the beginning of data. "
            "Defaults to seq_len - context_len to preserve the previous behavior."
        ),
    )
    parser.add_argument("--prediction-len", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument(
        "--window-sampling",
        choices=("random", "crossing", "pre", "post", "mixed"),
        default="mixed",
    )
    parser.add_argument(
        "--crossing-fraction",
        type=float,
        default=0.6,
        help="Fraction of mixed windows sampled around the reference-phi crossing.",
    )
    parser.add_argument(
        "--pre-fraction",
        type=float,
        default=0.2,
        help="Fraction of mixed windows sampled completely before the reference-phi crossing.",
    )
    parser.add_argument(
        "--reference-phi",
        type=float,
        default=0.0,
        help=(
            "Phi value used as the reference point for pre/crossing/post window "
            "sampling. Default 0.0 keeps the old bifurcation-centered behavior."
        ),
    )
    parser.add_argument(
        "--copy-full-arrays",
        action="store_true",
        help="Also copy full trajectory arrays into the output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overlap_len = (
        args.seq_len - args.context_len
        if args.overlap_len is None
        else args.overlap_len
    )
    if args.crossing_fraction < 0 or args.pre_fraction < 0:
        raise ValueError("Fractions must be non-negative")
    if args.crossing_fraction + args.pre_fraction > 1:
        raise ValueError("crossing_fraction + pre_fraction must be <= 1")

    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trajectories = np.load(args.input_dir / "trajectories.npy")
    phi = np.load(args.input_dir / "trajectory_phi.npy")
    phi_raw = np.load(args.input_dir / "trajectory_phi_raw.npy")
    regime_names = np.load(
        args.input_dir / "trajectory_regime_names.npy", allow_pickle=True
    )

    windows = build_training_windows(
        trajectories,
        phi,
        phi_raw,
        regime_names,
        windows_per_trajectory=args.windows_per_trajectory,
        context_len=args.context_len,
        seq_len=args.seq_len,
        overlap_len=overlap_len,
        rng=rng,
        window_sampling=args.window_sampling,
        crossing_fraction=args.crossing_fraction,
        pre_fraction=args.pre_fraction,
        reference_phi=args.reference_phi,
    )
    test = build_test_windows(
        trajectories,
        phi,
        phi_raw,
        test_len=args.context_len + args.prediction_len,
    )

    for name, value in windows.items():
        save_npy(args.output_dir / f"{name}.npy", value)
    for name, value in test.items():
        save_npy(args.output_dir / f"{name}.npy", value)

    if args.copy_full_arrays:
        for name in [
            "trajectories.npy",
            "trajectory_phi.npy",
            "trajectory_phi_raw.npy",
            "trajectory_regime_index.npy",
            "trajectory_regime_names.npy",
        ]:
            src = args.input_dir / name
            if src.exists():
                shutil.copy2(src, args.output_dir / name)

    source_metadata_path = args.input_dir / "metadata.json"
    source_metadata = {}
    if source_metadata_path.exists():
        with open(source_metadata_path, "r", encoding="utf-8") as f:
            source_metadata = json.load(f)

    unique_modes, counts = np.unique(
        windows["window_sampling_modes"], return_counts=True
    )
    metadata = {
        "description": "Training windows resampled from existing time-varying bifurcation trajectories.",
        "source_dir": str(args.input_dir),
        "source_args": source_metadata.get("args", {}),
        "resample_args": vars(args)
        | {
            "input_dir": str(args.input_dir),
            "output_dir": str(args.output_dir),
            "resolved_overlap_len": overlap_len,
        },
        "shapes": {
            "data": list(windows["data"].shape),
            "context": list(windows["context"].shape),
            "phi": list(windows["phi"].shape),
            "test": list(test["test"].shape),
            "test_phi": list(test["test_phi"].shape),
        },
        "sample_counts": {
            str(mode): int(count) for mode, count in zip(unique_modes, counts)
        },
        "window_sampling_convention": (
            "pre windows are fully before the first reference-phi crossing; crossing "
            "windows have context before the crossing and data spanning it; post "
            "windows start after the crossing."
        ),
        "window_layout_convention": (
            "context covers [start, start + context_len). data covers the last "
            "overlap_len context steps plus all steps after context until seq_len, "
            "so data_len = overlap_len + (seq_len - context_len)."
        ),
    }
    if "regimes" in source_metadata:
        metadata["regimes"] = source_metadata["regimes"]
    if "phi_convention" in source_metadata:
        metadata["phi_convention"] = source_metadata["phi_convention"]

    with open_for_write_text(args.output_dir / "metadata.json") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved resampled windows to {args.output_dir.resolve()}")
    for key, shape in metadata["shapes"].items():
        print(f"{key}: {shape}")
    print(f"sample_counts: {metadata['sample_counts']}")


if __name__ == "__main__":
    main()
