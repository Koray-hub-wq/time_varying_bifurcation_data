"""Generate non-autonomous 3D dynamical-system data with known parameter drift.

The output layout mirrors the current DynaMix training arrays and adds matched
external-input arrays for the known time-varying bifurcation parameter:

    data.npy          (T_train, S, 3)
    context.npy       (T_context, S, 3)
    phi.npy           (T_train, S, 1)       normalized parameter input
    context_phi.npy   (T_context, S, 1)
    phi_raw.npy       (T_train, S, 1)       physical parameter value
    context_phi_raw.npy
    test.npy          (T_context + T_pred, S_test, 3)
    test_phi.npy      (T_context + T_pred, S_test, 1)
    test_phi_raw.npy

For the first DynaMix modification, use phi.npy/test_phi.npy as known phi_t in
the expert transition term C phi_t. The raw parameter values are saved only for
analysis and plotting.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm


Array = np.ndarray
RhsFn = Callable[[float, Array, float], Array]


@dataclass(frozen=True)
class Regime:
    name: str
    family: str
    parameter: str
    boundary: float
    start: float
    end: float
    description: str
    rhs_key: str
    params: dict


def linear_parameter(t: float, t_end: float, start: float, end: float) -> float:
    if t_end <= 0:
        return start
    s = np.clip(t / t_end, 0.0, 1.0)
    return start + (end - start) * s


def constant_parameter_value(regime: Regime, position: str) -> float:
    if position == "start":
        return regime.start
    if position == "boundary":
        return regime.boundary
    if position == "end":
        return regime.end
    if position == "midpoint":
        return 0.5 * (regime.start + regime.end)
    raise ValueError(f"Unknown constant parameter position: {position}")


def parameter_value(t: float, t_end: float, regime: Regime, mode: str, constant_position: str) -> float:
    if mode == "linear":
        return linear_parameter(t, t_end, regime.start, regime.end)
    if mode == "constant":
        return constant_parameter_value(regime, constant_position)
    raise ValueError(f"Unknown parameter mode: {mode}")


def normalize_phi(phi_raw: Array, start: float, end: float, boundary: float) -> Array:
    scale = max(abs(start - boundary), abs(end - boundary), 1e-8)
    return (phi_raw - boundary) / scale


def lorenz_rhs(t: float, x: Array, p: float, *, sigma: float, beta: float) -> Array:
    return np.array(
        [
            sigma * (x[1] - x[0]),
            x[0] * (p - x[2]) - x[1],
            x[0] * x[1] - beta * x[2],
        ],
        dtype=np.float64,
    )


def rossler_rhs(t: float, x: Array, p: float, *, a: float, b: float) -> Array:
    return np.array(
        [
            -x[1] - x[2],
            x[0] + a * x[1],
            b + x[2] * (x[0] - p),
        ],
        dtype=np.float64,
    )


def chen_rhs(t: float, x: Array, p: float, *, a: float, b: float) -> Array:
    return np.array(
        [
            a * (x[1] - x[0]),
            (p - a) * x[0] - x[0] * x[2] + p * x[1],
            x[0] * x[1] - b * x[2],
        ],
        dtype=np.float64,
    )


def lu_rhs(t: float, x: Array, p: float, *, a: float, b: float) -> Array:
    return np.array(
        [
            a * (x[1] - x[0]),
            p * x[1] - x[0] * x[2],
            x[0] * x[1] - b * x[2],
        ],
        dtype=np.float64,
    )


def brusselator_rhs(t: float, x: Array, p: float, *, a: float, eps: float) -> Array:
    u, v, z = x
    return np.array(
        [
            a - (p + 1.0) * u + u * u * v,
            p * u - u * u * v,
            -z + eps * (u - a),
        ],
        dtype=np.float64,
    )


def vdp_rhs(t: float, x: Array, p: float, *, eps: float) -> Array:
    return np.array(
        [
            x[1],
            p * (1.0 - x[0] * x[0]) * x[1] - x[0],
            -x[2] + eps * x[0],
        ],
        dtype=np.float64,
    )


def pitchfork_rhs(t: float, x: Array, p: float, *, eps: float) -> Array:
    return np.array(
        [
            p * x[0] - x[0] ** 3 + eps * x[1],
            -x[1] + eps * x[2],
            -x[2] + eps * x[0],
        ],
        dtype=np.float64,
    )


def saddle_node_rhs(t: float, x: Array, p: float, *, eps: float) -> Array:
    return np.array(
        [
            p - x[0] ** 2 - 0.05 * x[0] ** 3 + eps * x[1],
            -x[1] + eps * x[2],
            -x[2] + eps * x[0],
        ],
        dtype=np.float64,
    )


def transcritical_rhs(t: float, x: Array, p: float, *, eps: float) -> Array:
    return np.array(
        [
            p * x[0] - x[0] ** 2 + eps * x[1],
            -x[1] + eps * x[2],
            -x[2] + eps * x[0],
        ],
        dtype=np.float64,
    )


def hopf_super_rhs(t: float, x: Array, p: float, *, omega: float, eps: float) -> Array:
    r2 = x[0] * x[0] + x[1] * x[1]
    return np.array(
        [
            p * x[0] - omega * x[1] - r2 * x[0] + eps * x[2],
            omega * x[0] + p * x[1] - r2 * x[1],
            -x[2] + eps * x[0],
        ],
        dtype=np.float64,
    )


def hopf_sub_rhs(t: float, x: Array, p: float, *, omega: float, eps: float) -> Array:
    r2 = x[0] * x[0] + x[1] * x[1]
    r4 = r2 * r2
    return np.array(
        [
            p * x[0] - omega * x[1] + r2 * x[0] - r4 * x[0] + eps * x[2],
            omega * x[0] + p * x[1] + r2 * x[1] - r4 * x[1],
            -x[2] + eps * x[0],
        ],
        dtype=np.float64,
    )


def duffing_pitchfork_rhs(t: float, x: Array, p: float, *, damping: float, eps: float) -> Array:
    return np.array(
        [
            x[1],
            -damping * x[1] + p * x[0] - x[0] ** 3 + eps * x[2],
            -x[2] + eps * x[0],
        ],
        dtype=np.float64,
    )


def sprott_a_rhs(t: float, x: Array, p: float) -> Array:
    return np.array(
        [
            x[1],
            -x[0] + x[1] * x[2] - p * x[1],
            1.0 - x[1] * x[1],
        ],
        dtype=np.float64,
    )


def hindmarsh_rose_rhs(
    t: float,
    x: Array,
    p: float,
    *,
    a: float,
    b: float,
    c: float,
    d: float,
    r: float,
    s: float,
    x_r: float,
) -> Array:
    return np.array(
        [
            x[1] - a * x[0] ** 3 + b * x[0] ** 2 - x[2] + p,
            c - d * x[0] ** 2 - x[1],
            r * (s * (x[0] - x_r) - x[2]),
        ],
        dtype=np.float64,
    )


RHS: dict[str, RhsFn] = {
    "lorenz": lorenz_rhs,
    "rossler": rossler_rhs,
    "chen": chen_rhs,
    "lu": lu_rhs,
    "brusselator": brusselator_rhs,
    "vdp": vdp_rhs,
    "pitchfork": pitchfork_rhs,
    "saddle_node": saddle_node_rhs,
    "transcritical": transcritical_rhs,
    "hopf_super": hopf_super_rhs,
    "hopf_sub": hopf_sub_rhs,
    "duffing_pitchfork": duffing_pitchfork_rhs,
    "sprott_a": sprott_a_rhs,
    "hindmarsh_rose": hindmarsh_rose_rhs,
}


def default_regimes() -> list[Regime]:
    sigma = 10.0
    beta = 8.0 / 3.0
    lorenz_hopf = sigma * (sigma + beta + 3.0) / (sigma - beta - 1.0)
    return [
        Regime("lorenz_pitchfork_rho", "Lorenz", "rho", 1.0, 0.5, 1.5,
               "origin loses stability; two nontrivial equilibria appear",
               "lorenz", {"sigma": sigma, "beta": beta}),
        Regime("lorenz_hopf_rho", "Lorenz", "rho", lorenz_hopf, 20.0, 30.0,
               "nontrivial equilibria lose stability near Hopf threshold",
               "lorenz", {"sigma": sigma, "beta": beta}),
        Regime("rossler_hopf_c", "Rossler", "c", 0.4, 0.2, 0.6,
               "small fixed-point oscillations emerge near c=a+b for a=b=0.2",
               "rossler", {"a": 0.2, "b": 0.2}),
        Regime("chen_origin_c", "Chen", "c", 17.5, 15.0, 20.0,
               "origin changes stability; nontrivial equilibria become relevant",
               "chen", {"a": 35.0, "b": 3.0}),
        Regime("lu_origin_c", "Lu", "c", 0.0, -1.0, 1.0,
               "origin crosses the c=0 stability threshold",
               "lu", {"a": 36.0, "b": 3.0}),
        Regime("brusselator_hopf_B", "Brusselator3D", "B", 2.0, 1.5, 2.5,
               "stable focus to limit cycle for A=1 at B=1+A^2",
               "brusselator", {"a": 1.0, "eps": 0.1}),
        Regime("vanderpol_hopf_mu", "VanDerPol3D", "mu", 0.0, -0.5, 0.5,
               "damped focus to self-sustained oscillation",
               "vdp", {"eps": 0.1}),
        Regime("normal_pitchfork_mu", "PitchforkNormalForm3D", "mu", 0.0, -1.0, 1.0,
               "one stable equilibrium to two stable symmetric equilibria",
               "pitchfork", {"eps": 0.05}),
        Regime("normal_saddle_node_mu", "SaddleNodeNormalForm3D", "mu", 0.0, 1.0, -1.0,
               "stable fixed-point branch disappears",
               "saddle_node", {"eps": 0.05}),
        Regime("normal_transcritical_mu", "TranscriticalNormalForm3D", "mu", 0.0, -1.0, 1.0,
               "two fixed-point branches exchange stability",
               "transcritical", {"eps": 0.05}),
        Regime("normal_supercritical_hopf_mu", "SupercriticalHopf3D", "mu", 0.0, -0.5, 0.5,
               "stable focus to stable limit cycle",
               "hopf_super", {"omega": 1.0, "eps": 0.05}),
        Regime("normal_subcritical_hopf_mu", "SubcriticalHopf3D", "mu", 0.0, -0.5, 0.5,
               "stable focus loses stability with finite-amplitude oscillation basin",
               "hopf_sub", {"omega": 1.0, "eps": 0.05}),
        Regime("duffing_pitchfork_mu", "Duffing3D", "mu", 0.0, -1.0, 1.0,
               "single-well to double-well restoring force",
               "duffing_pitchfork", {"damping": 0.2, "eps": 0.05}),
        Regime("sprott_a_damping_mu", "SprottA", "mu", 0.0, 0.5, -0.3,
               "damped oscillation to more irregular Sprott-A dynamics",
               "sprott_a", {}),
        Regime("hindmarsh_rose_current_I", "HindmarshRose", "I", 2.0, 1.0, 3.0,
               "quiescent regime toward spiking/bursting; threshold is numerical",
               "hindmarsh_rose",
               {"a": 1.0, "b": 3.0, "c": 1.0, "d": 5.0, "r": 0.006, "s": 4.0, "x_r": -1.6}),
    ]


def simulate_regime(
    regime: Regime,
    n_steps: int,
    dt: float,
    n_trajectories: int,
    rng: np.random.Generator,
    init_scale: float,
    rtol: float,
    atol: float,
    parameter_mode: str,
    constant_parameter: str,
) -> tuple[Array, Array, Array]:
    t_eval = np.arange(n_steps, dtype=np.float64) * dt
    t_end = float(t_eval[-1])
    rhs = RHS[regime.rhs_key]

    phi_raw = np.array(
        [
            parameter_value(t, t_end, regime, parameter_mode, constant_parameter)
            for t in t_eval
        ],
        dtype=np.float32,
    )
    phi = normalize_phi(phi_raw, regime.start, regime.end, regime.boundary).astype(np.float32)

    trajectories = np.empty((n_trajectories, n_steps, 3), dtype=np.float32)

    def ode(t: float, x: Array) -> Array:
        p = parameter_value(t, t_end, regime, parameter_mode, constant_parameter)
        dx = rhs(t, x, p, **regime.params)
        return np.clip(dx, -1e4, 1e4)

    for k in range(n_trajectories):
        x0 = rng.normal(loc=0.0, scale=init_scale, size=3)
        sol = solve_ivp(
            ode,
            (0.0, t_end),
            x0,
            t_eval=t_eval,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            max_step=dt * 5.0,
        )
        if not sol.success or sol.y.shape[1] != n_steps:
            raise RuntimeError(f"Integration failed for {regime.name}: {sol.message}")
        traj = sol.y.T
        if not np.all(np.isfinite(traj)):
            raise RuntimeError(f"Non-finite trajectory generated for {regime.name}")
        trajectories[k] = traj.astype(np.float32)

    return trajectories, phi[:, None], phi_raw[:, None]


def standardize_trajectories(trajectories: Array) -> tuple[Array, Array, Array]:
    mean = trajectories.mean(axis=1, keepdims=True)
    std = trajectories.std(axis=1, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((trajectories - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def first_phi_crossing(phi_series: Array) -> int | None:
    phi_flat = phi_series[:, 0] if phi_series.ndim == 2 else phi_series
    signs = np.signbit(phi_flat)
    candidates = np.flatnonzero(signs[:-1] != signs[1:])
    if candidates.size:
        return int(candidates[0] + 1)
    return None


def sample_window_start(
    phi_series: Array,
    n_steps: int,
    context_len: int,
    seq_len: int,
    data_len: int,
    rng: np.random.Generator,
    sampling_mode: str,
) -> tuple[int, int]:
    if sampling_mode == "random":
        return int(rng.integers(0, n_steps - seq_len + 1)), -1

    if sampling_mode != "crossing":
        raise ValueError(f"Unknown window sampling mode: {sampling_mode}")

    crossing_idx = first_phi_crossing(phi_series)
    if crossing_idx is None:
        raise ValueError(
            "window-sampling='crossing' requires phi to cross 0. "
            "Use --parameter-mode linear or choose random sampling."
        )

    data_offset = seq_len - data_len
    lower = max(0, crossing_idx - seq_len + 1)
    upper = min(
        n_steps - seq_len,
        crossing_idx - context_len,
        crossing_idx - data_offset,
    )
    if lower > upper:
        raise ValueError(
            "No valid crossing-centered window exists for the current lengths. "
            "Need context before the crossing and data containing the crossing; "
            "try smaller --context-len or larger --seq-len."
        )
    return int(rng.integers(lower, upper + 1)), crossing_idx


def build_training_windows(
    trajectories: Array,
    phi: Array,
    phi_raw: Array,
    regime_names: list[str],
    windows_per_trajectory: int,
    context_len: int,
    seq_len: int,
    rng: np.random.Generator,
    window_sampling: str,
) -> dict[str, Array]:
    if seq_len <= context_len:
        raise ValueError("seq_len must be greater than context_len")
    data_len = (seq_len - context_len) * 2
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
    window_regime = []

    out_idx = 0
    for series_idx in range(n_series):
        for _ in range(windows_per_trajectory):
            start, crossing_idx = sample_window_start(
                phi[series_idx],
                n_steps,
                context_len,
                seq_len,
                data_len,
                rng,
                window_sampling,
            )
            end = start + seq_len
            data_start = end - data_len
            context[:, out_idx, :] = trajectories[series_idx, start:start + context_len, :]
            data[:, out_idx, :] = trajectories[series_idx, data_start:end, :]
            phi_context[:, out_idx, :] = phi[series_idx, start:start + context_len, :]
            phi_data[:, out_idx, :] = phi[series_idx, data_start:end, :]
            phi_raw_context[:, out_idx, :] = phi_raw[series_idx, start:start + context_len, :]
            phi_raw_data[:, out_idx, :] = phi_raw[series_idx, data_start:end, :]
            start_indices[out_idx] = start
            crossing_indices[out_idx] = crossing_idx
            window_regime.append(regime_names[series_idx])
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
    parser.add_argument("--output-dir", type=Path, default=Path("generated"))
    parser.add_argument("--n-steps", type=int, default=10000)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--trajectories-per-regime", type=int, default=8)
    parser.add_argument("--windows-per-trajectory", type=int, default=32)
    parser.add_argument("--context-len", type=int, default=500)
    parser.add_argument("--seq-len", type=int, default=550)
    parser.add_argument("--prediction-len", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--init-scale", type=float, default=0.5)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument(
        "--parameter-mode",
        choices=("linear", "constant"),
        default="linear",
        help="Use the original linear parameter drift or hold the bifurcation parameter constant.",
    )
    parser.add_argument(
        "--constant-parameter",
        choices=("start", "boundary", "end", "midpoint"),
        default="start",
        help="Constant parameter value used when --parameter-mode constant is selected.",
    )
    parser.add_argument(
        "--window-sampling",
        choices=("random", "crossing"),
        default="random",
        help=(
            "random samples arbitrary windows; crossing samples windows whose context is before "
            "phi=0 and whose data segment contains the crossing."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    regimes = default_regimes()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_traj = []
    all_phi = []
    all_phi_raw = []
    series_regime_names = []
    regime_index = []

    for regime_idx, regime in enumerate(tqdm(regimes, desc="Simulating regimes")):
        traj, phi_one, phi_raw_one = simulate_regime(
            regime,
            n_steps=args.n_steps,
            dt=args.dt,
            n_trajectories=args.trajectories_per_regime,
            rng=rng,
            init_scale=args.init_scale,
            rtol=args.rtol,
            atol=args.atol,
            parameter_mode=args.parameter_mode,
            constant_parameter=args.constant_parameter,
        )
        traj, mean, std = standardize_trajectories(traj)
        all_traj.append(traj)
        all_phi.append(np.repeat(phi_one[None, :, :], args.trajectories_per_regime, axis=0))
        all_phi_raw.append(np.repeat(phi_raw_one[None, :, :], args.trajectories_per_regime, axis=0))
        series_regime_names.extend([regime.name] * args.trajectories_per_regime)
        regime_index.extend([regime_idx] * args.trajectories_per_regime)
        np.save(args.output_dir / f"{regime.name}_mean.npy", mean)
        np.save(args.output_dir / f"{regime.name}_std.npy", std)

    trajectories = np.concatenate(all_traj, axis=0)
    phi = np.concatenate(all_phi, axis=0)
    phi_raw = np.concatenate(all_phi_raw, axis=0)

    windows = build_training_windows(
        trajectories,
        phi,
        phi_raw,
        series_regime_names,
        windows_per_trajectory=args.windows_per_trajectory,
        context_len=args.context_len,
        seq_len=args.seq_len,
        rng=rng,
        window_sampling=args.window_sampling,
    )
    test = build_test_windows(
        trajectories,
        phi,
        phi_raw,
        test_len=args.context_len + args.prediction_len,
    )

    np.save(args.output_dir / "trajectories.npy", trajectories)
    np.save(args.output_dir / "trajectory_phi.npy", phi)
    np.save(args.output_dir / "trajectory_phi_raw.npy", phi_raw)
    np.save(args.output_dir / "trajectory_regime_index.npy", np.array(regime_index, dtype=np.int64))
    np.save(args.output_dir / "trajectory_regime_names.npy", np.array(series_regime_names))

    for name, value in windows.items():
        np.save(args.output_dir / f"{name}.npy", value)
    for name, value in test.items():
        np.save(args.output_dir / f"{name}.npy", value)

    metadata = {
        "description": "Time-varying bifurcation data with known normalized phi_t for DynaMix external input experiments.",
        "args": vars(args) | {"output_dir": str(args.output_dir)},
        "shapes": {
            "trajectories": list(trajectories.shape),
            "trajectory_phi": list(phi.shape),
            "data": list(windows["data"].shape),
            "context": list(windows["context"].shape),
            "phi": list(windows["phi"].shape),
            "test": list(test["test"].shape),
            "test_phi": list(test["test_phi"].shape),
        },
        "phi_convention": "phi arrays are normalized as (raw_parameter - bifurcation_boundary) / max(abs(start-boundary), abs(end-boundary)); boundary is therefore phi=0.",
        "window_sampling_convention": "crossing mode enforces context indices before the first phi=0 crossing and data indices spanning that crossing; crossing_indices.npy stores the full-trajectory crossing index or -1 for random sampling.",
        "regimes": [asdict(regime) for regime in regimes],
    }
    with open(args.output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved dataset to {args.output_dir.resolve()}")
    for key, shape in metadata["shapes"].items():
        print(f"{key}: {shape}")


if __name__ == "__main__":
    main()
