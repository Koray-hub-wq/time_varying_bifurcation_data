"""Simulate test trajectories for additional dynamic-system candidates.

This is an isolated sandbox version of generate_dynamic_regime_trajectories.py.
It writes the same core array layout:

    trajectories.npy              (S, T, 3)
    trajectory_phi.npy            (S, T, 1)
    trajectory_phi_raw.npy        (S, T, 1)
    trajectory_regime_index.npy   (S,)
    trajectory_regime_names.npy   (S,)
    metadata.json

The parameter boundaries are practical visual transition markers for testing,
not claims of exact analytical bifurcation points.
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
    transition_type: str
    description: str
    rhs_key: str
    params: dict
    init_center: tuple[float, float, float]
    init_scale: tuple[float, float, float]


def linear_parameter(t: float, t_end: float, start: float, end: float) -> float:
    if t_end <= 0.0:
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


def parameter_value(
    t: float, t_end: float, regime: Regime, mode: str, constant_position: str
) -> float:
    if mode == "linear":
        return linear_parameter(t, t_end, regime.start, regime.end)
    if mode == "constant":
        return constant_parameter_value(regime, constant_position)
    raise ValueError(f"Unknown parameter mode: {mode}")


def normalize_phi(phi_raw: Array, start: float, end: float, boundary: float) -> Array:
    scale = max(abs(start - boundary), abs(end - boundary), 1e-8)
    return (phi_raw - boundary) / scale


def chen_rhs(t: float, x: Array, p: float, *, b: float, c: float) -> Array:
    a = p
    return np.array(
        [
            a * (x[1] - x[0]),
            (c - a) * x[0] - x[0] * x[2] + c * x[1],
            x[0] * x[1] - b * x[2],
        ],
        dtype=np.float64,
    )


def thomas_rhs(t: float, x: Array, p: float) -> Array:
    b = p
    return np.array(
        [
            np.sin(x[1]) - b * x[0],
            np.sin(x[2]) - b * x[1],
            np.sin(x[0]) - b * x[2],
        ],
        dtype=np.float64,
    )


def chua_rhs(
    t: float,
    x: Array,
    p: float,
    *,
    beta: float,
    m0: float,
    m1: float,
) -> Array:
    alpha = p
    nonlinear = m1 * x[0] + 0.5 * (m0 - m1) * (
        abs(x[0] + 1.0) - abs(x[0] - 1.0)
    )
    return np.array(
        [
            alpha * (x[1] - x[0] - nonlinear),
            x[0] - x[1] + x[2],
            -beta * x[1],
        ],
        dtype=np.float64,
    )


def genesio_tesi_rhs(
    t: float, x: Array, p: float, *, b: float, c: float
) -> Array:
    a = p
    return np.array(
        [
            x[1],
            x[2],
            -(a * x[0]) - b * x[1] - c * x[2] + x[0] * x[0],
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
    x_rest: float,
) -> Array:
    current = p
    return np.array(
        [
            x[1] - a * x[0] ** 3 + b * x[0] ** 2 - x[2] + current,
            c - d * x[0] ** 2 - x[1],
            r * (s * (x[0] - x_rest) - x[2]),
        ],
        dtype=np.float64,
    )


RHS: dict[str, RhsFn] = {
    "chen": chen_rhs,
    "thomas": thomas_rhs,
    "chua": chua_rhs,
    "genesio_tesi": genesio_tesi_rhs,
    "hindmarsh_rose": hindmarsh_rose_rhs,
}


def default_regimes() -> list[Regime]:
    return [
        Regime(
            name="chen_a_regular_to_chaotic",
            family="Chen",
            parameter="a",
            boundary=30.0,
            start=22.0,
            end=38.0,
            transition_type="regular_to_chaotic",
            description="Chen system sweep toward stronger chaotic Lorenz-like motion.",
            rhs_key="chen",
            params={"b": 3.0, "c": 28.0},
            init_center=(-8.0, 0.0, 24.0),
            init_scale=(2.0, 2.0, 2.0),
        ),
        Regime(
            name="thomas_b_chaotic_to_regular",
            family="Thomas",
            parameter="b",
            boundary=0.208,
            start=0.17,
            end=0.28,
            transition_type="chaotic_to_regular",
            description="Thomas cyclic attractor with increasing damping.",
            rhs_key="thomas",
            params={},
            init_center=(1.0, -1.0, 1.0),
            init_scale=(0.4, 0.4, 0.4),
        ),
        Regime(
            name="chua_alpha_single_to_double_scroll",
            family="Chua",
            parameter="alpha",
            boundary=10.0,
            start=7.5,
            end=15.0,
            transition_type="regular_to_double_scroll",
            description="Chua circuit sweep from simpler oscillation toward double-scroll behavior.",
            rhs_key="chua",
            params={"beta": 14.286, "m0": -1.143, "m1": -0.714},
            init_center=(0.2, 0.0, 0.0),
            init_scale=(0.2, 0.2, 0.2),
        ),
        Regime(
            name="genesio_a_regular_to_chaotic",
            family="GenesioTesi",
            parameter="a",
            boundary=5.2,
            start=3.5,
            end=6.5,
            transition_type="regular_to_chaotic",
            description="Genesio-Tesi jerk system with increasing nonlinear folding.",
            rhs_key="genesio_tesi",
            params={"b": 2.92, "c": 1.2},
            init_center=(0.2, 0.0, 0.0),
            init_scale=(0.2, 0.2, 0.2),
        ),
        Regime(
            name="hindmarsh_rose_I_spiking_to_bursting",
            family="HindmarshRose",
            parameter="I",
            boundary=2.7,
            start=1.6,
            end=3.4,
            transition_type="spiking_to_bursting",
            description="Hindmarsh-Rose neuron sweep from lower-current oscillation toward bursting.",
            rhs_key="hindmarsh_rose",
            params={
                "a": 1.0,
                "b": 3.0,
                "c": 1.0,
                "d": 5.0,
                "r": 0.006,
                "s": 4.0,
                "x_rest": -1.6,
            },
            init_center=(-1.2, -8.0, 2.0),
            init_scale=(0.2, 0.5, 0.2),
        ),
    ]


def standardize_trajectories(trajectories: Array) -> tuple[Array, Array, Array]:
    mean = trajectories.mean(axis=1, keepdims=True)
    std = trajectories.std(axis=1, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (
        ((trajectories - mean) / std).astype(np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def simulate_regime(
    regime: Regime,
    n_steps: int,
    dt: float,
    n_trajectories: int,
    rng: np.random.Generator,
    rtol: float,
    atol: float,
    max_step_multiplier: float,
    burn_in_steps: int,
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
    phi = normalize_phi(phi_raw, regime.start, regime.end, regime.boundary).astype(
        np.float32
    )
    trajectories = np.empty((n_trajectories, n_steps, 3), dtype=np.float32)

    def ode_main(t: float, x: Array) -> Array:
        p = parameter_value(t, t_end, regime, parameter_mode, constant_parameter)
        dx = rhs(t, x, p, **regime.params)
        return np.clip(dx, -1e4, 1e4)

    def ode_burn_in(t: float, x: Array) -> Array:
        dx = rhs(t, x, regime.start, **regime.params)
        return np.clip(dx, -1e4, 1e4)

    for k in range(n_trajectories):
        x0 = rng.normal(
            loc=np.array(regime.init_center), scale=np.array(regime.init_scale)
        )

        if burn_in_steps > 0:
            burn = solve_ivp(
                ode_burn_in,
                (0.0, burn_in_steps * dt),
                x0,
                method="DOP853",
                rtol=rtol,
                atol=atol,
                max_step=dt * max_step_multiplier,
            )
            if not burn.success:
                raise RuntimeError(f"Burn-in failed for {regime.name}: {burn.message}")
            x0 = burn.y[:, -1]

        sol = solve_ivp(
            ode_main,
            (0.0, t_end),
            x0,
            t_eval=t_eval,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            max_step=dt * max_step_multiplier,
        )
        if not sol.success or sol.y.shape[1] != n_steps:
            raise RuntimeError(f"Integration failed for {regime.name}: {sol.message}")
        if not np.all(np.isfinite(sol.y)):
            raise RuntimeError(f"Non-finite trajectory generated for {regime.name}")

        trajectories[k] = sol.y.T.astype(np.float32)

    return trajectories, phi[:, None], phi_raw[:, None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "test_newds_output",
    )
    parser.add_argument("--n-steps", type=int, default=30000)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--trajectories-per-regime", type=int, default=2)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--max-step-multiplier", type=float, default=5.0)
    parser.add_argument("--burn-in-steps", type=int, default=5000)
    parser.add_argument(
        "--parameter-mode",
        choices=("linear", "constant"),
        default="linear",
    )
    parser.add_argument(
        "--constant-parameter",
        choices=("start", "boundary", "end", "midpoint"),
        default="start",
    )
    parser.add_argument(
        "--regime-filter",
        nargs="*",
        default=None,
        help="Optional list of regime names to simulate. Defaults to all regimes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    regimes = default_regimes()

    if args.regime_filter:
        requested = set(args.regime_filter)
        regimes = [regime for regime in regimes if regime.name in requested]
        missing = requested.difference(regime.name for regime in regimes)
        if missing:
            raise ValueError(f"Unknown regime names: {sorted(missing)}")
    if not regimes:
        raise ValueError("No regimes selected.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_traj = []
    all_phi = []
    all_phi_raw = []
    series_regime_names = []
    regime_index = []

    for regime_idx, regime in enumerate(tqdm(regimes, desc="Simulating new systems")):
        traj, phi_one, phi_raw_one = simulate_regime(
            regime=regime,
            n_steps=args.n_steps,
            dt=args.dt,
            n_trajectories=args.trajectories_per_regime,
            rng=rng,
            rtol=args.rtol,
            atol=args.atol,
            max_step_multiplier=args.max_step_multiplier,
            burn_in_steps=args.burn_in_steps,
            parameter_mode=args.parameter_mode,
            constant_parameter=args.constant_parameter,
        )
        traj, mean, std = standardize_trajectories(traj)
        all_traj.append(traj)
        all_phi.append(
            np.repeat(phi_one[None, :, :], args.trajectories_per_regime, axis=0)
        )
        all_phi_raw.append(
            np.repeat(phi_raw_one[None, :, :], args.trajectories_per_regime, axis=0)
        )
        series_regime_names.extend([regime.name] * args.trajectories_per_regime)
        regime_index.extend([regime_idx] * args.trajectories_per_regime)
        np.save(args.output_dir / f"{regime.name}_mean.npy", mean)
        np.save(args.output_dir / f"{regime.name}_std.npy", std)

    trajectories = np.concatenate(all_traj, axis=0)
    phi = np.concatenate(all_phi, axis=0)
    phi_raw = np.concatenate(all_phi_raw, axis=0)

    np.save(args.output_dir / "trajectories.npy", trajectories)
    np.save(args.output_dir / "trajectory_phi.npy", phi)
    np.save(args.output_dir / "trajectory_phi_raw.npy", phi_raw)
    np.save(
        args.output_dir / "trajectory_regime_index.npy",
        np.array(regime_index, dtype=np.int64),
    )
    np.save(
        args.output_dir / "trajectory_regime_names.npy", np.array(series_regime_names)
    )

    metadata = {
        "description": "Test trajectories for additional dynamic-system candidates.",
        "args": vars(args) | {"output_dir": str(args.output_dir)},
        "shapes": {
            "trajectories": list(trajectories.shape),
            "trajectory_phi": list(phi.shape),
            "trajectory_phi_raw": list(phi_raw.shape),
        },
        "regimes": [asdict(regime) for regime in regimes],
        "phi_convention": (
            "phi arrays are normalized as (raw_parameter - transition_boundary) / "
            "max(abs(start-boundary), abs(end-boundary)); boundary is therefore phi=0."
        ),
    }
    with open(args.output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved trajectories to {args.output_dir.resolve()}")
    for key, shape in metadata["shapes"].items():
        print(f"{key}: {shape}")


if __name__ == "__main__":
    main()
