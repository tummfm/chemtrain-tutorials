"""Loading, analysis, and visualization helpers for interface pinning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import re

import tomllib

import numpy as np

from units import EV_TO_KJ_PER_MOL


_COLUMNS = ("timestep", "lx", "ly", "lz", "pxx", "q")
_COLORS = {"reference": "black", "equilibration": "0.55", "production": "tab:blue"}


def _style_axes(axes) -> None:
    """Apply the compact plotting style used by the first tutorial."""
    for axis in np.atleast_1d(axes).flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.25)


@dataclass(frozen=True)
class ObservableSeries:
    """The six-column table written by the interface-pinning LAMMPS input."""

    label: str
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.values.ndim != 2 or self.values.shape[1] != len(_COLUMNS):
            raise ValueError(
                f"{self.label} must contain six columns: {', '.join(_COLUMNS)}"
            )
        if not np.isfinite(self.values).all():
            raise ValueError(f"{self.label} contains non-finite observables")

    @property
    def n_samples(self) -> int:
        return self.values.shape[0]

    @property
    def timestep(self) -> np.ndarray:
        return self.values[:, 0]

    @property
    def lx(self) -> np.ndarray:
        return self.values[:, 1]

    @property
    def ly(self) -> np.ndarray:
        return self.values[:, 2]

    @property
    def lz(self) -> np.ndarray:
        return self.values[:, 3]

    @property
    def pxx(self) -> np.ndarray:
        return self.values[:, 4]

    @property
    def q(self) -> np.ndarray:
        return self.values[:, 5]

    def time_ps(self, timestep_ps: float) -> np.ndarray:
        return self.timestep * timestep_ps


def load_observables(path: str | Path) -> ObservableSeries:
    """Load a LAMMPS ``fix ave/time`` interface-pinning table."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.atleast_2d(np.loadtxt(path, skiprows=2))
    return ObservableSeries(path.stem.replace("_", " "), values)


def load_pinning_parameters(path: str | Path) -> dict[str, float | int]:
    """Load and validate the ``[interface_pinning]`` calibration table."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        parameters = tomllib.load(handle).get("interface_pinning")
    if not isinstance(parameters, dict):
        raise ValueError(f"{path} does not contain an [interface_pinning] table")
    required = {
        "q_solid",
        "q_liquid",
        "center",
        "production_center",
        "kappa",
        "temperature",
        "pressure_gpa",
    }
    missing = required.difference(parameters)
    if missing:
        raise ValueError(f"{path} is missing pinning parameters: {', '.join(sorted(missing))}")
    return parameters


def load_phase_thermodynamics(path: str | Path) -> dict[str, dict[str, float]]:
    """Load the unbiased solid/liquid thermodynamic calibration averages.

    Energies are stored in LAMMPS ``metal`` units (eV per atom), volumes in
    Å³ per atom, and the normal pressure in bar.  The preparation run fixes
    the transverse box vectors and barostats the interface-normal x direction.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        contents = tomllib.load(handle)

    required = {
        "metadata": {"temperature_k", "pressure_gpa"},
        "solid": {
            "potential_energy_ev_per_atom",
            "total_energy_ev_per_atom",
            "enthalpy_ev_per_atom",
            "volume_angstrom3_per_atom",
            "temperature_k",
            "normal_pressure_bar",
        },
        "liquid": {
            "potential_energy_ev_per_atom",
            "total_energy_ev_per_atom",
            "enthalpy_ev_per_atom",
            "volume_angstrom3_per_atom",
            "temperature_k",
            "normal_pressure_bar",
        },
    }
    parsed: dict[str, dict[str, float]] = {}
    for section, fields in required.items():
        values = contents.get(section)
        if not isinstance(values, dict):
            raise ValueError(f"{path} does not contain a [{section}] table")
        missing = fields.difference(values)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"{path} [{section}] is missing: {missing_text}")
        parsed[section] = {field: float(values[field]) for field in fields}
        if not all(np.isfinite(value) for value in parsed[section].values()):
            raise ValueError(f"{path} [{section}] contains a non-finite value")
    return parsed


def lammps_atom_count(path: str | Path) -> int:
    """Read the atom count from a LAMMPS data-file header."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    pattern = re.compile(r"^\s*(\d+)\s+atoms\s*$")
    with path.open() as handle:
        for line in handle:
            match = pattern.match(line)
            if match:
                return int(match.group(1))
    raise ValueError(f"could not find an atom count in {path}")


def pinning_free_energy_difference(
    pinning: Mapping[str, float], production: ObservableSeries, n_atoms: int
) -> dict[str, float]:
    """Estimate the BCC--liquid free-energy difference from interface pinning.

    The LAMMPS ``metal`` bias uses eV, and the interface-pinning relation is
    ``Δμ = -κ (⟨Q⟩ - a) ΔQ / N``.  Returned free energies are in kJ/mol.
    """
    if n_atoms < 1:
        raise ValueError("n_atoms must be positive")
    delta_q = float(pinning.get("delta_q", pinning["q_solid"] - pinning["q_liquid"]))
    q_mean = float(production.q.mean())
    q_block_error = float(summary(production)["q_block_error"])
    kappa = float(pinning["kappa"])
    center = float(pinning["center"])
    delta_f_eV = -kappa * (q_mean - center) * delta_q
    delta_f_error_eV = abs(kappa * delta_q) * q_block_error
    return {
        "free_energy_difference_kj_per_mol": delta_f_eV * EV_TO_KJ_PER_MOL,
        "free_energy_difference_error_kj_per_mol": delta_f_error_eV * EV_TO_KJ_PER_MOL,
        "chemical_potential_difference_kj_per_mol": delta_f_eV * EV_TO_KJ_PER_MOL / n_atoms,
        "chemical_potential_difference_error_kj_per_mol": (
            delta_f_error_eV * EV_TO_KJ_PER_MOL / n_atoms
        ),
    }


def melting_temperature_estimate(
    free_energy: Mapping[str, float], thermodynamics: Mapping[str, Mapping[str, float]]
) -> dict[str, float]:
    """Convert an IP chemical-potential result into a one-step melting estimate.

    The convention is ``Δ = solid - liquid``.  The entropy difference is
    ``Δs = (Δh - Δμ) / T`` and the interface-pinning update is
    ``T_m = T + Δμ / Δs``.  The reported uncertainty propagates only the
    interface-pinning statistical error.
    """
    metadata = thermodynamics["metadata"]
    solid = thermodynamics["solid"]
    liquid = thermodynamics["liquid"]
    temperature = float(metadata["temperature_k"])
    if temperature <= 0.0:
        raise ValueError("thermodynamic calibration temperature must be positive")

    chemical_potential = float(
        free_energy["chemical_potential_difference_kj_per_mol"]
    )
    chemical_potential_error = float(
        free_energy["chemical_potential_difference_error_kj_per_mol"]
    )
    enthalpy_difference = (
        float(solid["enthalpy_ev_per_atom"])
        - float(liquid["enthalpy_ev_per_atom"])
    ) * EV_TO_KJ_PER_MOL
    entropy_difference = (enthalpy_difference - chemical_potential) / temperature
    if not np.isfinite(entropy_difference) or np.isclose(entropy_difference, 0.0):
        raise ValueError("solid-liquid entropy difference is zero or non-finite")

    melting_temperature = temperature + chemical_potential / entropy_difference
    temperature_error_from_pinning = abs(
        temperature * enthalpy_difference / (enthalpy_difference - chemical_potential) ** 2
    ) * chemical_potential_error
    return {
        "reference_temperature_k": temperature,
        "chemical_potential_difference_kj_per_mol": chemical_potential,
        "chemical_potential_difference_error_kj_per_mol": chemical_potential_error,
        "enthalpy_difference_kj_per_mol": enthalpy_difference,
        "entropy_difference_kj_per_mol_k": entropy_difference,
        "melting_temperature_k": melting_temperature,
        "melting_temperature_error_from_pinning_k": temperature_error_from_pinning,
    }


def load_runs(results_dir: str | Path) -> tuple[ObservableSeries, ObservableSeries]:
    """Load the equilibration and production observable tables."""
    results_dir = Path(results_dir)
    return (
        load_observables(results_dir / "interface_equilibration.txt"),
        load_observables(results_dir / "interface.txt"),
    )


def block_average(
    x: np.ndarray, y: np.ndarray, block_size: int = 25
) -> tuple[np.ndarray, np.ndarray]:
    """Return contiguous block means, dropping only an incomplete final block."""
    if block_size < 1:
        raise ValueError("block_size must be positive")
    n_blocks = len(y) // block_size
    if n_blocks == 0:
        return np.asarray(x), np.asarray(y)
    stop = n_blocks * block_size
    return (
        np.asarray(x[:stop]).reshape(n_blocks, block_size).mean(axis=1),
        np.asarray(y[:stop]).reshape(n_blocks, block_size).mean(axis=1),
    )


def summary(production: ObservableSeries, block_size: int = 25) -> dict[str, float | int]:
    """Compute compact global-order statistics and a block standard error."""
    _, q_blocks = block_average(production.timestep, production.q, block_size)
    q_block_error = (
        float(q_blocks.std(ddof=1) / np.sqrt(q_blocks.size)) if q_blocks.size > 1 else float("nan")
    )
    return {
        "samples": production.n_samples,
        "blocks": q_blocks.size,
        "q_mean": float(production.q.mean()),
        "q_std": float(production.q.std(ddof=1)),
        "q_block_error": q_block_error,
    }


def plot_evaluation(
    equilibration: ObservableSeries,
    production: ObservableSeries,
    pinning: Mapping[str, float],
    *,
    timestep_ps: float,
    block_size: int = 25,
):
    """Plot the global-order trace and its sampled distribution."""
    import matplotlib.pyplot as plt

    q_solid = float(pinning["q_solid"])
    q_liquid = float(pinning["q_liquid"])
    effective_center = float(pinning.get("effective_center", pinning["center"]))

    t_eq = equilibration.time_ps(timestep_ps)
    t_prod = production.time_ps(timestep_ps)
    t_block, q_block = block_average(t_prod, production.q, block_size)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    _style_axes(axes)

    axis = axes[0]
    axis.plot(
        t_eq,
        equilibration.q,
        color=_COLORS["equilibration"],
        ls="--",
        lw=1.2,
        label="Equilibration",
    )
    axis.plot(
        t_prod,
        production.q,
        color=_COLORS["production"],
        alpha=0.28,
        lw=0.8,
        label="Production",
    )
    axis.plot(
        t_block,
        q_block,
        color=_COLORS["production"],
        lw=2.2,
        label=f"{block_size}-sample mean",
    )
    for value, label, color in (
        (q_liquid, "Liquid reference", "tab:orange"),
        (q_solid, "Solid reference", "tab:green"),
        (effective_center, "Pinning center", _COLORS["reference"]),
    ):
        axis.axhline(value, color=color, lw=1.2, ls="--", label=label)
    axis.set(
        xlabel="Time [ps]",
        ylabel="Global order parameter Q",
        title="Interface-pinning coordinate",
    )
    axis.legend(frameon=False, fontsize=8, ncol=2)

    axis = axes[1]
    axis.hist(production.q, bins="auto", density=True, color=_COLORS["production"], alpha=0.75)
    for value, label, color in (
        (q_liquid, "Liquid reference", "tab:orange"),
        (q_solid, "Solid reference", "tab:green"),
        (effective_center, "Pinning center", _COLORS["reference"]),
        (float(production.q.mean()), "Production mean", _COLORS["production"]),
    ):
        axis.axvline(value, color=color, lw=1.4, ls="--", label=label)
    axis.set(
        xlabel="Global order parameter Q",
        ylabel="Probability density",
        title="Sampled order distribution",
    )
    axis.legend(frameon=False, fontsize=8)
    return fig


def save_figure(fig, stem: str | Path) -> tuple[Path, Path]:
    """Save a figure in portable raster and vector forms."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    svg = stem.with_suffix(".svg")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    return png, svg


def load_trajectory_frames(
    path: str | Path, frame_indices: Sequence[int] | None = None
):
    """Read selected LAMMPS dump frames as titanium ASE atoms objects."""
    from ase.io import read

    frames = read(Path(path), index=":", format="lammps-dump-text")
    if not frames:
        raise ValueError(f"No frames found in {path}")
    if frame_indices is None:
        frame_indices = (0, len(frames) // 2, len(frames) - 1)
    selected = []
    for index in dict.fromkeys(frame_indices):
        atoms = frames[index].copy()
        atoms.numbers = np.full(len(atoms), 22, dtype=int)
        selected.append(atoms)
    return selected


def trajectory_view(frames):
    """Plot periodic x-y projections that reveal crystalline and liquid slabs."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(
        1, len(frames), figsize=(5.4 * len(frames), 3.8), layout="constrained"
    )
    axes = np.atleast_1d(axes)
    _style_axes(axes)
    for index, (axis, atoms) in enumerate(zip(axes, frames), start=1):
        scaled_positions = atoms.get_scaled_positions(wrap=True)
        lengths = atoms.cell.lengths()
        if not np.allclose(atoms.cell.array, np.diag(lengths), atol=1.0e-8):
            raise ValueError("the top-down trajectory view requires an orthorhombic box")
        x = scaled_positions[:, 0] * lengths[0]
        y = scaled_positions[:, 1] * lengths[1]
        axis.scatter(
            x,
            y,
            color=_COLORS["production"],
            s=6.0,
            alpha=0.85,
            linewidths=0,
            rasterized=True,
        )
        axis.add_patch(Rectangle((0.0, 0.0), lengths[0], lengths[1], fill=False, lw=0.9, color="0.2"))
        axis.set(
            xlim=(0.0, lengths[0]),
            ylim=(0.0, lengths[1]),
            aspect="equal",
            xlabel="x [Å]",
            ylabel="y [Å]",
            title=f"Frame {index}",
        )
    fig.suptitle("Orthorhombic top-down interface view (projection along z)")
    return fig
