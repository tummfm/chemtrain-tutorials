"""Data and evaluation helpers for the titanium DiffTRe notebook.

The LAMMPS phase boxes use ``metal`` units.  This module converts them at the
boundary to the JAX-MD convention used by Chemtrain: nm, kJ/mol, ps, and g/mol.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from units import ANGSTROM_TO_NM, EV_TO_KJ_PER_MOL, GPA_TO_KJ_PER_MOL_NM3


@dataclass(frozen=True)
class PhaseBox:
    """A single periodic titanium configuration in JAX-MD units.

    ``positions`` and ``box`` are in nm; ``masses`` are in g/mol.
    """

    name: str
    path: Path
    positions: np.ndarray
    box: np.ndarray
    masses: np.ndarray

    @property
    def n_atoms(self) -> int:
        return len(self.positions)

    @property
    def volume_nm3(self) -> float:
        return float(abs(np.linalg.det(self.box)))

    @property
    def number_density_nm3(self) -> float:
        return self.n_atoms / self.volume_nm3

    @property
    def fractional_positions(self) -> np.ndarray:
        return self.positions @ np.linalg.inv(self.box)

    def as_statepoint(
        self, temperature: float, pressure_gpa: float
    ) -> dict[str, np.ndarray | float]:
        """Return a JAX-MD statepoint with pressure in kJ/(mol nm^3)."""
        return {
            "R": self.fractional_positions,
            "box": self.box,
            "mass": self.masses,
            "species": np.full(self.n_atoms, 22, dtype=np.int32),
            "temperature": float(temperature),
            "pressure": float(pressure_gpa) * GPA_TO_KJ_PER_MOL_NM3,
        }


def dump_timestep(path: str | Path) -> int:
    """Read the first timestep from a LAMMPS custom dump header."""
    path = Path(path)
    with path.open() as handle:
        marker = handle.readline().strip()
        value = handle.readline().strip()
    if marker != "ITEM: TIMESTEP":
        raise ValueError(f"{path} does not start with a LAMMPS timestep header")
    return int(value)


def load_phase_box(path: str | Path, name: str) -> PhaseBox:
    """Load a LAMMPS-metal data file and convert its geometry from Å to nm."""
    from ase.io import read

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    atoms = read(path, format="lammps-data", atom_style="atomic")
    positions = np.asarray(atoms.get_positions(), dtype=float) * ANGSTROM_TO_NM
    box = np.asarray(atoms.cell.array, dtype=float) * ANGSTROM_TO_NM
    if positions.ndim != 2 or positions.shape[1] != 3 or abs(np.linalg.det(box)) == 0.0:
        raise ValueError(f"{path} is not a three-dimensional periodic phase box")
    return PhaseBox(name, path, positions, box, np.full(len(atoms), 47.867))


def phase_box_paths(training_dir: str | Path) -> dict[str, Path]:
    """Return the standardized paths written by ``prepare_phase_boxes.lmp``."""
    input_dir = Path(training_dir) / "input"
    return {"bcc": input_dir / "bcc.lmpdat", "liquid": input_dir / "liquid.lmpdat"}


def missing_phase_boxes(training_dir: str | Path) -> list[Path]:
    """List phase inputs that still need the explicit LAMMPS preparation run."""
    return [path for path in phase_box_paths(training_dir).values() if not path.is_file()]


def load_phase_boxes(training_dir: str | Path) -> Mapping[str, PhaseBox]:
    """Load the BCC/liquid pair after the phase-box preparation has run."""
    paths = phase_box_paths(training_dir)
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Run prepare_phase_boxes.lmp first:\n{formatted}")
    return {name: load_phase_box(path, name) for name, path in paths.items()}


def phase_box_summary(phase_boxes: Mapping[str, PhaseBox]) -> dict[str, dict[str, float | int]]:
    """Return notebook-friendly size and density diagnostics."""
    return {
        name: {
            "atoms": box.n_atoms,
            "volume_nm3": box.volume_nm3,
            "number_density_nm^-3": box.number_density_nm3,
        }
        for name, box in phase_boxes.items()
    }


def stack_phase_statepoints(
    phase_boxes: Mapping[str, PhaseBox], temperature: float, pressure_gpa: float
) -> dict[str, np.ndarray]:
    """Stack the BCC/liquid pair into the fixed-shape arrays DiffTRe expects."""
    ordered = (phase_boxes["bcc"], phase_boxes["liquid"])
    atom_counts = {box.n_atoms for box in ordered}
    if len(atom_counts) != 1:
        raise ValueError("BCC and liquid boxes must contain the same number of atoms")
    states = [box.as_statepoint(temperature, pressure_gpa) for box in ordered]
    return {
        "R": np.stack([state["R"] for state in states]),
        "box": np.stack([state["box"] for state in states]),
        "mass": np.stack([state["mass"] for state in states]),
        "species": np.stack([state["species"] for state in states]),
        "temperature": np.asarray([state["temperature"] for state in states]),
        "pressure": np.asarray([state["pressure"] for state in states]),
    }


def training_history(
    trainer, reference_delta_mu: float, target_delta_mu: float, n_atoms: int
) -> dict[str, np.ndarray]:
    """Extract loss and BCC--liquid chemical-potential diagnostics."""
    if n_atoms < 1:
        raise ValueError("n_atoms must be positive")
    predictions = trainer.predictions
    if not {0, 1}.issubset(predictions):
        raise ValueError("expected BCC and liquid predictions at statepoints 0 and 1")
    epochs = sorted(set(predictions[0]).intersection(predictions[1]))
    if not epochs:
        raise ValueError("the trainer has no paired prediction history yet")

    def scalar(statepoint: int, epoch: int, key: str) -> float:
        value = np.asarray(predictions[statepoint][epoch][key])
        if value.size != 1:
            raise ValueError(f"{key} at statepoint {statepoint}, epoch {epoch} is not scalar")
        return float(value.reshape(()))

    epoch_losses = np.asarray(trainer.epoch_losses, dtype=float)
    correction = np.asarray(
        [
            scalar(0, epoch, "free_energy") - scalar(1, epoch, "free_energy")
            for epoch in epochs
        ]
    )
    return {
        "epoch": np.asarray(epochs, dtype=int),
        "loss_epoch": np.arange(epoch_losses.size, dtype=int),
        "loss": epoch_losses,
        "chemical_potential_difference": reference_delta_mu + correction / n_atoms,
        "chemical_potential_target": np.full(len(epochs), target_delta_mu),
    }


def plot_training_history(history: Mapping[str, np.ndarray]):
    """Plot the DiffTRe loss and BCC--liquid chemical-potential difference."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), layout="constrained")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.25)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    axes[0].semilogy(history["loss_epoch"], history["loss"], color="tab:blue", lw=1.8)
    axes[0].set(xlabel="Epoch", ylabel="Weighted loss", title="Training loss")

    axes[1].plot(
        history["epoch"],
        history["chemical_potential_difference"],
        color="tab:blue",
        lw=1.8,
        label="DiffTRe",
    )
    axes[1].plot(
        history["epoch"],
        history["chemical_potential_target"],
        color="black",
        lw=1.6,
        ls="--",
        label="Target",
    )
    axes[1].set(
        xlabel="Epoch",
        ylabel=r"$\Delta \mu_{BCC} - \Delta \mu_{liquid}$ [kJ/mol/atom]",
        title="Chemical-potential difference",
    )
    axes[1].legend(frameon=False)
    return fig


def plot_rdf_comparison(
    radii: np.ndarray,
    reference: np.ndarray,
    prediction: np.ndarray,
    labels: tuple[str, ...] = ("BCC", "Liquid"),
):
    """Compare the zero-weight structural references with the final model."""
    radii = np.asarray(radii)
    reference = np.asarray(reference)
    prediction = np.asarray(prediction)
    if reference.shape != prediction.shape or reference.ndim != 2:
        raise ValueError("RDF reference and prediction must have shape (state, bin)")
    if reference.shape[1] != radii.size or reference.shape[0] != len(labels):
        raise ValueError("RDF bins and state labels must match the RDF arrays")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(labels), figsize=(10.0, 3.8), layout="constrained")
    for index, axis in enumerate(np.atleast_1d(axes)):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.25)
        axis.plot(radii, reference[index], color="0.55", ls="--", lw=1.6, label="Foundation")
        axis.plot(radii, prediction[index], color="tab:blue", lw=1.8, label="DiffTRe")
        axis.set(xlabel="Distance [nm]", ylabel="RDF", title=labels[index])
        axis.legend(frameon=False)
    return fig


def export_mace_model(
    model_config,
    torch_model,
    params,
    equivariance_config,
    output_path: str | Path,
    head: str = "matpes_r2scan",
) -> Path:
    """Export trained MACE parameters as a communication-enabled metal bundle."""
    import functools

    from chemtrain.compose import mace_jax as mace_jax_compose
    from chemtrain.deploy import exporter, graphs
    from jax_md import space

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    displacement, _ = space.free()
    variables, apply_fn = mace_jax_compose.mace_jax_neighborlist_from_torch(
        model_config,
        torch_model,
        displacement,
        max_edge_multiplier=None,
        per_particle=True,
        species_mapping=mace_jax_compose.AtomicNumberMapping(max_number=100),
        scale_pot=1.0,
        scale_pos=1.0,
        equivariance_config=equivariance_config,
        head=head,
    )
    variables = {**variables, "params": params}

    class MaceExporter(exporter.Exporter):
        r_cutoff = model_config["r_max"]
        graph_type = graphs.SimpleSparseNeighborList
        nbr_order = [
            model_config["num_interactions"],
            2 * model_config["num_interactions"],
        ]
        unit_style = "metal"

        def __init__(self):
            self.model = functools.partial(apply_fn, variables)
            super().__init__()

        def energy_fn(self, position, particle_data, graph, comm=None):
            return self.model(
                position,
                graph.to_neighborlist(),
                species=particle_data["species"] + 1,
                comm=comm,
            )

    MaceExporter.displacement = displacement
    model = MaceExporter()
    model.export(
        communication=True,
        custom_calls=exporter.OPENEQUIVARIANCE_CUSTOM_CALLS,
    )
    model.save(output_path)
    return output_path


def save_difftre_results(
    trainer,
    model_config,
    torch_model,
    base_params,
    equivariance_config,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Save parameter deltas and export their sum with the foundation model."""
    from jax import tree_util

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params_path = output_dir / "final_params.pkl"
    trainer.save_energy_params(params_path, best=False, save_format=".pkl")
    full_params = tree_util.tree_map(lambda base, delta: base + delta, base_params, trainer.params)
    model_path = export_mace_model(
        model_config,
        torch_model,
        full_params,
        equivariance_config,
        output_dir / "mace_difftre.ptb",
    )
    return params_path, model_path
