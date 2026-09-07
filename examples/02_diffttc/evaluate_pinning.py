#!/usr/bin/env python
"""Run and evaluate interface pinning with the DiffTRe-exported MACE model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

from ip_helpers import (
    lammps_atom_count,
    load_pinning_parameters,
    load_runs,
    pinning_free_energy_difference,
    plot_evaluation,
    save_figure,
    summary,
)


TUTORIAL = Path(__file__).resolve().parent
TIMESTEP = 0.004
OUTPUT_FILES = (
    "interface_equilibration.txt",
    "equilibration.lammpstrj",
    "interface.txt",
    "interface.lammpstrj",
    "interface_final.lmpdat",
    "interface_evaluation.png",
    "interface_evaluation.svg",
    "evaluation.json",
)


def arguments() -> argparse.Namespace:
    """Read the final-evaluation inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0", help="CUDA device visible to LAMMPS")
    parser.add_argument(
        "--lmp",
        type=Path,
        default=TUTORIAL / "../../software/bin/lmp",
        help="LAMMPS executable",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=TUTORIAL / "training/output/mace_difftre.ptb",
        help="DiffTRe-exported MACE deployment bundle",
    )
    parser.add_argument(
        "--order-model",
        type=Path,
        default=TUTORIAL / "steinhardt_order_oeq.ptb",
        help="Steinhardt-order deployment bundle",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=TUTORIAL / "input/interface_start.lmpdat",
        help="Prepared solid-liquid interface data file",
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=TUTORIAL / "input/interface_pinning.toml",
        help="Calibrated interface-pinning parameters",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TUTORIAL / "training/output/final",
        help="Directory for the independent final evaluation",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow LAMMPS to replace an existing final evaluation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the LAMMPS command without running it",
    )
    return parser.parse_args()


def existing_outputs(output: Path) -> list[Path]:
    """Return final-evaluation artifacts already present in ``output``."""
    return [output / name for name in OUTPUT_FILES if (output / name).exists()]


def require_file(path: Path, label: str) -> Path:
    """Resolve and validate a required input file."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def lammps_command(args: argparse.Namespace, pinning: dict[str, float | int]) -> list[str]:
    """Build the production command using the calibrated interface restraint."""
    return [
        str(args.lmp),
        "-in",
        str(TUTORIAL / "pinning_production.lmp"),
        "-k",
        "on",
        "g",
        "1",
        "-sf",
        "kk",
        "-pk",
        "kokkos",
        "newton",
        "on",
        "neigh",
        "half",
        "-var",
        "input_data",
        str(args.input),
        "-var",
        "output_directory",
        str(args.output),
        "-var",
        "pinning_center",
        str(pinning["center"]),
        "-var",
        "kappa_value",
        str(pinning["kappa"]),
        "-var",
        "temperature",
        str(pinning["temperature"]),
        "-var",
        "pressure_gpa",
        str(pinning["pressure_gpa"]),
        "-var",
        "mace_model",
        str(args.model),
        "-var",
        "order_model",
        str(args.order_model),
    ]


def main() -> None:
    """Run final interface pinning and save the free-energy estimate."""
    args = arguments()
    args.output = args.output.resolve()
    outputs = existing_outputs(args.output)
    if outputs and not args.overwrite:
        names = ", ".join(path.name for path in outputs)
        print(f"Final IP evaluation already exists; refusing to overwrite: {names}")
        return

    args.lmp = require_file(args.lmp, "LAMMPS executable")
    args.model = require_file(args.model, "DiffTRe MACE model")
    args.order_model = require_file(args.order_model, "order model")
    args.input = require_file(args.input, "interface input")
    args.parameters = require_file(args.parameters, "pinning parameters")

    pinning = load_pinning_parameters(args.parameters)
    command = lammps_command(args, pinning)
    print(shlex.join(command))
    if args.dry_run:
        return

    args.output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    environment["JCN_PJRT_PATH"] = str(
        (TUTORIAL / "../../software/lib/pjrt").resolve()
    )
    environment["NVIDIA_TF32_OVERRIDE"] = "0"
    subprocess.run(command, cwd=TUTORIAL, env=environment, check=True)

    # The input file is unchanged, so this evaluates the updated model alone.
    equilibration, production = load_runs(args.output)
    free_energy = pinning_free_energy_difference(
        pinning,
        production,
        lammps_atom_count(args.input),
    )
    figure = plot_evaluation(
        equilibration, production, pinning, timestep_ps=TIMESTEP
    )
    figure_paths = save_figure(figure, args.output / "interface_evaluation")
    production_summary = summary(production)
    q_target = float(pinning["center"])
    q_mean = float(production_summary["q_mean"])
    q_drift = q_mean - q_target
    q_drift_error = float(production_summary["q_block_error"])
    delta_q = float(pinning["delta_q"])
    solid_fraction = (q_mean - float(pinning["q_liquid"])) / delta_q
    result = {
        "model": str(args.model),
        "temperature_k": float(pinning["temperature"]),
        "pressure_gpa": float(pinning["pressure_gpa"]),
        "q_target": q_target,
        "q_mean": q_mean,
        "q_drift": q_drift,
        "q_drift_error": q_drift_error,
        "solid_fraction": solid_fraction,
        "solid_fraction_drift": q_drift / delta_q,
        "evaluation_figures": [str(path) for path in figure_paths],
        **free_energy,
    }
    result_path = args.output / "evaluation.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved {result_path}")


if __name__ == "__main__":
    main()
