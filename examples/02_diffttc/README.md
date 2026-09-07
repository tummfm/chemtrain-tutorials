# Interface pinning and DiffTRe

This example refines a titanium MACE potential toward solid--liquid
coexistence. It first prepares an interface and phase boxes, then trains a
paired BCC/liquid DiffTRe model, and finally evaluates the exported model with
an independent 50/50 interface-pinning run.

## Quickstart

```bash
cd examples/02_diffttc
jupyter lab
```

1. `setup.ipynb` exports the foundation MACE model
   (`mace_mp_mh-1_matpes_r2scan_oeq.ptb`) and the Steinhardt-order model
   (`steinhardt_order_oeq.ptb`). It runs the foundation-model interface-pinning
   calculation and prepares the training phase boxes.

   It writes `input/interface_start.lmpdat`,
   `input/interface_pinning.toml`, and
   `input/phase_thermodynamics.toml`. The foundation pinning traces and plot
   are written to `results/`.

2. `difftre.ipynb` reads the BCC and liquid boxes in `training/input/`, trains
   the paired DiffTRe system, and writes its trainer state, figures, and log to
   `training/output/`.

   `training/output/final_params.pkl` contains the learned parameter change.
   `training/output/mace_difftre.ptb` combines that change with the foundation
   MACE parameters and is the model used for the final check.

3. `evaluate_difftre.ipynb` runs the final evaluation. Select the CUDA device
   and whether to replace an existing run in its configuration cell. The
   notebook calls `evaluate_pinning.py`, then plots the resulting
   interface-pinning trace.

   The final run writes to `training/output/final/`: observables and
   trajectories, `interface_final.lmpdat`, PNG/SVG plots, and
   `evaluation.json`.

## Supporting files

- `prepare.py` contains the preparation simulation used by `setup.ipynb`.
- `pinning_production.lmp` is the LAMMPS input used for both pinning runs.
- `difftre_helpers.py` implements DiffTRe data handling and model export.
- `evaluate_pinning.py` runs the final LAMMPS evaluation and saves its summary.
- `ip_helpers.py` loads the pinning data, computes the free energy, and makes
  the plots; `units.py` holds the unit conversion.
- `setup.html` and `difftre.html` are rendered copies of the first two
  notebooks.

## Results

`interface_pinning.toml` provides the solid and liquid reference order
parameters, the restraint strength, and the midpoint restraint center. The
final evaluation combines these values with the sampled production-average
order parameter to calculate the BCC--liquid free-energy difference and the
chemical-potential difference in `evaluation.json`.

The required preparation files are listed in `input/PREPARATION_OUTPUTS.txt`.
The standard foundation-run output files are listed in
`results/PINNING_OUTPUTS.txt`.
