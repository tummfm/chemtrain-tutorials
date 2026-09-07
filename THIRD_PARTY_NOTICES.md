# Third-Party Notices

The code, input scripts, documentation, and included example results in this
repository are released under the [MIT License](LICENSE). Third-party models,
reference data, and software are not covered by that license.

## Model parameters

- The titanium example exports the MACE-MH-1 foundation model with its
  `matpes_r2scan` output head selected. It is licensed under the
  [Academic Software License (ASL)](https://github.com/gabor1/ASL) and is not
  covered by this repository's MIT license.
- The Steinhardt-order model and the DiffTRe-trained model parameters are model
  artifacts, not MIT-licensed source code.

## Reference data

- Water positions and forces used in `examples/01_difftre` are from the water
  dataset supplied with the [Relative Entropy reference
  implementation](https://github.com/tummfm/relative-entropy). Cite Thaler,
  Stupp, and Zavadlav, [*Deep coarse-grained potentials via relative entropy
  minimization*](https://doi.org/10.1063/5.0124538), *J. Chem. Phys.* **157**,
  244103 (2022), when reusing them.
- The RDF and ADF targets are provided by the
  [DiffTRe reference implementation](https://github.com/tummfm/difftre) and
  are covered by its Apache-2.0 license.
