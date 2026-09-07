#!/usr/bin/env python3
"""Prepare the configuration and parameters used by interface pinning."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import tempfile


TUTORIAL = Path(__file__).resolve().parent
REPOSITORY = TUTORIAL.parents[1]
DEFAULT_LAMMPS = REPOSITORY / "software/bin/lmp"


LAMMPS_INPUT = r"""
variable defPress        equal 0.0
variable timestep        equal 0.004
variable tempMelt        equal 1941
variable tempLiquid      equal 2500
variable timeBoxEq       equal 5000
variable timeFixedEq     equal 2500
variable timeSolid       equal 5000
variable timeMelt        equal 2500
variable timeLiquidEq    equal 2500
variable timeLiquid      equal 5000
variable timeSetup       equal 2500
variable timeInterfaceEq equal 5000

variable lattice         equal 3.3246088723444838
variable mass            equal 47.867
variable xBoxSize        equal 32
variable yBoxSize        equal 8
variable zBoxSize        equal 8
variable crystalPlanes   equal 2*v_xBoxSize
variable boltzmann       equal 8.617333262145e-5
variable barAngstrom3ToEv equal 6.241509074e-7
variable productionSolidFraction equal 0.75

variable workPath        string ${work_path}
variable artifactPath    string ${artifact_path}

variable press           equal $(v_defPress*10000)
variable tempDamp        equal 100
variable pressDamp       equal 250

timer           loop
units           metal
dimension       3
boundary        p p p
atom_style      atomic
newton          on

package         kokkos atom/map no

lattice         bcc ${lattice}
region          box block 0 ${xBoxSize} 0 ${yBoxSize} 0 ${zBoxSize} units lattice
create_box      90 box
create_atoms    22 box
mass            * ${mass}

neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes
timestep        ${timestep}
comm_modify     mode single cutoff 14.0

variable        pinning_kappa equal 0.0
variable        pinning_center equal 0.0

pair_style      chemtrain cuda 0.6 &
                  comm on &
                  capacity/atom 1.1 &
                  capacity/edge 1.15 &
                  models main 1 order 1 &
                  global/input order steinhardt_kappa v_pinning_kappa &
                  global/input order steinhardt_center v_pinning_center &
                  output order steinhardt_order

pair_coeff      * * ${mace_model} ${order_model}

compute         Q all chemtrain/output/kk order steinhardt_order

thermo          100
thermo_style    custom step temp pe etotal press pxx vol lx ly lz c_Q
thermo_modify   format float %20.16g

dump            dump_all all atom 100 ${workPath}/dump.lammpstrj

velocity        all create ${tempMelt} 345632 mom yes dist gaussian
velocity        all zero linear

fix             box_npt all npt temp ${tempMelt} ${tempMelt} $(v_tempDamp*dt) &
                  aniso ${press} ${press} $(v_pressDamp*dt)

fix             momentum all momentum 100 linear 1 1 1

variable        boxLx equal lx
variable        boxLy equal ly
variable        boxLz equal lz
variable        boxSample equal 100
variable        boxRepeat equal $(v_timeBoxEq/v_boxSample)

fix             box_avg all ave/time ${boxSample} ${boxRepeat} ${timeBoxEq} &
                  v_boxLx v_boxLy v_boxLz ave one file ${workPath}/box.txt

run             ${timeBoxEq}

variable        lxMean equal $(f_box_avg[1]:%.17g)
variable        lyMean equal $(f_box_avg[2]:%.17g)
variable        lzMean equal $(f_box_avg[3]:%.17g)

unfix           box_avg
unfix           box_npt

change_box      all x final 0 ${lxMean} y final 0 ${lyMean} &
                  z final 0 ${lzMean} remap units box

fix             phase_npt all npt temp ${tempMelt} ${tempMelt} $(v_tempDamp*dt) &
                  x ${press} ${press} $(v_pressDamp*dt)

run             ${timeFixedEq}

variable        qSample equal 10
variable        qWindowSolid equal $(v_timeSolid/v_qSample)
variable        qStartSolid equal $(step+v_qSample)
variable        potentialEnergyPerAtom equal pe/count(all)
variable        totalEnergyPerAtom equal etotal/count(all)
variable        enthalpyPerAtom equal (etotal+pxx*vol*v_barAngstrom3ToEv)/count(all)
variable        volumePerAtom equal vol/count(all)
variable        sampleTemperature equal temp
variable        normalPressure equal pxx

# Sample the unbiased solid reference used to calibrate interface pinning. ---
fix             solid_average all ave/time ${qSample} 1 ${qSample} &
                  c_Q v_potentialEnergyPerAtom v_totalEnergyPerAtom &
                  v_enthalpyPerAtom v_volumePerAtom v_sampleTemperature &
                  v_normalPressure ave window ${qWindowSolid} start ${qStartSolid}

run             ${timeSolid}

variable        qSolid equal $(f_solid_average[1]:%.17g)
variable        solidPotentialEnergy equal $(f_solid_average[2]:%.17g)
variable        solidTotalEnergy equal $(f_solid_average[3]:%.17g)
variable        solidEnthalpy equal $(f_solid_average[4]:%.17g)
variable        solidVolume equal $(f_solid_average[5]:%.17g)
variable        solidTemperature equal $(f_solid_average[6]:%.17g)
variable        solidPressure equal $(f_solid_average[7]:%.17g)

unfix           solid_average

variable        crystalStep equal $(step)

write_dump      all custom ${workPath}/crystal.dump &
                  id type x y z vx vy vz ix iy iz modify sort id

unfix           phase_npt

fix             melt_npt all npt temp ${tempLiquid} ${tempLiquid} $(v_tempDamp*dt) &
                  x ${press} ${press} $(v_pressDamp*dt)

run             ${timeMelt}

unfix           melt_npt

fix             liquid_npt all npt temp ${tempMelt} ${tempMelt} $(v_tempDamp*dt) &
                  x ${press} ${press} $(v_pressDamp*dt)

run             ${timeLiquidEq}

variable        qWindowLiquid equal $(v_timeLiquid/v_qSample)
variable        qStartLiquid equal $(step+v_qSample)

# Sample the unbiased liquid reference at the same target state point. -------
fix             liquid_average all ave/time ${qSample} 1 ${qSample} &
                  c_Q v_potentialEnergyPerAtom v_totalEnergyPerAtom &
                  v_enthalpyPerAtom v_volumePerAtom v_sampleTemperature &
                  v_normalPressure ave window ${qWindowLiquid} start ${qStartLiquid}

run             ${timeLiquid}

variable        qLiquid equal $(f_liquid_average[1]:%.17g)
variable        liquidPotentialEnergy equal $(f_liquid_average[2]:%.17g)
variable        liquidTotalEnergy equal $(f_liquid_average[3]:%.17g)
variable        liquidEnthalpy equal $(f_liquid_average[4]:%.17g)
variable        liquidVolume equal $(f_liquid_average[5]:%.17g)
variable        liquidTemperature equal $(f_liquid_average[6]:%.17g)
variable        liquidPressure equal $(f_liquid_average[7]:%.17g)

unfix           liquid_average
unfix           liquid_npt

variable        qCenter equal $(0.5*(v_qSolid+v_qLiquid):%.17g)
variable        deltaQ equal $(v_qSolid-v_qLiquid:%.17g)
variable        kappa equal $(v_boltzmann*v_tempMelt*v_crystalPlanes*v_crystalPlanes/(v_deltaQ*v_deltaQ):%.17g)
variable        productionCenter equal $(v_productionSolidFraction*v_qSolid+(1.0-v_productionSolidFraction)*v_qLiquid:%.17g)

unfix           momentum

read_dump       ${workPath}/crystal.dump ${crystalStep} &
                  x y z vx vy vz ix iy iz box yes replace yes timestep no

variable        center equal lx/2
region          liquid_half plane v_center 0 0 1 0 0 units box
group           liquid region liquid_half

velocity        liquid create ${tempLiquid} 1234 mom yes dist gaussian
velocity        liquid zero linear

fix             half_melt liquid nvt temp ${tempLiquid} ${tempLiquid} $(v_tempDamp*dt)

run             ${timeSetup}

unfix           half_melt

velocity        all create ${tempMelt} 918273 mom yes dist gaussian
velocity        all zero linear

variable        pinning_center equal ${qCenter}
variable        pinning_kappa equal ${kappa}

fix             interface_npt all npt temp ${tempMelt} ${tempMelt} $(v_tempDamp*dt) &
                  x ${press} ${press} $(v_pressDamp*dt)

fix             interface_momentum all momentum 100 linear 1 1 1

run             ${timeInterfaceEq}

unfix           interface_momentum
unfix           interface_npt

write_data      ${artifactPath}/interface_start.lmpdat

print           "[interface_pinning]" &
                  file ${artifactPath}/interface_pinning.toml screen no

print           "q_solid = $(v_qSolid:%.17g)" &
                  append ${artifactPath}/interface_pinning.toml screen no

print           "q_liquid = $(v_qLiquid:%.17g)" &
                  append ${artifactPath}/interface_pinning.toml screen no

print           "center = $(v_qCenter:%.17g)" &
                  append ${artifactPath}/interface_pinning.toml screen no

print           "production_center = $(v_productionCenter:%.17g)" &
                  append ${artifactPath}/interface_pinning.toml screen no

print           "production_solid_fraction = $(v_productionSolidFraction:%.17g)" &
                  append ${artifactPath}/interface_pinning.toml screen no

print           "delta_q = $(v_deltaQ:%.17g)" &
                  append ${artifactPath}/interface_pinning.toml screen no

print           "crystal_planes = $(v_crystalPlanes:%.17g)" &
                  append ${artifactPath}/interface_pinning.toml screen no

print           "kappa = $(v_kappa:%.17g)" &
                  append ${artifactPath}/interface_pinning.toml screen no

print           "temperature = $(v_tempMelt:%.17g)" &
                  append ${artifactPath}/interface_pinning.toml screen no

print           "pressure_gpa = $(v_defPress:%.17g)" &
                  append ${artifactPath}/interface_pinning.toml screen no

print           "[metadata]" &
                  file ${artifactPath}/phase_thermodynamics.toml screen no

print           "temperature_k = $(v_tempMelt:%.17g)" &
                  append ${artifactPath}/phase_thermodynamics.toml screen no

print           "pressure_gpa = $(v_defPress:%.17g)" &
                  append ${artifactPath}/phase_thermodynamics.toml screen no

print           "energy_unit = 'eV/atom'" &
                  append ${artifactPath}/phase_thermodynamics.toml screen no

print           "volume_unit = 'angstrom^3/atom'" &
                  append ${artifactPath}/phase_thermodynamics.toml screen no

print           "pressure_unit = 'bar'" &
                  append ${artifactPath}/phase_thermodynamics.toml screen no

print           "barostat_direction = 'x'" &
                  append ${artifactPath}/phase_thermodynamics.toml screen no

print           "[solid]" &
                  append ${artifactPath}/phase_thermodynamics.toml screen no

print           "potential_energy_ev_per_atom = $(v_solidPotentialEnergy:%.17g)" &
                  append ${artifactPath}/phase_thermodynamics.toml screen no

print           "total_energy_ev_per_atom = $(v_solidTotalEnergy:%.17g)" &
                  append ${artifactPath}/phase_thermodynamics.toml screen no

print           "enthalpy_ev_per_atom = $(v_solidEnthalpy:%.17g)" &
                  append ${artifactPath}/phase_thermodynamics.toml screen no

print           "volume_angstrom3_per_atom = $(v_solidVolume:%.17g)" &
                  append ${artifactPath}/phase_thermodynamics.toml screen no

print           "temperature_k = $(v_solidTemperature:%.17g)" &
                  append ${artifactPath}/phase_thermodynamics.toml screen no

print           "normal_pressure_bar = $(v_solidPressure:%.17g)" &
                  append ${artifactPath}/phase_thermodynamics.toml screen no

print           "[liquid]" &
                  append ${artifactPath}/phase_thermodynamics.toml screen no

print           "potential_energy_ev_per_atom = $(v_liquidPotentialEnergy:%.17g)" &
                  append ${artifactPath}/phase_thermodynamics.toml screen no

print           "total_energy_ev_per_atom = $(v_liquidTotalEnergy:%.17g)" &
                  append ${artifactPath}/phase_thermodynamics.toml screen no

print           "enthalpy_ev_per_atom = $(v_liquidEnthalpy:%.17g)" &
                  append ${artifactPath}/phase_thermodynamics.toml screen no

print           "volume_angstrom3_per_atom = $(v_liquidVolume:%.17g)" &
                  append ${artifactPath}/phase_thermodynamics.toml screen no

print           "temperature_k = $(v_liquidTemperature:%.17g)" &
                  append ${artifactPath}/phase_thermodynamics.toml screen no

print           "normal_pressure_bar = $(v_liquidPressure:%.17g)" &
                  append ${artifactPath}/phase_thermodynamics.toml screen no

print           "Q_solid=$(v_qSolid:%.17g), Q_liquid=$(v_qLiquid:%.17g), center=$(v_qCenter:%.17g), kappa=$(v_kappa:%.17g)"
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="1", help="CUDA device index (default: 1)")
    parser.add_argument("--lammps", type=Path, default=DEFAULT_LAMMPS)
    args = parser.parse_args()

    lammps = args.lammps.resolve()
    mace_model = TUTORIAL / "mace_mp_mh-1_matpes_r2scan_oeq.ptb"
    order_model = TUTORIAL / "steinhardt_order_oeq.ptb"
    for required_file in (lammps, mace_model, order_model):
        if not required_file.is_file():
            raise FileNotFoundError(required_file)

    artifacts = TUTORIAL / "input"
    artifacts.mkdir(exist_ok=True)

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    environment["JCN_PJRT_PATH"] = str(REPOSITORY / "software/lib/pjrt")
    environment.setdefault("NVIDIA_TF32_OVERRIDE", "0")

    with tempfile.TemporaryDirectory(prefix="diffttc-preparation-") as temporary:
        command = [
            str(lammps),
            "-k", "on", "g", "1",
            "-sf", "kk",
            "-pk", "kokkos", "newton", "on", "neigh", "half",
            "-var", "work_path", temporary,
            "-var", "artifact_path", str(artifacts),
            "-var", "mace_model", str(mace_model),
            "-var", "order_model", str(order_model),
        ]
        subprocess.run(
            command,
            cwd=TUTORIAL,
            env=environment,
            input=LAMMPS_INPUT,
            text=True,
            check=True,
        )

    expected = (
        artifacts / "interface_start.lmpdat",
        artifacts / "interface_pinning.toml",
        artifacts / "phase_thermodynamics.toml",
    )
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(f"Preparation completed without expected files: {missing}")
    print(f"Prepared production inputs in {artifacts}")


if __name__ == "__main__":
    main()
