"""Example: anomalous (Bijvoet) refinement with TorchRef.

With a wavelength set and anomalous scatterers present (here Zn/Ca in
thermolysin at λ=1.2713 Å), the model computes distinct |F_calc| for the (+)
and (-) Friedel mates, so the refinement fits the Bijvoet differences instead
of their mean. Internally each mate keeps its true signed HKL for the
structure-factor calculation while sharing a canonical ASU index for
bookkeeping (binning, scaling, r-free); see ``anomalous_friedel_proposal.md``.

The refined MTZ is written phenix-style on the canonical ASU: normal display
maps (FWT/PHWT, DELFWT/PHDELWT) plus already-unstacked F-obs(+/-) /
F-model(+/-) columns — no negative-ASU indices, so it opens cleanly in Coot
and no ``unstack_anomalous`` step is needed.

Run:
    python example_anomalous_refinement.py
"""

import numpy as np
import torch
import reciprocalspaceship as rs

from torchref import LBFGSRefinement

# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
mtz_file = "asu_0_epoch_30.mtz"          # observed amplitudes/intensities
pdb_file = "2tli.pdb"                    # starting model (thermolysin: Zn, Ca)
wavelength = 1.2713                      # Å — drives f'/f'' anomalous corrections

pdb_out = "refined.pdb"
mtz_out = "refined.mtz"                  # phenix-style, already-unstacked columns
input_mtz = "input.mtz"

# --------------------------------------------------------------------------- #
# Prepare data
# --------------------------------------------------------------------------- #
# Stack to the full (+/-) reflection list so both Bijvoet mates are present as
# separate observations. On load, TorchRef maps both mates to the same canonical
# ASU index for bookkeeping but tracks a signed HKL per row for the structure-
# factor calculation, so the anomalous signal is preserved.
ds = rs.read_mtz(mtz_file)
ds = ds.stack_anomalous()
ds.write_mtz(input_mtz)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# --------------------------------------------------------------------------- #
# Refine
# --------------------------------------------------------------------------- #
refinement = LBFGSRefinement(
    data_file=input_mtz,
    pdb=pdb_file,
    device=device,
    wavelength=wavelength,   # set the experimental wavelength for f'/f''
)

rwork0, rfree0 = refinement.get_rfactor()
print(f"Initial: Rwork={rwork0:.4f}  Rfree={rfree0:.4f}")

refinement.refine_everything(macro_cycles=5)

rwork, rfree = refinement.get_rfactor()
print(f"Final  : Rwork={rwork:.4f}  Rfree={rfree:.4f}")

# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
refinement.write_out_pdb(pdb_out)
# anomalous=True writes a phenix-style MTZ on the canonical ASU:
#   * display maps  FWT/PHWT (2mFo-DFc), DELFWT/PHDELWT (mFo-DFc), with Friedel
#     mates merged by mean amplitude, so Coot renders normal maps;
#   * already-unstacked anomalous columns  F-obs(+/-), SIGF-obs(+/-),
#     F-model(+/-), PHIF-model(+/-), plus ANOM/PANOM.
refinement.write_out_mtz(mtz_out, anomalous=True)

# --------------------------------------------------------------------------- #
# Sanity checks
# --------------------------------------------------------------------------- #
out = rs.read_mtz(mtz_out)

# 1) Output is on the canonical ASU with no duplicate indices (the bug that
#    motivated this change) and already carries unstacked (+)/(-) columns.
assert not out.index.duplicated().any(), "ASU index should be unique"
assert {"F-obs(+)", "F-obs(-)", "F-model(+)", "F-model(-)"} <= set(out.columns)
print("✓ phenix-style anomalous columns present; ASU index unique")

# 2) The model carries a calculated anomalous signal: F-model(+) != F-model(-)
#    for acentric reflections (identical without a wavelength).
fm_plus = out["F-model(+)"].to_numpy(dtype="float32")
fm_minus = out["F-model(-)"].to_numpy(dtype="float32")
both = np.isfinite(fm_plus) & np.isfinite(fm_minus)
bijvoet = np.abs(fm_plus[both] - fm_minus[both])
print(f"  mean |F-model(+) - F-model(-)| = {bijvoet.mean():.3f}")
assert (bijvoet > 1e-3).any(), "expected a non-zero calculated anomalous difference"
print("✓ model reproduces a calculated Bijvoet difference")

print(f"\nWrote {pdb_out} and {mtz_out}. Open {mtz_out} in Coot:")
print("  2mFo-DFc -> FWT/PHWT, mFo-DFc -> DELFWT/PHDELWT, anomalous diff -> ANOM/PANOM")
