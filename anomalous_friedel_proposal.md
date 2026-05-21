# Anomalous / Friedel-mate handling — notes for developer discussion

**Context:** building a phenix-style MTZ export workflow on top of `LBFGSRefinement`.
The exported maps came out as nonsense in Coot, and `rs.unstack_anomalous` fails on the
output. Tracing both led to the same root cause in how Friedel mates are handled.

## Status (2026-05-20)

**Option B1 is now implemented** per the four decisions below (B1; merged display maps in
the ASU with anomalous columns unstacked; mean amplitude for merging; both mates counted in
the R-factor). It lives on uncommitted local changes for review — happy to revert or adjust
after the developer conversation. Working end-to-end on 1DAW; runnable example in
`example_anomalous_refinement.py`. Implemented pieces:

- **Defensive fix** in `canonicalize_hkl` — raises a clear `ValueError` instead of returning
  uninitialized `np.empty` memory for unmappable reflections (the `L=706272` wild entry, and
  the more dangerous *silent* garbage case). Regression test added.
- **Dual HKL representation** — `ReflectionData` now carries `friedel_flags` and a signed
  `hkl_anomalous` (fields on `CrystalDataset`, so device movement / state_dict / row-selection
  come for free), exposed via `hkl_for_sf()`. Canonical `hkl` is unchanged for bookkeeping.
- **Wavelength plumbing** — `Refinement.__init__` gains `wavelength` / `anomalous_threshold`,
  passed to `ModelFT` (previously hard-wired, so anomalous corrections never fired).
- **Signed-HKL routing** — `get_fcalc`, the x-ray target, the scaler (model + bulk solvent),
  and the CLI R-factor now evaluate structure factors at `hkl_for_sf()`. Verified: Bijvoet
  mates get distinct `|F_calc|` with a wavelength, identical without (backward compatible).
- **Phenix-style output** — `write_mtz(..., anomalous=True)` / `write_out_mtz(..., anomalous=True)`
  emit display maps + merged `F-obs`/`F-model` (mean amplitude) and unstacked `F-obs(+/-)`,
  `SIGF-obs(+/-)`, `F-model(+/-)`, `PHIF-model(+/-)`, `ANOM`/`PANOM` on the canonical ASU —
  no negative-ASU indices, no duplicate indices, no `unstack_anomalous` needed downstream.
- **R-factor** counts both mates (existing per-row aggregation; fcalc now signed).
- **Tests** in `tests/unit/io/test_anomalous_output.py` (representation, Bijvoet signal,
  output layout). Note: `pytest` is not installed in the working env; assertions were
  verified by running the test bodies directly.

### Not done / deliberately deferred
- Generated r-free is still assigned per-row (potential Bijvoet leakage); provided r-free
  from a stacked MTZ — the common case — is already safe. See smaller open items.
- `collection_scaler.py` (DatasetCollection path) was left on canonical HKL; only the
  single-dataset `Scaler` used by `LBFGSRefinement` was routed to signed HKL.

## Key finding that makes this tractable

The structure-factor engine **already produces correct Bijvoet differences when fed a
signed HKL** — no change to the SF math is required:

- `ReciprocalSymmetryExtractor` (`base/reciprocal/symmetry.py:257-266,330-333`) sums only
  over **rotation** operators; it does *not* Friedel-symmetrize. The FFT grid of real
  electron density is Hermitian, so the value at the `-h` cell is `conj(F(h))`. Passing a
  signed HKL therefore returns the correct conjugate grid value automatically.
- `ModelFT._apply_anomalous_correction` (`model/model_ft.py:712-773`) adds
  `ΔF(h)=Σ(f'+if'')·exp(2πi h·r)·occ` **at the HKL passed in**, and `ΔF(-h) ≠ conj(ΔF(h))`
  because of the `f''` sign pattern — this *is* the Bijvoet difference.
- So `F_calc(+h)=F_grid(h)+ΔF(h)` and `F_calc(-h)=conj(F_grid(h))+ΔF(-h)` both come out
  correctly, purely by giving each mate its signed index. When `wavelength is None` or
  there are no significant anomalous scatterers, the two collapse to equal `|F_calc|`
  again → fully backward compatible.

**Implication:** Option B is a *data-representation* change (keep a signed HKL for SF
evaluation alongside the canonical ASU index used for bookkeeping), **not** a rewrite of
the math engine. That substantially lowers the effort and risk estimate.

## Background: what the workflow does

- Input data is a (possibly anomalous) MTZ. The script `stack_anomalous()`-es it, writes
  `input.mtz`, and feeds it to `LBFGSRefinement`.
- After refinement it calls `write_out_mtz`, reads the result with reciprocalspaceship,
  renames columns to phenix conventions, and `unstack_anomalous()`-es the
  observed/model columns to produce `F-obs(+/-)`, `F-model(+/-)`, etc.

## Symptom 1 — scrambled maps (script-level, already understood)

The script was doing `ds = ds.set_index(index)` where `index` was captured from the
*input* MTZ. On load, `ReflectionData._canonicalize_in_place` re-sorts reflections into
canonical order and remaps HKLs, so the output rows are **not** in input order. Pasting
the input index positionally attaches every map coefficient to the wrong reflection.
Fix at the script level is to stop overriding the index — but see Symptom 2, which is the
real reason this needs torchref changes.

## Symptom 2 — `unstack_anomalous` cannot split the Bijvoet pairs

`rs.unstack_anomalous` expects the (+) half at `h,k,l` and the (−) half at `-h,-k,-l`
(distinct indices). After torchref's load, both members sit at the **same canonical ASU
index**, so the output MTZ has duplicate indices and there is nothing left to split on.

### Root cause (code references)

- `reflection_data.py:124-126` — load always calls
  `canonicalize_hkl(..., include_friedel=True)`.
- `reciprocal_symmetry.py:1444-1465` — when a reflection is not already in the ASU, its
  **Friedel mate** is mapped in (`equiv_neg = -(hkl @ R.T)`); if that lands in the ASU it
  becomes the canonical HKL, `friedel_flags[...] = True`, and the phase is negated
  (`reflection_data.py:141`). Both members of an acentric pair therefore collapse to the
  same canonical HKL.
- The `friedel_flags` array is a **local variable** — it is used to fix the phase and then
  discarded. It is never stored on the `ReflectionData` object.
- `reflection_data.py:2080-2087` — `write_mtz` writes only the canonical `self.hkl`. No
  Friedel flag, no original (signed) HKL. So the round-trip is **lossy**: nothing in the
  file records which duplicate row was the flipped (−) mate.

## Important related finding — the loss can't see the anomalous signal

`_canonicalize_in_place` is a reorder, not a merge (`reflection_data.py:128-134`), so the
two mates survive as separate rows with separate `F_obs`/`sigma` — but both carry the same
canonical HKL. The x-ray target gathers `F_calc` purely by HKL
(`grid_operations.py:104-115`), so **both rows read the same grid cell and get an identical
`|F_calc|`**. The loss becomes `(F_obs(+) − |F_calc(h)|)² + (F_obs(−) − |F_calc(h)|)²`,
i.e. it fits the *mean* of the Bijvoet pair.

The model does support anomalous scattering (f′/f″ via `wavelength`, `model_ft.py:85`),
which makes the reciprocal grid non-Hermitian (F(h) ≠ F(−h)\*). But because the (−) mate is
queried at its canonical +h instead of −h, that asymmetry never reaches the loss. **As it
stands, anomalous differences are not actually being refined against** — the duplicate-index
export problem is a symptom of this modeling choice.

## Key question for the developer

**Is fitting Bijvoet/anomalous differences a goal, or is anomalous data merely carried
through for display (anomalous difference maps, phenix-style columns)?** The answer picks
the fix:

## Proposed options

### Option A — make the round-trip lossless (minimal, fixes export only)
1. In `_canonicalize_in_place`, persist what is currently discarded: store
   `self.friedel_flags` (and optionally the pre-canonical signed HKL) as buffers.
2. Add a `write_mtz` mode (e.g. `anomalous=True` / `friedel="signed"`) that, for flagged
   rows, writes signed `-h,-k,-l` and re-negates the phase back, producing a properly
   stacked anomalous MTZ that `rs.unstack_anomalous` consumes directly.
- **Pros:** small, localized; fixes both export symptoms; no change to refinement behavior.
- **Cons:** does not address the loss-fits-the-mean issue — acceptable only if anomalous
  refinement is out of scope.

### Option B — don't collapse Friedel mates when anomalous is active (deeper, fixes the model)
- When `wavelength` is set / anomalous scattering is significant, canonicalize with
  `include_friedel=False` so each mate keeps a distinct signed HKL and queries the grid at
  +h vs −h → distinct `F_calc`, so the anomalous signal is actually fit. Output is then
  natively stackable.
- **Pros:** makes anomalous refinement meaningful *and* fixes export.
- **Cons:** larger blast radius — ASU bookkeeping, resolution binning, scaling, masks, and
  any code assuming canonical-ASU uniqueness need review; must confirm the grid is sampled
  correctly for both halves.

### Option C — script-only stopgap (no torchref change)
- Recover each output row's signed HKL by merging output↔input on
  `(canonical_h, canonical_k, canonical_l, F-obs)` (F-obs disambiguates the two members),
  then set that as the index before `unstack_anomalous`. Avoid reproducing the sort
  permutation directly — `torch.argsort` is not stable, so tie-breaking among duplicates is
  not reliable to mirror.
- **Pros:** unblocks the workflow today. **Cons:** fragile float join; doesn't fix anything
  upstream.

## Suggested test (whichever option)
Round-trip a stacked anomalous MTZ through `write_mtz` → `rs.read_mtz` →
`unstack_anomalous`; assert no duplicate indices in the unstacked result and that
`F-obs(+/-)` / `F-model(+/-)` columns are populated. Add a known-structure check.

## Four decisions to settle before implementing Option B

1. **Representation: B1 vs B2.** B1 = keep the canonical ASU index for bookkeeping
   (binning, scaling, r-free, merging) and carry a separate signed `hkl_anomalous` used
   only for SF evaluation/output. B2 = drop canonicalization and use a single signed
   representation throughout. B1 is far lower risk (the SF engine and all ASU-based
   bookkeeping are untouched); B2 is cleaner conceptually but touches binning/merging/ASU
   assumptions broadly. Recommend B1.
2. **Output division of labor.** Should `write_mtz` emit the **signed** HKL set and let the
   post-processing script call `rs.unstack_anomalous`, or should it unstack internally and
   write `F-obs(+/-)` / `F-model(+/-)` directly? (Display maps `FWT/PHWT`, `DELFWT/PHDELWT`
   stay in the canonical ASU either way, so Coot renders a normal map — this is why the
   `include_friedel=False` experiment's 2FOFCWT "looked sorta alright" while anom was
   garbage.)
3. **Display-map merge convention.** When collapsing Friedel mates for the ASU display
   maps, merge by mean amplitude, or some other convention, for parity with phenix?
4. **R-factor reporting.** `get_rfactor` → `scaler.rfactor()` will sum over the
   anomalous-expanded set (both mates). Report on merged amplitudes for phenix parity, or
   on the expanded set?

### Smaller open items
- Where should "un-canonicalization" live — a `write_mtz` flag, or a reusable
  `ReflectionData.to_signed_hkl()` / `to_stacked()` method?
- `friedel_flags` (and `hkl_anomalous`) should be persisted buffers (in `state_dict`) so
  save/load and `create_from_state_dict` round-trip.
- Generated r-free (`_generate_rfree_flags`) currently assigns per-row before
  canonicalize, so a Bijvoet pair can split across work/free (leakage). Should assign per
  canonical-ASU group. (Provided r-free from a stacked input MTZ is already safe.)
