"""Sigma_A-weighted map coefficients (Read, Acta Cryst. A42, 1986).

Estimates ``sigma_A`` per resolution bin by maximum likelihood, then builds
sigma_A-weighted ``2mFo-DFc`` and ``mFo-DFc`` map coefficients in the standard
crystallographic convention (acentric ``n=2``, centric ``n=1``). These match
the phenix ``2FOFCWT``/``PH2FOFCWT`` (best) and ``FOFCWT``/``PHFOFCWT``
(difference) columns.

The figure of merit ``m`` and scale ``D`` are::

    E_o = F_obs / sqrt(<F_obs^2>_bin),   E_c = |F_calc| / sqrt(<|F_calc|^2>_bin)
    X   = 2 sigma_A E_o E_c / (1 - sigma_A^2)         (acentric)
    m   = I1(X) / I0(X)  (acentric)   or   tanh(X/2) (centric)
    D   = sigma_A * sqrt(<F_obs^2>_bin / <|F_calc|^2>_bin)

with epsilon multiplicities taken as 1 (the effect on map display is small).

Notes
-----
Sigma_A weighting brings the *labelling* and noise behaviour in line with
phenix, but it does not by itself raise peak heights for an already
well-phased model -- with a good model ``m -> 1`` and ``D -> 1`` so
``2mFo-DFc -> 2Fo-Fc``. Peak height is governed by the model (B-factors,
coordinates), not the weighting.
"""

import numpy as np
from scipy.special import i0e, i1e

__all__ = ["estimate_sigmaa", "sigmaa_map_coefficients"]


def estimate_sigmaa(F_obs, F_calc_amp, centric, dHKL, work_mask, n_bins=20, n_grid=60):
    """Per-reflection figure of merit ``m`` and scale ``D``.

    Parameters
    ----------
    F_obs, F_calc_amp : np.ndarray
        Observed and calculated amplitudes (shape ``(N,)``), on the same scale.
    centric : np.ndarray of bool
        Centric flag per reflection.
    dHKL : np.ndarray
        Resolution (d-spacing, Angstrom) per reflection.
    work_mask : np.ndarray of bool
        True for working-set reflections (sigma_A is estimated from these).
    n_bins : int
        Number of equal-population resolution bins.
    n_grid : int
        Grid resolution for the per-bin sigma_A maximum-likelihood search.

    Returns
    -------
    m, D : np.ndarray
        Figure of merit and scale per reflection. Invalid reflections
        (non-finite / non-positive amplitudes) get ``m=0, D=0``.
    """
    F_obs = np.asarray(F_obs, dtype=np.float64)
    F_calc_amp = np.asarray(F_calc_amp, dtype=np.float64)
    centric = np.asarray(centric, dtype=bool)
    dHKL = np.asarray(dHKL, dtype=np.float64)
    work_mask = np.asarray(work_mask, dtype=bool)

    N = len(F_obs)
    m = np.zeros(N)
    D = np.zeros(N)

    valid = (
        np.isfinite(F_obs) & np.isfinite(F_calc_amp)
        & (F_obs > 0) & (F_calc_amp > 0)
        & np.isfinite(dHKL) & (dHKL > 0)
    )
    if valid.sum() < 2:
        return m, D
    # Cap bins so each holds a reasonable population.
    n_bins = int(max(1, min(n_bins, valid.sum() // 25)))

    s2 = np.zeros(N)
    s2[valid] = 1.0 / dHKL[valid] ** 2
    edges = np.quantile(s2[valid], np.linspace(0.0, 1.0, n_bins + 1))
    edges[-1] += 1e-6
    binid = np.clip(np.digitize(s2, edges) - 1, 0, n_bins - 1)

    sa_grid = np.linspace(0.05, 0.99, n_grid)
    log2 = np.log(2.0)
    for b in range(n_bins):
        inbin = valid & (binid == b)
        if not inbin.any():
            continue
        wbin = inbin & work_mask
        if wbin.sum() < 10:
            wbin = inbin
        Sig_o = np.mean(F_obs[wbin] ** 2)
        Sig_c = np.mean(F_calc_amp[wbin] ** 2)
        if Sig_o <= 0 or Sig_c <= 0:
            continue
        Eo = F_obs / np.sqrt(Sig_o)
        Ec = F_calc_amp / np.sqrt(Sig_c)
        eo, ec, cw = Eo[wbin], Ec[wbin], centric[wbin]

        # Per-bin ML sigma_A via grid search (overflow-safe log-likelihood).
        best_ll, best_sa = -np.inf, 0.5
        for sa in sa_grid:
            omm = 1.0 - sa * sa
            x = sa * eo * ec / omm  # >= 0
            log_i0 = np.log(i0e(2 * x)) + 2 * x          # log I0(2x)
            ll_ac = np.log(2 * eo / omm) - (eo ** 2 + sa ** 2 * ec ** 2) / omm + log_i0
            log_cosh = np.logaddexp(x, -x) - log2        # log cosh(x)
            ll_ce = (0.5 * np.log(2.0 / (np.pi * omm))
                     - (eo ** 2 + sa ** 2 * ec ** 2) / (2 * omm) + log_cosh)
            ll = np.where(cw, ll_ce, ll_ac)
            tot = ll[np.isfinite(ll)].sum()
            if tot > best_ll:
                best_ll, best_sa = tot, sa

        sa = best_sa
        omm = 1.0 - sa * sa
        D[inbin] = sa * np.sqrt(Sig_o / Sig_c)
        x = sa * Eo[inbin] * Ec[inbin] / omm
        cen = centric[inbin]
        m_ac = i1e(2 * x) / np.clip(i0e(2 * x), 1e-30, None)
        m_ce = np.tanh(np.clip(x, 0.0, 700.0))
        m[inbin] = np.where(cen, m_ce, m_ac)
    return m, D


def sigmaa_map_coefficients(
    F_obs, F_calc_amp, phi_c_deg, centric, dHKL, work_mask, n_bins=20
):
    """Sigma_A-weighted 2mFo-DFc and mFo-DFc map coefficients.

    Returns a dict with phenix-style column names::

        2FOFCWT, PH2FOFCWT   -- (n*m*Fo - D*Fc) * exp(i*phi_c), n=2/1 acen/cen
        FOFCWT,  PHFOFCWT    -- (m*Fo - D*Fc)   * exp(i*phi_c)

    plus the per-reflection ``m`` and ``D``. Amplitudes are non-negative; the
    sign of the real coefficient is folded into a 180-degree phase flip.
    """
    F_obs = np.asarray(F_obs, dtype=np.float64)
    F_calc_amp = np.asarray(F_calc_amp, dtype=np.float64)
    phi_c_deg = np.asarray(phi_c_deg, dtype=np.float64)
    centric = np.asarray(centric, dtype=bool)

    m, D = estimate_sigmaa(F_obs, F_calc_amp, centric, dHKL, work_mask, n_bins=n_bins)
    n_factor = np.where(centric, 1.0, 2.0)

    real_best = n_factor * m * F_obs - D * F_calc_amp
    real_diff = m * F_obs - D * F_calc_amp

    def amp_phase(real):
        amp = np.abs(real)
        ph = np.where(real < 0.0, phi_c_deg + 180.0, phi_c_deg)
        ph = np.mod(ph + 180.0, 360.0) - 180.0  # wrap to (-180, 180]
        return amp, ph

    a2, p2 = amp_phase(real_best)
    a1, p1 = amp_phase(real_diff)
    return {
        "2FOFCWT": a2,
        "PH2FOFCWT": p2,
        "FOFCWT": a1,
        "PHFOFCWT": p1,
        "m": m,
        "D": D,
    }
