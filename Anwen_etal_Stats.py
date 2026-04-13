#!/usr/bin/env python3
"""
Statistical Analysis of Mercury Isotope Geochemistry — Ediacaran-Cambrian Boundary
====================================================================================

Reproduces all supplementary figures (S1–S5) and statistical tables for:

    "Mercury isotopes record sedimentary host phase transitions
     across the Ediacaran-Cambrian boundary"
    Anwen Zhou, Wang Zheng, Swapan K. Sahoo, Theodore R. Them,
    Yogaraj Banerjee, Datu Adiatma, Jeremy D. Owens

This script performs:
    1. Breakpoint detection (PELT, Binary Segmentation, Dynamic Programming)
       with bootstrap confidence intervals (n = 1,000)
    2. Lead-lag (cross-correlation) analysis
    3. Full-dataset and unit-specific Pearson correlations
    4. Δ²⁰⁰Hg atmospheric source constraint analysis (ANOVA, Kruskal-Wallis)
    5. Boundary sensitivity analysis (±2 m, 25 scenarios)

Requirements:
    pip install pandas numpy scipy ruptures matplotlib seaborn openpyxl

Usage:
    python Hg_isotope_statistical_analysis_VERIFIED.py

    Or with custom paths:
    python Hg_isotope_statistical_analysis_VERIFIED.py \\
        --data_file "AnwenMS_Hg_isotope_Data_Table.xlsx" \\
        --output_dir "output"

Reproducibility:
    Random seed = 42. All outputs are deterministic.
    Boundaries: lower = 138.9 m, upper = 146.3 m (manuscript convention).
    Unit II contains n = 9 samples (139.5–146.3 m).

Version: 3.0.0 (Verified — 17 March 2026)
License: MIT
"""

# =============================================================================
# IMPORTS
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import os
import re
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d

try:
    import ruptures as rpt
except ImportError:
    print("Error: 'ruptures' package not found. Install with: pip install ruptures")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """All analysis parameters in one place for transparency."""

    # Stratigraphic unit boundaries (manuscript convention)
    LOWER_BOUNDARY = 138.9   # meters
    UPPER_BOUNDARY = 146.3   # meters

    # Bootstrap
    N_BOOTSTRAP    = 1000
    RANDOM_SEED    = 42

    # Breakpoint detection
    MIN_SEGMENT    = 3       # minimum samples per segment
    N_BREAKPOINTS  = 2       # expected number of breakpoints
    COST_MODEL     = "l2"    # least-squares cost function

    # Cross-correlation
    MAX_LAG_M      = 5.0     # meters
    INTERP_SPACING = 0.5     # meters

    # Sensitivity analysis
    BOUNDARY_SHIFTS    = [-2, -1, 0, +1, +2]   # meters
    ROBUSTNESS_THRESH  = 0.85                   # minimum acceptable r

    # Variables
    KEY_VARS   = ['THg (ppb)', 'δ202Hg', 'Δ199Hg', 'FeOxide (Wt %)']
    VAR_LABELS = ['THg (ppb)', 'δ²⁰²Hg (‰)', 'Δ¹⁹⁹Hg (‰)', 'Fe-oxide (wt%)']

    # Figure style
    UNIT_COLORS = {'Unit I': '#3498db', 'Unit II': '#e74c3c', 'Unit III': '#2ecc71'}
    DPI = 300


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(filepath):
    """
    Load and merge geochemistry (Table S3) and isotope (Table S2) data.

    Returns
    -------
    df : DataFrame
        Merged dataset, sorted by height, with NaN rows removed.
    sampling : dict
        Verified sampling interval statistics.
    """
    print("\n" + "=" * 70)
    print("1. LOADING DATA")
    print("=" * 70)

    # Table S3: geochemistry
    geo = pd.read_excel(filepath, sheet_name='Table S3', header=1)
    geo = geo[geo['Bed'].notna()].copy()
    geo = geo[~geo['Bed'].astype(str).str.contains('Note', na=False)]
    for col in geo.columns:
        if col not in ['Bed', 'Lithology', 'Age (Ma)'] and 'Unnamed' not in str(col):
            geo[col] = pd.to_numeric(geo[col], errors='coerce')
    geo['Height'] = geo['Strat. Height (m)']

    # Table S2: isotopes
    iso = pd.read_excel(filepath, sheet_name='Table S2')

    def extract_height(sid):
        m = re.search(r'WH09-?(\d+\.?\d*)', str(sid).strip())
        return float(m.group(1)) if m else np.nan

    iso['Height'] = iso['Sample ID'].apply(extract_height)
    iso = iso.rename(columns={
        'δ202Hg (‰)': 'δ202Hg', 'Δ199Hg (‰)': 'Δ199Hg',
        'Δ200Hg (‰)': 'Δ200Hg', 'Δ201Hg (‰)': 'Δ201Hg'
    })

    # Merge
    df = pd.merge(geo, iso[['Height', 'δ202Hg', 'Δ199Hg', 'Δ200Hg', 'Δ201Hg']],
                  on='Height', how='inner')
    df = df.dropna(subset=['THg (ppb)', 'FeOxide (Wt %)', 'δ202Hg', 'Δ199Hg'])
    df = df.sort_values('Height').reset_index(drop=True)

    # Sampling statistics
    h = df['Height'].values
    intervals = np.diff(h)
    sampling = {
        'n':              len(h),
        'h_min':          h.min(),
        'h_max':          h.max(),
        'study_interval': h.max() - h.min(),
        'mean':           intervals.mean(),
        'median':         np.median(intervals),
        'min':            intervals.min(),
        'max':            intervals.max(),
    }

    print(f"  Samples loaded: {sampling['n']}")
    print(f"  Range: {sampling['h_min']:.1f}–{sampling['h_max']:.1f} m")
    print(f"  Study interval: {sampling['study_interval']:.1f} m")
    print(f"  Mean sampling interval: {sampling['mean']:.2f} m")

    # Also prepare an outer-join version for sensitivity analysis
    # (the sensitivity script merges differently to capture edge samples)
    df_outer = pd.merge(geo, iso[['Height', 'Δ199Hg']], on='Height', how='outer')
    df_outer = df_outer.sort_values('Height').reset_index(drop=True)

    return df, df_outer, sampling


def assign_units(df, lower, upper):
    """Assign stratigraphic units based on boundary positions."""
    df = df.copy()
    df['Unit'] = 'Unit I'
    df.loc[(df['Height'] > lower) & (df['Height'] <= upper), 'Unit'] = 'Unit II'
    df.loc[df['Height'] > upper, 'Unit'] = 'Unit III'
    return df


# =============================================================================
# BREAKPOINT ANALYSIS
# =============================================================================

def _zscore(x):
    """Z-score normalization."""
    x = np.array(x, dtype=float)
    return (x - np.nanmean(x)) / np.nanstd(x)


def _detect_bkps(signal, method, n_bkps=2, min_size=3):
    """Run a single breakpoint detection algorithm."""
    signal = np.array(signal, dtype=float).flatten()
    if method == 'pelt':
        algo = rpt.Pelt(model="l2", min_size=min_size).fit(signal)
        pen = np.log(len(signal)) * signal.var()
        bkps = algo.predict(pen=pen)
        if len(bkps) > n_bkps + 1:
            bkps = bkps[:n_bkps] + [bkps[-1]]
    elif method == 'binseg':
        bkps = rpt.Binseg(model="l2", min_size=min_size).fit(signal).predict(n_bkps=n_bkps)
    elif method == 'dynp':
        bkps = rpt.Dynp(model="l2", min_size=min_size).fit(signal).predict(n_bkps=n_bkps)
    else:
        raise ValueError(f"Unknown method: {method}")
    return bkps[:-1] if bkps[-1] == len(signal) else bkps


def run_breakpoint_analysis(df, cfg):
    """
    Run breakpoint detection (3 algorithms + bootstrap CI) for all key variables.

    Returns
    -------
    results : dict
        Per-variable breakpoint positions, bootstrap distributions, and CIs.
    sync : dict
        Synchroneity metrics (spread, % of study interval).
    """
    print("\n" + "=" * 70)
    print("2. BREAKPOINT ANALYSIS")
    print("=" * 70)

    np.random.seed(cfg.RANDOM_SEED)
    heights = df['Height'].values
    results = {}

    for var in cfg.KEY_VARS:
        signal = _zscore(df[var].values)

        # Three deterministic algorithms
        pelt_idx   = _detect_bkps(signal, 'pelt',   cfg.N_BREAKPOINTS, cfg.MIN_SEGMENT)
        binseg_idx = _detect_bkps(signal, 'binseg', cfg.N_BREAKPOINTS, cfg.MIN_SEGMENT)
        dynp_idx   = _detect_bkps(signal, 'dynp',   cfg.N_BREAKPOINTS, cfg.MIN_SEGMENT)

        def idx_to_h(idxs):
            return [heights[min(i, len(heights) - 1)] for i in idxs]

        # Bootstrap (dynamic programming)
        boot_lo, boot_hi = [], []
        for _ in range(cfg.N_BOOTSTRAP):
            idx = np.sort(np.random.choice(len(signal), len(signal), replace=True))
            try:
                bkps = _detect_bkps(signal[idx], 'dynp', cfg.N_BREAKPOINTS, cfg.MIN_SEGMENT)
                if len(bkps) >= 2:
                    boot_lo.append(heights[min(bkps[0], len(heights) - 1)])
                    boot_hi.append(heights[min(bkps[1], len(heights) - 1)])
            except Exception:
                continue

        results[var] = {
            'pelt':   idx_to_h(pelt_idx),
            'binseg': idx_to_h(binseg_idx),
            'dynp':   idx_to_h(dynp_idx),
            'lower': {
                'mean':  np.mean(boot_lo), 'std':   np.std(boot_lo),
                'ci_lo': np.percentile(boot_lo, 2.5),
                'ci_hi': np.percentile(boot_lo, 97.5),
            },
            'upper': {
                'mean':  np.mean(boot_hi), 'std':   np.std(boot_hi),
                'ci_lo': np.percentile(boot_hi, 2.5),
                'ci_hi': np.percentile(boot_hi, 97.5),
            },
        }
        bl = results[var]['lower']
        bu = results[var]['upper']
        print(f"\n  {var}:")
        print(f"    Lower: {bl['mean']:.1f} ± {bl['std']:.1f} m "
              f"(95% CI: {bl['ci_lo']:.1f}–{bl['ci_hi']:.1f})")
        print(f"    Upper: {bu['mean']:.1f} ± {bu['std']:.1f} m "
              f"(95% CI: {bu['ci_lo']:.1f}–{bu['ci_hi']:.1f})")

    # Synchroneity
    lo_means = [results[v]['lower']['mean'] for v in cfg.KEY_VARS]
    hi_means = [results[v]['upper']['mean'] for v in cfg.KEY_VARS]
    study = heights.max() - heights.min()
    samp  = np.diff(heights).mean()
    lo_spread = max(lo_means) - min(lo_means)
    hi_spread = max(hi_means) - min(hi_means)

    sync = {
        'consensus_lower': np.mean(lo_means),
        'consensus_upper': np.mean(hi_means),
        'lower_spread':    lo_spread,
        'upper_spread':    hi_spread,
        'lower_pct':       lo_spread / study * 100,
        'upper_pct':       hi_spread / study * 100,
        'study_interval':  study,
        'mean_sampling':   samp,
    }
    print(f"\n  Synchroneity:")
    print(f"    Consensus lower: {sync['consensus_lower']:.1f} m")
    print(f"    Consensus upper: {sync['consensus_upper']:.1f} m")
    print(f"    Lower spread: {sync['lower_spread']:.1f} m "
          f"({sync['lower_pct']:.1f}% of study interval)")
    print(f"    Upper spread: {sync['upper_spread']:.1f} m "
          f"({sync['upper_pct']:.1f}% of study interval)")

    return results, sync


# =============================================================================
# LEAD-LAG ANALYSIS
# =============================================================================

def cross_correlation(df, ref_var, target_var, max_lag=5.0, spacing=0.5):
    """
    Cross-correlation between two variables on an interpolated regular grid.

    Returns
    -------
    dict with lags (m), correlations, optimal lag, and max/zero-lag r.
    """
    h = df['Height'].values
    h_grid = np.arange(h.min(), h.max(), spacing)
    x = interp1d(h, df[ref_var].values,    kind='linear', fill_value='extrapolate')(h_grid)
    y = interp1d(h, df[target_var].values,  kind='linear', fill_value='extrapolate')(h_grid)
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()

    max_s = int(max_lag / spacing)
    lags  = np.arange(-max_s, max_s + 1)
    lag_m = lags * spacing
    corrs = []
    for lag in lags:
        if lag < 0:
            corrs.append(np.corrcoef(x[:lag], y[-lag:])[0, 1])
        elif lag > 0:
            corrs.append(np.corrcoef(x[lag:], y[:-lag])[0, 1])
        else:
            corrs.append(np.corrcoef(x, y)[0, 1])
    corrs = np.array(corrs)
    mi = np.argmax(np.abs(corrs))
    return {
        'lags': lag_m, 'corrs': corrs,
        'opt_lag': lag_m[mi], 'max_r': corrs[mi],
        'zero_r': corrs[lags == 0][0]
    }


def run_lead_lag(df, cfg):
    """Run lead-lag analysis for THg, δ202Hg, Δ199Hg vs FeOxide."""
    print("\n" + "=" * 70)
    print("3. LEAD-LAG ANALYSIS")
    print("=" * 70)

    targets = ['THg (ppb)', 'δ202Hg', 'Δ199Hg']
    results = {}
    for t in targets:
        results[t] = cross_correlation(df, 'FeOxide (Wt %)', t,
                                       cfg.MAX_LAG_M, cfg.INTERP_SPACING)
        r = results[t]
        print(f"  {t}: optimal lag = {r['opt_lag']:.1f} m, "
              f"max r = {r['max_r']:.3f}, zero-lag r = {r['zero_r']:.3f}")
    return results


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def run_correlations(df, cfg):
    """Full-dataset and unit-specific Pearson correlations."""
    print("\n" + "=" * 70)
    print("4. CORRELATION ANALYSIS")
    print("=" * 70)

    variables = cfg.KEY_VARS
    results = {'full': {}, 'units': {}}

    # Full dataset
    print("\n  Full dataset:")
    for i, v1 in enumerate(variables):
        for j, v2 in enumerate(variables):
            if i < j:
                r, p = stats.pearsonr(df[v1], df[v2])
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                results['full'][f"{v1}|{v2}"] = {'r': r, 'p': p}
                print(f"    {v1} vs {v2}: r = {r:.3f}{sig}")

    # Unit-specific
    print("\n  Unit-specific:")
    pairs = [('THg (ppb)', 'FeOxide (Wt %)', 'THg-FeOx'),
             ('δ202Hg',    'FeOxide (Wt %)', 'd202-FeOx'),
             ('Δ199Hg',    'FeOxide (Wt %)', 'D199-FeOx')]

    for unit in ['Unit I', 'Unit II', 'Unit III']:
        sub = df[df['Unit'] == unit]
        n = len(sub)
        results['units'][unit] = {'n': n}
        for v1, v2, label in pairs:
            try:
                r, p = stats.pearsonr(sub[v1], sub[v2])
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                results['units'][unit][label] = {'r': r, 'p': p}
                print(f"    {unit} {label}: r = {r:.3f}{sig} (n = {n})")
            except Exception:
                results['units'][unit][label] = {'r': np.nan, 'p': np.nan}
                print(f"    {unit} {label}: NaN (constant input, n = {n})")

    return results


# =============================================================================
# Δ200Hg ANALYSIS
# =============================================================================

def run_delta200(df):
    """Δ200Hg analysis: per-unit statistics, ANOVA, Kruskal-Wallis."""
    print("\n" + "=" * 70)
    print("5. Δ200Hg ANALYSIS")
    print("=" * 70)

    results = {}
    for unit in ['Unit I', 'Unit II', 'Unit III']:
        vals = df[df['Unit'] == unit]['Δ200Hg'].dropna()
        t_stat, p_vs0 = stats.ttest_1samp(vals, 0)
        results[unit] = {
            'mean': vals.mean(), 'std': vals.std(), 'n': len(vals),
            'min': vals.min(), 'max': vals.max(), 'p_vs0': p_vs0,
        }
        print(f"  {unit}: {vals.mean():.4f} ± {vals.std():.4f}‰ "
              f"(n = {len(vals)}, p vs 0 = {p_vs0:.4f})")

    uv = [df[df['Unit'] == u]['Δ200Hg'].dropna().values
          for u in ['Unit I', 'Unit II', 'Unit III']]
    f_stat, f_p = stats.f_oneway(*uv)
    h_stat, h_p = stats.kruskal(*uv)
    results['anova']   = {'F': f_stat, 'p': f_p}
    results['kruskal'] = {'H': h_stat, 'p': h_p}
    print(f"\n  ANOVA:          F = {f_stat:.2f}, p = {f_p:.4f}")
    print(f"  Kruskal-Wallis: H = {h_stat:.2f}, p = {h_p:.4f}")

    # Δ200Hg vs THg and FeOxide
    r_thg,  p_thg  = stats.pearsonr(df['THg (ppb)'],       df['Δ200Hg'])
    r_feox, p_feox = stats.pearsonr(df['FeOxide (Wt %)'],  df['Δ200Hg'])
    results['vs_thg']  = {'r': r_thg,  'p': p_thg}
    results['vs_feox'] = {'r': r_feox, 'p': p_feox}
    print(f"  Δ200Hg vs THg:    r = {r_thg:.2f}, p = {p_thg:.3f}")
    print(f"  Δ200Hg vs FeOx:   r = {r_feox:.2f}, p = {p_feox:.3f}")

    return results


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================

def run_sensitivity(df_outer, cfg):
    """
    Systematically vary Unit II boundaries by ±2 m and recalculate
    THg–FeOxide and Δ199Hg–FeOxide correlations.

    Uses the outer-join dataframe to capture edge samples.
    """
    print("\n" + "=" * 70)
    print("6. BOUNDARY SENSITIVITY ANALYSIS")
    print("=" * 70)

    rows = []
    for ls in cfg.BOUNDARY_SHIFTS:
        for us in cfg.BOUNDARY_SHIFTS:
            lo = cfg.LOWER_BOUNDARY + ls
            hi = cfg.UPPER_BOUNDARY + us
            if lo >= hi:
                continue
            u2 = df_outer[(df_outer['Height'] > lo) & (df_outer['Height'] <= hi)]
            u2 = u2.dropna(subset=['THg (ppb)', 'FeOxide (Wt %)', 'Δ199Hg'])
            if len(u2) < 3:
                continue
            r_thg,  p_thg  = stats.pearsonr(u2['THg (ppb)'],  u2['FeOxide (Wt %)'])
            r_d199, p_d199 = stats.pearsonr(u2['Δ199Hg'],     u2['FeOxide (Wt %)'])
            rows.append({
                'lower_shift': ls, 'upper_shift': us,
                'lower': lo, 'upper': hi, 'n': len(u2),
                'r_thg': r_thg, 'p_thg': p_thg,
                'r_d199': r_d199, 'p_d199': p_d199,
            })

    df_s = pd.DataFrame(rows)
    ref = df_s[(df_s['lower_shift'] == 0) & (df_s['upper_shift'] == 0)].iloc[0]
    robust = (df_s['r_thg'] > cfg.ROBUSTNESS_THRESH).all()

    print(f"  Reference (0,0): n = {ref['n']:.0f}, "
          f"r_THg = {ref['r_thg']:.3f}, r_Δ199 = {ref['r_d199']:.3f}")
    print(f"  THg–FeOx across {len(df_s)} scenarios: "
          f"mean = {df_s['r_thg'].mean():.3f}, "
          f"min = {df_s['r_thg'].min():.3f}, max = {df_s['r_thg'].max():.3f}")
    print(f"  All r(THg–FeOx) > {cfg.ROBUSTNESS_THRESH}: "
          f"{'PASSED ✓' if robust else 'FAILED ✗'}")

    return df_s


# =============================================================================
# FIGURE S1: BREAKPOINT ANALYSIS (6-panel)
# =============================================================================

def figure_s1(df, bp_results, sync, cfg, outdir):
    """Breakpoint profiles, CI comparison, and correlation matrix."""
    print("\n  Creating Figure S1: Breakpoint Analysis...")

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)
    colors = cfg.UNIT_COLORS

    # Panels A–D: stratigraphic profiles
    for i, (var, label) in enumerate(zip(cfg.KEY_VARS, cfg.VAR_LABELS)):
        row, col = (0, i) if i < 3 else (1, 0)
        ax = fig.add_subplot(gs[row, col])

        for unit in ['Unit I', 'Unit II', 'Unit III']:
            sub = df[df['Unit'] == unit]
            ax.scatter(sub[var], sub['Height'], c=colors[unit], s=60, alpha=0.8,
                       edgecolors='white', linewidth=0.5, label=unit, zorder=3)

        # Unit shading
        ax.axhspan(df['Height'].min() - 1, cfg.LOWER_BOUNDARY, alpha=0.08, color='#3498db')
        ax.axhspan(cfg.LOWER_BOUNDARY, cfg.UPPER_BOUNDARY, alpha=0.12, color='#f1c40f')
        ax.axhspan(cfg.UPPER_BOUNDARY, df['Height'].max() + 1, alpha=0.08, color='#2ecc71')

        # Dynamic Programming breakpoints (always exactly 2 per variable)
        for h in bp_results[var]['dynp'][:2]:
            ax.axhline(h, color='red', linestyle='--', alpha=0.7, linewidth=1.5, zorder=2)

        if 'δ202' in var or 'Δ199' in var:
            ax.axvline(0, color='gray', linestyle=':', alpha=0.5)

        ax.set_xlabel(label, fontsize=11, fontweight='bold')
        ax.set_ylabel('Stratigraphic Height (m)', fontsize=11)
        ax.set_title(f'({chr(65 + i)})', fontsize=12, fontweight='bold', loc='left')
        ax.set_ylim(130, 157)
        if i == 0:
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    # Panel E: bootstrap CI comparison
    ax_e = fig.add_subplot(gs[1, 1])
    for i, (var, label) in enumerate(zip(cfg.KEY_VARS, cfg.VAR_LABELS)):
        bl = bp_results[var]['lower']
        bh = bp_results[var]['upper']
        ax_e.errorbar(bl['mean'], i - 0.15,
                      xerr=[[bl['mean'] - bl['ci_lo']], [bl['ci_hi'] - bl['mean']]],
                      fmt='o', color='#3498db', markersize=10, capsize=5,
                      label='Lower' if i == 0 else '')
        ax_e.errorbar(bh['mean'], i + 0.15,
                      xerr=[[bh['mean'] - bh['ci_lo']], [bh['ci_hi'] - bh['mean']]],
                      fmt='s', color='#2ecc71', markersize=10, capsize=5,
                      label='Upper' if i == 0 else '')

    ax_e.axvline(sync['consensus_lower'], color='#3498db', linewidth=2, alpha=0.4)
    ax_e.axvline(sync['consensus_upper'], color='#2ecc71', linewidth=2, alpha=0.4)
    ax_e.set_yticks(range(4))
    ax_e.set_yticklabels(cfg.VAR_LABELS, fontsize=9)
    ax_e.set_xlabel('Stratigraphic Height (m)', fontsize=11)
    ax_e.set_title('(E) Breakpoint Positions (95% CI)', fontsize=12, fontweight='bold', loc='left')
    ax_e.legend(loc='upper right', fontsize=8)

    text = (f"Lower spread: {sync['lower_spread']:.1f} m ({sync['lower_pct']:.1f}%)\n"
            f"Upper spread: {sync['upper_spread']:.1f} m ({sync['upper_pct']:.1f}%)\n"
            f"Mean sampling: {sync['mean_sampling']:.2f} m\n"
            f"n = {cfg.N_BOOTSTRAP:,} bootstrap replicates")
    ax_e.text(0.02, 0.02, text, transform=ax_e.transAxes, fontsize=8,
              verticalalignment='bottom',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel F: correlation matrix
    ax_f = fig.add_subplot(gs[1, 2])
    corr = df[cfg.KEY_VARS].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True, ax=ax_f,
                xticklabels=cfg.VAR_LABELS, yticklabels=cfg.VAR_LABELS,
                annot_kws={'size': 9})
    ax_f.set_title('(F) Full-Dataset Correlation', fontsize=12, fontweight='bold', loc='left')
    plt.setp(ax_f.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax_f.get_yticklabels(), rotation=0, fontsize=8)

    _save_fig(fig, outdir, 'Figure_S1_Breakpoint_Analysis')


# =============================================================================
# FIGURE S2: UNIT-SPECIFIC CORRELATIONS
# =============================================================================

def figure_s2(df, cfg, outdir):
    """Scatter plots of Hg proxies vs Fe-oxide, colored by unit."""
    print("  Creating Figure S2: Unit Correlations...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = cfg.UNIT_COLORS
    plot_vars = [('THg (ppb)', 'THg (ppb)'), ('δ202Hg', 'δ²⁰²Hg (‰)'),
                 ('Δ199Hg', 'Δ¹⁹⁹Hg (‰)')]

    for ax, (var, label) in zip(axes, plot_vars):
        for unit in ['Unit I', 'Unit II', 'Unit III']:
            sub = df[df['Unit'] == unit]
            ax.scatter(sub['FeOxide (Wt %)'], sub[var], c=colors[unit], s=80,
                       alpha=0.8, edgecolors='white', linewidth=0.5, label=unit, zorder=3)

        # Unit II regression
        u2 = df[df['Unit'] == 'Unit II']
        if len(u2) >= 3:
            slope, intercept, r, p, _ = stats.linregress(u2['FeOxide (Wt %)'], u2[var])
            x_line = np.linspace(u2['FeOxide (Wt %)'].min(),
                                 u2['FeOxide (Wt %)'].max(), 100)
            ax.plot(x_line, slope * x_line + intercept, 'r--', linewidth=2, alpha=0.7)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            ax.text(0.05, 0.95, f'Unit II: r = {r:.2f}{sig}\n(n = {len(u2)}, p = {p:.3f})',
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.set_xlabel('Fe-oxide (wt%)', fontsize=11, fontweight='bold')
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        if 'δ202' in var or 'Δ199' in var:
            ax.axhline(0, color='gray', linestyle=':', alpha=0.5)

    axes[0].legend(loc='upper left', fontsize=9)
    for i, ax in enumerate(axes):
        ax.set_title(f'({chr(65 + i)})', fontsize=12, fontweight='bold', loc='left')

    plt.tight_layout()
    _save_fig(fig, outdir, 'Figure_S2_Unit_Correlations')


# =============================================================================
# FIGURE S3: Δ200Hg
# =============================================================================

def figure_s3(df, d200, cfg, outdir):
    """Δ200Hg stratigraphic profile, boxplots, and cross-plots."""
    print("  Creating Figure S3: Δ200Hg...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = cfg.UNIT_COLORS

    # (A) Stratigraphic profile
    ax = axes[0, 0]
    for unit in ['Unit I', 'Unit II', 'Unit III']:
        sub = df[df['Unit'] == unit]
        ax.errorbar(sub['Δ200Hg'], sub['Height'], xerr=0.02, fmt='o',
                    color=colors[unit], markersize=8, alpha=0.8, label=unit, capsize=3)
    ax.axvspan(0.05, 0.25, alpha=0.15, color='blue', label='Atm. Hg(II)\n(Gratz et al., 2010)')
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axhspan(cfg.LOWER_BOUNDARY, cfg.UPPER_BOUNDARY, alpha=0.1, color='#f1c40f')
    ax.set_xlabel('Δ²⁰⁰Hg (‰)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Stratigraphic Height (m)', fontsize=11)
    ax.set_title('(A) Stratigraphic Profile', fontsize=12, fontweight='bold', loc='left')
    ax.legend(loc='upper right', fontsize=7)
    ax.set_ylim(130, 157)

    # (B) Box plots
    ax = axes[0, 1]
    unit_data = [df[df['Unit'] == u]['Δ200Hg'].dropna() for u in ['Unit I', 'Unit II', 'Unit III']]
    bp = ax.boxplot(unit_data, labels=['Unit I', 'Unit II', 'Unit III'],
                    patch_artist=True, widths=0.6)
    for patch, c in zip(bp['boxes'], colors.values()):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
    for i, ud in enumerate(unit_data):
        ax.scatter(np.repeat(i + 1, len(ud)), ud, c=list(colors.values())[i],
                   s=40, alpha=0.7, zorder=3, edgecolors='white')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(0.05, color='blue', linestyle='--', alpha=0.5, label='+0.05‰ threshold')
    ax.set_ylabel('Δ²⁰⁰Hg (‰)', fontsize=11, fontweight='bold')
    ax.set_title('(B) Distribution by Unit', fontsize=12, fontweight='bold', loc='left')
    ax.text(0.5, 0.95,
            f"ANOVA: F = {d200['anova']['F']:.2f}, p = {d200['anova']['p']:.2f}",
            transform=ax.transAxes, ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # (C) vs THg
    ax = axes[1, 0]
    for unit in ['Unit I', 'Unit II', 'Unit III']:
        sub = df[df['Unit'] == unit]
        ax.scatter(sub['THg (ppb)'], sub['Δ200Hg'], c=colors[unit], s=60, alpha=0.8)
    r, p = d200['vs_thg']['r'], d200['vs_thg']['p']
    ax.text(0.05, 0.95, f'r = {r:.2f}, p = {p:.2f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('THg (ppb)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Δ²⁰⁰Hg (‰)', fontsize=11, fontweight='bold')
    ax.set_title('(C) Δ²⁰⁰Hg vs THg', fontsize=12, fontweight='bold', loc='left')

    # (D) vs FeOxide
    ax = axes[1, 1]
    for unit in ['Unit I', 'Unit II', 'Unit III']:
        sub = df[df['Unit'] == unit]
        ax.scatter(sub['FeOxide (Wt %)'], sub['Δ200Hg'], c=colors[unit], s=60, alpha=0.8)
    r, p = d200['vs_feox']['r'], d200['vs_feox']['p']
    ax.text(0.05, 0.95, f'r = {r:.2f}, p = {p:.2f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Fe-oxide (wt%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Δ²⁰⁰Hg (‰)', fontsize=11, fontweight='bold')
    ax.set_title('(D) Δ²⁰⁰Hg vs Fe-oxide', fontsize=12, fontweight='bold', loc='left')

    plt.tight_layout()
    _save_fig(fig, outdir, 'Figure_S3_Delta200Hg')


# =============================================================================
# FIGURE S4: LEAD-LAG ANALYSIS
# =============================================================================

def figure_s4(ll_results, cfg, outdir):
    """Cross-correlation functions for THg, δ202Hg, Δ199Hg vs FeOxide."""
    print("  Creating Figure S4: Lead-Lag...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    targets = ['THg (ppb)', 'δ202Hg', 'Δ199Hg']
    labels  = ['THg', 'δ²⁰²Hg', 'Δ¹⁹⁹Hg']
    tcolors = ['#e74c3c', '#9b59b6', '#3498db']

    for ax, target, label, tc in zip(axes, targets, labels, tcolors):
        r = ll_results[target]
        ax.plot(r['lags'], r['corrs'], '-', color=tc, linewidth=2.5)
        ax.fill_between(r['lags'], r['corrs'], alpha=0.15, color=tc)
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(r['opt_lag'], color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        mi = np.argmax(np.abs(r['corrs']))
        ax.plot(r['lags'][mi], r['corrs'][mi], 'o', color='red', markersize=12,
                markeredgecolor='white', markeredgewidth=2, zorder=5)
        ax.set_xlabel('Lag (m)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Correlation (r)', fontsize=11)
        ax.set_title(f'{label} vs Fe-oxide', fontsize=12, fontweight='bold')
        ax.text(0.05, 0.08,
                f"Optimal lag: {r['opt_lag']:.1f} m\nr = {r['max_r']:.3f}",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        ax.set_xlim(-5.5, 5.5)

    plt.tight_layout()
    _save_fig(fig, outdir, 'Figure_S4_Lead_Lag')


# =============================================================================
# FIGURE S5: SENSITIVITY ANALYSIS
# =============================================================================

def figure_s5(df_sens, cfg, outdir):
    """Heatmaps, histogram, and r-vs-n plot for boundary sensitivity."""
    print("  Creating Figure S5: Sensitivity Analysis...")

    fig = plt.figure(figsize=(14, 10))

    # (A) THg–FeOxide heatmap
    ax1 = fig.add_subplot(2, 2, 1)
    piv = df_sens.pivot(index='lower_shift', columns='upper_shift', values='r_thg')
    im1 = ax1.imshow(piv.values, cmap='RdYlGn', vmin=0.87, vmax=0.95,
                     aspect='auto', origin='lower')
    ax1.set_xticks(range(len(piv.columns)))
    ax1.set_xticklabels([f'{x:+d}' for x in piv.columns])
    ax1.set_yticks(range(len(piv.index)))
    ax1.set_yticklabels([f'{x:+d}' for x in piv.index])
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax1.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=8,
                         fontweight='bold',
                         color='white' if v > 0.935 else 'black')
    ax1.set_xlabel('Upper boundary shift (m)', fontsize=10)
    ax1.set_ylabel('Lower boundary shift (m)', fontsize=10)
    ax1.set_title('(A) THg–FeOxide r', fontsize=12, fontweight='bold', loc='left')
    plt.colorbar(im1, ax=ax1, shrink=0.8, label='Pearson r')

    # (B) Δ199Hg–FeOxide heatmap
    ax2 = fig.add_subplot(2, 2, 2)
    piv2 = df_sens.pivot(index='lower_shift', columns='upper_shift', values='r_d199')
    im2 = ax2.imshow(piv2.values, cmap='RdYlGn_r', vmin=-0.9, vmax=-0.6,
                     aspect='auto', origin='lower')
    ax2.set_xticks(range(len(piv2.columns)))
    ax2.set_xticklabels([f'{x:+d}' for x in piv2.columns])
    ax2.set_yticks(range(len(piv2.index)))
    ax2.set_yticklabels([f'{x:+d}' for x in piv2.index])
    for i in range(len(piv2.index)):
        for j in range(len(piv2.columns)):
            v = piv2.values[i, j]
            if not np.isnan(v):
                ax2.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=8,
                         fontweight='bold',
                         color='white' if v < -0.8 else 'black')
    ax2.set_xlabel('Upper boundary shift (m)', fontsize=10)
    ax2.set_ylabel('Lower boundary shift (m)', fontsize=10)
    ax2.set_title('(B) Δ¹⁹⁹Hg–FeOxide r', fontsize=12, fontweight='bold', loc='left')
    plt.colorbar(im2, ax=ax2, shrink=0.8, label='Pearson r')

    # (C) Histogram
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(df_sens['r_thg'], bins=12, edgecolor='black', alpha=0.7, color='forestgreen')
    ax3.axvline(cfg.ROBUSTNESS_THRESH, color='red', linestyle='--', linewidth=2,
                label=f'Threshold (r = {cfg.ROBUSTNESS_THRESH})')
    ax3.axvline(df_sens['r_thg'].mean(), color='blue', linestyle='-', linewidth=2,
                label=f'Mean (r = {df_sens["r_thg"].mean():.3f})')
    ax3.set_xlabel('THg–FeOxide correlation (r)', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title('(C) Distribution of THg–FeOxide r', fontsize=12, fontweight='bold', loc='left')
    ax3.legend(fontsize=8)
    ax3.set_xlim(0.8, 1.0)

    # (D) r vs sample size
    ax4 = fig.add_subplot(2, 2, 4)
    sc = ax4.scatter(df_sens['n'], df_sens['r_thg'], c=df_sens['r_d199'],
                     cmap='RdYlGn_r', s=100, edgecolors='black', linewidth=0.5,
                     vmin=-0.9, vmax=-0.6)
    ax4.axhline(cfg.ROBUSTNESS_THRESH, color='red', linestyle='--', linewidth=2,
                label=f'Threshold (r = {cfg.ROBUSTNESS_THRESH})')
    ref = df_sens[(df_sens['lower_shift'] == 0) & (df_sens['upper_shift'] == 0)].iloc[0]
    ax4.scatter([ref['n']], [ref['r_thg']], s=200, facecolors='none',
                edgecolors='red', linewidth=2.5, zorder=5, label='Reference')
    ax4.set_xlabel('Sample size (n)', fontsize=10)
    ax4.set_ylabel('THg–FeOxide r', fontsize=10)
    ax4.set_title('(D) Correlation vs Sample Size', fontsize=12, fontweight='bold', loc='left')
    ax4.legend(fontsize=8, loc='lower right')
    plt.colorbar(sc, ax=ax4, shrink=0.8, label='Δ¹⁹⁹Hg–FeOxide r')

    plt.tight_layout()
    _save_fig(fig, outdir, 'Figure_S5_Sensitivity_Analysis')


# =============================================================================
# UTILITIES
# =============================================================================

def _save_fig(fig, outdir, basename):
    """Save figure in PNG, PDF, and TIFF at 300 dpi."""
    for fmt in ['png', 'pdf', 'tiff']:
        path = os.path.join(outdir, f'{basename}.{fmt}')
        fig.savefig(path, dpi=Config.DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def write_report(outdir, sampling, sync, corr_results, d200, ll_results, df_sens, cfg):
    """Write a plain-text summary report with all verified numbers."""
    path = os.path.join(outdir, 'Statistical_Analysis_Report.txt')
    with open(path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MERCURY ISOTOPE STATISTICAL ANALYSIS — VERIFIED REPORT\n")
        f.write("Ediacaran-Cambrian Boundary, Wuhe Section, South China\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Version: 3.0.0 (Verified)\n")
        f.write(f"Seed: {cfg.RANDOM_SEED} | Bootstrap: n = {cfg.N_BOOTSTRAP}\n")
        f.write(f"Boundaries: {cfg.LOWER_BOUNDARY} / {cfg.UPPER_BOUNDARY} m\n\n")

        f.write("1. SAMPLING\n" + "-" * 40 + "\n")
        f.write(f"  Samples: {sampling['n']}\n")
        f.write(f"  Range: {sampling['h_min']:.1f}–{sampling['h_max']:.1f} m\n")
        f.write(f"  Study interval: {sampling['study_interval']:.1f} m\n")
        f.write(f"  Mean sampling: {sampling['mean']:.2f} m\n\n")

        f.write("2. SYNCHRONEITY\n" + "-" * 40 + "\n")
        f.write(f"  Consensus lower: {sync['consensus_lower']:.1f} m\n")
        f.write(f"  Consensus upper: {sync['consensus_upper']:.1f} m\n")
        f.write(f"  Lower spread: {sync['lower_spread']:.1f} m "
                f"({sync['lower_pct']:.1f}%)\n")
        f.write(f"  Upper spread: {sync['upper_spread']:.1f} m "
                f"({sync['upper_pct']:.1f}%)\n\n")

        f.write("3. UNIT-SPECIFIC CORRELATIONS\n" + "-" * 40 + "\n")
        for unit in ['Unit I', 'Unit II', 'Unit III']:
            u = corr_results['units'][unit]
            f.write(f"  {unit} (n = {u['n']}):\n")
            for label in ['THg-FeOx', 'd202-FeOx', 'D199-FeOx']:
                if label in u:
                    r = u[label]['r']
                    p = u[label]['p']
                    if np.isnan(r):
                        f.write(f"    {label}: NaN\n")
                    else:
                        f.write(f"    {label}: r = {r:.3f} (p = {p:.4f})\n")

        f.write(f"\n4. Δ200Hg\n" + "-" * 40 + "\n")
        f.write(f"  ANOVA: F = {d200['anova']['F']:.2f}, p = {d200['anova']['p']:.4f}\n")
        f.write(f"  K-W:   H = {d200['kruskal']['H']:.2f}, p = {d200['kruskal']['p']:.4f}\n\n")

        f.write("5. LEAD-LAG\n" + "-" * 40 + "\n")
        for t in ['THg (ppb)', 'δ202Hg', 'Δ199Hg']:
            r = ll_results[t]
            f.write(f"  {t}: lag = {r['opt_lag']:.1f} m, "
                    f"r = {r['max_r']:.3f}\n")

        f.write(f"\n6. SENSITIVITY\n" + "-" * 40 + "\n")
        f.write(f"  Scenarios: {len(df_sens)}\n")
        f.write(f"  THg–FeOx: mean = {df_sens['r_thg'].mean():.3f}, "
                f"min = {df_sens['r_thg'].min():.3f}, "
                f"max = {df_sens['r_thg'].max():.3f}\n")
        f.write(f"  All > {cfg.ROBUSTNESS_THRESH}: "
                f"{'PASSED' if (df_sens['r_thg'] > cfg.ROBUSTNESS_THRESH).all() else 'FAILED'}\n")

    print(f"  Report saved: {path}")


# =============================================================================
# MAIN
# =============================================================================

def main(data_file=None, output_dir=None):
    """Run the complete analysis pipeline."""

    print("\n" + "=" * 70)
    print("MERCURY ISOTOPE STATISTICAL ANALYSIS — VERIFIED")
    print("Version 3.0.0 | Seed 42 | Boundaries 138.9 / 146.3 m")
    print("=" * 70)

    cfg = Config()
    np.random.seed(cfg.RANDOM_SEED)

    if data_file is None:
        data_file = 'AnwenMS_Hg_isotope_Data_Table.xlsx'
    if output_dir is None:
        output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(data_file):
        print(f"\nError: Data file not found: {data_file}")
        print("Please provide the path with --data_file")
        return

    # 1. Load data
    df, df_outer, sampling = load_data(data_file)
    df = assign_units(df, cfg.LOWER_BOUNDARY, cfg.UPPER_BOUNDARY)

    # 2. Breakpoint analysis
    bp_results, sync = run_breakpoint_analysis(df, cfg)

    # 3. Lead-lag analysis
    ll_results = run_lead_lag(df, cfg)

    # 4. Correlations
    corr_results = run_correlations(df, cfg)

    # 5. Δ200Hg
    d200 = run_delta200(df)

    # 6. Sensitivity analysis
    df_sens = run_sensitivity(df_outer, cfg)

    # 7. Generate figures
    print("\n" + "=" * 70)
    print("7. GENERATING FIGURES")
    print("=" * 70)

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1, 'ytick.major.width': 1,
    })

    figure_s1(df, bp_results, sync, cfg, output_dir)
    figure_s2(df, cfg, output_dir)
    figure_s3(df, d200, cfg, output_dir)
    figure_s4(ll_results, cfg, output_dir)
    figure_s5(df_sens, cfg, output_dir)

    # 8. Report
    write_report(output_dir, sampling, sync, corr_results, d200, ll_results, df_sens, cfg)

    # 9. Save sensitivity results as CSV
    df_sens.to_csv(os.path.join(output_dir, 'sensitivity_results.csv'), index=False)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs in: {output_dir}/")
    print("  Figure_S1_Breakpoint_Analysis.png/pdf/tiff")
    print("  Figure_S2_Unit_Correlations.png/pdf/tiff")
    print("  Figure_S3_Delta200Hg.png/pdf/tiff")
    print("  Figure_S4_Lead_Lag.png/pdf/tiff")
    print("  Figure_S5_Sensitivity_Analysis.png/pdf/tiff")
    print("  Statistical_Analysis_Report.txt")
    print("  sensitivity_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hg isotope statistical analysis — Verified v3.0.0"
    )
    parser.add_argument("--data_file", "-d", type=str,
                        default="AnwenMS_Hg_isotope_Data_Table.xlsx",
                        help="Path to the Excel data file")
    parser.add_argument("--output_dir", "-o", type=str,
                        default="output",
                        help="Output directory")
    args = parser.parse_args()
    main(data_file=args.data_file, output_dir=args.output_dir)
