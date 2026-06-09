"""
Shared visualization functions for regime outputs and backtest results.

Key design decision: all plot functions accept a `recession_state` parameter
(default 1) so the same function works for both HMM (recession = state 1)
and MSVAR (recession = state 0) without duplication.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .constants import COLOR_EXP, COLOR_REC, SHADE_COL, SHADE_A, LW

# NBER-dated U.S. recession periods used as reference shading (1990–2025 sample)
NBER_RECESSIONS = [
    ("1990-07-01", "1991-03-31"),   # Gulf War recession
    ("2001-03-01", "2001-11-30"),   # Dot-com bust
    ("2007-12-01", "2009-06-30"),   # Global Financial Crisis
    ("2020-02-01", "2020-04-30"),   # COVID-19 recession
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _contiguous_blocks(mask):
    """Return (start, end) index pairs for contiguous True runs in a boolean mask."""
    blocks, in_block = [], False
    for i, val in enumerate(mask):
        if val and not in_block:
            start, in_block = i, True
        elif not val and in_block:
            blocks.append((start, i - 1))
            in_block = False
    if in_block:
        blocks.append((start, len(mask) - 1))
    return blocks


def _fmt_matrix(M, labels):
    """Pretty-print a square matrix with row/column labels."""
    w      = max(len(l) for l in labels) + 2
    header = ' ' * w + ''.join(f'{l:>{w}}' for l in labels)
    rows   = [header]
    for i, label in enumerate(labels):
        row = f'{label:>{w}}' + ''.join(f'{M[i,j]:>{w}.5f}' for j in range(len(labels)))
        rows.append(row)
    return '\n'.join(rows)


def _state_colors(recession_state: int = 1):
    """Return a list of colors indexed by state, recession_state gets COLOR_REC."""
    colors = [COLOR_EXP, COLOR_EXP]
    colors[recession_state] = COLOR_REC
    colors[1 - recession_state] = COLOR_EXP
    return colors


def _style_ax(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.4)
    ax.margins(x=0.01)


# ---------------------------------------------------------------------------
# Regime sequence and probability plots
# ---------------------------------------------------------------------------

def plot_regime_states(dates, states, title="Regime States", ax=None):
    """Plot the inferred regime state sequence as a step function."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, states, drawstyle='steps-post', marker='o')
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Inferred State")
    ax.set_yticks(np.unique(states))
    ax.grid(True)
    plt.tight_layout()
    return ax


def plot_regime_probs(dates, probs, title="State Probabilities", ax=None):
    """
    Plot regime probabilities over time.

    Parameters
    ----------
    probs : (K, T) array of state probabilities
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    probs_df = pd.DataFrame(
        probs.T, index=dates,
        columns=[f"State {i}" for i in range(probs.shape[0])]
    )
    probs_df.plot(ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Series visualization with regime coloring / shading
# ---------------------------------------------------------------------------

def plot_series_colored(df, series_name, state_col, recession_state: int = 1, ax=None):
    """
    Plot a series with line color switching by regime.

    Parameters
    ----------
    df             : pd.DataFrame with DatetimeIndex
    series_name    : column to plot
    state_col      : column containing state labels (0 or 1)
    recession_state: which state index represents recession (gets COLOR_REC)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    dates  = df.index
    series = df[series_name].values
    states = df[state_col].values
    colors = _state_colors(recession_state)

    change_idx = np.where(np.diff(states) != 0)[0] + 1
    seg_starts = np.concatenate([[0], change_idx])
    seg_ends   = np.concatenate([change_idx, [len(states)]])

    for s, e in zip(seg_starts, seg_ends):
        end_ext = min(e + 1, len(states))
        ax.plot(dates[s:end_ext], series[s:end_ext],
                color=colors[states[s]], linewidth=LW)

    rec_label = f"State {recession_state} — Recession"
    exp_label = f"State {1 - recession_state} — Expansion"
    ax.legend(handles=[
        Line2D([0], [0], color=COLOR_EXP, lw=2, label=exp_label),
        Line2D([0], [0], color=COLOR_REC, lw=2, label=rec_label),
    ], loc='best', framealpha=0.9)
    return ax


def plot_series_shaded(df, series_name, state_col, recession_state: int = 1, ax=None):
    """
    Plot a series in black with recession periods shaded gray.

    Parameters
    ----------
    recession_state : which state index to shade (1 for HMM, 0 for MSVAR)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    dates  = df.index
    series = df[series_name].values
    states = df[state_col].values

    ax.plot(dates, series, color='#222222', linewidth=LW, zorder=3)

    for start_i, end_i in _contiguous_blocks(states == recession_state):
        ax.axvspan(dates[start_i], dates[end_i],
                   color=SHADE_COL, alpha=SHADE_A, zorder=2)

    ax.legend(handles=[
        Line2D([0], [0], color='#222222', lw=2, label=series_name),
        Patch(facecolor=SHADE_COL, alpha=SHADE_A,
              label=f'State {recession_state} — Recession'),
    ], loc='best', framealpha=0.9)
    return ax


# ---------------------------------------------------------------------------
# Backtest dashboard
# ---------------------------------------------------------------------------

def plot_backtest(backtest: dict, state_series=None):
    """
    Four-panel backtest dashboard: cumulative wealth, rolling Sharpe,
    drawdown, and monthly return distribution.

    Parameters
    ----------
    backtest     : dict with keys 'cum_returns' and 'returns'
    state_series : optional pd.Series of {0,1} — shades recession periods
    """
    cum  = backtest['cum_returns']
    rets = backtest['returns']

    strategy_colors = {
        'HMM Dynamic':        '#1a6faf',
        'Equal Weight (1/n)': '#e07b00',
        '60/40':              '#444444',
    }
    col_colors = {}
    for col in cum.columns:
        matched = next((c for k, c in strategy_colors.items() if k in col), '#888888')
        col_colors[col] = matched

    fig = plt.figure(figsize=(14, 11))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.32)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, :])

    # Panel 1: cumulative wealth
    for col in cum.columns:
        ax1.plot(cum.index, cum[col], label=col, color=col_colors[col], lw=1.8)
    if state_series is not None:
        _shade_recessions(ax1, state_series)
    ax1.set_title('Cumulative Wealth  (start = 1)', fontweight='bold')
    ax1.set_ylabel('Portfolio Value')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.axhline(1, color='gray', lw=0.7, ls='--')
    _style_ax(ax1)

    # Panel 2: rolling 12-month Sharpe
    window = 12
    for col in rets.columns:
        roll_sh = (rets[col].rolling(window).mean() /
                   rets[col].rolling(window).std().clip(1e-12)) * np.sqrt(window)
        ax2.plot(rets.index, roll_sh, label=col, color=col_colors[col], lw=1.4)
    ax2.axhline(0, color='gray', lw=0.7, ls='--')
    ax2.set_title(f'Rolling {window}-Month Sharpe', fontweight='bold')
    ax2.set_ylabel('Sharpe Ratio')
    _style_ax(ax2)

    # Panel 3: drawdown
    for col in cum.columns:
        roll_max = cum[col].expanding().max()
        dd = (cum[col] / roll_max - 1) * 100
        ax3.fill_between(cum.index, dd, 0, color=col_colors[col], alpha=0.35, label=col)
        ax3.plot(cum.index, dd, color=col_colors[col], lw=0.8)
    ax3.set_title('Drawdown (%)', fontweight='bold')
    ax3.set_ylabel('Drawdown (%)')
    _style_ax(ax3)

    # Panel 4: return distribution
    data_bp = [rets[col].dropna().values * 100 for col in rets.columns]
    bp = ax4.boxplot(data_bp, patch_artist=True, widths=0.5,
                     medianprops=dict(color='black', lw=2))
    for patch, col in zip(bp['boxes'], rets.columns):
        patch.set_facecolor(col_colors[col])
        patch.set_alpha(0.7)
    ax4.set_xticklabels(rets.columns, fontsize=9)
    ax4.axhline(0, color='gray', lw=0.7, ls='--')
    ax4.set_title('Monthly Return Distribution (%)', fontweight='bold')
    ax4.set_ylabel('Return (%)')
    _style_ax(ax4)

    fig.suptitle('Backtest Results', fontsize=14, fontweight='bold', y=1.01)
    return fig


def plot_regime_nber(df, series_name, state_col, recession_state: int = 1,
                     title: str = None, ylabel: str = None, ax=None):
    """
    Plot a series with two overlapping visual cues:
    1. Line color switches by model-classified regime (blue = expansion, red = recession).
    2. NBER-dated recessions are shaded light gray in the background.

    Parameters
    ----------
    df             : pd.DataFrame with DatetimeIndex
    series_name    : column to plot
    state_col      : column containing integer state labels (0 or 1)
    recession_state: state index treated as recession (gets COLOR_REC line color)
    title          : optional axes title
    ylabel         : optional y-axis label (defaults to series_name)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    dates  = df.index
    series = df[series_name].values
    states = df[state_col].values
    colors = _state_colors(recession_state)

    # Colored line segments by classified regime
    change_idx = np.where(np.diff(states) != 0)[0] + 1
    seg_starts = np.concatenate([[0], change_idx])
    seg_ends   = np.concatenate([change_idx, [len(states)]])
    for s, e in zip(seg_starts, seg_ends):
        end_ext = min(e + 1, len(states))
        ax.plot(dates[s:end_ext], series[s:end_ext],
                color=colors[states[s]], linewidth=LW)

    # NBER recession shading (gray background bands)
    nber_added = False
    for rec_start, rec_end in NBER_RECESSIONS:
        rs, re = pd.Timestamp(rec_start), pd.Timestamp(rec_end)
        if re < dates[0] or rs > dates[-1]:
            continue
        ax.axvspan(max(rs, dates[0]), min(re, dates[-1]),
                   color='#d4d4d4', alpha=0.6, zorder=0,
                   label='NBER Recession' if not nber_added else '_nolegend_')
        nber_added = True

    if title:
        ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel or series_name)
    _style_ax(ax)

    handles = [
        Line2D([0], [0], color=COLOR_EXP, lw=2, label='Expansion (model)'),
        Line2D([0], [0], color=COLOR_REC, lw=2, label='Recession (model)'),
        Patch(facecolor='#d4d4d4', alpha=0.6, label='NBER Recession'),
    ]
    ax.legend(handles=handles, loc='best', framealpha=0.9)
    return ax


def _shade_recessions(ax, state_series, recession_state: int = 1):
    """Shade recession periods on an axes object."""
    states = state_series.values
    dates  = state_series.index
    in_rec = False
    for i, s in enumerate(states):
        if s == recession_state and not in_rec:
            start, in_rec = dates[i], True
        elif s != recession_state and in_rec:
            ax.axvspan(start, dates[i - 1], color='#cccccc', alpha=0.4, zorder=0)
            in_rec = False
    if in_rec:
        ax.axvspan(start, dates[-1], color='#cccccc', alpha=0.4, zorder=0)
