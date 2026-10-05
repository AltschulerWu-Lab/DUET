"""Shared utilities for NIS-seq error analysis.

Functions for loading Brunello library data, loading spot sequence files,
and BLAS-accelerated library matching.

Alphabet convention: ATCG ordering (A=0, T=1, C=2, G=3) throughout,
matching DNAEncoder in duet.codebook_evaluator.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize

from duet.codebook_evaluator import DNAEncoder, HammingDistance, one_hot_encode
from duet.plotting import DNA_BASE_PALETTE, FIGURE_WIDTHS

# Base-to-index mapping (ATCG ordering, matches DNAEncoder)
BASE_TO_IDX = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
IDX_TO_BASE = {0: 'A', 1: 'T', 2: 'C', 3: 'G'}
NUM_BASES = 4
COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

# ---------------------------------------------------------------------------
# Task 1: Library and Data Loading
# ---------------------------------------------------------------------------

def load_brunello_library(path: str | Path) -> pd.DataFrame:
    """Load Brunello sgRNA library and compute reverse complements.

    Args:
        path: Path to Brunello_sgRNAs.txt (tab-separated, no header,
              columns: Gene, Sequence).

    Returns:
        DataFrame with columns: Gene, Sequence, ReverseComplement, ID.
        ID is Gene_N where N is the occurrence count within that gene.
    """
    df = pd.read_csv(path, sep='\t', names=['Gene', 'Sequence'])
    df['ReverseComplement'] = df['Sequence'].apply(
        lambda x: ''.join(COMPLEMENT[b] for b in reversed(x))
    )
    df['ID'] = df['Gene'] + '_' + (df.groupby('Gene').cumcount() + 1).astype(str)
    return df


def load_spot_data(path: str | Path, has_header: bool = False) -> pd.DataFrame:
    """Load spot sequence data from a tab-separated file.

    Args:
        path: Path to spot data file.
        has_header: If True, first row is a header. If False (default),
                    columns are assigned as [tile, x, y, sequence, intensity].

    Returns:
        DataFrame with columns: tile, x, y, sequence, intensity.
    """
    if has_header:
        return pd.read_csv(path, sep='\t')
    else:
        return pd.read_csv(
            path, sep='\t',
            names=['tile', 'x', 'y', 'sequence', 'intensity'],
            header=None,
        )


def filter_valid_reads(df: pd.DataFrame, seq_length: int = 14) -> pd.DataFrame:
    """Filter reads to valid length and ATCG-only bases.

    Args:
        df: DataFrame with 'sequence' column.
        seq_length: Expected sequence length (default 14 for Brunello).

    Returns:
        Filtered DataFrame (copy). Prints how many reads were dropped.
    """
    n_total = len(df)

    # Filter by length
    valid_length = df['sequence'].str.len() == seq_length
    n_wrong_length = (~valid_length).sum()

    # Filter by valid bases (ATCG only)
    valid_bases = df['sequence'].str.match(r'^[ATCG]+$')
    n_invalid_bases = (valid_length & ~valid_bases).sum()

    mask = valid_length & valid_bases
    result = df[mask].copy()

    if n_wrong_length > 0 or n_invalid_bases > 0:
        print(f"  Filtered: {n_wrong_length} wrong length, "
              f"{n_invalid_bases} invalid bases, "
              f"{len(result)}/{n_total} kept ({len(result)/n_total:.1%})")

    return result


# ---------------------------------------------------------------------------
# Task 2: BLAS-Accelerated Matching
# ---------------------------------------------------------------------------

def encode_sequences(sequences: list[str]) -> np.ndarray:
    """Encode DNA sequences to integer array using ATCG ordering (A=0,T=1,C=2,G=3).

    Uses vectorized numpy operations for speed. For 32M sequences of length 14,
    DNAEncoder.encode() takes hours (nested Python loops); this runs in seconds.

    Args:
        sequences: List of DNA strings (all same length, ATCG only).

    Returns:
        numpy array of shape (N, seq_length), dtype int8.
    """
    seq_length = len(sequences[0])
    # Convert all sequences to a single ASCII byte array, then reshape
    raw = np.frombuffer(''.join(sequences).encode('ascii'), dtype=np.uint8)
    raw = raw.reshape(-1, seq_length)

    # Vectorized lookup: A(65)->0, T(84)->1, C(67)->2, G(71)->3
    lookup = np.zeros(256, dtype=np.int8)
    lookup[ord('A')] = 0
    lookup[ord('T')] = 1
    lookup[ord('C')] = 2
    lookup[ord('G')] = 3
    return lookup[raw]


def match_to_library(
    read_sequences: np.ndarray,
    precomputed,
    chunk_size: int = 10000,
) -> dict:
    """Match reads to library using BLAS-accelerated Hamming distance.

    Args:
        read_sequences: Encoded read sequences, shape (M, L), dtype int8.
        precomputed: Precomputed BLAS state from
                     HammingDistance.precompute_transmitted(library_encoded).
        chunk_size: Number of reads to process per BLAS chunk.

    Returns:
        dict with keys:
            'match_index': int32 array (M,) -- index of best-matching library entry
            'best_distance': int8 array (M,) -- Hamming distance to best match
            'second_best_distance': int8 array (M,) -- distance to second-best match
            'tied_indices': dict mapping read index (int) to list of tied library
                           indices, only for reads with ties (distance == best for
                           multiple library entries)
    """
    M = read_sequences.shape[0]

    match_index = np.empty(M, dtype=np.int32)
    best_distance = np.empty(M, dtype=np.int8)
    second_best_distance = np.empty(M, dtype=np.int8)
    tied_indices = {}

    hamming = HammingDistance(alphabet_size=NUM_BASES)

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        chunk = read_sequences[start:end]

        # BLAS: phi_obs @ psi_tx.T gives Hamming distances
        distances = hamming.compute_with_precomputed(chunk, precomputed)

        # Best match
        best_idx = np.argmin(distances, axis=1)
        best_dist = distances[np.arange(len(chunk)), best_idx]

        # Second-best: use partition for efficiency
        partitioned = np.partition(distances, 1, axis=1)
        second_best_dist = partitioned[:, 1]

        match_index[start:end] = best_idx
        best_distance[start:end] = best_dist.astype(np.int8)
        second_best_distance[start:end] = np.clip(
            second_best_dist, 0, 127
        ).astype(np.int8)

        # Identify ties: a read has ties iff second_best == best
        tie_local_indices = np.where(second_best_dist == best_dist)[0]
        for local_i in tie_local_indices:
            global_i = start + local_i
            tie_mask = distances[local_i] == best_dist[local_i]
            tied_indices[global_i] = np.where(tie_mask)[0].tolist()

    return {
        'match_index': match_index,
        'best_distance': best_distance,
        'second_best_distance': second_best_distance,
        'tied_indices': tied_indices,
    }


# ---------------------------------------------------------------------------
# Task 3: Error Statistics Computation
# ---------------------------------------------------------------------------

@dataclass
class ErrorStatistics:
    """Container for error analysis results."""
    per_position_error_rates: np.ndarray      # shape (L,)
    per_position_base_errors: np.ndarray      # shape (L, 4) -- per-base error rate at each position
    channel_matrix: np.ndarray                 # shape (4, 4) -- aggregated transition matrix
    positional_channel_matrices: np.ndarray    # shape (L, 4, 4)
    positional_channel_counts: np.ndarray      # shape (L, 4, 4) -- unnormalized counts feeding positional_channel_matrices
    n_reads: int


def compute_error_statistics(
    read_sequences: np.ndarray,
    library_encoded: np.ndarray,
    match_result: dict,
    intensity: np.ndarray,
    intensity_threshold: float = 0.0,
) -> ErrorStatistics:
    """Compute per-position error rates and transition matrices.

    Uses vectorized NumPy for the common case (unambiguous matches, ~83%
    of reads). Tied reads are handled in a separate Python loop but are
    a small minority.

    Args:
        read_sequences: Encoded read sequences, shape (M, L), dtype int8.
        library_encoded: Encoded library, shape (N, L), dtype int8.
        match_result: Output from match_to_library().
        intensity: Intensity values, shape (M,).
        intensity_threshold: Minimum intensity to include a read.

    Returns:
        ErrorStatistics with per-position errors, channel matrix, etc.
    """
    L = read_sequences.shape[1]

    # Apply intensity threshold
    mask = intensity >= intensity_threshold
    indices = np.where(mask)[0]
    n_reads = len(indices)

    if n_reads == 0:
        return ErrorStatistics(
            per_position_error_rates=np.zeros(L),
            per_position_base_errors=np.zeros((L, NUM_BASES)),
            channel_matrix=np.zeros((NUM_BASES, NUM_BASES)),
            positional_channel_matrices=np.zeros((L, NUM_BASES, NUM_BASES)),
            positional_channel_counts=np.zeros((L, NUM_BASES, NUM_BASES)),
            n_reads=0,
        )

    tied = match_result['tied_indices']

    # Split into unambiguous and tied reads
    tied_set = set(tied.keys())
    is_tied = np.array([idx in tied_set for idx in indices])
    untied_indices = indices[~is_tied]
    tied_indices_arr = indices[is_tied]

    # --- Vectorized path for unambiguous reads ---
    transition_counts = np.zeros((NUM_BASES, NUM_BASES), dtype=np.float64)
    pos_transition_counts = np.zeros((L, NUM_BASES, NUM_BASES), dtype=np.float64)

    if len(untied_indices) > 0:
        obs_seqs = read_sequences[untied_indices]          # (U, L)
        lib_idxs = match_result['match_index'][untied_indices]
        true_seqs = library_encoded[lib_idxs]              # (U, L)

        # Transition counts: for each position, count (true_base, obs_base) pairs
        for pos in range(L):
            true_col = true_seqs[:, pos]
            obs_col = obs_seqs[:, pos]
            np.add.at(transition_counts, (true_col, obs_col), 1.0)
            np.add.at(pos_transition_counts[pos], (true_col, obs_col), 1.0)

    # --- Slow path for tied reads (small minority) ---
    for idx in tied_indices_arr:
        read_seq = read_sequences[idx]
        lib_indices_list = tied[idx]
        weight = 1.0 / len(lib_indices_list)
        for lib_idx in lib_indices_list:
            lib_seq = library_encoded[lib_idx]
            for pos in range(L):
                true_base = lib_seq[pos]
                obs_base = read_seq[pos]
                transition_counts[true_base, obs_base] += weight
                pos_transition_counts[pos, true_base, obs_base] += weight

    # --- Derive all statistics from transition counts ---
    pos_error_counts = np.zeros(L, dtype=np.float64)
    for pos in range(L):
        pos_error_counts[pos] = pos_transition_counts[pos].sum() - np.trace(pos_transition_counts[pos])
    per_position_error_rates = pos_error_counts / n_reads

    pos_base_error_counts = np.zeros((L, NUM_BASES), dtype=np.float64)
    for pos in range(L):
        for obs_base in range(NUM_BASES):
            pos_base_error_counts[pos, obs_base] = (
                pos_transition_counts[pos, :, obs_base].sum()
                - pos_transition_counts[pos, obs_base, obs_base]
            )
    per_position_base_errors = pos_base_error_counts / n_reads

    # Normalize transition matrices (row-stochastic)
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    channel_matrix = np.divide(
        transition_counts, row_sums,
        out=np.zeros_like(transition_counts),
        where=row_sums > 0,
    )

    pos_row_sums = pos_transition_counts.sum(axis=2, keepdims=True)
    positional_channel_matrices = np.divide(
        pos_transition_counts, pos_row_sums,
        out=np.zeros_like(pos_transition_counts),
        where=pos_row_sums > 0,
    )

    return ErrorStatistics(
        per_position_error_rates=per_position_error_rates,
        per_position_base_errors=per_position_base_errors,
        channel_matrix=channel_matrix,
        positional_channel_matrices=positional_channel_matrices,
        positional_channel_counts=pos_transition_counts,
        n_reads=n_reads,
    )


# ---------------------------------------------------------------------------
# Task 4: Plotting Helpers
# ---------------------------------------------------------------------------

def plot_intensity_distribution(df: pd.DataFrame, title: str = "Intensity Distribution",
                                 ax=None):
    """Plot histogram of read intensities."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(FIGURE_WIDTHS["two_thirds_page"], 3.5))
    sns.histplot(data=df, x='intensity', bins=100, ax=ax)
    ax.set_xlabel('Read Intensity')
    ax.set_ylabel('Count')
    ax.set_title(title)
    return ax


def plot_per_position_error_rates(error_rates: np.ndarray, label: str = None,
                                   ax=None, show_average: bool = True, **kwargs):
    """Plot per-position error rates as a line plot.

    When label is None (single-line mode), draws a labeled gray average line.
    When label is set (multi-line overlay mode), draws the average line in the
    same color as the data line so it's clear which average belongs to which series.
    Set show_average=False to suppress the average line entirely.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(FIGURE_WIDTHS["half_page"], 2.5))
    positions = np.arange(1, len(error_rates) + 1)
    line, = ax.plot(positions, error_rates, marker='o', markersize=6, label=label, **kwargs)
    avg = error_rates.mean()
    if show_average:
        if label is None:
            # Single-line mode: gray average with label
            ax.axhline(y=avg, color='gray', linestyle=':', linewidth=1.5,
                       label=f'Average ({avg:.4f})')
        else:
            # Multi-line mode: match the data line color, no legend entry
            ax.axhline(y=avg, color=line.get_color(), linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('Sequence Position')
    ax.set_ylabel('Error Rate')
    ax.set_ylim(bottom=0)
    if label is not None:
        ax.legend()
    return ax


def plot_channel_matrix(matrix: np.ndarray, title: str = "Channel Transition Matrix",
                        ax=None):
    """Plot 4x4 channel transition matrix as a heatmap."""
    labels = ['A', 'T', 'C', 'G']
    if ax is None:
        fig, ax = plt.subplots(figsize=(FIGURE_WIDTHS["two_thirds_page"], 4.0))
    sns.heatmap(
        matrix, annot=True, fmt='.4f', cmap='viridis',
        xticklabels=labels, yticklabels=labels, ax=ax,
        cbar_kws={'label': 'P(observed | true)'},
    )
    ax.set_xlabel('Observed Base')
    ax.set_ylabel('True Base')
    ax.set_title(title)
    ax.set_yticklabels(labels, rotation=0)
    return ax


def plot_positional_channel_matrices(matrices: np.ndarray, ncols: int = 7):
    """Plot L x 4 x 4 positional channel matrices as a grid of heatmaps.

    Cell values are annotated to 2 decimals. A shared colorbar is placed to
    the right of the grid. ATCG tick labels are colored with DNA_BASE_PALETTE
    so the source-base key is consistent across the notebook's publication
    figures.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    L = matrices.shape[0]
    labels = ['A', 'T', 'C', 'G']
    source_colors = [DNA_BASE_PALETTE[b] for b in labels]
    nrows = (L + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3 * ncols + 1.5, 3 * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    for pos in range(L):
        row, col = divmod(pos, ncols)
        ax = axes[row, col]
        sns.heatmap(
            matrices[pos], annot=True, fmt='.2f', cmap='viridis',
            xticklabels=labels, yticklabels=labels, ax=ax,
            cbar=False, vmin=0, vmax=1,
        )
        ax.set_title(f'Position {pos + 1}')
        ax.set_yticklabels(labels, rotation=0, fontsize=8)
        ax.set_xticklabels(labels, fontsize=8)
        # Color ATCG tick labels to match the per-true-base palette
        for tick_label, color in zip(ax.get_yticklabels(), source_colors):
            tick_label.set_color(color)
        for tick_label, color in zip(ax.get_xticklabels(), source_colors):
            tick_label.set_color(color)

    # Hide unused axes
    for pos in range(L, nrows * ncols):
        row, col = divmod(pos, ncols)
        axes[row, col].set_visible(False)

    # Shared colorbar anchored to the full grid
    sm = ScalarMappable(cmap='viridis', norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    visible_axes = [ax for ax in axes.ravel() if ax.get_visible()]
    cbar = fig.colorbar(sm, ax=visible_axes, shrink=0.7, aspect=30, pad=0.015)
    cbar.set_label('P(observed | true)', fontsize=11, labelpad=6)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle('Positional Channel Matrices', fontsize=14)
    return fig


def plot_per_base_error_rates(per_position_base_errors: np.ndarray, ax=None):
    """Plot per-position error rates broken down by observed base.

    Args:
        per_position_base_errors: shape (L, 4) -- error rate for each base at each position.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(FIGURE_WIDTHS["half_page"], 2.5))
    labels = ['A', 'T', 'C', 'G']
    colors = [DNA_BASE_PALETTE[base] for base in ("A", "T", "C", "G")]
    positions = np.arange(1, per_position_base_errors.shape[0] + 1)

    for base_idx in range(NUM_BASES):
        ax.plot(positions, per_position_base_errors[:, base_idx],
                marker='o', markersize=5, label=f'{labels[base_idx]} errors',
                color=colors[base_idx])

    ax.set_xlabel('Sequence Position')
    ax.set_ylabel('Error Rate')
    ax.set_title('Per-Position Error Rates by Base')
    ax.set_ylim(bottom=0)
    ax.legend()
    return ax


# ---------------------------------------------------------------------------
# Publication-quality figure: position x channel error structure
# ---------------------------------------------------------------------------

def plot_position_channel_error_heatmap(
    positional_channel_matrices: np.ndarray,
    per_position_error_rates: Optional[np.ndarray] = None,
    figsize: tuple = (FIGURE_WIDTHS["full_page"], 5.4),
    cmap: str = 'rocket_r',
    log_scale: bool = True,
    title: Optional[str] = None,
    group_colors: Optional[list] = None,
):
    """Publication-quality heatmap of position x channel errors.

    Arranges all 12 off-diagonal transitions (4 true bases x 3 destinations)
    as rows and sequence positions as columns, grouped by true base. Includes
    a top marginal showing per-position error rate, a right marginal showing
    mean error rate per transition across positions, and a log-scale colorbar.

    Args:
        positional_channel_matrices: (L, 4, 4) row-stochastic matrices. Entry
            [p, i, j] is P(observed=j | true=i) at position p.
        per_position_error_rates: Optional (L,) empirical per-position error
            rate (weighted by base frequency). Falls back to the mean
            off-diagonal row-sum if not provided.
        figsize: figure size in inches.
        cmap: matplotlib/seaborn colormap name.
        log_scale: use log color scale (recommended — errors span ~1e-3 to ~5e-2).
        title: optional figure suptitle.
        group_colors: 4-element list of colors for the four true-base groups
            (used in the right marginal). Defaults to DNA_BASE_PALETTE (ATCG).

    Returns:
        (fig, axes_dict) where axes_dict contains 'main', 'top', 'right', 'cbar'.
    """
    labels = ['A', 'T', 'C', 'G']
    L = positional_channel_matrices.shape[0]
    if group_colors is None:
        group_colors = [DNA_BASE_PALETTE[b] for b in labels]

    # Build (12, L) error matrix, rows grouped by true base
    error_matrix = np.zeros((12, L), dtype=np.float64)
    row_labels = []
    for ti, true_base in enumerate(labels):
        within = 0
        for oi, obs_base in enumerate(labels):
            if ti == oi:
                continue
            error_matrix[ti * 3 + within] = positional_channel_matrices[:, ti, oi]
            row_labels.append(f'{true_base}→{obs_base}')
            within += 1

    # Marginals
    if per_position_error_rates is None:
        off_diag_sum = 1.0 - np.diagonal(
            positional_channel_matrices, axis1=1, axis2=2
        )  # (L, 4) off-diag prob per (pos, true base)
        per_position_error_rates = off_diag_sum.mean(axis=1)  # uniform-base-freq fallback
    per_transition_mean = error_matrix.mean(axis=1)

    # Color normalization
    positive_vals = error_matrix[error_matrix > 0]
    if positive_vals.size == 0:
        raise ValueError("positional_channel_matrices has no positive off-diagonal entries.")
    vmin = positive_vals.min()
    vmax = error_matrix.max()
    norm = LogNorm(vmin=vmin, vmax=vmax) if log_scale else Normalize(vmin=0.0, vmax=vmax)

    # Figure layout: 2x3 grid (top strip over main; right strip; colorbar)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        2, 3,
        width_ratios=[14.0, 2.2, 0.35],
        height_ratios=[1.4, 12.0],
        wspace=0.06, hspace=0.05,
        figure=fig,
    )
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_cbar = fig.add_subplot(gs[1, 2])

    # --- Main heatmap ---
    # Clip zeros to vmin when log-scaling so they render as the lightest color
    # (rather than emitting log(0) warnings or appearing as masked/bad values).
    plot_matrix = np.where(error_matrix > 0, error_matrix, vmin) if log_scale else error_matrix
    im = ax_main.imshow(
        plot_matrix, aspect='auto', cmap=cmap, norm=norm, interpolation='none',
    )
    ax_main.set_xticks(np.arange(L))
    ax_main.set_xticklabels(np.arange(1, L + 1), fontsize=9)
    ax_main.set_yticks(np.arange(12))
    ax_main.set_yticklabels(row_labels, fontsize=9, family='DejaVu Sans Mono')
    ax_main.set_xlabel('Sequence position', fontsize=11, labelpad=6)
    ax_main.tick_params(length=0, pad=3)
    for spine in ax_main.spines.values():
        spine.set_visible(False)

    # Group separators (white lines between the four true-base blocks)
    for y in (2.5, 5.5, 8.5):
        ax_main.axhline(y, color='white', linewidth=2.5, zorder=3)

    # True-base group labels on the far left (in axis-fraction coords)
    for i, base in enumerate(labels):
        ax_main.text(
            -0.085, i * 3 + 1, base,
            transform=ax_main.get_yaxis_transform(),
            ha='right', va='center',
            fontsize=13, fontweight='bold', color=group_colors[i],
        )
    # Left banner label ("true base") rotated
    ax_main.text(
        -0.115, 5.5, 'True base',
        transform=ax_main.get_yaxis_transform(),
        ha='right', va='center', rotation=90,
        fontsize=10, color='#333333',
    )

    # --- Top marginal: per-position error rate ---
    ax_top.bar(
        np.arange(L), per_position_error_rates,
        color='#4a4a4a', edgecolor='none', width=0.82,
    )
    ax_top.set_ylabel('Per-pos.\nerror rate', fontsize=8, labelpad=4)
    ax_top.tick_params(axis='y', labelsize=7, length=2)
    ax_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    for spine in ('top', 'right', 'bottom'):
        ax_top.spines[spine].set_visible(False)
    ax_top.spines['left'].set_linewidth(0.6)
    ax_top.spines['left'].set_color('#666666')
    ax_top.set_ylim(bottom=0)

    # --- Right marginal: mean transition prob across positions ---
    bar_colors_right = [group_colors[i // 3] for i in range(12)]
    ax_right.barh(
        np.arange(12), per_transition_mean,
        color=bar_colors_right, edgecolor='none', height=0.82,
    )
    # sharey with ax_main already inverts ylim (imshow origin='upper' puts row 0
    # at the top). Do NOT call ax_right.invert_yaxis() — it would propagate back
    # through sharey and flip ax_main.
    ax_right.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax_right.set_xlabel('Mean across\npositions', fontsize=8, labelpad=4)
    ax_right.tick_params(axis='x', labelsize=7, length=2)
    for spine in ('top', 'right', 'left'):
        ax_right.spines[spine].set_visible(False)
    ax_right.spines['bottom'].set_linewidth(0.6)
    ax_right.spines['bottom'].set_color('#666666')
    ax_right.set_xlim(left=0)

    # --- Colorbar ---
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label('P(observed | true)', fontsize=9, labelpad=6)
    cbar.ax.tick_params(labelsize=8, length=2)
    cbar.outline.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=12, y=0.995)

    return fig, {'main': ax_main, 'top': ax_top, 'right': ax_right, 'cbar': ax_cbar}


# ---------------------------------------------------------------------------
# Publication-quality figure: position x channel error structure (line form)
# ---------------------------------------------------------------------------

def plot_position_channel_error_lines(
    positional_channel_matrices: np.ndarray,
    figsize: tuple = (FIGURE_WIDTHS["half_page"], 2.7),
    source_colors: Optional[list] = None,
    title: Optional[str] = None,
    log_scale: bool = True,
):
    """Grouped line plot of position x channel errors.

    12 lines (one per off-diagonal transition). Color encodes the true (source)
    base; line style and marker encode destination rank within each source
    (solid/circle = 1st destination, dashed/square = 2nd, dotted/triangle = 3rd;
    ordering is ATCG with self removed). Each line is labeled at its right
    endpoint with the full X->Y transition for unambiguous identification.

    Args:
        positional_channel_matrices: (L, 4, 4) row-stochastic matrices. Entry
            [p, i, j] is P(observed=j | true=i) at position p.
        figsize: figure size in inches.
        source_colors: 4-element list of colors for true bases A/T/C/G.
            Defaults to DNA_BASE_PALETTE (matches other plots in this module).
        title: optional axis title.
        log_scale: use log y-axis (recommended — errors span ~1e-3 to ~5e-2).

    Returns:
        (fig, ax)
    """
    from matplotlib.lines import Line2D

    labels = ['A', 'T', 'C', 'G']
    L = positional_channel_matrices.shape[0]
    if source_colors is None:
        source_colors = [DNA_BASE_PALETTE[b] for b in labels]

    dest_linestyles = ['-', '--', ':']
    dest_markers = ['o', 's', '^']

    positions = np.arange(1, L + 1)

    fig, ax = plt.subplots(figsize=figsize)

    end_label_entries = []  # (y_at_last_pos, text, color) — used for collision-aware placement

    for ti, true_base in enumerate(labels):
        dest_rank = 0
        for oi, obs_base in enumerate(labels):
            if ti == oi:
                continue
            y = positional_channel_matrices[:, ti, oi]
            label_text = f'{true_base}→{obs_base}'
            ax.plot(
                positions, y,
                color=source_colors[ti],
                linestyle=dest_linestyles[dest_rank],
                marker=dest_markers[dest_rank],
                markersize=5,
                markeredgewidth=0.0,
                linewidth=1.4,
                alpha=0.92,
                label=label_text,
            )
            end_label_entries.append((y[-1], label_text, source_colors[ti]))
            dest_rank += 1

    # --- End-of-line labels with collision-aware vertical offsets ---
    # Sort by y (descending), then greedily nudge labels that are too close on
    # log scale. Nudge is in log space to stay proportional to spacing.
    ax.set_xticks(positions)
    if log_scale:
        ax.set_yscale('log')
    ax.set_xlim(positions[0] - 0.5, positions[-1] + 2.8)

    # Force axis autoscale so we can read ylim before placing labels
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()  # best-effort for display-space math; safe if no backend

    # Greedy collision avoidance in log space
    sorted_entries = sorted(end_label_entries, key=lambda t: t[0], reverse=True)
    if log_scale:
        positive = [e[0] for e in sorted_entries if e[0] > 0]
        min_pos = min(positive) if positive else 1e-6
        # Minimum log-space gap between labels (in decades)
        min_gap = 0.10
        placed_log_y = []
        placements = []  # (y_display, text, color)
        for y, text, color in sorted_entries:
            y_safe = max(y, min_pos)
            log_y = np.log10(y_safe)
            # Push down if too close to any previously placed (larger) label
            for ply in placed_log_y:
                if ply - log_y < min_gap:
                    log_y = ply - min_gap
            placed_log_y.append(log_y)
            placements.append((10 ** log_y, text, color))
    else:
        placements = [(y, text, color) for (y, text, color) in sorted_entries]

    x_label = positions[-1] + 0.35
    for y_disp, text, color in placements:
        ax.annotate(
            text,
            xy=(positions[-1], y_disp),  # not used for draw; xytext does the drawing
            xytext=(x_label, y_disp),
            ha='left', va='center',
            fontsize=8,
            color=color,
            fontweight='medium',
            annotation_clip=False,
        )

    # Axis cosmetics
    ax.set_xlabel('Sequence position', fontsize=11)
    ylabel = 'P(observed | true)' + ('  (log scale)' if log_scale else '')
    ax.set_ylabel(ylabel, fontsize=11)
    # ax.grid(True, which='major', axis='both', alpha=0.28, linestyle='--', linewidth=0.6)
    # if log_scale:
    #     ax.grid(True, which='minor', axis='y', alpha=0.12, linestyle=':', linewidth=0.5)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#444444')
    ax.spines['bottom'].set_color('#444444')

    # Compact legend: true-base color key only. Line-style/marker meaning
    # belongs in the manuscript figure caption (not drawn in the figure).
    true_base_handles = [
        Line2D([0], [0], color=source_colors[i], linewidth=3, label=labels[i])
        for i in range(4)
    ]
    ax.legend(
        handles=true_base_handles,
        loc='upper left',
        frameon=True,
        framealpha=0.92,
        edgecolor='#cccccc',
        fontsize=9,
        title='True base',
        title_fontsize=9,
        ncol=4,
        columnspacing=1.2,
        handletextpad=0.5,
    )

    if title:
        ax.set_title(title, fontsize=12, pad=10)

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Simplified figure: 4 lines, one per true (source) base
# ---------------------------------------------------------------------------

def _wilson_ci(k: np.ndarray, n: np.ndarray, z: float = 1.96):
    """Vectorized Wilson score interval for a binomial proportion.

    For cells with n == 0 or non-finite inputs, returns (p, p) so no bar is drawn.
    n may be fractional (tied reads contribute weighted counts in this module);
    treated as effective sample size.
    """
    n_safe = np.where(n > 0, n, 1.0)
    p = np.where(n > 0, k / n_safe, 0.0)
    denom = 1.0 + z * z / n_safe
    center = (p + z * z / (2.0 * n_safe)) / denom
    half = (z / denom) * np.sqrt(p * (1.0 - p) / n_safe + z * z / (4.0 * n_safe * n_safe))
    lo = np.clip(center - half, 0.0, 1.0)
    hi = np.clip(center + half, 0.0, 1.0)
    empty = n <= 0
    lo = np.where(empty, p, lo)
    hi = np.where(empty, p, hi)
    return lo, hi


def plot_per_true_base_error_rates(
    positional_channel_matrices: np.ndarray,
    figsize: tuple = (FIGURE_WIDTHS["half_page"], 2.5),
    source_colors: Optional[list] = None,
    title: Optional[str] = None,
    log_scale: bool = True,
    counts: Optional[np.ndarray] = None,
    ci_level: float = 0.95,
):
    """4-line plot of per-position error rate conditional on each true base.

    For each true base b in {A, T, C, G}, plots P(error | true=b) at each
    sequence position, i.e. 1 - positional_channel_matrices[pos, b, b]. This
    aggregates over destinations — complementary to the 12-line view which
    resolves destination.

    Args:
        positional_channel_matrices: (L, 4, 4) row-stochastic matrices.
        figsize: figure size in inches.
        source_colors: 4-element color list for A/T/C/G. Defaults to
            DNA_BASE_PALETTE (matches other plots in this module).
        title: optional axis title.
        log_scale: use log y-axis.
        counts: optional (L, 4, 4) unnormalized count tensor feeding the
            row-normalized matrices (e.g. stats.positional_channel_counts).
            When provided, Wilson score CIs are drawn as vertical error bars
            at each point.
        ci_level: two-sided confidence level for the error bars.

    Returns:
        (fig, ax)
    """
    labels = ['A', 'T', 'C', 'G']
    L = positional_channel_matrices.shape[0]
    if source_colors is None:
        source_colors = [DNA_BASE_PALETTE[b] for b in labels]

    # (L, 4) per-true-base error rate
    per_source_error = 1.0 - np.diagonal(
        positional_channel_matrices, axis1=1, axis2=2
    )

    ci_lo = ci_hi = None
    if counts is not None:
        if counts.shape != positional_channel_matrices.shape:
            raise ValueError(
                f"counts shape {counts.shape} must match "
                f"positional_channel_matrices shape {positional_channel_matrices.shape}"
            )
        n_per_base = counts.sum(axis=2)           # (L, 4): total reads per (pos, true_base)
        k_per_base = n_per_base - np.diagonal(    # (L, 4): error counts
            counts, axis1=1, axis2=2
        )
        # Two-sided Wilson z. scipy is an indirect dep but we avoid importing
        # it here; 1.96 covers 95% and the common alternatives are hardcoded.
        _Z = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}
        if ci_level not in _Z:
            raise ValueError(f"ci_level must be one of {sorted(_Z)}, got {ci_level}")
        ci_lo, ci_hi = _wilson_ci(k_per_base, n_per_base, z=_Z[ci_level])

    positions = np.arange(1, L + 1)

    fig, ax = plt.subplots(figsize=figsize)

    for ti, true_base in enumerate(labels):
        if ci_lo is None:
            ax.plot(
                positions, per_source_error[:, ti],
                color=source_colors[ti],
                marker='o',
                markersize=6,
                markeredgewidth=0.0,
                linewidth=1.8,
                alpha=0.95,
                label=true_base,
            )
        else:
            p = per_source_error[:, ti]
            yerr = np.vstack([p - ci_lo[:, ti], ci_hi[:, ti] - p])
            ax.errorbar(
                positions, p,
                yerr=yerr,
                color=source_colors[ti],
                marker='o',
                markersize=6,
                markeredgewidth=0.0,
                linewidth=1.8,
                alpha=0.95,
                label=true_base,
                capsize=2,
                elinewidth=1.0,
            )

    ax.set_xlabel('Sequence position', fontsize=11)
    ylabel = 'P(error | true base)' + ('  (log scale)' if log_scale else '')
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(positions)
    ax.set_xlim(positions[0] - 0.5, positions[-1] + 0.5)
    if log_scale:
        ax.set_yscale('log')

    # ax.grid(True, which='major', axis='both', alpha=0.28, linestyle='--', linewidth=0.6)
    # if log_scale:
    #     ax.grid(True, which='minor', axis='y', alpha=0.12, linestyle=':', linewidth=0.5)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#444444')
    ax.spines['bottom'].set_color('#444444')

    ax.legend(
        loc='upper left',
        frameon=True,
        framealpha=0.92,
        edgecolor='#cccccc',
        fontsize=9,
        title='True base',
        title_fontsize=9,
        ncol=4,
        columnspacing=1.2,
        handletextpad=0.5,
    )

    if title:
        ax.set_title(title, fontsize=12, pad=10)

    fig.tight_layout()
    return fig, ax
