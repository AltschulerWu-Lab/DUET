from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from duet.plotting import FIGURE_WIDTHS
from duet.codebook_evaluator import ErrorCorrectionMetrics
from duet.evaluator_config import EvaluatorConfig, create_evaluator
from duet.pep_accessor import PEPAccessor


def _print_summary_stats(groups):
    # Per-group summary statistics
    for label, vals, _ in groups:
        vals = np.asarray(vals)
        print(
            f'{label}:\n'
            f'  Min: {np.min(vals):.4f},\n'
            f'   5th Percentile: {np.percentile(vals, 5):.4f},\n'
            f'  10th Percentile: {np.percentile(vals, 10):.4f},\n'
            f'  Median: {np.median(vals):.4f},\n'
            f'  Mean: {np.mean(vals):.4f},\n'
            f'  90th Percentile: {np.percentile(vals, 90):.4f},\n'
            f'  Max: {np.max(vals):.4f}\n'
        )


def _plot_grouped_hist(
    groups,
    *,
    bins=50,
    kde=True,
    x_label='',
    y_label='Density',
    title='',
    print_summary=False,
    histplot_kwargs=None,
    figsize: tuple | None = None,
):
    """
    groups: list of tuples (label, values_array, color)
    """
    all_vals = np.concatenate([np.asarray(vals) for _, vals, _ in groups])
    bin_edges = np.histogram_bin_edges(all_vals, bins=bins)

    if figsize is None:
        figsize = (FIGURE_WIDTHS["half_page"], 2.2)
    fig, ax = plt.subplots(figsize=figsize)
    for label, vals, color in groups:
        vals = np.asarray(vals)

        plot_kwargs = {
            "bins": bin_edges,
            "kde": kde,
            "stat": "density",
            "color": color,
            "alpha": 0.45,
            "edgecolor": None,
            "label": label,
        }
        if histplot_kwargs:
            plot_kwargs.update(histplot_kwargs)

        sns.histplot(
            vals,
            ax=ax,
            **plot_kwargs,
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()

    if print_summary:
        _print_summary_stats(groups)
    return fig, ax


def plot_codeword_probabilities(
    groups,
    bins=50,
    kde=True,
    print_summary=False,
    histplot_kwargs=None,
    figsize: tuple | None = None,
):
    """
    groups: list of tuples (label, codeword_probs_array, color)
    """
    return _plot_grouped_hist(
        groups,
        bins=bins,
        kde=kde,
        x_label='Codeword Probability',
        y_label='Density',
        title='Codeword Probability Distribution (Initial vs Optimized)',
        print_summary=print_summary,
        histplot_kwargs=histplot_kwargs,
        figsize=figsize,
    )


def plot_activity_scores(
    groups,
    bins=50,
    kde=True,
    print_summary=False,
    histplot_kwargs=None,
    figsize: tuple | None = None,
):
    """
    groups: list of tuples (label, scores_array, color)
    """
    return _plot_grouped_hist(
        groups,
        bins=bins,
        kde=kde,
        x_label='Activity Score',
        y_label='Density',
        title='Activity Score Distribution (Initial vs Optimized)',
        print_summary=print_summary,
        histplot_kwargs=histplot_kwargs,
        figsize=figsize,
    )


def plot_dual_objectives(
    groups,
    scatter_kwargs=None,
    kdeplot_kwargs=None,
    figsize: tuple | None = None,
):
    """
    groups: list of tuples (label, codeword_probs_array, activity_scores_array, color)
    Plots a 2D scatter: x = codeword probabilities, y = activity scores.
    """
    if figsize is None:
        figsize = (FIGURE_WIDTHS["half_page"], 2.6)
    fig, ax = plt.subplots(figsize=figsize)
    for label, probs, scores, color in groups:
        probs = np.asarray(probs)
        scores = np.asarray(scores)

        _scatter_kwargs = {
            "s": 18,
            "alpha": 0.5,
            "color": color,
            "label": label,
            "edgecolors": 'none'
        }
        if scatter_kwargs:
            _scatter_kwargs.update(scatter_kwargs)
        ax.scatter(
            probs, scores,
            **_scatter_kwargs
        )

        _kdeplot_kwargs = {
            "color": color,
            "levels": 5,
            "linewidths": 1,
            "alpha": 0.6
        }
        if kdeplot_kwargs:
            _kdeplot_kwargs.update(kdeplot_kwargs)
        sns.kdeplot(x=probs, y=scores, ax=ax, **_kdeplot_kwargs)

    ax.set_xlabel('Codeword Probability')
    ax.set_ylabel('Activity Score')
    ax.set_title('Codeword Probability vs Activity Score')
    ax.legend()
    return fig, ax


# =============================================================================
# PEP Matrix Visualizations
# =============================================================================


def _get_baseline_indices(
    baseline_sequences: List[str],
    candidate_sequences: List[str] | np.ndarray,
) -> np.ndarray:
    """Map baseline codebook sequences to their indices in the candidate set.

    Args:
        baseline_sequences: List of sequences from a baseline codebook.
        candidate_sequences: Full candidate sequence library.

    Returns:
        Array of indices into candidate_sequences corresponding to baseline_sequences.

    Raises:
        ValueError: If any baseline sequence is not found in candidates.
    """
    if isinstance(candidate_sequences, np.ndarray):
        candidate_list = candidate_sequences.tolist()
    else:
        candidate_list = candidate_sequences

    seq_to_idx = {seq: i for i, seq in enumerate(candidate_list)}

    indices = []
    for seq in baseline_sequences:
        if seq not in seq_to_idx:
            raise ValueError(f"Sequence not found in candidates: {seq}")
        indices.append(seq_to_idx[seq])

    return np.array(indices, dtype=np.int64)


def plot_pep_clustermap(
    pep_matrix: np.ndarray,
    indices: np.ndarray,
    title: str,
    output_path: Path,
    cmap: str = "viridis",
    figsize: tuple = (10, 10),
) -> float:
    """Generate a clustered heatmap of a PEP matrix subset.

    Uses a custom distance metric (1 - PEP) to cluster codewords that are
    highly confusable with each other together. The PEP matrix is symmetrized
    for clustering, and the same linkage is applied to both rows and columns
    to ensure a symmetric heatmap display.

    Args:
        pep_matrix: Full PEP matrix (n_candidates x n_candidates).
        indices: Indices of codewords to include in the subset.
        title: Title for the plot.
        output_path: Path to save the figure (without extension).
        cmap: Colormap for the heatmap.
        figsize: Figure size.

    Returns:
        Total PEP sum (off-diagonal sum of the subset matrix).
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    # Subset the PEP matrix
    subset_pep = pep_matrix[np.ix_(indices, indices)]

    # Compute total PEP sum (off-diagonal)
    total_pep = float(subset_pep.sum() - np.trace(subset_pep))

    # Symmetrize PEP matrix for clustering (average of P(j|i) and P(i|j))
    sym_pep = (subset_pep + subset_pep.T) / 2

    # Create distance matrix: high PEP → small distance → clustered together
    dist_matrix = 1 - sym_pep
    np.fill_diagonal(dist_matrix, 0)  # self-distance = 0

    # Compute linkage once and apply to both axes for symmetric ordering
    condensed_dist = squareform(dist_matrix)
    linkage_matrix = linkage(condensed_dist, method="average")

    # Create clustermap with same linkage for rows and columns
    g = sns.clustermap(
        subset_pep,  # Display original PEP values
        row_linkage=linkage_matrix,
        col_linkage=linkage_matrix,
        cmap=cmap,
        figsize=figsize,
        xticklabels=False,
        yticklabels=False,
        dendrogram_ratio=(0.1, 0.1),
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
    )
    g.fig.suptitle(f"{title}\nTotal PEP Sum: {total_pep:.4f}", y=1.02)

    # Save figures
    g.savefig(f"{output_path}.svg", bbox_inches="tight")
    g.savefig(f"{output_path}.png", dpi=150, bbox_inches="tight")
    plt.close(g.fig)

    return total_pep


def generate_pep_debug_visualizations(
    candidates,  # CandidatePool
    evaluator_config: EvaluatorConfig,
    duet_pep_matrix: np.ndarray | PEPAccessor,
    best_indices: np.ndarray,
    baseline_results: Dict[str, Tuple[List[str], List[str], np.ndarray, float, ErrorCorrectionMetrics]],
    output_dir: Path,
    alphabet_size: int = 2,
    n_samples: int | None = None,
) -> pd.DataFrame:
    """Generate PEP matrix debug visualizations for all codebooks.

    Creates clustered heatmaps for each codebook under two PEP matrices:
    1. DUET PEP: The PEP matrix used during DUET optimization
    2. Evaluator PEP: A fresh PEP matrix computed with evaluator_config

    Args:
        candidates: CandidatePool instance with full candidate library.
        evaluator_config: EvaluatorConfig for ground truth PEP computation.
        duet_pep_matrix: PEP matrix used during DUET optimization.
        best_indices: Indices of DUET-selected codewords.
        baseline_results: Dict mapping method name to
            (sequences, genes, per_codeword_accuracies, mean_accuracy, error_metrics).
        output_dir: Directory to save debug visualizations.
        alphabet_size: Alphabet size (2 for binary MERFISH).
        n_samples: Monte Carlo sample count for uint16 count matrices.

    Returns:
        DataFrame with PEP sum summary for each method and PEP source.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute Evaluator PEP matrix on full candidate library
    # Note: Even if config.duet.pep is None and falls back to evaluator_config,
    # the two matrices will differ due to different RNG states (seed=None by default)
    print("Computing Evaluator PEP matrix for debug visualization...")
    evaluator_pep_evaluator = create_evaluator(
        candidates.sequences,
        evaluator_config,
        alphabet_size=alphabet_size,
    )
    evaluator_count_matrix, eval_n_samples = evaluator_pep_evaluator.compute_pep_matrix(
        n_jobs=evaluator_config.num_cpus
    )

    # Convert counts → probabilities for visualization.
    # PEPAccessor.get_rows already returns float64 probabilities, so no
    # divide-by-n_samples is needed on that path (n_samples is also None there).
    if isinstance(duet_pep_matrix, PEPAccessor):
        duet_pep_float = duet_pep_matrix.get_rows(np.arange(duet_pep_matrix.N))
    elif n_samples is not None:
        duet_pep_float = duet_pep_matrix.astype(np.float64) / n_samples
    else:
        duet_pep_float = duet_pep_matrix.astype(np.float64)
    evaluator_pep_float = evaluator_count_matrix.astype(np.float64) / eval_n_samples

    # Collect PEP sums for summary
    pep_summary_rows = []

    # Define codebooks to visualize
    codebooks = {
        "DUET": best_indices,
    }

    # Add baseline codebooks
    for method_name, (seqs, _, _, _, _) in baseline_results.items():
        # Create a safe filename from method name
        safe_name = method_name.replace(" ", "_").replace("(", "").replace(")", "")
        codebooks[safe_name] = _get_baseline_indices(seqs, candidates.sequences)

    # Generate heatmaps for each codebook under each PEP matrix
    pep_sources = {
        "DUET_PEP": duet_pep_float,
        "Evaluator_PEP": evaluator_pep_float,
    }

    for codebook_name, indices in codebooks.items():
        for pep_name, pep_matrix in pep_sources.items():
            title = f"{codebook_name} - {pep_name}"
            safe_filename = f"pep_{codebook_name.lower()}_{pep_name.lower()}"
            output_path = output_dir / safe_filename

            pep_sum = plot_pep_clustermap(
                pep_matrix=pep_matrix,
                indices=indices,
                title=title,
                output_path=output_path,
            )

            pep_summary_rows.append({
                "Method": codebook_name,
                "PEP_Source": pep_name,
                "Total_PEP_Sum": pep_sum,
            })

            print(f"  Saved {safe_filename}.svg/.png (PEP sum: {pep_sum:.4f})")

    return pd.DataFrame(pep_summary_rows)