"""
Optical crowding simulation for multiplexed FISH experiments.

Simulates the effect of molecular crowding from shared decoding probes
by placing transcripts in a simulated cell and detecting spatial conflicts
where two transcripts sharing a decoding probe are closer than the
optical diffraction limit.

Reproduces the molecular-crowding panels (Fig 4C-K) of Boström, Zapała &
Adameyko, "Boosting multiplexing capabilities for error-robust spatial
transcriptomic methods using a set exchange approach", Science Advances 11,
eadr4026 (2025). doi:10.1126/sciadv.adr4026

Typical usage:
    >>> import pandas as pd
    >>> from duet.optical_crowding import simulate_optical_crowding
    >>>
    >>> codebook = pd.DataFrame({
    ...     "Gene": ["GeneA", "GeneB", "GeneC"],
    ...     "Sequence": ["1100", "1010", "0110"],
    ... })
    >>> expression = pd.DataFrame({
    ...     "gene_name": ["GeneA", "GeneB", "GeneC"],
    ...     "mean_raw_counts": [100.0, 200.0, 50.0],
    ... })
    >>> result = simulate_optical_crowding(
    ...     codebook, expression, expression_col="mean_raw_counts",
    ...     total_reads=1000, seed=42,
    ... )
    >>> print(f"Conflict rate: {result.conflict_fraction:.2%}")
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from duet.bostrom_codebook import build_panel_codebook, load_bostrom_codebook

if TYPE_CHECKING:  # annotation only; matplotlib loads inside plot_crowding_spatial
    import matplotlib.pyplot as plt


@dataclass
class CrowdingResult:
    """Results from a single optical crowding simulation run.

    Attributes:
        conflict_fraction: Fraction of transcripts with at least one conflict.
        identified_fraction: Fraction of transcripts without conflicts.
        n_transcripts: Total number of transcripts simulated.
        n_conflicted: Number of transcripts with at least one conflict.
        transcripts_df: DataFrame with columns (gene, x, y, conflicted).
        barcode_length: Number of bit positions in the codebook.
        hamming_weight: Hamming weight of the codewords (None if mixed).
    """

    conflict_fraction: float
    identified_fraction: float
    n_transcripts: int
    n_conflicted: int
    transcripts_df: pd.DataFrame
    barcode_length: int
    hamming_weight: Optional[int]


def compute_diffraction_limit(
    wavelength_nm: float = 500.0,
    numerical_aperture: float = 1.4,
    expansion_factor: float = 1.0,
    model: str = "abbe",
) -> float:
    """Compute the optical diffraction limit in micrometers.

    Args:
        wavelength_nm: Emission wavelength in nanometers.
        numerical_aperture: Numerical aperture of the objective.
        expansion_factor: Expansion factor (1.0 = none; 3.0 = 3x improves resolution).
        model: "abbe" -> lambda/(2*NA) (default; matches Boström et al. 2025);
               "rayleigh" -> 0.61*lambda/NA.

    Returns:
        Diffraction limit in micrometers (smaller under expansion).
    """
    wavelength_um = wavelength_nm / 1000.0
    if model == "abbe":
        base = wavelength_um / (2.0 * numerical_aperture)
    elif model == "rayleigh":
        base = 0.61 * wavelength_um / numerical_aperture
    else:
        raise ValueError(f"Unknown diffraction model {model!r}; expected 'abbe' or 'rayleigh'.")
    return base / expansion_factor


def generate_constant_weight_codebook(
    n_genes: int,
    hamming_weight: int,
    barcode_length: int,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate a random constant-weight binary codebook.

    Enumerates all binary codewords of the given length and Hamming weight,
    then randomly selects n_genes codewords.

    Args:
        n_genes: Number of genes (codewords) to generate.
        hamming_weight: Number of 1-bits in each codeword.
        barcode_length: Length of each codeword in bits.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns 'Gene' (integer index) and 'Sequence'
        (binary string).

    Raises:
        ValueError: If n_genes exceeds C(barcode_length, hamming_weight).
    """
    n_available = math.comb(barcode_length, hamming_weight)
    if n_genes > n_available:
        raise ValueError(
            f"Cannot generate {n_genes} codewords with barcode_length="
            f"{barcode_length} and hamming_weight={hamming_weight}. "
            f"Only {n_available} codewords available."
        )

    rng = np.random.default_rng(seed)

    # Enumerate all constant-weight codewords
    all_codewords = []
    for positions in combinations(range(barcode_length), hamming_weight):
        bits = ["0"] * barcode_length
        for pos in positions:
            bits[pos] = "1"
        all_codewords.append("".join(bits))

    # Randomly sample n_genes codewords
    indices = rng.choice(len(all_codewords), size=n_genes, replace=False)
    selected = [all_codewords[i] for i in indices]

    return pd.DataFrame(
        {"Gene": [f"Gene_{i}" for i in range(n_genes)], "Sequence": selected}
    )


def load_set_codebook(path: Union[str, Path], barcode_length: int) -> pd.DataFrame:
    """Back-compat shim: Boström Set-format codebook -> DataFrame{Gene, Sequence}.

    Thin wrapper over duet.bostrom_codebook.load_bostrom_codebook(fmt="set").
    'Gene' values are placeholder row labels (``row_0``, ``row_1``, …); assign
    real gene names with ``build_panel_codebook``.
    """
    seqs = load_bostrom_codebook(path, barcode_length, fmt="set")
    return pd.DataFrame(
        {"Gene": [f"row_{i}" for i in range(len(seqs))], "Sequence": seqs}
    )


def simulate_transcriptome(
    gene_names: np.ndarray,
    gene_expression: np.ndarray,
    total_reads: int = 100_000,
    cell_size_um: float = 100.0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Simulate transcript positions in a cell.

    Samples transcripts according to gene expression levels and places
    them uniformly at random in a square cell.

    Args:
        gene_names: Array of gene names (length G).
        gene_expression: Array of expression values (length G, non-negative).
        total_reads: Total number of transcripts to simulate.
        cell_size_um: Side length of the square cell in micrometers.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns 'gene' (str), 'x' (float), 'y' (float).
    """
    rng = np.random.default_rng(seed)

    # Normalize expression to probability distribution
    expr = np.asarray(gene_expression, dtype=np.float64)
    expr = np.maximum(expr, 0.0)
    total = expr.sum()
    if total <= 0:
        raise ValueError("Total expression is zero; cannot create distribution.")
    probs = expr / total

    # Sample transcript identities
    gene_indices = rng.choice(len(gene_names), size=total_reads, p=probs)
    genes = np.asarray(gene_names)[gene_indices]

    # Sample positions uniformly in [0, cell_size_um]^2
    x = rng.uniform(0, cell_size_um, size=total_reads)
    y = rng.uniform(0, cell_size_um, size=total_reads)

    return pd.DataFrame({"gene": genes, "x": x, "y": y})


def detect_conflicts(
    transcripts_df: pd.DataFrame,
    codebook_df: pd.DataFrame,
    diffraction_limit_um: float,
    neighborhood: str = "box",
) -> np.ndarray:
    """Detect spatial conflicts from shared decoding probes.

    Two transcripts conflict if they share >=1 readout position (a common '1'
    bit) AND lie within the diffraction limit. ``neighborhood`` selects the
    distance metric: "box" -> Chebyshev (L-inf): two transcripts are neighbors
    when ``|dx| <= diffraction_limit_um`` AND ``|dy| <= diffraction_limit_um``
    (a square extending ±``diffraction_limit_um`` from each point; matches the
    MinDist box in Boström et al. SimulatingCode.R). "disk" -> Euclidean (L2)
    circle of radius ``diffraction_limit_um``.

    Args:
        transcripts_df: DataFrame with columns 'gene', 'x', 'y'.
        codebook_df: DataFrame with columns 'Gene', 'Sequence' (binary str).
        diffraction_limit_um: Minimum resolvable distance in micrometers.
        neighborhood: "box" (Chebyshev) or "disk" (Euclidean).

    Returns:
        Boolean array of length len(transcripts_df), True if conflicted.
    """
    p = {"box": np.inf, "disk": 2.0}.get(neighborhood)
    if p is None:
        raise ValueError(f"Unknown neighborhood {neighborhood!r}; expected 'box' or 'disk'.")

    n_transcripts = len(transcripts_df)
    conflicted = np.zeros(n_transcripts, dtype=bool)
    gene_to_barcode = dict(zip(codebook_df["Gene"], codebook_df["Sequence"]))
    barcodes = transcripts_df["gene"].map(gene_to_barcode)
    barcode_length = len(codebook_df["Sequence"].iloc[0])
    positions = np.column_stack([transcripts_df["x"].values, transcripts_df["y"].values])

    for bit in range(barcode_length):
        mask = barcodes.str[bit] == "1"
        indices = np.where(mask.values)[0]
        if len(indices) < 2:
            continue
        tree = KDTree(positions[indices])
        pairs = tree.query_pairs(diffraction_limit_um, p=p)
        for i, j in pairs:
            conflicted[indices[i]] = True
            conflicted[indices[j]] = True
    return conflicted


def simulate_optical_crowding(
    codebook_df: pd.DataFrame,
    gene_expression_df: pd.DataFrame,
    expression_col: str = "mean_raw_counts",
    gene_col: str = "gene_name",
    total_reads: int = 100_000,
    cell_size_um: float = 100.0,
    wavelength_nm: float = 500.0,
    numerical_aperture: float = 1.4,
    expansion_factor: float = 1.0,
    diffraction_model: str = "abbe",
    neighborhood: str = "box",
    placement: str = "multinomial",
    seed: Optional[int] = None,
) -> CrowdingResult:
    """Simulate optical crowding for a codebook and gene expression profile.

    Generates a simulated cell with transcripts from the FULL transcriptome
    (all genes in expression data), then detects conflicts only among
    transcripts belonging to genes in the codebook. This models the real
    experimental scenario where total_reads represents the full mRNA content
    of a cell but only codebook genes are targeted for decoding.

    Args:
        codebook_df: DataFrame with columns 'Gene' and 'Sequence'.
            'Sequence' must be binary strings (e.g., '0010110100001000').
        gene_expression_df: DataFrame with a gene-name column (named by
            ``gene_col``) and the column named by ``expression_col``. Should
            contain the FULL transcriptome (not just codebook genes) so that
            total_reads are distributed across all genes proportionally.
        expression_col: Name of the expression column in gene_expression_df.
        gene_col: Name of the gene-name column in gene_expression_df.
        total_reads: Total mRNA transcripts in the cell (full transcriptome).
        cell_size_um: Side length of the square cell in micrometers.
        wavelength_nm: Fluorophore emission wavelength in nanometers.
        numerical_aperture: Objective numerical aperture.
        expansion_factor: Expansion microscopy factor (1.0 = no expansion).
        diffraction_model: "abbe" or "rayleigh" (see compute_diffraction_limit).
        neighborhood: "box" (Chebyshev) or "disk" (Euclidean) conflict metric.
        placement: "multinomial" (sample total_reads via multinomial over all
            expressed genes) or "deterministic" (per-gene counts =
            round(total_reads * prop), prop normalized over all expressed genes).
        seed: Random seed for reproducibility.

    Returns:
        CrowdingResult with conflict statistics and transcript data.
        The transcripts_df contains only codebook-gene transcripts.
    """
    codebook_genes = set(codebook_df["Gene"])
    expression_genes = set(gene_expression_df[gene_col])
    common_genes = codebook_genes & expression_genes

    if not common_genes:
        raise ValueError(
            "No genes in common between codebook and expression data."
        )

    # Simulate full transcriptome using ALL genes in expression data
    all_expr = gene_expression_df[gene_expression_df[expression_col] > 0]

    diffraction_limit = compute_diffraction_limit(
        wavelength_nm, numerical_aperture, expansion_factor, model=diffraction_model
    )

    cb = codebook_df[codebook_df["Gene"].isin(common_genes)].copy()

    if placement == "multinomial":
        all_transcripts = simulate_transcriptome(
            gene_names=all_expr[gene_col].values,
            gene_expression=all_expr[expression_col].values,
            total_reads=total_reads,
            cell_size_um=cell_size_um,
            seed=seed,
        )
        codebook_mask = all_transcripts["gene"].isin(common_genes)
        transcripts = all_transcripts[codebook_mask].reset_index(drop=True)
    elif placement == "deterministic":
        # round(total_reads * prop) per gene, prop normalized over ALL expressed
        # genes (matches SimulatingCode.R); place only codebook-gene transcripts.
        rng = np.random.default_rng(seed)
        total_expr = all_expr[expression_col].to_numpy(float).sum()
        cb_expr = all_expr[all_expr[gene_col].isin(common_genes)]
        rows = []
        for gname, gexpr in zip(cb_expr[gene_col].values, cb_expr[expression_col].values):
            # int(round(...)) is intentionally banker's (round-half-to-even)
            # rounding to match R's round() in SimulatingCode.R.
            c = int(round(total_reads * (gexpr / total_expr)))
            if c <= 0:
                continue
            rows.append(pd.DataFrame({
                "gene": gname,
                "x": rng.uniform(0, cell_size_um, c),
                "y": rng.uniform(0, cell_size_um, c),
            }))
        transcripts = (pd.concat(rows, ignore_index=True) if rows
                       else pd.DataFrame(columns=["gene", "x", "y"]))
    else:
        raise ValueError(f"Unknown placement {placement!r}; expected 'multinomial' or 'deterministic'.")

    if len(transcripts) == 0:
        raise ValueError(
            "No transcripts from codebook genes were sampled. "
            "Check that codebook genes have non-zero expression."
        )

    # Detect conflicts among codebook-gene transcripts
    conflicted = detect_conflicts(transcripts, cb, diffraction_limit, neighborhood=neighborhood)
    transcripts = transcripts.copy()
    transcripts["conflicted"] = conflicted

    n_conflicted = int(conflicted.sum())
    n_total = len(transcripts)

    # Determine hamming weight (None if mixed)
    weights = cb["Sequence"].apply(lambda s: s.count("1"))
    hw = int(weights.iloc[0]) if weights.nunique() == 1 else None

    return CrowdingResult(
        conflict_fraction=n_conflicted / n_total if n_total > 0 else 0.0,
        identified_fraction=1.0 - (n_conflicted / n_total) if n_total > 0 else 1.0,
        n_transcripts=n_total,
        n_conflicted=n_conflicted,
        transcripts_df=transcripts,
        barcode_length=len(cb["Sequence"].iloc[0]),
        hamming_weight=hw,
    )


def _published_set_path(codebook_dir: Union[str, Path], n_genes: int, hw: int, barcode_length: int) -> Path:
    """Resolve the Set-format codebook file with >= n_genes codewords for (hw, L).

    Looks under ``codebook_dir/HW{hw}`` for files named
    ``{barcode_length}Bit_HW{hw}_HD4_finalsize{N}Set.csv`` and returns the first
    whose final size N >= n_genes.
    """
    d = Path(codebook_dir) / f"HW{hw}"
    matches = sorted(d.glob(f"{barcode_length}Bit_HW{hw}_HD4_finalsize*Set.csv"))
    for m in matches:
        finalsize = int(m.stem.split("finalsize")[1].removesuffix("Set"))
        if finalsize >= n_genes:
            return m
    raise FileNotFoundError(
        f"No Set codebook for HW{hw}, L={barcode_length}, >= {n_genes} codewords in {d}"
    )


def run_crowding_sweep(
    gene_expression_df: pd.DataFrame,
    n_genes_list: List[int],
    hamming_weights: List[int],
    barcode_lengths: Dict[Tuple[int, int], int],
    expression_col: str = "mean_raw_counts",
    gene_col: str = "gene_name",
    n_trials: int = 5,
    total_reads: int = 100_000,
    cell_size_um: float = 100.0,
    wavelength_nm: float = 500.0,
    numerical_aperture: float = 1.4,
    expansion_factor: float = 1.0,
    panel_pool: str = "detectable",
    codebook_source: str = "random",
    codebook_dir: Optional[Union[str, Path]] = None,
    assign_by_expression: bool = True,
    diffraction_model: str = "abbe",
    neighborhood: str = "box",
    placement: str = "multinomial",
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Run optical crowding simulation across multiple gene sets and HWs.

    For each combination of gene set size and Hamming weight, builds a
    codebook (either a random constant-weight codebook or a published
    Boström Set-format constant-weight codebook, per ``codebook_source``)
    and runs the crowding simulation over multiple trials. By default the
    panel is sampled from the detectable gene pool (genes whose expected
    round(total_reads * prop) >= 1; see ``panel_pool``). The full
    transcriptome expression data is used to simulate total_reads across all
    genes; only transcripts from the selected panel genes are analyzed for
    conflicts.

    Args:
        gene_expression_df: DataFrame with a gene-name column (named by
            ``gene_col``) and expression_col. Should contain the full
            transcriptome (all genes).
        n_genes_list: List of gene set sizes to simulate (e.g., [1000, 5000]).
        hamming_weights: List of Hamming weights (e.g., [4, 5, 6]).
        barcode_lengths: Dict mapping (n_genes, hamming_weight) tuples to
            the barcode length to use for that condition.
        expression_col: Name of the expression column in gene_expression_df.
        gene_col: Name of the gene-name column in gene_expression_df.
        n_trials: Number of independent trials per condition.
        total_reads: Total mRNA transcripts per cell (full transcriptome).
        cell_size_um: Cell side length in micrometers.
        wavelength_nm: Fluorophore emission wavelength in nanometers.
        numerical_aperture: Objective numerical aperture.
        expansion_factor: Expansion microscopy factor.
        panel_pool: "detectable" (sample panel uniformly from genes whose
            round(total_reads * prop) >= 1, prop over all expressed; the
            Boström "detectable pool" fix) or "all_expressed" (any expr>0 gene).
        codebook_source: "random" (random constant-weight codebook) or
            "published" (load Set-format codebook from codebook_dir).
        codebook_dir: Directory of published Set-format codebooks; required
            when codebook_source="published".
        assign_by_expression: If True, sort the sampled panel by descending
            expression before assigning codewords (highest expr -> first row).
        diffraction_model: "abbe" or "rayleigh" (see compute_diffraction_limit).
        neighborhood: "box" (Chebyshev) or "disk" (Euclidean) conflict metric.
        placement: "multinomial" or "deterministic" (see
            simulate_optical_crowding).
        seed: Base random seed (incremented per trial).

    Returns:
        DataFrame with columns: n_genes, hamming_weight, trial,
        conflict_fraction, identified_fraction, n_conflicted, n_transcripts,
        barcode_length.
    """
    rng = np.random.default_rng(seed)
    results = []

    expressed = gene_expression_df[gene_expression_df[expression_col] > 0].copy()
    if panel_pool == "detectable":
        prop = expressed[expression_col].to_numpy(float)
        prop = prop / gene_expression_df[expression_col].to_numpy(float).sum()
        pool = expressed[np.round(total_reads * prop) >= 1].copy()
    elif panel_pool == "all_expressed":
        pool = expressed
    else:
        raise ValueError(f"Unknown panel_pool {panel_pool!r}; expected 'detectable' or 'all_expressed'.")

    for n_genes in n_genes_list:
        if n_genes > len(pool):
            raise ValueError(
                f"Requested {n_genes} genes but pool has {len(pool)} ({panel_pool})."
            )
        for trial in range(n_trials):
            trial_rng = np.random.default_rng(rng.integers(0, 2**31))
            gene_idx = trial_rng.choice(len(pool), size=n_genes, replace=False)
            selected = pool.iloc[gene_idx]
            if assign_by_expression:
                selected = selected.sort_values(expression_col, ascending=False)
            gene_order = selected[gene_col].values

            for hw in hamming_weights:
                bl = barcode_lengths[(n_genes, hw)]
                if codebook_source == "random":
                    codebook = generate_constant_weight_codebook(
                        n_genes=n_genes, hamming_weight=hw, barcode_length=bl,
                        seed=trial_rng.integers(0, 2**31),
                    )
                    codebook["Gene"] = gene_order
                elif codebook_source == "published":
                    if codebook_dir is None:
                        raise ValueError("codebook_source='published' requires codebook_dir")
                    set_path = _published_set_path(codebook_dir, n_genes, hw, bl)
                    seqs = load_set_codebook(set_path, bl)["Sequence"].tolist()
                    codebook = build_panel_codebook(gene_order, seqs)
                else:
                    raise ValueError(
                        f"Unknown codebook_source {codebook_source!r}; expected 'random' or 'published'."
                    )

                result = simulate_optical_crowding(
                    codebook_df=codebook,
                    gene_expression_df=gene_expression_df,
                    expression_col=expression_col,
                    gene_col=gene_col,
                    total_reads=total_reads,
                    cell_size_um=cell_size_um,
                    wavelength_nm=wavelength_nm,
                    numerical_aperture=numerical_aperture,
                    expansion_factor=expansion_factor,
                    diffraction_model=diffraction_model,
                    neighborhood=neighborhood,
                    placement=placement,
                    seed=trial_rng.integers(0, 2**31),
                )

                results.append({
                    "n_genes": n_genes,
                    "hamming_weight": hw,
                    "trial": trial,
                    "conflict_fraction": result.conflict_fraction,
                    "identified_fraction": result.identified_fraction,
                    "n_conflicted": result.n_conflicted,
                    "n_transcripts": result.n_transcripts,
                    "barcode_length": bl,
                })

    return pd.DataFrame(results)


def plot_crowding_spatial(
    results: Dict[int, CrowdingResult],
    cell_size_um: float = 100.0,
    point_size: float = 0.3,
    title_prefix: str = "",
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[np.ndarray] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot spatial scatter of identified vs missed transcripts per HW.

    Produces one subplot per Hamming weight, showing identified transcripts
    in blue and missed (conflicted) transcripts in red, similar to
    Figure 4 panels C, F, I of Boström et al. (2025).

    Args:
        results: Dict mapping hamming_weight (int) to CrowdingResult.
        cell_size_um: Side length of the square cell in micrometers.
        point_size: Marker size for scatter plot points.
        title_prefix: Text prepended to each subplot title
            (e.g., "1000 genes").
        figsize: Figure size as (width, height). If None, auto-computed.
        ax: Optional array of pre-existing Axes to plot into. Must have
            length >= len(results). If None, a new figure is created.

    Returns:
        Tuple of (Figure, array of Axes).
    """
    import matplotlib.pyplot as plt

    hw_list = sorted(results.keys())
    n = len(hw_list)

    if ax is None:
        if figsize is None:
            figsize = (4 * n, 4)
        fig, axes = plt.subplots(1, n, figsize=figsize)
        if n == 1:
            axes = np.array([axes])
    else:
        axes = np.asarray(ax).ravel()
        fig = axes[0].get_figure()

    color_identified = "#3b4cc0"
    color_missed = "#d32f2f"

    for i, hw in enumerate(hw_list):
        result = results[hw]
        df = result.transcripts_df
        identified = df[~df["conflicted"]]
        missed = df[df["conflicted"]]

        axes[i].scatter(
            identified["x"], identified["y"],
            s=point_size, c=color_identified, alpha=0.5,
            linewidths=0, rasterized=True,
        )
        axes[i].scatter(
            missed["x"], missed["y"],
            s=point_size, c=color_missed, alpha=0.7,
            linewidths=0, rasterized=True,
        )
        axes[i].set_xlim(0, cell_size_um)
        axes[i].set_ylim(0, cell_size_um)
        axes[i].set_aspect("equal")
        pct = result.conflict_fraction * 100
        title = f"{title_prefix}\nHW{hw}" if title_prefix else f"HW{hw}"
        axes[i].set_title(title, fontsize=10, fontweight="bold")
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    # Add legend to center subplot
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=color_identified, markersize=6,
               label="Resolved"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=color_missed, markersize=6,
               label="Missed"),
    ]
    mid = n // 2
    axes[mid].legend(
        handles=legend_elements, loc="lower center",
        bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8, frameon=False,
    )

    return fig, axes
