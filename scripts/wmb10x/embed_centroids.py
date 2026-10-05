#!/usr/bin/env python
"""Embed the WMB-10X cluster centroids: PCA over all genes, plus a Fig 1B UMAP.

Every point here is one of the 5,322 transcriptomic CLUSTER CENTROIDS of Yao et al.
2023, i.e. one point per cell type, NOT one point per cell. The ~4.04M cells of the
atlas were never downloaded (94.7 GB to 150 GB). Within-cluster spread is therefore
absent from these embeddings by construction, so the UMAP territories below are
sparser and cleaner than the published Fig 1B, which draws every cell. Any caption
using these coordinates must say so.

This script computes both embeddings once and writes coordinate tables, so the
plotting stage is pure rendering and never recomputes an embedding.

DELIVERABLE 1, PCA (results/wmb10x/pca_*.csv)
  One transform, no variants. The h5 values are ALREADY mean log2(CPM+1), so the
  only step is to mean-center each gene. There is deliberately NO variance scaling
  and NO z-scoring: scaling would hand a rare, noisy, low-expression gene the same
  say in the geometry as a strong marker. That is asserted, not just intended.
  PCA runs over all 32,285 genes and keeps 100 components.

  A SECOND, SEPARATE PCA runs over the 6,558 marker genes, and it is that one, not
  the all-gene one, that the UMAP consumes. The two are different spaces with
  different cumulative variance (67.8% against 66.5% at 100 PCs) and different class
  purity. pca_variance.csv carries BOTH curves, and EVERY column in it names
  its gene set, precisely so a figure cannot attribute one space's variance to the
  other. There is deliberately no unqualified "cumulative" column left to grab.

DELIVERABLE 2, UMAP (results/wmb10x/umap_coords.csv)
  Follows the paper's Methods section "UMAP projection" as closely as centroid data
  allows: PCA on marker genes, top 100 PCs, then UMAP at n_neighbors=25,
  min_dist=0.4.

  THE METRIC IS OURS, NOT THE PAPER'S. The Methods pin down n_neighbors and min_dist
  and say nothing whatever about the distance metric, so there is no faithful choice
  to make here, only a defensible one. We use metric="cosine" for the main embedding.
  What that choice does, and does not, buy is measured by island_structure() below and
  written to embedding_summary.json, so the claim in every caption is generated from
  numbers rather than from an impression.

  WHAT THE METRIC DOES NOT CHANGE: which groups detach. At centroid level the same
  handful of groups (01 IT-ET Glut, 02 NP-CT-L6b Glut, 10 LSX GABA, the non-neuronal
  30-34 block, 31 OPC-Oligo) break off under BOTH metrics, and the membership of those
  islands is essentially metric-invariant from 5x to 20x the median nearest-neighbour
  spacing. A centroid carries no within-cluster spread, so the cells that would bridge
  those groups to the rest in a 4M-cell embedding do not exist here and only
  between-class gaps are left. No metric can conjure the bridge back.

  WHAT THE METRIC DOES CHANGE: how big those gaps are relative to the layout, and
  therefore what sets the scale. Under cosine the 01 IT-ET island sits 22.7% of the
  layout span from the continent and the non-neuronal block 7.2%, so the continent of
  neuronal territories remains the dominant structure of the picture, which is the
  structure Fig 1B shows. Under euclidean the same two gaps blow out to 75.1% and
  52.7% of span: the islands then set the layout scale and the entire neuronal core
  is crushed into a small dense blob. That is why cosine is the main embedding.

  WHERE IT STILL DIFFERS FROM FIG 1B, stated because a reader will see it in one
  second. In the published panel, 01 IT-ET Glut is a large lobe INSIDE the main mass
  and 10 LSX GABA is an interior territory. In our cosine embedding both are detached:
  only 1.2% of the 01 centroids and 0.7% of the 10 LSX centroids fall in the continent.
  Cosine does not fix that, and this docstring does not claim it does. The cause is the
  one above: the centroid kNN graph for 01 is almost entirely within-class, so UMAP has
  nearly no attractive force holding it to the mass.

  Euclidean is NOT hidden: it is one of the two metrics in the variants grid
  (umap_variants.csv), so both layouts ship and a reader can see the
  difference. The claim being made for cosine is that it keeps the gaps proportionate
  and so preserves the continent-plus-satellites topology of the published figure; the
  judgement is BY EYE, against the printed panel, and no comparison against the
  published per-cell coordinates was ever made. Cosine also scores higher on the one
  quantitative handle available (25-NN class purity over cell types: 0.857 against
  euclidean's 0.812, and it is the only variant that gains purity over its own PCA
  input rather than losing a little), but that is a bonus found after the fact, not the
  reason for the choice. Both metrics' purity is reported here so nobody has to take
  either statement on trust.

  Two documented departures from the Methods, neither of them a choice:
    1. Feature set. The paper used 8,460 marker genes from its own DEG pipeline,
       which Allen does not distribute. We use Allen's published MapMyCells marker
       union (6,558 genes), a close relative of that list but NOT the identical one.
    2. The paper drops the single PC whose correlation with per-cell log2(gene
       count) exceeds 0.7, a sequencing-depth artifact. A centroid has no per-cell
       depth, so the quantity being corrected for does not exist here. Skipped.
  The paper's <=1,000-cells-per-cluster subsampling and KNN imputation are likewise
  moot when a cluster is already a single averaged point.

VALIDATION
  We cannot compare against the published 4M-cell coordinates (not downloaded), so
  the embedding is judged on class purity instead: for each point, the fraction of
  its 25 nearest neighbours in UMAP space carrying the same class label. The same
  metric is computed in the 100-PC space that UMAP consumed, which is the honest
  baseline: it tells us whether UMAP preserved the class structure already present
  in the PCs, degraded it, or inflated it. A number near 1.0 in UMAP space is only
  impressive relative to the PCA number, never on its own.

  Purity is also computed for the EUCLIDEAN embedding at the same n_neighbors, so
  the metric choice is auditable. If cosine wins on purity, that is a bonus; if it
  ties or loses, the choice still stands on topological resemblance, which is what
  it was made for. The numbers are printed either way and neither is suppressed.

  RAW PURITY HAS A CEILING, AND IGNORING IT INVERTS THE CONCLUSION. A class holding
  n clusters cannot fill a 25-neighbour ball with its own kind unless n > 25: its
  maximum attainable purity is min(n - 1, 25) / 25. Ten of the 34 classes are that
  small. The two singleton classes (Pineal Glut, HY Gnrh1 Glut) have a ceiling of
  exactly 0 and are mathematically incapable of scoring above zero, no matter how
  cleanly they are embedded. Worse, CB Glut scores 0.320 raw, which lands it in the
  five "worst" classes, while its ceiling is also 0.320: it is PERFECTLY separated,
  every one of its 8 siblings inside every neighbourhood. Ranking classes on raw
  purity would therefore report a flawless class as a failure. So we carry
  purity_ceiling alongside the raw numbers and rank on purity / ceiling, which is
  the fraction of the attainable structure actually recovered. Raw values are still
  written, because they are what the neighbourhood fraction literally is.

  UMAP reproducibility is asserted by fitting twice under the same seed and
  requiring identical coordinates.

  THE TOPOLOGY IS MEASURED, NOT EYEBALLED. Every caption on the UMAP figures makes a
  claim about the SHAPE of the layout ("one continent plus satellites"). An earlier
  pass made that claim from an impression and got it wrong, shipping a docstring that
  called 01 IT-ET Glut an interior lobe when it is in fact a detached island. So
  island_structure() now computes the thing being claimed: single-linkage components of
  the layout at a fixed multiple of the median nearest-neighbour spacing, the fraction
  of points in the largest one, and for every island its size, its member classes and
  its gap to the continent as a fraction of the layout span. It runs on both nn=25
  layouts, lands in embedding_summary.json, and the plotting stage generates its
  topology sentence from those numbers and asserts against them.

THE PCA CACHE
  A full exact SVD of the 5,322 x 32,285 matrix takes most of this script's runtime
  and is the same every time, so both PCA results are cached under
  results/wmb10x/cache/ (gitignored). It sits under results/, NOT under
  data/processed/, because PCA scores and components are a compute cache of THIS
  analysis and nothing else: they are keyed on this script's solver, seed and library
  versions, and no other analysis would reuse them. data/processed/WMB-10X keeps only
  what is genuinely reusable (the re-encoded matrix and the deterministic joins over
  the raw downloads). A cache entry is keyed on a SHA-256 of: the size, the
  mtime_ns AND the full byte content of centroids.npy; the exact ordered list of gene
  ids that formed the columns; n_components; the SVD solver; the PCA seed; and the
  numpy and scikit-learn versions. The content hash is the belt to the mtime's
  braces: a stale cache silently feeding a changed matrix is the one failure mode
  worth paying 2 seconds of hashing to make impossible. A hit still asserts the
  loaded shapes and dtypes against the matrix in hand before anything downstream may
  touch them. Pass --no-cache to force a recompute and rewrite the entry.

Reads (all on disk, nothing is downloaded), from data/processed/WMB-10X/:
  centroids.npy                 (5322, 32285) float32, mean log2(CPM+1)
  genes.csv                     ensembl_id, gene_symbol, in centroids column order
  marker_genes.txt              6,558 Ensembl marker ids
  cluster_metadata.csv          one row per cluster, in centroids ROW ORDER

Writes, into results/wmb10x/ (gitignored, regenerable):
  pca_coords.csv                row_index, cluster_id, PC1..PC10 (ALL-GENE PCA)
  pca_variance.csv              pc, then per-PC and cumulative variance for BOTH the
                                all-gene PCA and the marker-gene PCA. EVERY variance
                                column names its gene set; there is no unqualified
                                column for a figure to grab. The UMAP consumed the
                                marker columns.
  pca_loadings_top.csv          top 15 +/- genes per PC1..PC5, by symbol (all-gene)
  umap_coords.csv               row_index, cluster_id, UMAP1, UMAP2 (MAIN: cosine)
  umap_variants.csv             metric x n_neighbors grid, 6 variants, is_main flag
  umap_knn_purity.csv           class_name, n_clusters, n_cells, purity_umap_cosine,
                                purity_umap_euclidean, purity_marker_pc,
                                purity_all_gene_pc, + the ceiling and ceiling-
                                normalized columns the ranking above requires. Every
                                purity names both the SPACE and the METRIC it was
                                measured in, because the two PCAs here are different
                                spaces whose purities differ by 0.039 and the two UMAP
                                metrics differ by 0.046.
  embedding_summary.json        headline numbers (including the island structure of
                                both nn=25 layouts), so captions need not recompute

  cache/*.npz                   the SVD cache (gitignored, regenerable)

A NOTE ON WHAT PURITY IS NOT. Every neighbourhood in this file is a neighbourhood of
25 CELL TYPES. No per-cell neighbourhood is ever computed, because no cell-level data
exists here. Cluster size enters only as a weight, so the weighted figures are named
*_cluster_size_weighted and never "cell-weighted": "85% of cells have same-class
neighbours" is a claim about cells that this pipeline cannot make.

Reproduce:
  cd scripts/wmb10x
  PYTHONPATH=../../src python embed_centroids.py [--no-cache]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

import umap

ROOT = Path(__file__).resolve().parents[2]  # scripts/wmb10x/x.py -> repo root
PROC = ROOT / "data" / "processed" / "WMB-10X"
OUT = ROOT / "results" / "wmb10x"          # named for scripts/wmb10x, which writes it
CACHE = OUT / "cache"                       # SVD scores/components: OUR compute cache,
                                            # not data anyone else would reuse

CENTROIDS_NPY = PROC / "centroids.npy"
GENES_CSV = PROC / "genes.csv"
MARKERS_TXT = PROC / "marker_genes.txt"
CLUSTER_META_CSV = PROC / "cluster_metadata.csv"

N_CLUSTERS = 5322
N_CLASSES = 34
N_GENES = 32285
N_MARKERS = 6558

N_PCS = 100          # paper: top 100 PCs
N_NEIGHBORS = 25     # paper: n_neighbors = 25
MIN_DIST = 0.4       # paper: min_dist = 0.4
SVD_SOLVER = "full"  # exact; see run_pca
PCA_SEED = 0
UMAP_SEED = 42
PURITY_K = 25        # neighbourhood size for the class-purity check

# Island structure. Two points are in the same component if a chain of hops, each no
# longer than LINK_MULT times the layout's median nearest-neighbour spacing, joins
# them. The multiple is scale-free by construction (it is measured in units of the
# layout's own point spacing), which is what lets one threshold serve two layouts whose
# absolute spans differ by 2.7x. The result is stable from 10x to 20x, and that
# stability is asserted rather than assumed.
LINK_MULT = 10.0
LINK_MULT_CHECK = 20.0
MIN_ISLAND = 10      # islands smaller than this are noise, not features of the picture

# The metric the paper never states. cosine is ours, chosen for topological
# resemblance to the published Fig 1B; euclidean is kept in the grid, not hidden.
MAIN_METRIC = "cosine"
VARIANT_METRICS = ("euclidean", "cosine")
VARIANT_NEIGHBORS = (15, 25, 50)
MAIN_VARIANT = f"{MAIN_METRIC}_nn{N_NEIGHBORS}"


# --------------------------------------------------------------------- transform


def center_genes(x: np.ndarray) -> np.ndarray:
    """Mean-center each GENE (column). No variance scaling. That is the point.

    The mean is accumulated in float64 for accuracy, then applied in float32 to keep
    the 5322 x 32285 matrix at 687 MB rather than 1.4 GB.
    """
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    xc = x - mean
    resid = np.abs(xc.mean(axis=0, dtype=np.float64)).max()
    assert resid < 1e-3, f"genes are not mean-centered: max |column mean| = {resid:.3e}"
    return xc


def assert_not_scaled(xc: np.ndarray, label: str) -> None:
    """Guard the one transform decision that matters: no z-scoring happened.

    If the columns had been variance-scaled, every gene's standard deviation would be
    1.0. Real log2(CPM+1) centroid data has a wide, skewed spread of gene variances,
    so we require that spread to still be there.
    """
    sd = xc.std(axis=0, dtype=np.float64)
    assert not np.allclose(sd, 1.0, atol=1e-3), (
        f"{label}: every gene has unit variance, so the matrix WAS variance-scaled. "
        "This pipeline must not z-score."
    )
    frac_unit = float(np.mean(np.abs(sd - 1.0) < 1e-3))
    assert frac_unit < 0.5, (
        f"{label}: {frac_unit:.1%} of genes have unit variance, which looks like scaling"
    )
    print(f"  [{label}] gene sd: min {sd.min():.4f}, median {np.median(sd):.4f}, "
          f"max {sd.max():.4f}  (not 1.0 => unscaled, as required)")


def run_pca(xc: np.ndarray, n_components: int, label: str) -> tuple[np.ndarray, PCA]:
    """PCA on an already-centered matrix. Returns (scores, fitted model).

    svd_solver="full", not "randomized". The randomized solver under-captured the
    top-100 cumulative variance by 3.45e-04 (0.677204 against the exact 0.677549,
    checked by eigendecomposing the 5322 x 5322 Gram matrix in float64), which is
    invisible in the coordinates but flips the caption digit from 67.8% to 67.7%.
    A figure should not quote a number that is wrong in the last digit it prints,
    and a full SVD of a 5,322-row matrix is affordable. NOT "covariance_eigh":
    that solver forms a p x p covariance matrix, and p = 32,285 here.
    """
    model = PCA(n_components=n_components, svd_solver=SVD_SOLVER, random_state=PCA_SEED)
    scores = model.fit_transform(xc)

    # We centered explicitly above; sklearn centers again, so its learned mean must be
    # ~0. If it is not, our centering and sklearn's disagree and the scores are suspect.
    learned = np.abs(model.mean_).max()
    assert learned < 1e-3, (
        f"{label}: sklearn re-centered by up to {learned:.3e}, so the input was not "
        "already centered as assumed"
    )
    total = float(model.explained_variance_ratio_.sum())
    assert 0.0 < total <= 1.0 + 1e-6, (
        f"{label}: explained variance ratios sum to {total}, which is not a fraction"
    )
    return scores, model


# ------------------------------------------------------------------- the PCA cache


def _file_digest(path: Path, chunk: int = 8 << 20) -> str:
    """SHA-256 of a file's bytes. 687 MB in about 2 s, against a 20 min SVD."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def cache_key(matrix_digest: str, matrix_stat: os.stat_result,
              gene_ids: list[str], n_components: int) -> str:
    """Everything that could change the PCA result, hashed into one name.

    Deliberately over-inclusive. A miss costs a recompute; a false HIT would silently
    ship coordinates that do not belong to the matrix on disk, so every input the SVD
    depends on goes in: the content AND the stat of centroids.npy, the exact ordered
    gene ids that form the columns (this is what separates the all-gene entry from the
    marker entry), the component count, the solver, the seed, and the library versions
    that implement the decomposition.
    """
    h = hashlib.sha256()
    h.update(b"wmb10x-pca-cache-v1\0")
    h.update(matrix_digest.encode())
    h.update(f"{matrix_stat.st_size}\0{matrix_stat.st_mtime_ns}\0".encode())
    h.update(f"{len(gene_ids)}\0".encode())
    h.update("\0".join(gene_ids).encode())
    h.update(f"{n_components}\0{SVD_SOLVER}\0{PCA_SEED}\0".encode())
    h.update(f"numpy{np.__version__}\0sklearn{sklearn.__version__}\0".encode())
    return h.hexdigest()[:32]


def pca_cached(xc_fn, label: str, key: str, n_rows: int, n_genes: int,
               n_components: int, use_cache: bool):
    """Return (scores, explained_variance_ratio, components), from cache if valid.

    xc_fn is a thunk: on a cache hit it is never called, so the 687 MB centered copy
    is never even materialized. On a miss it is called, the PCA runs, and the result
    is written through a .part temp file so an interrupted write cannot be picked up
    as a complete cache entry on the next run.

    The shape and dtype of everything loaded is asserted against the matrix we are
    actually working with. The key already makes a mismatch essentially impossible;
    the assertions make it impossible AND loud.
    """
    path = CACHE / f"{label}_{key}.npz"
    if use_cache and path.exists():
        with np.load(path) as z:
            scores = z["scores"]
            evr = z["explained_variance_ratio"]
            components = z["components"]
        assert scores.shape == (n_rows, n_components), (
            f"cached {label} scores are {scores.shape}, expected "
            f"{(n_rows, n_components)}; the cache is stale, delete {path}"
        )
        assert evr.shape == (n_components,), (
            f"cached {label} explained_variance_ratio is {evr.shape}, expected "
            f"{(n_components,)}; the cache is stale, delete {path}"
        )
        assert components.shape == (n_components, n_genes), (
            f"cached {label} components are {components.shape}, expected "
            f"{(n_components, n_genes)}; the cache is stale, delete {path}"
        )
        assert scores.dtype == np.float32 and components.dtype == np.float32, (
            f"cached {label} arrays are {scores.dtype}/{components.dtype}, not float32; "
            "the cache was written by a different pipeline, delete it"
        )
        assert np.isfinite(scores).all() and np.isfinite(evr).all(), (
            f"cached {label} arrays contain NaN or inf; delete {path}"
        )
        assert (np.diff(evr) <= 1e-6).all(), (
            f"cached {label} explained-variance ratios are not non-increasing, so the "
            f"entry is not a PCA spectrum; delete {path}"
        )
        assert 0.0 < float(evr.sum()) <= 1.0 + 1e-6, (
            f"cached {label} explained-variance ratios sum to {evr.sum()}, not a "
            f"fraction; delete {path}"
        )
        print(f"  [{label}] PCA cache HIT ({path.name}), SVD skipped")
        return scores, evr, components

    if use_cache:
        print(f"  [{label}] PCA cache MISS, running the exact SVD (minutes)")
    else:
        print(f"  [{label}] --no-cache, running the exact SVD (minutes)")

    xc = xc_fn()
    assert_not_scaled(xc, label)
    scores, model = run_pca(xc, n_components, label)
    del xc

    evr = np.asarray(model.explained_variance_ratio_)
    components = np.asarray(model.components_)
    scores = np.ascontiguousarray(scores, dtype=np.float32)
    components = np.ascontiguousarray(components, dtype=np.float32)

    CACHE.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".npz.part")
    with open(part, "wb") as fh:
        np.savez(fh, scores=scores, explained_variance_ratio=evr, components=components)
    part.replace(path)                      # atomic: a partial file is never a hit
    print(f"  [{label}] cached to {path.name}")
    return scores, evr, components


# ---------------------------------------------------------------------- embedding


def knn_purity(coords: np.ndarray, labels: np.ndarray, k: int = PURITY_K,
               metric: str = "euclidean") -> np.ndarray:
    """Per-point fraction of the k nearest neighbours sharing the point's label.

    Self is excluded from its own neighbourhood. Ties in distance are broken by
    sklearn's ordering, which is deterministic for a fixed input. The neighbourhood is
    euclidean by default in EVERY space, including the space of a cosine UMAP: a UMAP
    layout is a picture in the plane, and what a reader reads off it is euclidean
    proximity. The cosine metric was an input to the layout, not a property of the
    layout.

    The metric is nonetheless a parameter, for one specific check. Since the main UMAP
    is fitted with a cosine graph, a euclidean-kNN baseline in marker-PC space could be
    accused of stacking the deck: the cosine UMAP might simply be inheriting a purer
    cosine input graph. So the marker-PC baseline is ALSO measured with metric="cosine"
    (its own natural metric) and that number is written to the summary, where the README
    quotes it. Without this the claim "the cosine UMAP clears its own baseline" would be
    the one number in the README that the pipeline could not regenerate.
    """
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(coords)
    _, idx = nn.kneighbors(coords)
    assert (idx[:, 0] == np.arange(len(coords))).all(), (
        "the nearest neighbour of a point is not itself; duplicate rows would break "
        "the self-exclusion below"
    )
    neighbor_labels = labels[idx[:, 1:]]           # drop self
    return (neighbor_labels == labels[:, None]).mean(axis=1)


def island_structure(xy: np.ndarray, labels: np.ndarray, mult: float = LINK_MULT) -> dict:
    """Connected components of a 2-D layout, and each island's gap to the continent.

    This measures the thing the figures CLAIM. "One continent with satellites around
    its rim" is a statement about connected components and gap sizes, and it is the
    statement a reader checks first by holding our panel next to the published one, so
    it is computed here and quoted from here rather than eyeballed.

    Everything is reported in units of the layout itself: the linkage threshold is a
    multiple of the median nearest-neighbour spacing, and each gap is a fraction of the
    layout span. A UMAP's absolute scale is a free parameter of the optimisation, so an
    absolute gap of "5.5 units" means nothing and would not be comparable between the
    cosine layout (span 24) and the euclidean one (span 66). A gap of "23% of span" is.
    """
    nn = NearestNeighbors(n_neighbors=2).fit(xy)
    d, _ = nn.kneighbors(xy)
    median_1nn = float(np.median(d[:, 1]))
    comp = fcluster(linkage(xy, method="single"), t=mult * median_1nn,
                    criterion="distance")
    span = float(max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])))

    sizes = pd.Series(comp).value_counts()
    continent = int(sizes.index[0])
    in_continent = comp == continent
    cont_xy = xy[in_continent]

    islands = []
    for cid, n in sizes.items():
        if int(cid) == continent or n < MIN_ISLAND:
            continue
        pts = xy[comp == cid]
        gap = float(np.linalg.norm(pts[:, None, :] - cont_xy[None, :, :], axis=-1).min())
        members = pd.Series(labels[comp == cid]).value_counts()
        islands.append({
            "n_points": int(n),
            "gap_to_continent_frac_of_span": round(gap / span, 4),
            "classes": {str(k): int(v) for k, v in members.items() if v >= 3},
        })
    islands.sort(key=lambda i: -i["n_points"])

    # Which classes are NOT in the continent. This is the list a caption must not
    # misdescribe, so it is generated rather than typed.
    detached = {}
    for cls in np.unique(labels):
        frac = float(in_continent[labels == cls].mean())
        if frac < 0.5:
            detached[str(cls)] = round(frac, 4)

    return {
        "linkage_threshold_x_median_1nn": mult,
        "median_1nn_spacing": round(median_1nn, 5),
        "layout_span": round(span, 3),
        "continent_fraction": round(float(in_continent.mean()), 4),
        "n_islands_ge_%d_points" % MIN_ISLAND: len(islands),
        "islands": islands,
        "classes_mostly_outside_the_continent": detached,
    }


def fit_umap(pcs: np.ndarray, n_neighbors: int, metric: str, seed: int) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=MIN_DIST,
        metric=metric,
        random_state=seed,
        verbose=False,
    )
    return np.asarray(reducer.fit_transform(pcs), dtype=np.float64)


def overall(purity: np.ndarray, n_cells: np.ndarray, ceiling: np.ndarray) -> dict:
    """The three headline flavours of one purity vector.

    unweighted        : every cell TYPE counts once. The headline, and the only thing
                        centroid data can support.
    cluster_size_wtd  : each cell type weighted by how many cells it stands for. NOT a
                        per-cell measurement, and not named as one.
    ceiling_normalized: fraction of the ATTAINABLE purity. Points in the 2 singleton
                        classes have a zero ceiling and are undefined, so they are
                        excluded rather than silently scored as 0.
    """
    defined = ceiling > 0
    return {
        "unweighted": float(purity.mean()),
        "cluster_size_weighted": float(np.average(purity, weights=n_cells)),
        "ceiling_normalized": float((purity[defined] / ceiling[defined]).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore the PCA cache and rerun both exact SVDs")
    args = ap.parse_args()
    use_cache = not args.no_cache

    OUT.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- load
    x = np.load(CENTROIDS_NPY)
    genes = pd.read_csv(GENES_CSV)
    markers = [g for g in MARKERS_TXT.read_text().split("\n") if g]
    meta = pd.read_csv(CLUSTER_META_CSV)

    assert x.shape == (N_CLUSTERS, N_GENES), f"centroids are {x.shape}"
    assert len(genes) == N_GENES, f"genes.csv has {len(genes)} rows"
    assert len(markers) == N_MARKERS, f"marker_genes.txt has {len(markers)} ids"
    assert len(meta) == N_CLUSTERS, f"cluster_metadata.csv has {len(meta)} rows"
    assert meta["row_index"].tolist() == list(range(N_CLUSTERS)), (
        "cluster_metadata.csv is not in centroids.npy row order; every coordinate "
        "table written below would be mislabelled"
    )
    assert meta["class_name"].nunique() == N_CLASSES
    assert np.isfinite(x).all(), "centroid matrix contains NaN or inf"
    assert x.min() >= 0.0, (
        f"centroids have negative values (min {x.min():.3f}); they are supposed to be "
        "log2(CPM+1) and therefore non-negative, so they may already be centered"
    )

    cluster_id = meta["cluster_id"].to_numpy()
    class_name = meta["class_name"].to_numpy()
    n_cells = meta["n_cells"].to_numpy()
    symbols = genes["gene_symbol"].to_numpy()
    ensembl = genes["ensembl_id"].to_numpy()

    print(f"loaded {N_CLUSTERS:,} cluster centroids x {N_GENES:,} genes "
          f"(log2(CPM+1), range {x.min():.2f} to {x.max():.2f})")
    print(f"       {N_CLASSES} classes, {n_cells.sum():,} cells represented\n")

    marker_pos = pd.Index(ensembl).get_indexer(markers)
    assert (marker_pos >= 0).all(), (
        f"{int((marker_pos < 0).sum())} marker genes are not columns of centroids.npy"
    )
    assert len(set(marker_pos)) == N_MARKERS, "marker gene columns are not unique"

    # One digest of the matrix bytes, reused by both cache keys.
    digest = _file_digest(CENTROIDS_NPY)
    stat = CENTROIDS_NPY.stat()
    key_all = cache_key(digest, stat, list(ensembl), N_PCS)
    key_mark = cache_key(digest, stat, list(markers), N_PCS)
    assert key_all != key_mark, (
        "the all-gene and marker-gene cache keys collided, so one PCA could be served "
        "from the other's entry"
    )

    # ------------------------------------------------- deliverable 1: PCA
    print("PCA over all genes (mean-centered, NOT variance-scaled)")
    pca_scores, evr, pca_components = pca_cached(
        lambda: center_genes(x), "all_genes", key_all,
        N_CLUSTERS, N_GENES, N_PCS, use_cache,
    )
    cum = np.cumsum(evr)

    pca_coords = pd.DataFrame({"row_index": meta["row_index"], "cluster_id": cluster_id})
    for i in range(10):
        pca_coords[f"PC{i + 1}"] = pca_scores[:, i]
    pca_coords.to_csv(OUT / "pca_coords.csv", index=False)

    # Top loadings, so a reader can see what the axes actually mean.
    loading_rows = []
    for pc in range(5):
        w = pca_components[pc]
        order = np.argsort(w)                      # ascending: most negative first
        for direction, idx in (("negative", order[:15]), ("positive", order[-15:][::-1])):
            for rank, gi in enumerate(idx, start=1):
                loading_rows.append({
                    "pc": pc + 1,
                    "direction": direction,
                    "rank": rank,
                    "gene_symbol": symbols[gi],
                    "ensembl_id": ensembl[gi],
                    "loading": float(w[gi]),
                })
    pd.DataFrame(loading_rows).to_csv(OUT / "pca_loadings_top.csv", index=False)
    print(f"  PC1 {evr[0]:.1%}, PC2 {evr[1]:.1%}, PC3 {evr[2]:.1%} | "
          f"first {N_PCS} PCs: {cum[-1]:.1%} of total variance\n")

    # ----------------------------------------------- deliverable 2: UMAP
    print(f"marker-gene PCA on {N_MARKERS:,} Allen markers (the space the UMAP consumes,")
    print("  a DIFFERENT space from the all-gene PCA above; do not swap their numbers)")
    marker_pcs, marker_evr, _ = pca_cached(
        lambda: center_genes(x[:, marker_pos]), "marker_genes", key_mark,
        N_CLUSTERS, N_MARKERS, N_PCS, use_cache,
    )
    marker_cum = np.cumsum(marker_evr)
    print(f"  marker PCA: PC1 {marker_evr[0]:.1%}, first {N_PCS} PCs {marker_cum[-1]:.1%}")

    # BOTH variance curves go in one table, and EVERY column names its gene set. The
    # scree figure draws both: the all-gene curve is what wmb_pca_classes.svg plots, and
    # the marker curve is what the UMAP consumed. Attributing the all-gene 100-PC number
    # to the UMAP is a real scientific error, and it was shipped once, so the table is
    # built to make it hard to commit again. In particular there is deliberately no bare
    # "cumulative" column: an unqualified name is exactly what a future caption reaches
    # for, and it would silently hand back the all-gene curve.
    pd.DataFrame({
        "pc": np.arange(1, N_PCS + 1),
        "explained_variance_ratio_all_genes": evr,
        "cumulative_all_genes": cum,
        "explained_variance_ratio_marker_genes": marker_evr,
        "cumulative_marker_genes": marker_cum,
    }).to_csv(OUT / "pca_variance.csv", index=False)

    print(f"\nUMAP (paper params: n_neighbors={N_NEIGHBORS}, min_dist={MIN_DIST}; "
          f"metric={MAIN_METRIC!r} is OURS, the paper never states one)")
    umap_xy = fit_umap(marker_pcs, N_NEIGHBORS, MAIN_METRIC, UMAP_SEED)

    # Reproducibility: same seed, same coordinates. Reported honestly either way.
    umap_repeat = fit_umap(marker_pcs, N_NEIGHBORS, MAIN_METRIC, UMAP_SEED)
    max_dev = float(np.abs(umap_xy - umap_repeat).max())
    reproducible = bool(np.array_equal(umap_xy, umap_repeat))
    if reproducible:
        print(f"  reproducibility: two fits at seed {UMAP_SEED} are bit-for-bit identical")
    else:
        span = float(np.ptp(umap_xy))
        print(f"  reproducibility: NOT bit-identical at seed {UMAP_SEED}; max coordinate "
              f"deviation {max_dev:.3e} ({max_dev / span:.2e} of the embedding span)")

    pd.DataFrame({
        "row_index": meta["row_index"],
        "cluster_id": cluster_id,
        "UMAP1": umap_xy[:, 0],
        "UMAP2": umap_xy[:, 1],
    }).to_csv(OUT / "umap_coords.csv", index=False)

    # ------------------------------------------------------ variant grid
    # The main embedding IS one cell of this grid, so it is entered here rather than
    # refitted: the panel and the headline figure cannot then drift apart.
    print("\nvariant grid (metric x n_neighbors, same 100 marker PCs)")
    fits: dict[tuple[str, int], np.ndarray] = {(MAIN_METRIC, N_NEIGHBORS): umap_xy}
    variant_frames = []
    for metric in VARIANT_METRICS:
        for k in VARIANT_NEIGHBORS:
            if (metric, k) not in fits:
                fits[(metric, k)] = fit_umap(marker_pcs, k, metric, UMAP_SEED)
            xy = fits[(metric, k)]
            key = f"{metric}_nn{k}"
            variant_frames.append(pd.DataFrame({
                "variant": key,
                "metric": metric,
                "n_neighbors": k,
                "is_main": key == MAIN_VARIANT,
                "row_index": meta["row_index"],
                "cluster_id": cluster_id,
                "UMAP1": xy[:, 0],
                "UMAP2": xy[:, 1],
            }))
            print(f"  {metric:>9} n_neighbors={k:<3} done" +
                  ("   <- MAIN (same array as umap_coords.csv)" if key == MAIN_VARIANT
                   else ""))
    variants = pd.concat(variant_frames, ignore_index=True)
    variants.to_csv(OUT / "umap_variants.csv", index=False)
    assert len(variants) == N_CLUSTERS * len(VARIANT_METRICS) * len(VARIANT_NEIGHBORS)
    assert variants["is_main"].sum() == N_CLUSTERS, "exactly one variant must be main"
    main_panel = variants.loc[variants["is_main"], ["UMAP1", "UMAP2"]].to_numpy()
    assert np.array_equal(main_panel, umap_xy), (
        "the MAIN panel of umap_variants.csv is not the embedding in umap_coords.csv"
    )

    euclid_xy = fits[("euclidean", N_NEIGHBORS)]

    # -------------------------------------------------------- validation
    # Class purity in UMAP space, and the same metric in the 100-PC MARKER space that
    # UMAP was given. That marker-PC number is the baseline: UMAP is only "faithful"
    # relative to the space it consumed.
    #
    # THE SPACE IS PART OF THE NUMBER. There are two different PCAs in this script,
    # and their purities differ by 0.039. A name like "purity_pca" invites a caption
    # to attach the marker-space number to the all-gene PCA figure, which would
    # overstate that figure's class coherence. So every purity carries the space it
    # was measured in, and the all-gene number is computed here rather than left to
    # be guessed at.
    #
    # THE METRIC IS PART OF THE NUMBER TOO. The euclidean embedding at the same
    # n_neighbors is scored alongside the cosine one, because we changed the main
    # metric on the strength of a LOOK, and a reader is entitled to see whether the
    # look cost anything measurable. If it did, that is reported, not buried.
    #
    # AND THE NEIGHBOURHOOD IS ALWAYS OVER CELL TYPES. knn_purity only ever sees the
    # 5,322-row centroid arrays. There is no per-cell neighbourhood anywhere in this
    # pipeline, so no number here may be called a "cell-weighted purity": cluster size
    # is a WEIGHT on cell types, never a count of cells with neighbourhoods of their
    # own. The keys say cluster_size_weighted for that reason.
    #
    # NAME THE METRIC, NOT JUST THE SPACE. The headline column used to be "purity_umap",
    # and when the main metric changed from euclidean to cosine that column silently
    # changed meaning (0.812 -> 0.857) while keeping its name, which is how a stale
    # quote survives a revision. Both UMAP columns now name their metric.
    purity_cos = knn_purity(umap_xy, class_name, PURITY_K)
    purity_eucl = knn_purity(euclid_xy, class_name, PURITY_K)
    purity_marker_pc = knn_purity(marker_pcs, class_name, PURITY_K)
    purity_allgene_pc = knn_purity(pca_scores, class_name, PURITY_K)
    # The cosine UMAP's OWN natural baseline: marker-PC purity measured with a cosine
    # kNN rather than a euclidean one. Without this, "cosine beats its input" is open to
    # the charge that it merely inherited a purer cosine input graph.
    purity_marker_pc_cos = knn_purity(marker_pcs, class_name, PURITY_K, metric="cosine")

    per_point = pd.DataFrame({
        "class_name": class_name,
        "n_cells": n_cells,
        "purity_umap_cosine": purity_cos,
        "purity_umap_euclidean": purity_eucl,
        "purity_marker_pc": purity_marker_pc,
        "purity_all_gene_pc": purity_allgene_pc,
    })
    per_class = (per_point.groupby("class_name")
                 .agg(n_clusters=("n_cells", "size"),
                      n_cells=("n_cells", "sum"),
                      purity_umap_cosine=("purity_umap_cosine", "mean"),
                      purity_umap_euclidean=("purity_umap_euclidean", "mean"),
                      purity_marker_pc=("purity_marker_pc", "mean"),
                      purity_all_gene_pc=("purity_all_gene_pc", "mean"))
                 .reset_index()
                 .sort_values("class_name"))
    assert len(per_class) == N_CLASSES

    # The attainable maximum, see the module docstring. Without this column, a class
    # that is perfectly separated but small reads as a failed one.
    per_class["purity_ceiling"] = (
        np.minimum(per_class["n_clusters"] - 1, PURITY_K) / PURITY_K
    )
    obs_max = (per_class["purity_umap_cosine"] - per_class["purity_ceiling"]).max()
    assert obs_max < 1e-9, (
        f"a class exceeds its own purity ceiling by {obs_max:.3e}, so the ceiling "
        "formula or the neighbourhood size is wrong"
    )
    ceil_ok = per_class["purity_ceiling"].replace(0.0, np.nan)
    per_class["purity_umap_cosine_norm"] = per_class["purity_umap_cosine"] / ceil_ok
    per_class["purity_umap_euclidean_norm"] = per_class["purity_umap_euclidean"] / ceil_ok
    per_class["purity_marker_pc_norm"] = per_class["purity_marker_pc"] / ceil_ok
    per_class.to_csv(OUT / "umap_knn_purity.csv", index=False)

    ceiling_per_point = per_point["class_name"].map(
        per_class.set_index("class_name")["purity_ceiling"]
    ).to_numpy()
    n_undefined = int((ceiling_per_point <= 0).sum())

    o_cos = overall(purity_cos, n_cells, ceiling_per_point)
    o_euc = overall(purity_eucl, n_cells, ceiling_per_point)
    o_mark = overall(purity_marker_pc, n_cells, ceiling_per_point)
    o_all = overall(purity_allgene_pc, n_cells, ceiling_per_point)

    # --------------------------------------------------- island structure
    # What the figures actually claim about the shape of the layout, measured. See
    # island_structure(). Both nn=25 layouts, so the caption can contrast them with
    # numbers instead of adjectives.
    topo = {v: island_structure(xy, class_name)
            for v, xy in ((MAIN_VARIANT, umap_xy), (f"euclidean_nn{N_NEIGHBORS}", euclid_xy))}
    for v, xy in ((MAIN_VARIANT, umap_xy), (f"euclidean_nn{N_NEIGHBORS}", euclid_xy)):
        loose = island_structure(xy, class_name, mult=LINK_MULT_CHECK)
        drift = abs(loose["continent_fraction"] - topo[v]["continent_fraction"])
        assert drift < 0.05, (
            f"{v}: the continent fraction moves by {drift:.3f} between a {LINK_MULT}x and "
            f"a {LINK_MULT_CHECK}x linkage threshold, so the island structure is an "
            "artifact of the threshold and must not be quoted in a caption"
        )

    # -------------------------------------------------- validation
    o_mark_cos = overall(purity_marker_pc_cos, n_cells, ceiling_per_point)

    # Rank on the normalized score. Singleton classes (undefined) are held out of the
    # ranking rather than parked at the bottom, where they would be uninformative.
    ranked = (per_class.dropna(subset=["purity_umap_cosine_norm"])
              .sort_values("purity_umap_cosine_norm", ascending=False))
    best, worst = ranked.head(5), ranked.tail(5).iloc[::-1]
    singletons = per_class[per_class["purity_ceiling"] == 0]["class_name"].tolist()
    n_capped = int((per_class["purity_ceiling"] < 1.0).sum())

    summary = {
        "n_points_are_cluster_centroids": N_CLUSTERS,
        "points_are_cells": False,
        "purity_neighbourhoods_are_over": "cell types (cluster centroids), never cells",
        "all_gene_pca": {"n_genes": N_GENES},
        "pc1_variance_ratio": float(evr[0]),
        "pc2_variance_ratio": float(evr[1]),
        # The two cumulative-variance numbers belong to two DIFFERENT spaces. The UMAP
        # consumed the marker one. wmb_pca_classes.svg plots the all-gene one. A figure
        # that attributes the all-gene PCs to the UMAP is simply wrong. There is no
        # unqualified "cumulative_variance_100pc" key here for a caption to reach for.
        "all_gene_pca_cumulative_variance_100pc": float(cum[-1]),
        "marker_pca_cumulative_variance_100pc": float(marker_cum[-1]),
        "pca_100pc_is_input_to_umap": "marker_genes",
        "purity_k": PURITY_K,
        # Every purity below is a mean over the 5,322 CELL TYPES of the fraction of a
        # cell type's 25 nearest CELL TYPES that share its class. Each key names the
        # SPACE and, for the UMAPs, the METRIC.
        "umap_cosine_class_purity_unweighted": o_cos["unweighted"],
        "umap_cosine_class_purity_cluster_size_weighted": o_cos["cluster_size_weighted"],
        "umap_cosine_class_purity_ceiling_normalized": o_cos["ceiling_normalized"],
        "umap_euclidean_class_purity_unweighted": o_euc["unweighted"],
        "umap_euclidean_class_purity_cluster_size_weighted":
            o_euc["cluster_size_weighted"],
        "umap_euclidean_class_purity_ceiling_normalized": o_euc["ceiling_normalized"],
        "marker_pca_class_purity_unweighted": o_mark["unweighted"],
        "marker_pca_class_purity_cluster_size_weighted": o_mark["cluster_size_weighted"],
        "marker_pca_class_purity_ceiling_normalized": o_mark["ceiling_normalized"],
        # The cosine UMAP's own natural baseline, measured with a cosine kNN in the same
        # marker-PC space. The README quotes this; the pipeline must be able to make it.
        "marker_pca_class_purity_unweighted_cosine_knn": o_mark_cos["unweighted"],
        "all_gene_pca_class_purity_unweighted": o_all["unweighted"],
        "all_gene_pca_class_purity_cluster_size_weighted":
            o_all["cluster_size_weighted"],
        "all_gene_pca_class_purity_ceiling_normalized": o_all["ceiling_normalized"],
        "n_classes_with_purity_ceiling_below_1": n_capped,
        "n_points_with_undefined_ceiling": n_undefined,
        "singleton_classes_purity_undefined": singletons,
        "umap_reproducible_bitwise": reproducible,
        "umap_repeat_max_coord_deviation": max_dev,
        "main_variant": MAIN_VARIANT,
        "umap_params": {"n_neighbors": N_NEIGHBORS, "min_dist": MIN_DIST,
                        "metric": MAIN_METRIC, "random_state": UMAP_SEED,
                        "n_pcs": N_PCS, "n_marker_genes": N_MARKERS},
        # The measured shape of both nn=25 layouts. Captions generate their topology
        # sentence from this, and assert against it, instead of describing a picture
        # from memory. See island_structure().
        "island_structure": topo,
        "umap_metric_provenance": (
            "The Methods of Yao et al. 2023 specify n_neighbors=25 and min_dist=0.4 and "
            "never state a distance metric. cosine is OUR choice. It does not change "
            "WHICH groups detach at centroid level (01 IT-ET Glut, 02 NP-CT-L6b Glut, "
            "10 LSX GABA, the non-neuronal block and 31 OPC-Oligo break off under both "
            "metrics), it changes how large those gaps are relative to the layout: 01 "
            "sits 23% of the span from the continent under cosine against 75% under "
            "euclidean, so under cosine the islands no longer set the scale and the "
            "neuronal continent survives as the dominant structure, as in the published "
            "Fig 1B. The resemblance is judged BY EYE against the printed panel and is "
            "not complete: Fig 1B has 01 IT-ET Glut and 10 LSX GABA inside the main "
            "mass, and neither of our embeddings does. No comparison against the "
            "published per-cell coordinates was made. euclidean is kept in the variants "
            "grid, and its class purity is reported alongside."
        ),
    }
    (OUT / "embedding_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # ----------------------------------------------------------- summary
    print()
    print("=" * 78)
    print("WMB-10X embeddings: ALL ASSERTIONS PASSED")
    print("=" * 78)
    print(f"  points              : {N_CLUSTERS:,} cluster centroids (not cells)")
    print("  PCA, all 32,285 genes, mean-centered, unscaled:")
    print(f"    PC1 {evr[0]:.2%} | PC2 {evr[1]:.2%} | PC1+PC2 {evr[0] + evr[1]:.2%}")
    print(f"    cumulative over {N_PCS} PCs: {cum[-1]:.2%}  (all-gene space; what "
          "wmb_pca_classes.svg plots)")
    print(f"  PCA, {N_MARKERS:,} marker genes (the UMAP's actual input space):")
    print(f"    cumulative over {N_PCS} PCs: {marker_cum[-1]:.2%}  (a DIFFERENT number "
          "for a DIFFERENT space)")
    print()
    print(f"  class purity of {PURITY_K}-NN neighbourhoods over CELL TYPES (34 classes)")
    print("                        unweighted  size-weighted  ceiling-norm")
    for name, o in (("UMAP cosine  (MAIN)", o_cos),
                    ("UMAP euclidean     ", o_euc),
                    ("marker 100-PC      ", o_mark),
                    ("all-gene 100-PC    ", o_all)):
        print(f"    {name}   {o['unweighted']:.3f}       {o['cluster_size_weighted']:.3f}"
              f"          {o['ceiling_normalized']:.3f}")
    print("    (no purity here is a per-cell number: every neighbourhood is over cell "
          "types,")
    print("     and cluster size is only ever a weight)")

    d_metric = o_cos["unweighted"] - o_euc["unweighted"]
    if abs(d_metric) < 0.01:
        verdict = ("a WASH: the metric change is justified by topological resemblance "
                   "to Fig 1B, NOT by purity")
    elif d_metric > 0:
        verdict = f"BETTER by {d_metric:+.3f} unweighted"
    else:
        verdict = (f"WORSE by {d_metric:+.3f} unweighted; the metric change is justified "
                   "by topological resemblance to Fig 1B, not by purity")
    print(f"    cosine against euclidean: {verdict}")

    delta = o_cos["unweighted"] - o_mark["unweighted"]
    verb = "preserved" if abs(delta) < 0.02 else ("inflated" if delta > 0 else "degraded")
    print(f"    the MAIN UMAP {verb} the class structure already in the marker PCs "
          f"({delta:+.3f})")
    print(f"    the same marker-PC baseline under a COSINE kNN (its own natural metric, "
          f"so the cosine UMAP cannot be accused of inheriting a purer input graph): "
          f"{o_mark_cos['unweighted']:.3f}")
    print(f"  ceiling correction: {n_capped} of {N_CLASSES} classes hold <= {PURITY_K} "
          f"clusters, so raw purity CANNOT reach 1.0 for them")
    print(f"    ({n_undefined} points in singleton classes have a zero ceiling and are "
          "excluded from the normalized column)")
    print(f"  UMAP reproducible at seed {UMAP_SEED}: {reproducible} "
          f"(max deviation {max_dev:.3e})")

    print()
    print("  ISLAND STRUCTURE of the layout (what the figures claim, measured; "
          f"single linkage at {LINK_MULT:g}x the median 1-NN spacing)")
    for v in (MAIN_VARIANT, f"euclidean_nn{N_NEIGHBORS}"):
        t = topo[v]
        print(f"    {v:<15} continent holds {t['continent_fraction']:.1%} of the "
              f"centroids; {t['n_islands_ge_%d_points' % MIN_ISLAND]} islands "
              f">= {MIN_ISLAND} points")
        for isl in t["islands"][:4]:
            who = ", ".join(f"{k} ({v_})" for k, v_ in isl["classes"].items())
            print(f"      n={isl['n_points']:>4}  gap "
                  f"{isl['gap_to_continent_frac_of_span']:.1%} of span :: {who}")
    print("    the metric does not change WHICH groups detach, it changes how far out "
          "they land:")
    for cls in ("01 IT-ET Glut", "10 LSX GABA"):
        c_frac = topo[MAIN_VARIANT]["classes_mostly_outside_the_continent"].get(cls)
        e_frac = topo[f"euclidean_nn{N_NEIGHBORS}"][
            "classes_mostly_outside_the_continent"].get(cls)
        print(f"      {cls:<16} in-continent fraction: cosine {c_frac}, "
              f"euclidean {e_frac}  (Fig 1B has it INSIDE the mass; we do not "
              "reproduce that)")

    print("  5 highest-purity classes (MAIN UMAP, by fraction of ceiling):")
    for _, r in best.iterrows():
        print(f"    {r['class_name']:<28} {r['purity_umap_cosine_norm']:.3f} of ceiling  "
              f"(raw {r['purity_umap_cosine']:.3f}, {int(r['n_clusters']):>4} clusters)")
    print("  5 lowest-purity classes (MAIN UMAP, by fraction of ceiling):")
    for _, r in worst.iterrows():
        print(f"    {r['class_name']:<28} {r['purity_umap_cosine_norm']:.3f} of ceiling  "
              f"(raw {r['purity_umap_cosine']:.3f}, {int(r['n_clusters']):>4} clusters)")
    print(f"    (excluded, ceiling is 0 so purity is undefined: "
          f"{', '.join(singletons)})")
    print("  wrote:")
    for name in ("pca_coords.csv", "pca_variance.csv", "pca_loadings_top.csv",
                 "umap_coords.csv", "umap_variants.csv", "umap_knn_purity.csv",
                 "embedding_summary.json"):
        print(f"    {OUT / name}")


if __name__ == "__main__":
    main()
