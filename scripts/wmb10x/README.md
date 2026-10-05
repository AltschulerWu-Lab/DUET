# WMB-10X cell-type landscape: PCA and a Fig. 1B UMAP

A PCA of gene expression and a reconstruction of the Fig. 1B UMAP from the Allen
whole-mouse-brain 10x scRNA-seq atlas (Yao et al. 2023, Nature). Six SVGs, plus the
coordinate tables they were drawn from.

The **publication artefact is `figures/wmb_umap_panel.svg`**, an 89 mm
(`FIGURE_WIDTHS["half_page"]`) panel meant to be assembled with other panels in Inkscape.
It ships whole and also split into a scatter and a key, for a compiled figure whose panels
share one key. It carries **no panel letter** (letters are added when the panels are
assembled into a figure), and these scripts never write a compiled figure. The other three
SVGs are full-page diagnostics.

The folder also holds `build_whole_brain_cpm.py`, which builds the whole-brain expression
table that the MERFISH experiments read (see
[The whole-brain expression table](#the-whole-brain-expression-table)). It shares the two
large atlas inputs with the scripts above but is not part of the figure pipeline.

## The one thing to know before you reuse anything here

**Every point is a cell type, not a cell.** All figures plot the 5,322 transcriptomic
**cluster centroids** of the atlas, one point per cluster, each the mean log2(CPM+1)
profile of that cluster. They are not the ~4.04M individual cells that Yao et al. drew
in Fig. 1B.

The reason is a deliberate cost decision: cell-level expression is a 94.7 GB download
(10xv3 subset) to about 150 GB (full atlas) from the ABC Atlas S3 bucket, and it was
declined. The per-cluster mean matrix is 1.4 GB, and that is what these scripts read.

What the choice costs, stated plainly:

- **Within-cluster spread is absent by construction.** A real UMAP of 4M cells has
  fuzzy, filled islands. These islands are sparser and cleaner because each cluster
  contributes exactly one point. The figure looks like Fig. 1B in topology, not in
  texture.
- **Abundance is not in the geometry.** A cluster of 50 cells and one of 50,000 are both
  one point. Only `wmb_pca_classes.svg` (full page) restores any sense of abundance: there,
  point *diameter* increases with log10(cluster cell count), rescaled between a fixed
  minimum and maximum. That is an affine map, not a proportional one: the smallest cluster
  still gets a visible dot. Its size key (10 / 100 / 1,000 / 10,000 / 100,000 cells)
  brackets the data, since the median cluster holds **122** cells and 91% hold under 1,000.
  **The 89 mm panel drops the encoding entirely** (see "Why the panel has no size key"
  below), and `wmb_umap_variants.svg` never had one, because the only claim it makes is
  topological.
- **Every purity number below is a neighborhood of 25 cell types.** No per-cell
  neighborhood is computed anywhere, because no cell-level data exists here. Cluster size
  enters only as a weight, which is why the weighted columns are named
  `*_cluster_size_weighted` and never "cell-weighted". "85% of cells have same-class
  neighbors" is a claim about cells that this pipeline cannot make.

**This caveat is no longer printed on the face of any figure, and that is deliberate.** The
figures used to carry a footnote line saying it. A publication panel takes its caveat from
its caption, and a footnote baked into the SVG would be typeset twice on the page, in a
font the journal never set. So the footnote is gone from every figure, panel and diagnostic
alike, and `results/wmb10x/captions.md` is now the **only** place the caveat is written. It
is rewritten by `plot_embeddings.py` on every run, from the measured numbers, so it cannot
drift from the figures; the panel's caption is written to be pasted into a legend as it
stands. **A figure from here must not travel without its caption.** `plot_embeddings.py`
asserts on every save that no figure has grown a footnote back.

## Where everything lives

| Directory | Holds | Tracked? |
|---|---|---|
| `scripts/wmb10x/` | the four pipeline scripts, `build_whole_brain_cpm.py` and this README | yes |
| `data/raw/WMB-10X/` | the 5 downloaded ABC metadata files (16 MB) and the two large inputs (1.4 GB) | no, gitignored |
| `data/processed/WMB-10X/` | `centroids.npy`, `genes.csv`, `marker_genes.txt`, `cluster_metadata.csv` | no, gitignored |
| `examples/data/processed/WMB-10X/` | `whole_brain_per_gene_cpm.csv`, written by `build_whole_brain_cpm.py` | no, the atlas values are not redistributed |
| `results/wmb10x/` | the coordinate tables, `captions.md`, `figures/*.svg`, and `cache/` | no, `results/` is blanket-gitignored |

The split is on reusability, not on size. `data/processed/WMB-10X` holds only things
another analysis could legitimately pick up: `centroids.npy` is a dtype-cast re-encoding
of the h5 matrix, and the three small tables are deterministic joins over the raw
downloads. The PCA cache is **not** in that category, which is why it sits in
`results/wmb10x/cache/` instead: it is SVD scores and components keyed on this script's
solver, seed and library versions, a compute cache of this analysis and of nothing else.

The results directory is named for the scripts directory that writes it: `scripts/wmb10x`
-> `results/wmb10x`.

## Data provenance

| Input | Size | Where it comes from |
|---|---|---|
| `wmb_precomputed_stats.h5` | 1.4 GB | `data/raw/WMB-10X/`, downloaded if absent (see below). `f["sum"]` is the (5,322 clusters x 32,285 genes) mean log2(CPM+1) matrix; `f["taxonomy_tree"]` carries the 4-level hierarchy and the per-cluster `alias`. Its `f["n_cells"]` is a placeholder of all 1s and is never used. |
| `wmb_gene.csv` | 2.3 MB | Ensembl id to gene symbol. Resolved the same way as the h5. ABC key `metadata/WMB-10X/20241115/gene.csv`. |
| `cluster_annotation_term.csv` | 0.9 MB | ABC `metadata/WMB-taxonomy/20231215/`. Class and subclass names plus Allen's official `color_hex_triplet`. |
| `cluster.csv` | 0.13 MB | ABC `metadata/WMB-taxonomy/20231215/`. Per-cluster cell counts. They sum to 4,042,976, the paper's cell total. |
| `membership_pivoted.csv` | 0.53 MB | ABC `metadata/WMB-taxonomy/20231215/views/`. Used as an independent cross-check on the cluster join. |
| `term_with_counts.csv` | 0.9 MB | ABC `metadata/WMB-taxonomy/20231215/views/`. Taxonomy terms with cluster and cell counts. |
| `mouse_markers_230821.json` | 13.5 MB | ABC `mapmycells/WMB-10X/20240831/`. Marker genes per taxonomy node; the union over the 340 nodes is 6,558 genes, all present in the h5 columns. |

The five metadata files total about 16 MB and are fetched by `download_wmb_metadata.py`,
which records the exact S3 key of each, skips files already present, and writes through a
`.part` temp file so an interrupted fetch cannot masquerade as a complete one. Cell-level
expression (94.7 GB to 150 GB) is never downloaded by any step here.

### The two large inputs

The h5 and `wmb_gene.csv` are **not** part of that 16 MB fetch. `resolve_large_input()` in
`download_wmb_metadata.py` looks for each one in `data/raw/WMB-10X/<name>`. If the file is
not there, it downloads it from the ABC bucket into that folder (h5 key
`mapmycells/WMB-10X/20240831/precomputed_stats_ABC_revision_230821.h5`). Both
`build_centroid_matrix.py` and `build_whole_brain_cpm.py` **print the path they read** at the
top of every run, so no run leaves you guessing which bytes it read.

If you keep the two files elsewhere, link each file into `data/raw/WMB-10X/`. Link the
files, not the folder: the download step writes into that folder, and a folder link would
write through it.

**Join trap, asserted in code.** `cluster.csv`'s `label` (`CS20230722_0001`) looks like the
h5's cluster id (`CS20230722_CLUS_0001`) with a token dropped. It is not. The label is the
zero-padded `cluster_alias`, and the alias is not the cluster index: alias 1 is
`CS20230722_CLUS_0326`. Rewriting the string matches 5,030 of the 5,322 clusters and
attaches the wrong cell count to every one of them. The correct key is the `alias` field
that each cluster carries in the h5 taxonomy's `name_mapper`. That join is bijective, it is
asserted total, and the names and classes it produces are cross-checked against
`membership_pivoted.csv` (0 mismatches out of 5,322).

## The PCA transform

One transform, no variants:

1. The h5 values are **already** mean log2(CPM+1). No further normalization, no log, no
   CPM step.
2. **Mean-center each gene.**
3. **No variance scaling and no z-scoring.** Deliberate: scaling to unit variance would
   give a rare, noisy, low-expression gene the same say in the geometry as a strong
   marker. `assert_not_scaled()` checks after the fact that the spread of per-gene
   standard deviations is still wide (all genes: min 0.000, median 0.201, max 4.346), so a
   scaling step cannot creep back in silently.
4. Exact SVD (`svd_solver="full"`). The randomized solver under-captured the 100-PC
   cumulative variance by 3e-4, enough to move a caption digit.

`wmb_pca_classes.svg` and `wmb_pca_scree.svg` show the PCA over **all 32,285 genes**.
The UMAP consumes a **separate** PCA over the 6,558 marker genes. These are two different
spaces and their purities are not interchangeable (see Validation).

## The UMAP recipe: the paper's Methods against what we did

| Step | Yao et al. Methods | Here | Same? |
|---|---|---|---|
| Feature set | 8,460 marker genes from their own DEG pipeline | 6,558 genes, the union of Allen's published MapMyCells marker sets | **No.** Allen does not distribute the 8,460-gene list. Ours is a close relative of it, not that list. |
| Input matrix | per-cell, KNN-imputed, capped at 1,000 cells per cluster | one mean profile per cluster | Not applicable. A cluster is already a single averaged point, so subsampling and imputation have nothing to act on. |
| Dimensionality reduction | PCA, top 100 PCs | PCA, top 100 PCs | Yes. |
| Depth correction | drop the one PC with r > 0.7 against per-cell log2(gene count) | not done | **No.** A centroid has no per-cell sequencing depth, so the quantity being corrected for does not exist. Skipped, not forgotten. |
| `n_neighbors` | 25 | 25 | Yes. |
| `min_dist` | 0.4 | 0.4 | Yes. |
| metric | **never stated** | `cosine` | **Our call, not theirs.** See below. |
| seed | not stated | `random_state=42`, bit-for-bit reproducible on refit | n/a |

Two of these are forced, not chosen: one is a file Allen does not publish, the other a
correction that is undefined on centroids. The metric is different. The Methods leave it
free and we picked it, which is the next section.

### The metric is the one free parameter, and we chose it

The Methods pin down `n_neighbors` and `min_dist` and say **nothing at all** about the
distance metric. There is therefore no faithful choice available here, only a defensible
one, and euclidean would have been just as unforced as cosine.

**What the metric does not change: which groups detach.** On centroid data the same handful
of groups (01 IT-ET Glut, 02 NP-CT-L6b Glut, 10 LSX GABA, the non-neuronal 30-34 block, 31
OPC-Oligo) breaks off under *both* metrics, and island membership is essentially
metric-invariant from 5x to 20x the median nearest-neighbour spacing. A centroid carries no
within-cluster spread, so the cells that would bridge those groups to the rest in a
4M-cell embedding do not exist here, and only between-class gaps are left. No metric can
put the bridge back.

**What the metric does change: how big the gaps are, and therefore what sets the scale.**

| | 01 IT-ET Glut island | non-neuronal island | continent holds |
|---|---|---|---|
| **cosine** (main) | 23% of layout span | 7% | 82% of the cell types |
| euclidean | 75% of layout span | 53% | 83% of the cell types |

Under cosine the islands stay proportionate and the continent of neuronal territories
remains the dominant structure of the picture, which is what Fig. 1B shows. Under euclidean
the same gaps blow out to most of the layout; since a UMAP is scaled to whatever spans it,
those islands then set the scale and the entire neuronal core is compressed into a small
dense blob. **That is why cosine is the main embedding.**

**Where it still does not match Fig. 1B, stated because a reader will see it in one
second.** In the published panel, 01 IT-ET Glut is a large lobe *inside* the main mass and
10 LSX GABA is an interior territory. In our cosine embedding both are **detached**: only
1.2% of the 01 centroids and 0.7% of the 10 LSX centroids fall in the continent. Cosine
does not fix that and no metric can, for the reason above. So the resemblance to Fig. 1B is
real but **partial**, it is **qualitative and by eye**, and no comparison against the
published per-cell coordinates was ever made (see Validation).

None of this is guesswork about a picture. `embed_centroids.py` measures the connected
components of each layout (`island_structure()`), writes them to
`results/wmb10x/embedding_summary.json`, and `plot_embeddings.py` generates its topology sentence
from those numbers and asserts against them. An earlier pass wrote that sentence from an
impression and claimed 01 IT-ET Glut was an interior lobe; it is an island, and the numbers
now make that claim impossible to ship.

Cosine also scores higher on the one quantitative handle we have (25-NN class purity over
cell types, 0.857 against 0.812, and it is the only variant that *gains* purity over its
own PCA input instead of losing a little), but that was found afterwards and is a bonus,
not the reason.

**Euclidean is not hidden.** It is one of the two metrics in `wmb_umap_variants.svg` and
in `results/wmb10x/umap_variants.csv`, and its purity is reported next to cosine's, so the choice
is auditable and reversible by anyone who disagrees with it.

## Validation, and what was not validated

**We did not compare against the published per-cell UMAP coordinates.** Those coordinates
sit inside the cell-level metadata that was never downloaded, so no point-for-point
agreement with Fig. 1B has been measured, and none is claimed. The similarity to Fig. 1B is
qualitative and by eye.

What was measured is **class purity of k-NN neighborhoods**: for each of the 5,322
centroids, the fraction of its 25 nearest neighbors (in cell-type space, always) that carry
the same one of the 34 class labels. The number only means something against a baseline, so
it is computed in three spaces:

| Space | Class purity (unweighted) | Cluster-size weighted | Ceiling-normalized |
|---|---|---|---|
| UMAP, cosine (**the reconstruction**) | **0.857** | 0.888 | 0.865 |
| UMAP, euclidean (same `n_neighbors`) | 0.812 | 0.862 | 0.818 |
| 100 marker-gene PCs (what the UMAP consumed) | **0.820** | 0.836 | 0.828 |
| 100 all-gene PCs (what `wmb_pca_classes.svg` plots) | 0.781 | 0.806 | 0.788 |

Read it honestly. A high purity in UMAP space is unimpressive on its own; it is only
informative next to the PC baseline the UMAP started from. Against that baseline, **the
cosine UMAP is the only variant that adds class coherence rather than shedding a little**
(0.857 against its 0.820 input), while euclidean lands slightly *below* its own input
(0.812). This is not an artifact of measuring the baseline with the wrong metric: the
marker-PC baseline re-measured with a **cosine** k-NN, its own natural metric, gives
**0.827** (`marker_pca_class_purity_unweighted_cosine_knn` in
`results/wmb10x/embedding_summary.json`), so the cosine UMAP still clears its own natural baseline
by +0.030.

None of that is why cosine was chosen. It was chosen for resemblance to Fig. 1B, by eye,
and the purity gain was found afterwards.

**The topology was also measured, not just looked at.** `island_structure()` computes the
single-linkage components of each layout at a fixed multiple of the median
nearest-neighbour spacing (stable from 10x to 20x, and that stability is asserted), the
fraction of points in the largest component, and every island's size, member classes and
gap to the continent as a fraction of the layout span. Both nn=25 layouts are measured and
the results are in `results/wmb10x/embedding_summary.json` under `island_structure`. This exists
because the topology is the thing the figures claim, and claiming it from an impression is
how the "01 is an interior lobe" error got shipped.

**Raw purity has a ceiling.** A class holding n clusters cannot fill a 25-neighbor ball
with its own kind unless n > 25, so its maximum attainable raw purity is min(n-1, 25)/25.
Ten of the 34 classes are that small, and the two singleton classes (15 HY Gnrh1 Glut, 25
Pineal Glut) have a ceiling of exactly 0 and cannot score above zero however cleanly they
embed. Ranking classes on raw purity therefore reports perfectly separated small classes as
failures: 29 CB Glut scores 0.320 raw, which is exactly its ceiling, meaning every one of
its 8 siblings is inside every neighborhood. `results/wmb10x/umap_knn_purity.csv` carries
`purity_ceiling` and the normalized columns, and the ceiling-normalized column of the table
above is the fraction of the *attainable* structure each space actually recovers. Rank on
those, never on the raw values.

Also asserted: cluster count 5,322, class count 34, cell-count sum 4,042,976, every cluster
resolving to exactly one class and one color, explained-variance ratios monotone and summing
to at most 1, and the UMAP reproducing its coordinates exactly on a refit under the same
seed.

## How to rerun

The environment is **`scanpy_env`** (`environments/scanpy.yml`). It has `umap-learn`
(0.5.11) and everything else the scripts need (numpy, pandas, h5py, scikit-learn,
matplotlib). `bash experiments/wmb10x_landscape/run.sh` runs the four steps below with
the right settings. To run them by hand, call the environment's interpreter by path, as
`run.sh` does, so the variables below reach the scripts unchanged. The scripts hardcode
no paths: every path is derived from `__file__`.

```bash
cd scripts/wmb10x   # every path below is derived from __file__, so cwd is only convenience

PY=$(conda run -n scanpy_env python -c 'import sys; print(sys.executable)')
export PYTHONPATH=$(git rev-parse --show-toplevel)/src  # for duet.plotting
export MPLCONFIGDIR=$TMPDIR      # matplotlib cache, keeps it out of the repo
export NUMBA_CACHE_DIR=$TMPDIR   # umap compiles through numba and needs a writable cache

$PY download_wmb_metadata.py   # ~16 MB, idempotent, skips whatever is already present
$PY build_centroid_matrix.py   # h5 + metadata -> data/processed/WMB-10X/ (687 MB), ~1 min
$PY embed_centroids.py         # PCA + UMAP + purity -> results/wmb10x/*.csv, ~6 min cold
$PY plot_embeddings.py         # -> results/wmb10x/figures/*.svg + captions.md, ~1 min
```

`PYTHONPATH` is needed because the figures must go through `duet.plotting.apply_style()`,
the repo's shared publication style. The 34-class palette is the one sanctioned exception:
it uses Allen's official `color_hex_triplet`, since the shared palettes do not cover a
34-class taxonomy.

`embed_centroids.py` is the slow step. It does two exact SVDs (all-gene and marker-gene) and
then seven UMAP fits (the main one, its reproducibility refit, and the six variants). It is
CPU-bound and single-machine: about 6 minutes cold, about 3.5 with the SVDs cached. The
SVD results are cached under `results/wmb10x/cache/` (gitignored), keyed on a SHA-256 over
the byte content **and the mtime** of `centroids.npy`, the ordered gene list, the component
count, the solver, the seed and the library versions, so a stale entry cannot survive a
changed matrix. Because the mtime is in the key, rerunning `build_centroid_matrix.py`
invalidates the cache even when it rewrites byte-identical content: that costs one cold SVD
and is the intended trade, since a false cache hit is the far worse failure. Pass
`--no-cache` to force a recompute. The seven UMAP fits, not the SVDs, set the floor.

## Files

### Figures

**The panel** (89 mm, `FIGURE_WIDTHS["half_page"]`, for Inkscape assembly). No panel
letters: they are added when the figure is assembled.

| File | What |
|---|---|
| `figures/wmb_umap_panel.svg` | **The publication artefact.** 5,322 centroids, cosine, colored by class, equal aspect, bare UMAP axes, and the 34-entry color key **below** the scatter (3.16 x 1.53 in at 3 columns, 7 pt). Self-contained: 3.50 x 4.81 in. |
| `figures/wmb_umap_panel_nokey.svg` | The same scatter alone: points and axis arrows, no text but `UMAP1` / `UMAP2`. 3.50 x 3.22 in. |
| `figures/wmb_umap_key.svg` | The 34-entry key alone, on its own 89 mm canvas. 3.50 x 1.57 in. |

The split pair exists for the case where one shared key serves several panels of a compiled
figure. It is drawn from the same arrays as the whole panel, and `assert_identical_scatter`
then compares the two **rendered** collections point for point (offsets, marker areas, RGBA)
**and the size of the axes each was drawn into, in points on the page**, so they cannot
drift: a shared key placed beside a scatter that has moved, or that has been drawn at
another scale, is a key to a picture that is not there.

That second half of the check is there because its absence shipped a bug. The two figures
used to be handed to `constrained_layout` with different padding (the keyed one tuned the
engine, the key-less one took its defaults), so the engine gave them axes of different
widths and the key-less scatter printed **1.1% larger**. The old fingerprint compared data
coordinates and pt^2 marker areas, both of which are invariant to the size of the axes, so
it passed, and would have passed at any divergence. The panel family now takes no layout
engine at all: both figures place the axes at an explicitly computed rect (`_panel_figure`),
so they are the same picture at the same size **by construction**, and the assertion
confirms it on the rendered canvas (243.64 x 223.27 pt in both files, which is also what the
two SVGs' rasters measure). The diagnostics keep `constrained_layout`: they have tick
labels, axis labels and several panes to reconcile, and no twin they must match.

**Why the key is under the scatter and not in it.** At 89 mm the axes is 3.38 x 3.10 in.
The 34-entry key at the stylesheet's 7 pt is 2.13 x 2.18 in at two columns, so to sit
inside the axes it would need a corner void about 70% of the panel height, and the tallest
void this layout offers in either upper corner is **0.10 in**. That is measured
(`corner_void`), printed on every run, and asserted, so if the embedding ever changes shape
enough that a key *would* fit inside it, the run says so. The full-page figure could put its
key in the void; this one cannot, and the fix for that is a taller panel, never smaller
type. **Type is never reduced below the stylesheet to force a fit.**

**Why the panel has no size key.** It has no size encoding to key. The full-page figures map
point diameter affinely onto log10(cell count) and at 183 mm that reads. At 89 mm it does
not, and this was rendered both ways and looked at before it was cut: 91.4% of clusters hold
under 1,000 cells, so the encoding spends nearly half its diameter ramp (1.1 to 3.3 of a 1.1
to 6.0 pt range) on the top 8.6% of the points. The whole bulk of the data therefore lands
inside one step a reader cannot resolve at print size, while the fat dots of the 53 clusters
over 10,000 cells swallow their neighbours and destroy the streak texture that a uniform dot
resolves. An encoding a reader cannot decode is not information, it is ink, and it was
costing a key besides. The panel uses a uniform **2.2 pt** dot (1.8 goes faint at print, 2.6
starts closing the gaps in the continent) and spends the freed room on a bigger, cleaner
point. Cluster size is in `cluster_metadata.csv`, and the caption says the encoding is
absent.

**Nothing on any figure is allowed to nearly touch the data.** `assert_no_ink_under` used to
be a binary overlap test, which passes just as happily at 0.1 pt of gap as at 10, and 0.1 pt
is a label touching the data. It now **measures** the gap between every annotation's text box
and the nearest dot's ink, in points on the page, with the marker radius subtracted, and
holds it to a floor of 3 pt (a hair over one dot diameter, the smallest gap that still reads
as a gap at print size). The floor caught two real things:

- On the panel, the `UMAP1` label cleared the 02 NP-CT-L6b island by **1.3 pt**. The axis
  arrows were anchored *inside* the corner of the point cloud; they now sit a fifth of an
  arrow length outside it, in the axes padding, still plainly attached to the cloud, and the
  gap is **6.4 pt**. The run prints both label clearances every time.
- In the variants grid, the "not a rendering failure" note used to sit inside panel c,
  threaded through the corridor between the outlying islands and the crushed continent. At
  183 mm the best row cleared the data by **3.1 pt**, but that was measured before the
  cosine row existed: once constrained_layout had re-flowed the finished grid, the note sat
  **1.5 pt** from the data. At 180 mm no row in the corridor reaches the floor.
  The note is now one line hung 4 pt under the euclidean row (panels a to c), outside every
  panel, in paper the width-limited frames already left empty, so the canvas and every
  panel keep their size. It cannot reach a centroid there. `assert_text_clears_axes` holds
  it 3 pt off every panel frame, title and letter and off the key, on the finished layout,
  which is frozen before the check so that the layout checked is the layout written.

**The diagnostics** (full page, `FIGURE_WIDTHS["full_page"]`). Not publication panels.

| File | What |
|---|---|
| `figures/wmb_umap_variants.svg` | Six panels (a to f): the same embedding recomputed for 2 metrics x 3 neighborhood sizes. **Within** a metric the layout is stable across all three neighborhood sizes; **across** metrics it is not, and that is the point of the grid, since the metric is the parameter the Methods leave free. The framed panel **e** (cosine, 25 neighbours) is the embedding the panel above shows at publication size. |
| `figures/wmb_pca_classes.svg` | PC1 vs PC2 of the all-gene PCA, equal aspect, colored by class. The one figure that still carries the size encoding, and its size key. |
| `figures/wmb_pca_scree.svg` | Per-PC variance of the all-gene PCA (bars) plus **both** cumulative curves: all-gene (the space `wmb_pca_classes.svg` plots) and marker-gene (the space the UMAP consumed). Each is annotated with its own 100-PC value, 67.8% and 66.5%. They are different spaces over different gene sets, so neither percentage may be quoted as the other's. |

Every SVG comes out at exactly the `FIGURE_WIDTHS` entry it names (249.4 pt = 88 mm for the
three panel files, 510.2 pt = 180 mm for the three diagnostics), which is
asserted on every save. The stylesheet's `savefig.bbox: tight` is switched off for these figures:
shrink-wrapping the canvas to the ink gave three different widths across four figures, and
any rescale afterwards drifts the type away from the stylesheet target. Drop them into
Inkscape at 100%.

`wmb_umap_fig1b.svg` is gone. It was the full-page version of the same cosine embedding, and
`wmb_umap_panel.svg` replaces it: two renderings of one embedding at two widths, with two
different size treatments, is exactly the drift this pipeline spends its assertions
preventing.

### Tables

None of these are committed: `results/` is blanket-gitignored and
`data/processed/WMB-10X` is gitignored by name. Every one of them is regenerated by the
chain above, and the UMAP tables come back bit-for-bit at seed 42.

| File | What |
|---|---|
| `data/processed/WMB-10X/cluster_metadata.csv` | One row per cluster, in `centroids.npy` row order: cluster id and name, supertype, subclass, class id, class name, Allen class color, cell count. The join key for everything else, which is why it lives with the matrix it indexes rather than in `results/`. |
| `results/wmb10x/pca_coords.csv` | PC1 to PC10 of the all-gene PCA, per cluster. |
| `results/wmb10x/pca_variance.csv` | Explained-variance ratio and cumulative, PCs 1 to 100, for **both** PCAs: `*_all_genes` (32,285 genes) and `*_marker_genes` (6,558 genes, the space the UMAP consumed). **Every** column names its gene set, and there is deliberately no unqualified `cumulative` column, because an unqualified name is exactly what a caption reaches for when it means the other space. `plot_embeddings.py` asserts on load that no such column has come back. |
| `results/wmb10x/pca_loadings_top.csv` | Top 15 positive and negative genes on PC1 to PC5, by symbol. An interpretability check, not an input to anything. |
| `results/wmb10x/umap_coords.csv` | UMAP1, UMAP2 per cluster for the main embedding (cosine, n_neighbors=25). |
| `results/wmb10x/umap_variants.csv` | The same, for all six metric x n_neighbors variants (31,932 rows), with an `is_main` flag. 2.7 MB, the largest table written here. Kept deliberately, and regenerable bit-for-bit at seed 42. |
| `results/wmb10x/umap_knn_purity.csv` | Per class: cluster count, cell count, purity in UMAP-cosine / UMAP-euclidean / marker-PC / all-gene-PC space, the purity ceiling, and the ceiling-normalized values. Every purity column names the **space** it was measured in and, for the UMAPs, the **metric**: `purity_umap_cosine` and `purity_umap_euclidean`, never a bare `purity_umap`. (It used to be bare, and when the main metric changed from euclidean to cosine that column silently changed meaning, 0.812 to 0.857, under an unchanged name.) |
| `results/wmb10x/embedding_summary.json` | The headline numbers, so a caption never has to recompute an embedding. Includes `island_structure` (the measured connected-component structure of both nn=25 layouts) and `marker_pca_class_purity_unweighted_cosine_knn`, the 0.827 baseline quoted above. |
| `results/wmb10x/captions.md` | The figure captions, regenerated on every plotting run, and the **only** place the centroid caveat is now written (there is no on-figure footnote any more). The panel's caption leads and is written to be pasted into a supplementary legend as it stands: what a point is, the UMAP parameters, whose choice the metric was, where it does *not* match the published Fig. 1B, and that colour is the only class cue. |
| `results/wmb10x/cache/*.npz` | The SVD cache. Not data, and not a result either: a compute cache, deletable at the cost of one cold SVD. |

### Scripts (committed)

| File | What |
|---|---|
| `download_wmb_metadata.py` | The ~16 MB ABC metadata fetch, idempotent. Also owns `resolve_large_input()`, the h5 / `wmb_gene.csv` lookup described above. |
| `build_centroid_matrix.py` | h5 + metadata to `data/processed/WMB-10X/{centroids.npy, genes.csv, marker_genes.txt, cluster_metadata.csv}`. Where the join is asserted, and where the resolved h5 path is printed. |
| `embed_centroids.py` | The PCA, the UMAP, the variants, and the purity metrics. |
| `plot_embeddings.py` | Pure rendering. Reads only the tables the other two wrote, never recomputes an embedding, so a figure cannot disagree with the coordinates it was drawn from. |
| `build_whole_brain_cpm.py` | Not part of the figure pipeline. Builds the whole-brain expression table of the MERFISH experiments from the h5 and `wmb_gene.csv` (see below). |

## Four figure decisions worth knowing about

**No class labels are drawn inside any point cloud, and there is no zoom inset.** Classes
are identified by the 34-entry color key alone, exactly as Fig. 1B identifies them. An
earlier pass stamped the two-digit class codes into the clouds, and every clutter defect in
those figures came from it: a knot of overlapping label boxes in the PCA core, a row of
labels banished under the plot trailing a fan of leader lines, and a "04" sitting on class
01's cloud that read as a mislabel even though it was correctly anchored on its own class.
All of that machinery (anchors, declutter, banishment, leader lines, and the assertions that
existed only to police it) is deleted. The zoom inset on the non-neuronal island is gone
too: it existed only because the euclidean embedding crushed that island to about 2 mm on
the page.

**Colour is the only class identification, so its failures are disclosed in full.** Nine of
the 561 class pairs in Allen's palette are closer than a CIE76 dE of 25.
`plot_embeddings.py` computes them and prints them into every caption, together with how far
apart the two classes actually sit, because "position tells them apart" is a claim that has
to be checked rather than asserted:

- For most of them position *does* rescue the reader: 08 CNU-MGE GABA / 20 MB GABA (dE 12)
  are 18% of the layout span apart, 06 CTX-CGE GABA / 20 MB GABA (dE 13) are 36% apart, and
  so on. Every confusable pair is either under 5.1% of span apart or over 12.3%, with
  nothing in between, and the plotting code asserts that the gap is still there before it
  makes the binary claim.
- For the **non-neuronal block it does not**. 30 Astro-Epen, 32 OEC, 33 Vascular and 34
  Immune are four browns and greys whose closest pair differs by only dE 15, and they share
  one small rim satellite (30 / 34 are 3.9% of span apart, 32 / 34 are 5.1%). Nothing on the
  figure separates them. Read that satellite as non-neuronal and go to the tables for more.
  The palette is Allen's own and is **not** repainted here, so the same degeneracy is
  present in the published Fig. 1B.

**The UMAP panels are rotated for display, and the variants are scaled per panel.** A UMAP
encodes pairwise distances and nothing else: it has no canonical orientation and its
absolute scale is a free parameter of the optimization, not a property of the data. Each
panel is rigidly rotated (and, in the variants grid, reflected into the main panel's
orientation). Both are isometries and an assertion checks that every pairwise distance
survives. In the variants grid **each row shares one frame shape**, and each panel is scaled
independently, so apparent cluster size is **not** comparable between panels, only topology
is. Where a panel's own layout is a different shape from its row's (cosine at 50 neighbours,
panel f), it is padded and not stretched, so it does not fill its frame: an anisotropic
scale would distort the embedding, and white space is the cheaper price. The two rows differ
in frame shape because the euclidean layouts are wide thin streaks and the cosine ones are
nearly square; forcing one shape on all six leaves half the figure white. Rescaling all six
to a common RMS radius was tried and rejected: the euclidean embeddings put two islands far
from a compact core, so no single linear scale shows both the islands and the core. The
euclidean panels look nearly empty, which is the finding rather than a bug, and the note
under that row says so.

**The scree carries both PCAs.** `wmb_pca_scree.svg` used to plot the all-gene curve and
annotate its 100-PC cumulative as "input to the UMAP". It is not: the UMAP consumed the
*marker-gene* PCA, a different space over a different gene set. Both cumulative curves are
now drawn and separately annotated, and no annotation attributes one space's variance to the
other's role. The legend names gene sets, not source filenames.

## The whole-brain expression table

The MERFISH experiments (`experiments/merfish_zhang2023_v2`, `experiments/merfish_2000_genes`
and `experiments/merfish_expression_prior`) read a whole-brain mean CPM per gene,
`examples/data/processed/WMB-10X/whole_brain_per_gene_cpm.csv`, as their `expression:` table
and to order their baseline panels. The values come from the atlas, whose licence is
CC BY-NC 4.0, so the table is not in the repository. Build it once, in `scanpy_env`:

```bash
python scripts/wmb10x/build_whole_brain_cpm.py   # reads data/raw/WMB-10X/, downloads if absent
```

`--stats` and `--genes` point at copies of the two inputs kept elsewhere, and `--out`
writes the table to another path. The inputs used for the paper have the sha256
`b21ca985652fb25f9608f99005139a40757133a76fbe845ae5b175c5c26a447b`
(`wmb_precomputed_stats.h5`) and
`ec5a211c23fab5cce8a06f01bb0a68af087fb979b647ab42d635201f98de2f69` (`wmb_gene.csv`).

The script turns each cluster's mean log2(CPM+1) back into CPM (2^x - 1), averages over
the 5,322 clusters without weights, maps Ensembl ids to symbols, sums the 22 symbols that
several ids map to, keeps the genes above 0 and sorts them by descending CPM. The result
has 30,599 rows. The script then compares the sha256 of its output with that of the table
used in the paper (`a8314155531fefb3e7abfb4993f1f4c184ece9cdbf541fdf38cc2503584fd8e4`)
and warns if they differ. Its docstring explains each step, including the CSV round trip
that makes the output byte-identical.

## Seam

WMB-10X is scRNA-seq, not MERFISH, and this pipeline is not a MERFISH analysis, so it sits
at `scripts/wmb10x` on its own rather than under the MERFISH tree. The thread back to the
MERFISH experiments is the pair of large inputs in `data/raw/WMB-10X/`:
`build_whole_brain_cpm.py` reads the same h5 and `wmb_gene.csv` to build the expression
table those experiments use, and `experiments/merfish_expression_prior` reads them too.
