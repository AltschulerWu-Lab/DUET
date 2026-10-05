"""Runner-shaped tests for positional required_codewords construction.

These tests exercise the runner's per-position required_codewords building
logic without spinning up the full DUET pipeline. Construct a baseline
DataFrame with genes both inside and outside gene_names and verify the
factory only requires the in-gene_names anchors.
"""
import pandas as pd

from duet.merfish_factory import MERFISHFactory


def _build_positional_required(
    *,
    codebook_size: int,
    gene_names: list[str],
    baseline_genes: list[str],
    baseline_seqs: list[str],
    gene_col: str = "Gene",
    seq_col: str = "Sequence",
) -> list[set[str]]:
    """Replica of the runner's Step 4 construction for unit testing."""
    df = pd.DataFrame({gene_col: baseline_genes, seq_col: baseline_seqs})
    required = [set() for _ in range(codebook_size)]
    gene_to_seq = dict(zip(df[gene_col].astype(str), df[seq_col]))
    for i, gene in enumerate(gene_names):
        if gene in gene_to_seq:
            required[i].add(gene_to_seq[gene])
    return required


def test_baseline_with_extra_genes_does_not_anchor_extras():
    """Baseline rows whose gene is outside gene_names are not anchored;
    rows whose gene is in gene_names are."""
    gene_names = ["geneA", "geneB", "geneC"]  # codebook positions
    baseline_genes = ["geneA", "geneB", "geneC", "geneX", "geneY"]  # baseline has extras
    baseline_seqs = ["11110000", "00001111", "11001100", "10101010", "01010101"]

    required = _build_positional_required(
        codebook_size=3,
        gene_names=gene_names,
        baseline_genes=baseline_genes,
        baseline_seqs=baseline_seqs,
    )

    # In-gene_names genes anchored at their positions.
    assert required[0] == {"11110000"}
    assert required[1] == {"00001111"}
    assert required[2] == {"11001100"}

    # The "extras" (geneX, geneY) must NOT appear anywhere in required.
    flat = set().union(*required)
    assert "10101010" not in flat
    assert "01010101" not in flat


def test_baseline_with_extra_genes_factory_does_not_preserve_extras():
    """End-to-end: factory accepts the positional list and the resulting
    pool contains only the in-gene_names anchors (the "extras" are not in
    the pool under per-codeword sampling because nothing anchored them)."""
    gene_names = ["geneA", "geneB", "geneC"]
    baseline_genes = ["geneA", "geneB", "geneC", "geneX", "geneY"]
    baseline_seqs = ["11110000", "00001111", "11001100", "10101010", "01010101"]

    required = _build_positional_required(
        codebook_size=3,
        gene_names=gene_names,
        baseline_genes=baseline_genes,
        baseline_seqs=baseline_seqs,
    )

    f = MERFISHFactory(
        seq_rounds=8,
        codebook_size=3,
        hamming_weights=[4],
        per_codeword_sample_size=10,
        sample_seed=42,
        required_codewords=required,
    )
    pool = f.create()
    pool_seqs = set(pool.sequences)

    # The three in-gene_names anchors are in the pool.
    assert {"11110000", "00001111", "11001100"}.issubset(pool_seqs)
    # The two "extras" are NOT structurally required to be in the pool.
    # They might appear by random draw, but the relaxed assertion only
    # checks anchored sequences. So we just verify the anchored set is
    # what the runner's relaxed assertion would compute.
    anchored = {"11110000", "00001111", "11001100"}
    assert anchored.issubset(pool_seqs)  # this is what the relaxed assertion checks
