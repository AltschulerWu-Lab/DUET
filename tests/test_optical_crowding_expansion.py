# tests/test_optical_crowding_expansion.py
import numpy as np
import pandas as pd
from duet.optical_crowding import compute_diffraction_limit, simulate_optical_crowding


def test_diffraction_limit_divides_by_expansion():
    base = compute_diffraction_limit(wavelength_nm=500.0, numerical_aperture=1.4,
                                     expansion_factor=1.0, model="abbe")
    assert np.isclose(base, 0.5 / (2 * 1.4))  # (500/1000)/(2*NA)
    for E in (2.0, 4.0):
        assert np.isclose(
            compute_diffraction_limit(500.0, 1.4, expansion_factor=E, model="abbe"),
            base / E,
        )


def test_expansion_reduces_crowding_monotonically():
    rng = np.random.default_rng(0)
    genes = [f"g{i}" for i in range(50)]
    # constant-weight-ish toy codebook (length 16, HW 4) is fine for the monotonicity check
    seqs = []
    for _ in genes:
        bits = ["0"] * 16
        for p in rng.choice(16, size=4, replace=False):
            bits[p] = "1"
        seqs.append("".join(bits))
    codebook = pd.DataFrame({"Gene": genes, "Sequence": seqs})
    expr = pd.DataFrame({"gene_name": genes, "mean_raw_counts": rng.uniform(1, 100, size=50)})

    def conflict(E):
        return simulate_optical_crowding(
            codebook, expr, expression_col="mean_raw_counts", gene_col="gene_name",
            total_reads=20000, cell_size_um=20.0, expansion_factor=E, seed=42,
        ).conflict_fraction

    c1, c2, c4 = conflict(1.0), conflict(2.0), conflict(4.0)
    assert c1 >= c2 >= c4, f"crowding should fall with expansion: {c1:.3f},{c2:.3f},{c4:.3f}"
