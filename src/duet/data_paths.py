"""Where the library factories look for their default reference tables.

The Weissman (Horlbeck et al. 2016) and CRISPick factories read default
tables from the source repository's ``data/`` folder, so their defaults
resolve only from a source checkout. Set the ``DUET_DATA_DIR`` environment
variable to a directory laid out like the repository's ``data/`` folder to use
them from an installed package.

Not every default table ships with DUET. The hCRISPRi-v2.1 table does. The
hCRISPRa-v2 and CRISPick tables do not: build them yourself, as the error for
a missing table explains.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR_ENV = "DUET_DATA_DIR"

# The repository's data/ folder, seen from src/duet/. Resolves only in a
# source checkout (or an editable install).
SOURCE_DATA_DIR = Path(__file__).parent.parent.parent / "data"

# Where each default table comes from, for error messages. "shipped" says
# whether the DUET repository includes the table; "obtain" (one sentence, no
# final period) tells the user how to get a table that is not shipped.
_SOURCES = {
    "processed/Horlbeck_2016/CRISPRi_v2_1.csv": {
        "source": (
            "the hCRISPRi-v2.1 library of Horlbeck et al. 2016, eLife 5:e19760 "
            "(Supplementary Table S3, sheet 'hCRISPRi-v2.1')"
        ),
        "shipped": True,
    },
    "processed/Horlbeck_2016/CRISPRa.csv": {
        "source": (
            "the hCRISPRa-v2 library of Horlbeck et al. 2016, eLife 5:e19760 "
            "(Supplementary Table S5)"
        ),
        "shipped": False,
        "obtain": (
            "Build it from that supplementary table (CC BY 4.0) as a CSV with "
            "the columns Gene, Sequence, Rank and Activity score, with the "
            "non-targeting controls in the Gene group negative_control"
        ),
    },
    "processed/CRISPick/crispick_ensembl_aggrCFD_top20_candidates.csv": {
        "source": (
            "the first 20 CRISPick picks per human gene (Broad Institute GPP; "
            "SpCas9 CRISPRko, Ensembl gene IDs, aggregate CFD) plus the "
            "CRISPick no-site and one-site intergenic control guides"
        ),
        "shipped": False,
        "obtain": (
            "CRISPick output is under the Broad Institute GPP terms of use, so "
            "build your own copy: download the three CRISPick files that "
            "scripts/data_processing/build_crispick_candidates.py names into "
            "data/raw/CRISPick, then run `python "
            "scripts/data_processing/build_crispick_candidates.py --raw-dir "
            "data/raw/CRISPick` from the repository root"
        ),
    },
}


def default_data_file(relpath: str, source_default: Path) -> Path:
    """Resolve a factory's default table.

    Args:
        relpath: Path of the table relative to the data directory, for example
            ``"processed/Horlbeck_2016/CRISPRi_v2_1.csv"``.
        source_default: The factory's source-checkout default, used when
            ``DUET_DATA_DIR`` is unset.

    Returns:
        ``$DUET_DATA_DIR/relpath`` if the variable is set, else
        ``source_default``.
    """
    override = os.environ.get(DATA_DIR_ENV)
    if override:
        return Path(override) / relpath
    return Path(source_default)


def missing_data_message(path: Path) -> str:
    """Error text for a missing table, naming where a default table comes from."""
    path = Path(path)
    msg = f"CSV file not found: {path}"
    for relpath, entry in _SOURCES.items():
        if path.as_posix().endswith(relpath):
            source = entry["source"]
            if entry["shipped"]:
                msg += (
                    f". This default table is {source}, preprocessed as in "
                    "the DUET source repository "
                    "(https://github.com/AltschulerWu-Lab/DUET, "
                    f"file data/{relpath})."
                )
            else:
                msg += (
                    f". This default table is {source}. It is not shipped "
                    f"with DUET. {entry['obtain']}."
                )
            msg += (
                f" To use a copy stored elsewhere, pass csv_path=..., or set "
                f"{DATA_DIR_ENV} to a directory that contains {relpath}."
            )
            break
    return msg
