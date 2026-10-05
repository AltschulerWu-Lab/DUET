Noise-channel matrices
======================

Fitted sequencing and readout error models that the experiments in
experiments/ read. docs/noise_channels.md (section "Channels in the
repository") shows how to load each one with duet.channels.

Convention
----------

- A channel matrix T has the transmitted symbol in rows and the observed
  symbol in columns: T[i, j] = P(observe j | transmitted i). Each row sums
  to 1.
- DNA matrices are in A, T, C, G order (build them with alphabet="ATCG").
- Binary (MERFISH) matrices are in 0, 1 order, 0 = off and 1 = on (build
  them with alphabet="01").
- A file with a leading round axis, shape (rounds, 4, 4), holds one matrix
  per sequencing round. A file of shape (rounds,) holds one error rate per
  round.


merfish_zhang2023_channel.npy
-----------------------------

Shape (2, 2). The binary readout channel of the MERFISH experiments
(experiments/merfish_2000_genes, merfish_zhang2023_v2 and
merfish_expression_prior):

    P(1->0) = 0.0561   (row 1: an "on" bit read as off)
    P(0->1) = 0.0145   (row 0: an "off" bit read as on)

Readout error rates were estimated with MERlin v0.1.6 from four samples of
the Zhang et al. (2023) whole-mouse-brain MERFISH data, pooled, and fitted
as a 2x2 binary channel. The decoding and error-rate scripts are not part of
this repository.

Source data: Zhuang X, Jung W, Zhang M (2023), Brain Image Library,
doi:10.35077/act-bag, licensed under CC BY-SA 4.0. Article: Zhang et al.
(2023), Nature, doi:10.1038/s41586-023-06808-9.
Changes: error rates estimated from the raw data and fitted as a 2x2 binary
channel.
Licence of this file: CC BY-SA 4.0
(https://creativecommons.org/licenses/by-sa/4.0/). The MIT licence of the
repository covers the code only.

This work used data from the Brain Image Library (RRID:SCR_017272), which is supported by the
National Institutes of Mental Health of the National Institutes of Health
under award number R24-MH-114793.


NIS-seq channels
----------------

Fitted to spot-level base calls of two HeLa IL-1β screens (E-9 and E-9C,
eight files) of Fandrey et al. (2025), Nat. Biotechnol.,
doi:10.1038/s41587-024-02516-5. Each read (14 nt) is matched to the Brunello
library, a per-screen intensity threshold at the 50th percentile is applied,
and the transmitted versus observed bases are counted in four classes:

    uniform              (4, 4)      symmetric channel, one error rate
    positional           (14,)       one error rate per round
    channel              (4, 4)      asymmetric channel, pooled over rounds
    positional_channel   (14, 4, 4)  one asymmetric matrix per round

The files differ in how many reads were drawn from each of the eight spot
files ({kind} is one of the four classes):

- archive/nisseq_hela_{kind}_subsample.npy
  100,000 reads per file. Read by experiments/ops_crispick_genome_wide (the
  genome-wide design). Despite its name, archive/ holds these live inputs;
  it is not a store of old files.
- nisseq/nisseq_hela_{kind}_pctl50_subsample50000.npy
  50,000 reads per file. The matrices in
  experiments/ops_crispri_cross_eval/noise_matrices/ come from these fits
  (the per-round ones cut to the first 10 rounds).
- nisseq/nisseq_hela_{kind}_pctl50_subsample5000.npy
  5,000 reads per file. Used only by the provenance check of
  experiments/nisseq_error_analysis.

experiments/nisseq_error_analysis rebuilds all of them and checks them byte
for byte against these files.

The NIS-seq channel matrices are fitted from spot-level base calls that the
authors of Fandrey et al. (2025) shared on request, and are included only
with their permission. If one is missing from your copy, rebuild it with
experiments/nisseq_error_analysis from the spot calls, which are available
from those authors on request.
