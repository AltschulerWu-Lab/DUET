# pep_grid_glyph: Fig 1b pairwise-error grid glyph

Fig 1 is a schematic drawn in Inkscape; no Fig 1 panel comes from an experiment. Its only script-made element is the pairwise-error-probability (PEP) grid glyph in Fig 1b, left ("Precompute pairwise error probability"). This folder regenerates that glyph.

**The values are illustrative placeholders, not results.** Nothing here is computed from a noise channel, a decoder, a candidate pool or any data file, and nothing is random, so there is no seed.
- The grid is the full 3-bit binary codebook, in Hamming-weight order (000, 001, 010, 100, 011, 101, 110, 111). Rows are the transmitted codeword and columns the decoded one. The diagonal (correct decoding) is blank.
- Each off-diagonal chip is coloured by the Hamming distance d between the two codewords: d = 1 `#b5473f`, d = 2 `#daa39f`, d = 3 `#f5e5e4`. That ordering (nearer codewords are likelier errors) is what a binary symmetric channel with a flip rate below 0.5 gives.
- The three colours are set-points picked by eye, not probabilities on a colour scale. They are the terracotta anchor `#b5473f` blended 0, 0.50 and 0.86 of the way to white.
- The colour bar has no numbers, only "High probability" / "Low probability".

## Run

```bash
conda activate duet          # the DUET environment (environment.yml) with the benchmark extra (run.sh checks it imports this clone's duet)
bash experiments/pep_grid_glyph/run.sh
```

- Works from any cwd. run.sh cd's into this folder, stops before any work unless `config.yaml` exists and the active env imports `duet` from this repo's `src/`, and then runs `generate_pep_grid.py --outdir <outdir>`.
- It then prints the sha256 of each file it wrote and compares the panel's sha256 with that of the paper's glyph (`SHIPPED_SHA256` in `run.sh`, `13b0473c…`; see below). A match prints `OK`. A difference prints a `WARNING` but does not fail the run.
- CPU only, about 3 s wall clock, almost all of it Python and matplotlib start-up. No PEP cache, no scratch, no GPU.
- `config.yaml` has two keys. `outdir` is relative to this folder. `rx_variants: true` also writes the five corner-radius comparison files (see below). A missing or null key stops run.sh, and so does an `rx_variants` value other than true or false.
- To run the script alone: `python generate_pep_grid.py --outdir <dir> [--rx-variants]`.

## Notes on the drawing

- **`duet.plotting`.** The script imports `duet.plotting` and calls `apply_style()`, like every plotting script in this repository.
  - **The stylesheet changes nothing in the output.** `apply_style()` sets matplotlib rcParams (Arial, 8 pt, and so on). This script writes the SVG as text and never draws through matplotlib, so the rcParams never reach the file.
  - **Fonts** come from the SVG's own CSS: `font-family: Arial, Helvetica, sans-serif; font-size: 12px`. Twelve SVG units is 3.1 pt at Figure 1's placed scale. At assembly the two labels were re-set by hand to 8 pt (30.85 units), which is the stylesheet's `font.size`.
  - **The canvas is not a `FIGURE_WIDTHS` width.** It is a unitless 545 × 395 viewBox, which Figure 1 places at 0.09146755 mm per unit (49.9 × 36.1 mm).
- **Black** comes from `OKABE_ITO["black"]`, which is the same `#000000`.
- **Hex literals.** No shared palette holds white or the terracotta set-points, so those stay as literals in the script, with their derivation in a comment. `sequential_shades("#b5473f", 3)` gives `#e5bfbc` / `#cd837d`, which are visibly different d = 2 and d = 3 colours.
- **Output location.** The script writes to `--outdir` (the config's `outdir`).
- **Corner-radius comparisons.** `pairwise_error_final_rx{6,4,3,2,0}.svg` are opt-in (`rx_variants`, default false). The panel is `pairwise_error_final.svg`, drawn at radius 4, so it is the same file as `rx4`.

## Outputs

Under `results/experiments/pep_grid_glyph/`:

| file | use |
|---|---|
| `pairwise_error_final.svg` | **Fig 1b glyph**, before the hand edits made at assembly (below) |
| `pairwise_error_final_rx{6,4,3,2,0}.svg` | only with `rx_variants: true`; corner-radius comparisons, not in the paper |
| `logs/generate.log` | script output, the sha256 of each file written, and the result of the sha256 comparison |

## Compare after a re-run

**The output is byte-identical on every machine.** `pairwise_error_final.svg` has sha256 `13b0473c6c7384308f939d9a16accd60cd0995032c556446aa0689266d7733a2` (16,161 B, no trailing newline): the same bytes as the SVG that was placed into Fig 1b. run.sh checks this. It was identical in every recorded full and smoke run, and the `rx` files equalled the original script's output for every radius.

**Fig 1b is a composite.** Only the chips, the colour bar and its two labels come from this script. A comparison with Figure 1 (rendered at the same scale) found:
- **Chips.** All 64 cells match the same pattern: 24 d = 1, 24 d = 2, 8 d = 3 and 8 blank on the diagonal. The raster RGB values are identical: (181, 71, 63), (218, 163, 159) and (245, 229, 228).
- **Size.** Chip size and pitch match: 4.03 mm per cell in the PDF, which is 44 units at about 0.0915 mm per unit. Chip corner radius and the white gutters also match.
- **Gradient.** The colour-bar gradient has the same two end colours.

Differences, all hand edits in Inkscape at assembly (not reproduced here):
1. **Codeword glyphs.** Figure 1 does not use the script's glyphs; it uses its own.
   - The script draws 7 × 12-unit boxes (0.64 × 1.10 mm placed), filled `#000000`, with the column glyphs below the grid.
   - Figure 1's boxes are nearly square (about 1.13 × 1.19 mm), filled `#333333`, with the column glyphs above the grid.
   - Figure 1 also leaves a wider gap between the row glyphs and the chips.
2. **Colour bar.** It was reshaped to about 1.25× wider and 0.89× shorter: 1.74 × 9.57 mm in the PDF, against 1.37 × 10.73 mm from the script. It was also moved from the vertical middle third of the grid down to the bottom rows.
3. **Labels.** "High probability" and "Low probability" were re-set to 8 pt (3.1 pt as generated) and repositioned.
4. **Axis titles.** "Decoded codeword" (top), "Transmitted codeword" (left, rotated) and the panel title are Figure 1 text, not script output.

Regenerating this SVG therefore does not rebuild Fig 1b. The Fig 1b panel has no numbers to compare.

## Determinism

The output depends only on constants in the script. There is no seed, no randomness, no input file and no device dependence. The same code gives the same bytes on every run.

## Smoke test

`VARIANT=smoke bash experiments/pep_grid_glyph/run.sh` reads the tracked `config.smoke.yaml`. It takes the same code path and writes under `results/smoke/pep_grid_glyph/`. It sets `rx_variants: true`, so the optional files are exercised too:

```yaml
outdir: ../../results/smoke/pep_grid_glyph
rx_variants: true
```

Expected:
- it writes six files, and `pairwise_error_final.svg` (= `rx4`) has the sha256 above;
- the sha256 prefixes of the others are rx6 `94b872aa`, rx3 `7e2d40e3`, rx2 `81e7ab4c` and rx0 `738b1c46`.

run.sh stops before writing anything in each of these cases:
- a missing config;
- a missing `rx_variants` key;
- an invalid `rx_variants` value;
- an env without `duet`.

## Inputs

None. `apply_style()` reads the shared stylesheet `src/duet/plotting/duet_publication.mplstyle`, but the stylesheet does not affect this output.
