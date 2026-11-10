# Reproducibility

Run all commands from the repository root. The frozen S sampling entry points require Python 3.12. The validated environment uses Python 3.12.14, Stim 1.16.0, PyMatching 2.3.1 and NumPy 2.4.4. Plotting dependencies are listed separately; the full dependency list also includes OR-Tools for distance searches.

## Obtain the pinned dependency

```bash
git clone --branch master --recurse-submodules https://github.com/QEC-FPGA/trivalent-logical-operations.git
cd trivalent-logical-operations
```

For an existing clone, run `git submodule update --init --recursive`. `third_party/midout` is pinned to an exact `midout_3-CX` commit; do not use `--remote` to switch to a version that has not been validated with these results. Plotting from counts does not load midout; lattice-surgery circuit generation requires it.

A directory extracted from the Zenodo code ZIP has no Git metadata. To generate lattice-surgery circuits, obtain the same pinned dependency from the extracted repository root:

```bash
git clone --no-checkout https://github.com/QEC-FPGA/midout_3-CX.git third_party/midout
git -C third_party/midout checkout --detach ef3bb5ef1f70a40280c60370d7d3071cfdfdbc06
```

The code ZIP's `MANIFEST.json` records this dependency URL and commit. The submodule source is not bundled in the ZIP.

## Replot the archived counts

```bash
python -m pip install -r requirements-plot.txt
python tools/check_results.py
python reproduce_figures.py
```

Outputs in `reproduced_figures/` include the H, S baseline, waiting-time comparison and lattice-surgery plots. Plotting does not require the large data archives.

Error rates are failures divided by samples for a complete test circuit, without normalization by round count. Absolute-rate intervals use 95% Wilson intervals; ratio intervals combine two 97.5% Clopper-Pearson intervals. Different figures may reuse the same sampling cohort, so their counts must not be added as independent samples.

## Read and verify frozen artifacts

Select archives using [DATA.md](DATA.md) and download them from [Zenodo](https://doi.org/10.5281/zenodo.24113458) to a directory of your choice. Extracted members reside under `bundles/`. Each archive includes member-level SHA-256 checksums; archive checksums are in `release-assets.json`.

```bash
python tools/check_results.py --assets-dir /path/to/downloads --members
python tools/check_s_reports.py /path/to/downloads/fig10-11-records.tar.gz
python tools/check_surgery_reports.py /path/to/downloads/fig12-confirmation.tar.gz
tar -xzf /path/to/downloads/distance-evidence.tar.gz
python tools/check_distance_results.py
```

`check_distance_results.py` verifies archived partition coverage, models and evidence hashes; **it does not repeat the exhaustive search**. The optional large pair-index archive reuses cached computation and is not required for plotting or checking stored results.

## Generate circuits and check witnesses

```bash
python -m pip install -r requirements.txt
python src/h/generate_one.py --kind h_flow --distance 5 --basis X -o out/H_d5_X.stim
python src/s/run.py check
python src/s/run.py list
python src/s_lowp/run.py check
python src/surgery/generate.py --d 5 --p 0.001 --measurement ZZ --out out/surgery_d5_ZZ
python tools/replay_witnesses.py
python tools/check_d3_certificate.py
```

The H example generates an ideal circuit. Use the frozen files in the Figure 9 archive for the SI1000 noisy circuits and DEMs underlying the published counts. The S `run.py` supports `prepare/pilot/budget/confirm/report`; parameter groups are defined in the corresponding `protocol.json`. The low-error-rate package also includes a frozen protocol and generator code; see [DATA.md](DATA.md) for exact input coverage. Do not overwrite the plan identities in archived reports with a new plan. The frozen source retains a historical `mechanism` subcommand that depends on an unavailable `mechanism_seeds/` directory; it is not a supported reproduction entry point. Use the independent witness checks above.

The lattice-surgery generator uses the supplied d=5 reference circuit and modified midout dependency; see [LICENSE.md](../LICENSE.md) for the licensing scope. Generating a circuit alone does not establish a valid noise-model decomposition or a distance certificate.

## Run new sampling

Select a circuit/DEM pair from an archive and check its hashes against the figure tables or the archive's `stats.csv`. The following command runs independent sampling without replacing archived counts:

```bash
python tools/sample_frozen.py --circuit path/to/circuit.stim --dem path/to/matching.dem --shots 100000 --seed 12345 --out out/new_sample.json
```

The sampler uses correlated PyMatching and records input hashes, the random seed, batch size, failure counts and a multi-observable residual histogram. Random-number algorithms, threads, batch sizes and dependency versions can affect individual samples. Aggregate the original shards to verify existing exact counts. Full-scale sampling at low error rates is computationally expensive.

## Run a new distance search

```bash
python -m pip install -r requirements.txt
python tools/search_distance.py --case d5_m2-3_W3_XY --cap 4 --seconds 900 --workers 2 --out out/search_d5_XY
```

This entry point runs a new full-signature constraint search from an extracted model and writes to a new directory. `UNKNOWN` means the search did not finish within its budget and does not establish a lower bound. Archived results and indexes for four d=5/d=7 cases can be reused directly; the original partition-enumeration code, full-signature models, physical fault realizations and algorithms are supplied. A new CP-SAT run is not guaranteed to match the runtime of the archived partition algorithm within the specified budget.

W3 witnesses of weights 11, 12 and 14 establish upper bounds for their respective circuits. Strict-distance searches and the local movement diagnostics in Appendix C use distinct circuits and boundaries; do not transfer fault-instruction offsets between circuits.

## Frozen source and optional caches

The two S protocols and the distance search retain their original source layouts to preserve generation behavior and the code digests in sampling records. Run each package from its own entry point and keep its internal dependencies in place.

Full fault evidence is extracted to `bundles/distance-evidence/`. To reuse separately downloaded pair indexes, extract only the arrays so the core member manifest is preserved:

```bash
tar -xzf /path/to/downloads/strict-distance-pair-indexes.tar.gz --strip-components=2 --wildcards -C bundles/distance-evidence "*/results/*/pair_index.npy"
```

Count checks, plotting and evidence replay are generally desktop-scale tasks. Full low-error-rate sampling and exhaustive distance searches require substantially larger computational budgets.
