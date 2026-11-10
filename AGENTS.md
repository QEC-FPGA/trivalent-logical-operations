# Reproducing the research results

Use [README.md](README.md) for the project entry point and
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for complete commands.

## Start with the archived counts

Run commands from the repository root. Use Python 3.12 and the dependency
versions in `requirements.txt`; plotting alone needs `requirements-plot.txt`.

```bash
python tools/check_results.py
python reproduce_figures.py
```

The first command checks 275 table rows and 254 H sampling shards. Plots are
written to `reproduced_figures/`. Large-scale Monte Carlo is not needed to
replot the archived counts.

## Scientific conventions

- Rates are failures per complete test circuit, not failures per round.
- Preserve cohort identity, exact shard membership, input hashes, seeds,
  decoder options and recorded batch sizes. Do not merge different cohorts
  just because distance, channel and physical error rate match.
- For independent sampling use `tools/sample_frozen.py` and a new output path.
  Keep the archived counts and frozen inputs unchanged.
- A replayed fault witness establishes an upper bound. An incomplete or
  timed-out search does not establish a distance lower bound.
- `tools/check_distance_results.py` checks archived partition coverage and
  artifact bindings; it does not perform a new exhaustive search.

## Dependencies and evidence

Keep `third_party/midout` at its recorded submodule commit. Preserve each frozen
S package's source layout: its file bytes and relative paths participate in the
historical code digest. Do not replace recorded input bytes with newly generated
files that have different hashes.

Before reporting a successful reproduction, state which checks were run.
The 10 unmatched frozen S inputs and 29 historical rows with incomplete shard
membership remain limitations; see [docs/DATA.md](docs/DATA.md).
License and upstream attribution are defined in [LICENSE.md](LICENSE.md).
