# Data and archives

The tables for Figures 9-12 in `data/counts/` contain 275 rows; Table C1 contains another seven rows comparing round counts. `data/h/stats.csv` contains 254 H sampling shards, with shared settings in `protocol.json`. `data/w3/` and `data/s_lowp/` contain 15 S reports. `data/fig10_d13_cohorts.json` fixes the cohort combinations for three d=13 points in Figure 10. The two Stim files in `data/injection/` define injection and ideal propagation checks; they are not noisy sampling data for a complete device.

Download the data archives from [Zenodo](https://doi.org/10.5281/zenodo.24113458).

| Archive | Contents | Use |
| --- | --- | --- |
| `fig09-frozen-circuits.tar.gz` | 108 H/memory circuit sets, detector error models (DEMs) and input parameters | Reuse exact inputs for new sampling |
| `fig10-11-records.tar.gz` | 189,792 S shard records, 68 shared configurations and 101 frozen input files | Check report membership and counts; reuse S inputs |
| `fig12-confirmation.tar.gz` | 81 circuit/DEM/ideal-boundary sets and 503 rows of four-bit residual statistics | Recompute joint-measurement failure counts or resample |
| `distance-evidence.tar.gz` | d=3 certificates, four d=5/d=7 search archives and 12 Appendix C/W3 fault cases | Verify distance evidence and replay faults; not needed for plotting |
| `strict-distance-pair-indexes.tar.gz` | Two precomputed d=5 pair indexes | Optional separate cache that avoids rebuilding indexes; not included in the core download package |

Archive sizes and SHA-256 checksums are recorded in `release-assets.json`. Each archive also includes a member manifest. The d=5/d=7 partition results in `distance-evidence` are bound to their referenced shards, models, fault signatures and physical realizations, allowing the archived search coverage and results to be checked. The d=3 checks use fixed Stim circuits as inputs.

## Statistical conventions and record identity

Each row of the S archive's `stats.csv` is one completed production shard. It retains the cohort, condition, seed, sample count, failure count, input hashes and original shard hash. `contexts.json` stores one copy of each shared circuit and decoder configuration. `report_ids` identifies the reports using a shard; an empty value does not make the shard invalid. The 15 reports reference 122,700 W3 shards and 28,200 low-error-rate shards in total. Set digests in the reports verify exact membership.

`record_sha256` identifies the original JSON bytes before normalization; it does not imply that those individual JSON files are also included. An empty `batch_size` means that the original record did not report it. `data/s_cohort_counts.csv` contains per-cohort aggregates. Do not merge cohorts solely because d, W and p match.

Figure 12 retains all 16 residual bins. Every bin except zero counts as an overall failure. The least significant bit indicates a joint-measurement failure; the other three bits indicate failures of the retained observables. These bins allow joint, retained, simultaneous and overall failure counts to be recomputed.

## Coverage and limitations

- The archive supplies 101 of the 111 frozen input files referenced by S shards. The remaining ten concern five conditions; their exact hashes are listed in `data/s_input_coverage.json`. The archived counts and plots can be checked, but not every original sampling input is available byte for byte.
- Another 29 historical aggregate rows in Figure 10 lack complete shard membership. Use the published count table for these rows; records with matching parameters in the archive cannot automatically be treated as their complete source.
- The d=3 certificate check recomputes fault-signature exclusions and replays upper-bound witnesses for the specified circuits. The d=5/d=7 checks verify stored models, hashes and partition coverage; they do not repeat the full exhaustive search. W3 witnesses of weights 11, 12 and 14 establish upper bounds only.
