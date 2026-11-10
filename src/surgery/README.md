# Joint-measurement circuits

Run `python generate.py --d 5 --p 0.001 --measurement ZZ --out /path/to/new-output`.
The output directory must be new or empty. `reference/` binds the d=5 physical
schedule; the modified midout implementation in `../../third_party/midout/src/midout/` supplies layout
and extraction primitives. See `../../LICENSE.md` for license scope.

`generate.py` creates ideal/noisy circuits and metadata. It does not sample or
certify distance. The Figure 12 archive contains the actual frozen sampling
circuits, DEMs and residual-count records used by the paper.

`prepare.py` applies strict detector-basis repair and exports sampling circuits
and DEMs. For example, run `python src/surgery/prepare.py --distances 5
--measurements ZZ --p 0.001 --repair-seconds 120 --out out/new-surgery` from
the repository root (as one line). A repair timeout is a failure to prepare,
not permission to drop faults.
