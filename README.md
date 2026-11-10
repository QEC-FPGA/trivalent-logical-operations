# Universal logical operations on trivalent surface-code hardware

Code and reproducibility materials for *Universal logical operations and their circuit implementations for surface codes on trivalent hardware architectures* (《三连通硬件架构上表面码的通用逻辑操作及其线路实现》).

This repository provides code and small datasets for circuit generation, sampling, plotting and distance verification. Frozen circuits, per-shard statistics and fault evidence are available in the [Zenodo data archive](https://doi.org/10.5281/zenodo.24113458).

```bash
git clone --branch master --recurse-submodules https://github.com/QEC-FPGA/trivalent-logical-operations.git
cd trivalent-logical-operations
python -m pip install -r requirements-plot.txt
python tools/check_results.py
python reproduce_figures.py
```

Figures 9-12 can be replotted from the archived counts without new sampling. Outputs are written to `reproduced_figures/`.

| Path | Contents |
| --- | --- |
| `data/` | 275 rows of figure counts, H sampling shards, S reports, exact cohort membership and injection circuits |
| `src/h/`, `src/s/`, `src/s_lowp/` | H/S circuit generators and two frozen S sampling protocols |
| `src/surgery/` | ZZ/XX/XZ joint-measurement circuits and decoder-input compilation |
| `src/strict_search/` | Full fault signatures, exact constraint searches and partition enumeration |
| `tools/` | Data checks, fault replay, independent sampling and distance-search entry points |
| `third_party/midout` | The [midout_3-CX](https://github.com/QEC-FPGA/midout_3-CX) construction dependency, pinned to an exact commit |

See [Reproducibility](docs/REPRODUCIBILITY.md) for commands and [Data and archives](docs/DATA.md) for archive contents and coverage limitations. [AGENTS.md](AGENTS.md) provides concise conventions for automated reproduction; the same commands apply to all readers.

The noisy simulations cover H, S and joint-measurement components. They do not characterize a complete CNOT or magic-state distillation device. Distance results apply to the specified circuits, boundaries and fault models. Original contributions are provided under MIT; see [LICENSE.md](LICENSE.md) for the scope and required third-party attribution.
