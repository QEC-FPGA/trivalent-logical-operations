"""Missing loader dependency only; does not load any historical geometry.

load_source is reproduced from the read-only reference ZIP bundled with the
user's archive. The caller explicitly selects active source_work_topcrop.
"""
import sys
from pathlib import Path


def load_source(source: Path):
    sys.path.insert(0, str(source))
    from d7_stage1 import Stage1
    from d7_readout import (P, annotate, body_after_init,
                            complete_detectors, ideal_y_preparation,
                            noisy_version, observable_indices)
    return Stage1, P, annotate, body_after_init, complete_detectors, ideal_y_preparation, noisy_version, observable_indices
