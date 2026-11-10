"""Recheck the archived lower bound and replay without relying on a matching graph."""
from pathlib import Path
import json, gzip, hashlib
import stim
from functools import reduce
from operator import xor
from replay_witnesses import inject, response

ROOT = Path(__file__).resolve().parents[1] / 'bundles/distance-evidence/d3'
def circuit(p):
    return stim.Circuit(p.read_text(encoding='utf-8-sig'))
def signatures(c):
    result = set()
    for e in c.detector_error_model(decompose_errors=False, flatten_loops=True):
        if e.type != 'error':
            continue
        value = 0
        for t in e.targets_copy():
            if t.is_relative_detector_id():
                value ^= 1 << t.val
            elif t.is_logical_observable_id():
                value ^= 1 << (c.num_detectors + t.val)
            else:
                raise ValueError('Unexpected target')
        result.add(value)
    return result

for channel in ('XY', 'ZZ'):
    folder = ROOT / channel
    c = circuit(folder / 'full_original.stim')
    cert = json.loads((folder / 'signature_exclusion_certificate.json').read_text())
    result = json.loads((folder / 'result.json').read_text())
    assert hashlib.sha256(str(c).encode()).hexdigest() == result['canonical_circuit_sha256']
    values = set(map(lambda s: int(s, 16), cert['signatures_hex']))
    with gzip.open(folder / 'all_physical_branches.jsonl.gz', 'rt') as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == result['physical_branches']
    assert {int(row['signature_hex'], 16) for row in rows} == values
    assert signatures(c) == values - {0}
    target = 1 << c.num_detectors
    assert target == int(cert['target_hex'], 16)
    assert target not in values
    assert not any(v ^ target in values for v in values)
    replay = circuit(folder / 'weight3_replayed.stim')
    assert all(e.args_copy() == [1.0] for e in replay.detector_error_model() if e.type == 'error')
    assert reduce(xor, signatures(replay), 0) == target
    measurements = replay.compile_sampler(seed=53).sample(256)
    d, l = c.without_noise().compile_m2d_converter().convert(measurements=measurements, separate_observables=True)
    assert not d.any() and l.all()
    witness = json.loads((folder / 'weight3_witness.json').read_text())
    assert len(witness) == 3
    assert response(inject(c, witness)) == ([], [0])
    print(channel, 'PASS: lower bound 3 and replayed upper bound 3')
