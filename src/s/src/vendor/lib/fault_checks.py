"""Physical-fault replays and bounded interface enumeration, with full global parities."""
from pathlib import Path
import argparse
import json
import re
import stim

NOISE = {'X_ERROR', 'Y_ERROR', 'Z_ERROR', 'DEPOLARIZE1', 'DEPOLARIZE2'}

def mpp_products(op):
    products=[];joined=False
    for t in op.targets_copy():
        if t.is_combiner: joined=True;continue
        if not joined: products.append([])
        products[-1].append(t);joined=False
    return products


def signature(op, D):
    v = 0
    for t in op.targets_copy():
        assert not t.is_separator()
        v ^= 1 << (t.val if t.is_relative_detector_id() else D)
    return v


def propagate(c, events):
    by_op = {}
    for i, e in enumerate(events): by_op.setdefault(e['instruction_offset'], []).append((i, e))
    x = [0] * c.num_qubits
    z = x.copy()
    meas = []
    rows = []
    logical = 0
    for index, op in enumerate(c):
        name = op.name
        qs = [t.value for t in op.targets_copy()]
        if name in NOISE:
            for i, e in by_op.get(index, []):
                for q, p in e['paulis'].items():
                    q = int(q)
                    if p in 'XY': x[q] ^= 1 << i
                    if p in 'ZY': z[q] ^= 1 << i
            continue
        if name in ('TICK', 'QUBIT_COORDS'): continue
        if name == 'R':
            for q in qs: x[q] = z[q] = 0
        elif name == 'H':
            for q in qs: x[q], z[q] = z[q], x[q]
        elif name in ('S', 'S_DAG'):
            for q in qs: z[q] ^= x[q]
        elif name == 'CX':
            for a, b in zip(qs[::2], qs[1::2]): x[b] ^= x[a]; z[a] ^= z[b]
        elif name == 'CZ':
            for a, b in zip(qs[::2], qs[1::2]): z[a] ^= x[b]; z[b] ^= x[a]
        elif name in ('M', 'MX', 'MY'):
            for q in qs: meas.append(x[q] if name == 'M' else z[q] if name == 'MX' else x[q] ^ z[q])
        elif name == 'MPP':
            for product in mpp_products(op):
                v=0
                for t in product:
                    if t.is_x_target or t.is_y_target: v ^= z[t.value]
                    if t.is_z_target or t.is_y_target: v ^= x[t.value]
                meas.append(v)
        elif name in ('DETECTOR', 'OBSERVABLE_INCLUDE'):
            v = 0
            for r in qs: v ^= meas[len(meas) + r]
            if name == 'DETECTOR': rows.append(v)
            else: logical ^= v
        else: raise ValueError(name)
    columns = [0] * len(events)
    for j, row in enumerate(rows + [logical]):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] ^= 1 << j
            row ^= bit
    return columns


def injected(c, events):
    at = {}
    sites = set()
    for e in events:
        site = (e['instruction_offset'], tuple(e['noise_targets']))
        assert site not in sites, 'Multiple selected branches at one physical noise site'
        sites.add(site)
        at.setdefault(e['instruction_offset'], []).append(e)
    out = stim.Circuit()
    for i, op in enumerate(c):
        if op.name in NOISE:
            for e in at.get(i, []):
                ts = [getattr(stim, 'target_' + p.lower())(int(q)) for q, p in e['paulis'].items()]
                out.append('E', ts, 1)
        else: out.append(op)
    v = 0
    for op in out.detector_error_model(decompose_errors=False):
        if op.type == 'error':
            assert op.args_copy()[0] == 1
            v ^= signature(op, c.num_detectors)
    return v, out


def unpack(v, D):
    return {'D': [i for i in range(D) if v >> i & 1], 'L': (v >> D) & 1}


def noise_sites(c):
    tick = 0
    for i, op in enumerate(c):
        if op.name == 'TICK': tick += 1
        if op.name not in NOISE: continue
        qs = [t.value for t in op.targets_copy()]
        size = 2 if op.name == 'DEPOLARIZE2' else 1
        for j in range(0, len(qs), size):
            yield i, tick, op, qs[j:j + size]


def replay_patterns(folder, source, tick_shift=0, keep_earliest=False):
    c = stim.Circuit.from_file(str(folder / 'd9_YX_SI1000_body.stim'))
    s = source.read_text(encoding='utf-8')
    pattern_text = re.findall(r'### events\s+```json\s+(.*?)\s+```', s, re.S)
    sites = list(noise_sites(c))
    out = []
    for j, raw in enumerate(pattern_text):
        events = []
        invalid = []
        original_events = json.loads(raw)
        earliest = min(e['tick'] for e in original_events)
        for ev in original_events:
            support = {int(q): p for q, p in ev['pauli_support'].items()}
            mapped_tick = ev['tick'] + (0 if keep_earliest and ev['tick'] == earliest else tick_shift)
            matches = [(i, tick, op, qs) for i, tick, op, qs in sites
                       if tick == mapped_tick and op.name == ev['original_noise'] and set(support) <= set(qs)]
            if len(matches) != 1:
                invalid.append({'source_event': ev, 'matching_sites': len(matches)})
                continue
            i, tick, op, qs = matches[0]
            events.append({'instruction_offset': i, 'tick': tick, 'noise_targets': qs,
                           'noise_instruction': str(op), 'paulis': support,
                           'source_line_one_based': ev['line_one_based'], 'source_tick': ev['tick']})
        r = {'pattern_one_based': j + 1, 'event_mapping': 'Specified tick map, same Pauli qubits and noise type, unique valid physical site; NOT proof of identity to missing C circuit',
             'tick_shift': tick_shift, 'keep_earliest_event_at_original_tick': keep_earliest,
             'events': events, 'unmapped_events': invalid}
        if not invalid:
            cols = propagate(c, events)
            total = 0
            for v in cols: total ^= v
            independent, replay = injected(c, events)
            assert independent == total
            r.update(total=unpack(total, c.num_detectors), independent_Stim_replay='PASS',
                     status='COUNTEREXAMPLE' if total == 1 << c.num_detectors else 'DETECTED' if total & ((1 << c.num_detectors) - 1) else 'NO_EFFECT')
            if r['status'] == 'COUNTEREXAMPLE':
                replay.to_file(str(folder / f'pattern_{j + 1}_shift{tick_shift}_keep{int(keep_earliest)}_injected.stim'))
        else: r['status'] = 'NOT_MAPPABLE'
        out.append(r)
    return out


def audit_front(folder, ch):
    report = json.loads((folder / 'report.json').read_text())
    d = report['distance']
    c = stim.Circuit.from_file(str(folder / f'd{d}_{ch}_SI1000_body.stim'))
    coords = c.get_final_qubit_coordinates()
    offset = 2 if ch == 'YX' else 0  # prep replacing the builder's Z reset
    windows = []
    for seg in report['resources']['segments']:
        if re.fullmatch(r'grow_\d+_to_\d+_MY', seg['name']):
            width = int(seg['name'].split('_')[3])
            tick = seg['tick_start_in_builder'] + offset
            windows.append({'name': seg['name'], 'tick_inclusive': [tick - 16, tick + 16],
                            'x_inclusive': [d - width - 2, d - width + 4],
                            'y_inclusive': [d - 2, d + 3]})
            unit = next((u for u in report.get('units', []) if u['new_width'] == width), None)
            if unit:
                for axis, key in enumerate(('x_inclusive', 'y_inclusive')):
                    values = [coords[q][axis] for q in unit['touched_qubits']]
                    windows[-1][key] = [min(windows[-1][key][0], min(values)),
                                       max(windows[-1][key][1], max(values))]
                windows[-1]['tick_inclusive'][0] = min(tick - 16, tick - unit['ticks'])
                windows[-1]['covers_entire_physical_bridge'] = True
    events = []
    selected = {}
    for i, tick, op, qs in noise_sites(c):
        matches = [w['name'] for w in windows if w['tick_inclusive'][0] <= tick <= w['tick_inclusive'][1]
                   and any(w['x_inclusive'][0] <= coords[q][0] <= w['x_inclusive'][1]
                           and w['y_inclusive'][0] <= coords[q][1] <= w['y_inclusive'][1] for q in qs)]
        if not matches: continue
        selected.setdefault(i, []).extend(qs)
        if op.name == 'DEPOLARIZE2':
            branches = [a + b for a in 'IXYZ' for b in 'IXYZ' if a + b != 'II']
        elif op.name == 'DEPOLARIZE1': branches = list('XYZ')
        else: branches = [op.name[0]]
        for branch in branches:
            events.append({'instruction_offset': i, 'tick': tick, 'noise_targets': qs,
                           'noise_kind': op.name, 'windows': matches,
                           'paulis': {q: p for q, p in zip(qs, branch) if p != 'I'}})
    cols = propagate(c, events)
    filtered = stim.Circuit()
    for i, op in enumerate(c):
        if op.name not in NOISE: filtered.append(op)
        elif i in selected: filtered.append(op.name, selected[i], op.gate_args_copy())
    ds = {signature(op, c.num_detectors) for op in filtered.detector_error_model(decompose_errors=False)
          if op.type == 'error'}
    assert ds == set(cols) - {0}, 'Independent physical propagation/Stim signature mismatch'
    target = 1 << c.num_detectors
    representatives = {}
    counter = None
    for i, sig in enumerate(cols):
        if sig == target:
            counter = [i]
            break
        other = representatives.get(sig ^ target)
        if other is not None:
            a, b = events[i], events[other]
            if (a['instruction_offset'], a['noise_targets']) != (b['instruction_offset'], b['noise_targets']):
                counter = [i, other]
                break
        representatives.setdefault(sig, i)
    result = {'channel': ch, 'windows': windows, 'physical_Pauli_branches': len(events),
              'distinct_nonzero_signatures': len(ds), 'independent_Stim_signature_set_match': True,
              'global_parities_retained': True, 'one_and_two_fault_counterexample': counter,
              'scope': 'Faults restricted to explicit interface windows; detectors and logical observable from full circuit. No global distance lower bound.'}
    if counter is not None:
        v, replay = injected(c, [events[i] for i in counter])
        assert v == target
        result['counterexample_events'] = [events[i] for i in counter]
        replay.to_file(str(folder / f'{ch}_front_counterexample.stim'))
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('folder', type=Path)
    ap.add_argument('--patterns', type=Path)
    ap.add_argument('--tick-shift', type=int, default=0)
    ap.add_argument('--keep-earliest', action='store_true')
    ap.add_argument('--front', action='store_true')
    a = ap.parse_args()
    if a.patterns:
        r = replay_patterns(a.folder, a.patterns, a.tick_shift, a.keep_earliest)
        suffix = '' if a.tick_shift == 0 else f'_shift{a.tick_shift}_keep{int(a.keep_earliest)}'
        (a.folder / f'patterns_replay{suffix}.json').write_text(json.dumps(r, indent=2))
        print(json.dumps({status: sum(x['status'] == status for x in r) for status in {x['status'] for x in r}}), flush=True)
    if a.front:
        r = {ch: audit_front(a.folder, ch) for ch in ('YX', 'ZZ')}
        (a.folder / 'front_audit.json').write_text(json.dumps(r, indent=2))
        print(json.dumps(r), flush=True)
