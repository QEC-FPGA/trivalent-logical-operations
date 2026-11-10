"""Experimental two-site wall growth. Generation and flow checks do not prove distance."""
from pathlib import Path
import argparse
import json
import sys
import stim

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'lib'))
import generate as base
from code_algebra import physical_code, rank, gauge_summary


def builder(d, wide=None, twist=None):
    if d < 5 or d % 2 != 1:
        raise ValueError('Odd distance >=5 required')
    wide = d if wide is None else wide
    twist = d if twist is None else twist
    if min(wide, twist) < 1 or (wide + twist) % 2:
        raise ValueError('Positive counts with matching parity required to retain final B -> fullwall A')
    b = base.Stage1(d)
    b.right_seam_width = d - 3
    b.data_init('Z')
    for j in range(d):
        b.standard_round('B' if j % 2 == 0 else 'A', j + 1)
    b.expansion()
    for j in range(wide):
        b.wide_round('A' if j % 2 == 0 else 'B', d + 1 + j)
    first = 'B' if wide % 2 else 'A'
    second = 'A' if first == 'B' else 'B'
    for width in range(2, d - 2, 2):
        c = stim.Circuit()
        c.append('MY', [b.data(i, d + 1) for i in range(d - width, d - width + 2)])
        c.append('TICK')
        b.append_segment(f'grow_{width - 2}_to_{width}_MY', c)
        mid = base.Stage1(d)
        mid.right_seam_width = width
        if width < d - 3:
            for j, phase in enumerate((first, second)):
                mid.twist_round(phase, 1000 + width * 10 + j)
                b.append_segment(f'grow_{width}_{phase}', list(mid.segments.values())[-1])
        else:
            for j in range(twist):
                phase = first if j % 2 == 0 else second
                # Preserve the legacy last-B tick omission at the left-MY interface.
                index = 2 * d + 5 if j == twist - 1 else 2000 + j
                mid.twist_round(phase, index)
                b.append_segment(f'twist_final_{j + 1}_{phase}', list(mid.segments.values())[-1])
    b.my_seam('left')
    for j in range(d):
        b.fullwall_round('A' if j % 2 == 0 else 'B', j + 1)
    b.h_lower()
    b.shrink()
    return b


def algebra(b):
    d = b.d
    n = b.circuit.num_qubits
    data = {b.data(i, j + 1) for i in range(d) for j in range(d)}
    stabs = []
    for q in range(n):
        if q not in data:
            p = stim.PauliString(n + 1)
            p[q] = 'Z'
            stabs.append(p)
    for check in base.standard_stabilizers(d):
        p = stim.PauliString(n + 1)
        for i in range(d):
            for j in range(d):
                p[b.data(i, j + 1)] = check[d * i + j]
        stabs.append(p)
    for axis in 'XZ':
        p = base.logicals(b, 'input')[axis].copy() + stim.PauliString(1)
        p[n] = axis
        stabs.append(p)
    sim = stim.TableauSimulator(seed=101)
    sim.set_inverse_tableau(stim.Tableau.from_stabilizers(stabs).inverse())
    result = {'input': 'Encoded Bell pair; no fixed logical Y constraint', 'cuts': {},
              'raw_endpoint_unions': {}, 'dressed_distance': 'NOT_COMPUTED',
              'union_caution': 'Only consecutive MY-only cuts are in one physical Pauli frame. Across extraction Clifford circuits, raw unions are diagnostics, not valid gauge-fixing certificates without transporting generators.'}
    previous = None
    names = list(b.segments)
    last_wide = [nm for nm in names if nm.startswith('wide_')][-1]
    last_twist = [nm for nm in names if nm.startswith('twist_final_')][-1]
    for name, c in b.segments.items():
        if name == 'data_init':
            continue
        sim.do(c)
        selected = name.startswith(('grow_', 'handoff_before_')) or name in (last_wide, last_twist, 'wall_MY_3', 'shrink')
        if not selected:
            continue
        rows = physical_code(sim, n)
        result['cuts'][name] = {'rank': rank(rows), 'k': n - rank(rows)}
        assert n - rank(rows) == 1, name
        if previous is not None:
            pn, pr = previous
            g = gauge_summary(pr, rows, n)
            my_only = name.endswith('_MY') or name == 'wall_MY_3'
            g['MY_only_same_frame'] = my_only
            result['raw_endpoint_unions'][pn + ' -> ' + name] = g
            if my_only:
                assert g['protected_k'] == 1, (pn, name, g)
        previous = (name, rows)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--distance', type=int, required=True)
    ap.add_argument('--wide', type=int)
    ap.add_argument('--twist', type=int)
    ap.add_argument('--channels', nargs='+', choices=['YX', 'ZZ'], default=['YX', 'ZZ'])
    ap.add_argument('--body-only', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    d = args.distance
    w = args.wide if args.wide is not None else d
    t = args.twist if args.twist is not None else d
    b = builder(d, w, t)
    args.out.mkdir(parents=True, exist_ok=True)
    r = base.resources(b, w, t)
    r.update(handoff_rounds=d - 5, total_extraction_rounds=3 * d + w + t - 5,
             growth_widths=list(range(2, d - 2, 2)), growth_MY_steps=(d - 3) // 2)
    report = {'distance': d, 'profile': 'moving_two_sites', 'status': 'EXPERIMENTAL; NOT DISTANCE VERIFIED',
              'resources': r, 'logical_audit': base.logical_audit(b), 'channels': {},
              'algebra': algebra(b)}
    if d == 7 and w == t == 7:
        report['d7_exact_keep2_body_regression'] = b.circuit == base.build_builder(7).circuit
        assert report['d7_exact_keep2_body_regression']
    base.engine.with_coordinates(base.engine.body_after_init(b), b).to_file(str(args.out / 'body.stim'))
    for ch in args.channels:
        if args.body_only:
            break
        print('Building', d, w, t, ch, flush=True)
        report['channels'][ch] = base.build_channel(d, ch, .001, b, args.out)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'distance': d, 'wide': w, 'twist': t,
                      'blocks': r['total_extraction_rounds'], 'ticks': r['body_ticks_excluding_input_prep'],
                      'layers': r['two_qubit_gate_layers'], 'k_and_logical_flows': 'PASS'}), flush=True)


if __name__ == '__main__':
    main()
