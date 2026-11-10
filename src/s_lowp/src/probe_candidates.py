"""Odd-k reconstruction probes. Frozen baseline imports are read-only.

The independent reference is a projected wide code, before calling twist_round.
Candidate circuits are not certified until code, flow and physical faults pass.
"""
from pathlib import Path
import argparse, json, sys, time
import stim

ROOT = Path(__file__).resolve().parent
OLD = ROOT
sys.path.insert(0, str(OLD))
import movement as old
ORIGINAL_EXTRACTION = old.extraction
ORIGINAL_INITIAL_CODE = old.initial_code


def mask(p):
    n = len(p)
    return sum((int(v in (1, 2)) << q) | (int(v in (2, 3)) << (q+n)) for q, v in enumerate(p))


def basis(rows):
    b = {}
    for v in rows:
        while v:
            q = v.bit_length()-1
            if q in b: v ^= b[q]
            else:
                b[q] = v
                break
    return b


def residual(v, b):
    while v:
        q = v.bit_length()-1
        if q not in b: return v
        v ^= b[q]
    return 0


def restricted_basis(checks, keep):
    """Stabilizer subgroup supported only on keep, by exact elimination."""
    n=len(checks[0]); allowed=sum((1<<q)|(1<<(n+q)) for q in keep)
    outside=((1<<(2*n))-1)^allowed; piv={}; rows=[]
    for p in checks:
        v=mask(p)
        while v & outside:
            j=(v & outside).bit_length()-1
            if j in piv: v^=piv[j]
            else:
                piv[j]=v; break
        if not v & outside: rows.append(v)
    return basis(rows)


def restrict_logical(p, checks, keep):
    n=len(p); outside=((1<<(2*n))-1)^sum((1<<q)|(1<<(n+q)) for q in keep)
    piv={}
    for s in checks:
        v=mask(s)
        while v & outside:
            j=(v & outside).bit_length()-1
            if j in piv: v^=piv[j]
            else:
                piv[j]=v; break
    v=mask(p)
    while v & outside:
        j=(v & outside).bit_length()-1
        if j not in piv: raise ValueError('Logical cannot be supported on retained data')
        v^=piv[j]
    out=stim.PauliString(n)
    for q in range(n): out[q]='IXZY'[(v>>q & 1)+2*(v>>(n+q) & 1)]
    return out


def sim_from(checks, logicals):
    n = len(logicals['X'])
    st = [p+stim.PauliString(1) for p in checks]
    for ax in 'XZ':
        p = logicals[ax]+stim.PauliString(1)
        p[n] = ax
        st.append(p)
    sim = stim.TableauSimulator(seed=81)
    sim.set_inverse_tableau(stim.Tableau.from_stabilizers(st).inverse())
    return sim, n


def projected_code(d, k, last_phase='A'):
    # Never calls the odd-k template. The starting wide code is the k=0 baseline.
    checks, logicals = ORIGINAL_INITIAL_CODE(d, 0, last_phase)
    sim, n = sim_from(checks, logicals)
    if k: sim.do(old.my_growth(d, 0, k))
    return old.split_ref(sim.canonical_stabilizers(), n)


def candidate(d, k, phase, gate='S', swap=False, toggle=False):
    c = ORIGINAL_EXTRACTION(d, k, phase)
    if not k % 2: return c
    b = old.base.Stage1(d)
    tip = b.anc(d-k, d+1)
    chunks = []; chunk = stim.Circuit()
    for op in c:
        if op.name == 'TICK':
            chunks.append(chunk); chunk = stim.Circuit()
        elif op.name in ('S', 'S_DAG'):
            assert [t.value for t in op.targets_copy()] == [tip]
        elif op.name == 'H' and toggle:
            qs = set(t.value for t in op.targets_copy()) ^ {tip}
            if qs: chunk.append('H', sorted(qs))
        else: chunk.append(op)
    assert len(chunks) == 8 and len(chunk) == 0
    moment = (3 if phase == 'A' else 4) if swap else (4 if phase == 'A' else 3)
    if gate != 'I': chunks[moment].append(gate if phase == 'A' else gate+'_DAG', [tip])
    out = stim.Circuit()
    for chunk in chunks:
        out += chunk; out.append('TICK')
    return out


def test(d, k, gate, swap, toggle):
    checks, lin = projected_code(d, k)
    sim, n = sim_from(checks, lin)
    b0 = basis(map(mask, checks))
    body = stim.Circuit()
    result = dict(d=d, k=k, gate=gate, swap=swap, toggle=toggle, cuts=[])
    prior = None
    geom=old.base.Stage1(d)
    keep={geom.data(i,j) for i in range(d) for j in range(1,geom.h)}
    keep-={geom.data(i,d+1) for i in range(d-k,d)}
    projected_data=restricted_basis(checks,keep)
    for j in range(6):
        phase = 'B' if j % 2 == 0 else 'A'
        c = candidate(d, k, phase, gate, swap, toggle)
        body += c; sim.do(c)
        try: checks_out, lout = old.split_ref(sim.canonical_stabilizers(), n)
        except (ValueError, AssertionError) as e:
            result.update(logical_survival=False, error=str(e)); return result
        bo = basis(map(mask, checks_out))
        cut = dict(phase=phase, outside_projected_span=sum(bool(residual(mask(p), b0)) for p in checks_out))
        if phase == 'A':
            cut['periodic'] = prior is not None and all(not residual(mask(p), prior) for p in checks_out)
            prior = bo
            db=restricted_basis(checks_out,keep)
            cut['data_rank']=len(db)
            cut['projected_data_rank']=len(projected_data)
            cut['data_matches_projection']=(len(db)==len(projected_data) and all(not residual(v,projected_data) for v in db.values()))
        result['cuts'].append(cut)
    probe = stim.Circuit()
    for p in checks: old.compiler.mpp(probe, p)
    probe += body
    result['logical_survival'] = all(old.base.flow(probe, lin[a], lout[a])['exists'] for a in 'XZ')
    dl={a:restrict_logical(lin[a],checks,keep) for a in 'XZ'}
    result['projected_logical_identity']={a:old.base.flow(probe,dl[a],dl[a])['exists'] for a in 'XZ'}
    data_checks=[]
    for v in projected_data.values():
        p=stim.PauliString(n)
        for q in range(n): p[q]='IXZY'[(v>>q & 1)+2*(v>>(n+q) & 1)]
        data_checks.append(p)
    measurements=body.solve_flow_measurements([stim.Flow(output=p) for p in data_checks])
    result['projected_checks_not_measured']=sum(x is None for x in measurements)
    result['final_logicals'] = {a:str(p) for a,p in lout.items()}
    return result


def snapshot_distance(checks, logical, seconds=20):
    from ortools.sat.python import cp_model
    n=len(logical); model=cp_model.CpModel()
    x=[model.new_bool_var(f'x{q}') for q in range(n)]
    z=[model.new_bool_var(f'z{q}') for q in range(n)]
    w=[model.new_bool_var(f'w{q}') for q in range(n)]
    for q in range(n): model.add_max_equality(w[q],[x[q],z[q]])
    def terms(p):
        return [v for q,a in enumerate(p) for v in ([x[q]] if a==3 else [z[q]] if a==1 else [x[q],z[q]] if a==2 else [])]
    for p in checks: model.add_bool_xor([*terms(p), True])
    model.add_bool_xor(terms(logical)); model.minimize(sum(w))
    solver=cp_model.CpSolver(); solver.parameters.max_time_in_seconds=seconds
    solver.parameters.num_search_workers=2; solver.parameters.random_seed=31415
    start=time.monotonic(); status=solver.solve(model)
    r=dict(status=solver.status_name(status), lower_bound=solver.best_objective_bound, seconds=time.monotonic()-start)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        p=stim.PauliString(n)
        for q in range(n): p[q]='IXZY'[solver.value(x[q])+2*solver.value(z[q])]
        assert all(p.commutes(s) for s in checks) and not p.commutes(logical)
        r.update(weight=p.weight, pauli=str(p))
    return r


if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--d',type=int,default=9)
    ap.add_argument('--k',type=int,default=1); ap.add_argument('--snapshot',action='store_true')
    args=ap.parse_args(); results=[]
    for gate in ('S','SQRT_X','I'):
        for swap in (False,True):
            for toggle in (False,True):
                r=test(args.d,args.k,gate,swap,toggle); results.append(r)
                print(json.dumps({k:v for k,v in r.items() if k!='final_logicals'}),flush=True)
    (ROOT/f'probe_d{args.d}_k{args.k}.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    if args.snapshot:
        checks,logicals=projected_code(args.d,args.k)
        r=snapshot_distance(checks,logicals['X'])
        (ROOT/f'projected_distance_d{args.d}_k{args.k}.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
        print('PROJECTED_DISTANCE',json.dumps(r),flush=True)
