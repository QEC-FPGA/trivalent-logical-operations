"""Physical counterexamples only; these routines never establish a lower bound."""
from support import *
import terminal_move
from physical import circuit,sites,replay
from explicit_family import build_events
from study import qubit

def terminal_events(c,d,q,anchor):
    assert q>=3 and q%2==1 and d%2==1 and q<d
    loc={}
    for i,t,op,qs in sites(c):
        if t-anchor not in (-8,-4,-1):continue
        for v in qs:loc.setdefault((t,v),[]).append((i,op,qs))
    events=[]
    def add(dt,support,kind,part):
        pa={qubit(v,d):p for v,p in support}
        for i,op,qs in loc.get((anchor+dt,next(iter(pa))),[]):
            if op.name!=kind or not set(pa)<=set(qs):continue
            events.append(dict(instruction_offset=i,tick=anchor+dt,noise_kind=op.name,
                noise_targets=qs,noise_args=op.gate_args_copy(),paulis={str(v):p for v,p in pa.items()},part=part))
            return
        raise ValueError(('Terminal family has no real physical site',dt,support,kind))
    # Final MY is in the extraction readout tick: use its actual measurement
    # error, never a fictitious idle-depolarization site on a measured qubit.
    add(-1,[(('q',q-2,d+1),'X')],'X_ERROR','terminal_MY_readout')
    for x in range(q-1,d-1):add(-1,[(('q',x,d+2),'X')],'DEPOLARIZE1','right_extension')
    add(-8,[(('a',d,d+2),'X')],'X_ERROR','right_ancilla')
    for x in range(2,q-2,2):
        add(-4,[(('a',x,d+1),'X'),(('q',x,d+1),'Z')],'DEPOLARIZE2','paired_hook')
    add(-1,[(('q',0,d+1),'X')],'X_ERROR','terminal_MY_readout')
    assert len(events)==d-(q-3)//2
    return events

def prepare_seeds(spec):
    folder=case_dir(spec);out=folder/'seed_check.json';cs=sha(folder/'full_original.stim')
    if out.exists():
        result=read(out);assert result['circuit_sha256']==cs
        if result['upper_bound'] is not None:
            for rel,h in result['artifacts_sha256'].items():assert sha(folder/rel)==h
        return result
    c=circuit(folder/'full_original.stim');m=read(folder/'metadata.json');upper=None;k=0;records=[]
    for j,length in enumerate(spec['moves']):
        final=j==len(spec['moves'])-1
        label=f'wall_MY_{length}' if final else f'grow_{k}_to_{k+length}_MY'
        st=next(s for s in m['stages'] if s['stage']==label)
        phase='A' if j==0 else m['phase_sequence'][sum(spec['holds'][:j])-1]['phase']
        anchor=st['start_tick']+int(final)
        if not final and length<2:
            # This inherited local family is derived for jumps >=2. The exact
            # fault model still contains every physical fault of this move.
            records.append(dict(k=k,length=length,phase=phase,status='CONSTRUCTION_NOT_APPLICABLE_TO_ONE_STEP'))
            k+=length
            continue
        try:
            events=terminal_events(c,spec['d'],length,anchor) if final else build_events(c,spec['d'],k,length,phase,anchor)
            for e in events:
                pos=next((s for s in reversed(m['stages']) if s['start_tick']<=e['tick']),None)
                e.update(stage=pos['stage'],local_tick=e['tick']-pos['start_tick'])
            r=replay(c,events)
            records.append(dict(k=k,length=length,phase=phase,kind='terminal' if final else 'internal',
                status=r['status'],physical_weight=r['physical_weight']))
            if r['status']=='UNDETECTED_LOGICAL' and (upper is None or r['physical_weight']<upper):
                replay(c,events,folder/'seed_witness');upper=r['physical_weight']
        except ValueError as exc:
            records.append(dict(k=k,length=length,phase=phase,status='NOT_MAPPED',detail=str(exc)))
        k+=length
    paths=['seed_witness/witness_events.json','seed_witness/witness_injected.stim'] if upper is not None else []
    result=dict(circuit_sha256=cs,upper_bound=upper,checks=records,artifacts_sha256={p:sha(folder/p) for p in paths},
        scope='Constructed physical internal/terminal families only; no distance lower bound; no exact-distance claim')
    write(out,result)
    print('PHYSICAL SEEDS',spec['id'],'upper',upper,flush=True)
    return result

def graph_seed(spec,destination):
    """Optional graph subset heuristic; invoked in a time-limited subprocess."""
    from fault_model import read_cover,materialize
    import stim
    nr,D,cols=read_cover(result_dir(spec)/'reduced.cover')
    model=stim.DetectorErrorModel();lookup={}
    for j,col in enumerate(cols):
        model.append('error',.001,[stim.target_logical_observable_id(0) if v==D else stim.target_relative_detector_id(v) for v in col]);lookup[col]=j
    try:
        chain=model.shortest_graphlike_error(ignore_ungraphlike_errors=True)
        selected=[lookup[tuple(sorted(t.val if t.is_relative_detector_id() else D for t in op.targets_copy()))] for op in chain if op.type=='error']
        return materialize(spec,selected,destination)
    except ValueError:return None

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--graph-case');ap.add_argument('--dest',type=Path)
    a=ap.parse_args()
    if a.graph_case:
        spec=next(s for s in plan() if s['id']==a.graph_case)
        graph_seed(spec,a.dest)
    else:
        for spec in plan():prepare_seeds(spec)
