"""Actual fault replays and restricted counterexample search, never lower bounds."""
from pipeline import *
from physical import circuit,replay,find,normalized,validate_events
from explicit_family import build_events
import time,collections

def ticks(c):
    t=0
    for i,o in enumerate(c):
        yield i,t,o
        if o.name=='TICK':t+=1

def annotate_events(c,meta,events):
    xy=c.get_final_qubit_coordinates();out=[]
    for e in events:
        z=dict(e);t=e['tick']
        stages=[s for s in meta['stages'] if s['start_tick']<=t<s['start_tick']+s['ticks']]
        z['stage']=stages[-1]['stage'] if stages else 'boundary'
        z['coordinates']={str(q):xy.get(int(q)) for q in e['paulis']};out.append(z)
    return out

def seed_replays(p,c,meta):
    name={9:'d9_W1_six',15:'d15_W1_eleven'}.get(p['d'])
    if p['d']==13 and p['moves']==[3,3,3,1]:name='d13_multi_move_eleven'
    if not name:return []
    sc=circuit(PACKAGE/'mechanism_seeds'/f'{name}_original.stim')
    se=read(PACKAGE/'mechanism_seeds'/f'{name}_events.json')['events']
    old_first=next(t for i,t,o in ticks(sc) if o.name=='MY')
    new_first=next(s['start_tick'] for s in meta['stages'] if s['stage'].startswith('grow_') and s['stage'].endswith('_MY'))
    old_xy=sc.get_final_qubit_coordinates();new_inv={tuple(v):q for q,v in c.get_final_qubit_coordinates().items()}
    translated=[]
    try:
        for raw in se:
            e=normalized(raw);op=sc[e['instruction_offset']]
            target=[new_inv[tuple(old_xy[int(q)])] for q in e['noise_targets']]
            pa={new_inv[tuple(old_xy[int(q)])]:v for q,v in e['paulis'].items()}
            translated.append(find(c,e['tick']+new_first-old_first,pa,op.name,target))
        r=replay(c,translated)
    except (ValueError,KeyError) as exc:
        r=dict(status='NO_IDENTICAL_PHYSICAL_PLACEMENT',reason=str(exc))
    r.update(kind='multi_move_chain_aligned_to_first_movement' if name=='d13_multi_move_eleven' else 'old_W1_chain_aligned_to_first_movement',seed=name,
        limitation='A failed placement is not a detected chain and does not exclude retimed variants')
    return [r]

def search_corridor(p,c,meta,seconds):
    """Full syndrome parity within a declared spatial/time corridor only."""
    import importlib.util
    from physical import fc
    spec=importlib.util.spec_from_file_location('full_boundary_reverse',Path(__file__).resolve().parent/'vendor/lib/reverse_faults.py')
    rev=importlib.util.module_from_spec(spec)
    previous=sys.modules.get('fault_checks');sys.modules['fault_checks']=fc
    try:spec.loader.exec_module(rev)
    finally:
        if previous is None:sys.modules.pop('fault_checks',None)
        else:sys.modules['fault_checks']=previous
    FaultIndex=rev.FaultIndex
    from ortools.sat.python import cp_model
    started=time.monotonic();idx=FaultIndex(c);xy=c.get_final_qubit_coordinates();D=c.num_detectors
    first=next(s['start_tick'] for s in meta['stages'] if s['stage'].startswith('grow_') and s['stage'].endswith('_MY'))
    wide=next(s['start_tick'] for s in meta['stages'] if s['stage'].startswith('wide_'))
    walls=[s['start_tick'] for s in meta['stages'] if s['stage'].startswith('fullwall')]
    stop=min(walls)+16 if walls else c.num_ticks
    eligible={}
    for i,op in enumerate(idx.ops):
        t=idx.ticks[i]
        if not max(0,wide-8)<=t<=stop or op.name not in ('X_ERROR','Y_ERROR','Z_ERROR','DEPOLARIZE1','DEPOLARIZE2'):continue
        qs=[v.value for v in op.targets_copy()];size=2 if op.name=='DEPOLARIZE2' else 1
        for j in range(0,len(qs),size):
            q=qs[j:j+size];coords=[xy.get(v) for v in q]
            if any(v is None or len(v)<2 for v in coords):continue
            if all(0<=v[0]<=p['d'] and p['d']/3<=v[1]<=p['d']+1.5 for v in coords):
                eligible[(i,j)]=(t<=first and size==2 and all(v[0]<=2 for v in coords))
    choices={};entry_codes=set()
    class Collector(dict):
        def setdefault(self,sig,code):
            i=code>>32;j=(code&0xffffffff)>>4;branch=code&15
            if sig==0 or (i,j) not in eligible:return
            # Retain one Pauli representative per signature at the same real site.
            key=(sig,i,j);choices.setdefault(key,code)
            if eligible[(i,j)] and branch in (1,4):entry_codes.add(code)
    idx.reps=Collector();idx.enumerate()
    data=[(k[0],code,k[1:]) for k,code in choices.items()]
    # A non-entry representative must not hide an entry branch with identical response.
    present_codes={x[1] for x in data}
    for code in entry_codes:
        if code not in present_codes:
            e=idx.event(code)
            from physical import fc
            sig=fc.propagate(c,[e])[0];data.append((sig,code,(code>>32,(code&0xffffffff)>>4)));present_codes.add(code)
    model=cp_model.CpModel();vs=[model.new_bool_var(f'e{i}') for i in range(len(data))]
    incidence=collections.defaultdict(list);sitevars=collections.defaultdict(list);entry=[]
    for j,(sig,code,site) in enumerate(data):
        sitevars[site].append(vs[j])
        if code in entry_codes:entry.append(vs[j])
        while sig:
            bit=sig&-sig;incidence[bit.bit_length()-1].append(vs[j]);sig^=bit
    base=dict(scope='Restricted space/time corridor and a left-X entrance; no lower-bound interpretation',
        d=p['d'],W=p['W'],step=p['step'],candidate_branches=len(data),
        corridor=dict(start_tick=max(0,wide-8),end_tick=stop,x=[0,p['d']],y=[p['d']/3,p['d']+1.5]),
        cap=p['d']-1,time_limit=seconds)
    if not entry or D not in incidence:return dict(**base,status='NO_CANDIDATE_IN_DECLARED_REGION')
    for bit,items in incidence.items():model.add(sum(items)==2*model.new_int_var(0,len(items)//2,f'p{bit}')+int(bit==D))
    for items in sitevars.values():model.add(sum(items)<=1)
    model.add(sum(entry)>=1);model.add(sum(vs)<=p['d']-1);model.minimize(sum(vs))
    solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;solver.parameters.random_seed=18341
    left=seconds;attempts=[]
    while left>0:
        solver.parameters.max_time_in_seconds=left;t=time.monotonic();status=solver.solve(model);left-=time.monotonic()-t
        attempts.append(solver.status_name(status))
        if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):break
        selected=[i for i,v in enumerate(vs) if solver.value(v)]
        # Require the logical response component to include the left entrance.
        unused=set(selected);has_connected_entrance=False
        while unused:
            component={unused.pop()};support=0;changed=True
            while changed:
                changed=False
                for i in component:support|=data[i][0]&((1<<D)-1)
                for i in list(unused):
                    if data[i][0]&support:component.add(i);unused.remove(i);changed=True
            parity=0
            for i in component:parity^=data[i][0]
            if parity==1<<D and any(data[i][1] in entry_codes for i in component):has_connected_entrance=True
        if not has_connected_entrance:
            model.add(sum(vs[i] for i in selected)<=len(selected)-1);continue
        ev=[idx.event(data[i][1]) for i in selected];r=replay(c,ev)
        assert r['status']=='UNDETECTED_LOGICAL'
        return dict(**base,**r,solver_status=solver.status_name(status),attempts=attempts,
                    seconds=time.monotonic()-started,global_minimum_claimed=False)
    return dict(**base,status='NO_WITNESS_WITHIN_SEARCH_LIMIT',solver_status=attempts[-1] if attempts else 'NOT_RUN',
                attempts=attempts,seconds=time.monotonic()-started,global_minimum_claimed=False)

def one(job):
    p,seconds=job;binding(p);folder=case_folder(p);meta=read(folder/'metadata.json')
    c=circuit(folder/'full_original.stim');out=ROOT/'mechanism'/p['case'];out.mkdir(parents=True,exist_ok=True)
    local=[];k=0
    for step in p['moves']:
        if step>=2:
            stage=next(s for s in meta['stages'] if s['stage']==f'grow_{k}_to_{k+step}_MY')
            ev=build_events(c,p['d'],k,step,'A',stage['start_tick'])
            r=replay(c,ev);r.update(k=k,length=step,phase_before='A',events=annotate_events(c,meta,ev));local.append(r)
        k+=step
    seeds=seed_replays(p,c,meta);write(out/'local_and_seed_replays.json',dict(local=local,seeds=seeds,
        circuit_sha256=sha(folder/'full_original.stim'),note='All W are odd and intermediate extraction counts are 2; each move follows A'))
    if seconds>0 and p['channel']=='XY':
        sparse=circuit(compiled_folder(p)/'full_matching.stim')
        r=search_corridor(p,sparse,meta,seconds)
        if r.get('status')=='UNDETECTED_LOGICAL':
            # Restore original detector labels without changing any physical fault.
            ev=[find(c,e['tick'],{int(q):v for q,v in e['paulis'].items()},e['noise_kind'],e['noise_targets']) for e in r['events']]
            checked=replay(c,ev,out/'left_entry_witness');assert checked['status']=='UNDETECTED_LOGICAL'
            r['original_basis_replay']=checked;r['events']=annotate_events(c,meta,ev)
        r['original_circuit_sha256']=sha(folder/'full_original.stim');write(out/'left_entry_search.json',r)
    return dict(case=p['case'],status='MECHANISM_CHECKED_NO_DISTANCE_LOWER_BOUND')

def run(group,workers,ids,seconds):
    if seconds<0:raise ValueError('Nonnegative search time required')
    ps=selected(group,ids);cases={p['case']:p for p in ps}
    return parallel(one,[(p,seconds) for p in cases.values()],workers)
