"""Ascending-weight exhaustive decision search using complete fault signatures.

The parity/cardinality search is exact CP-SAT, not a graphlike approximation.
Unassignable signature sets are excluded only after exact site matching fails.
An INFEASIBLE result proves absence up to that cap; UNKNOWN proves nothing new.
"""
import gzip,time,math
import ortools
from ortools.sat.python import cp_model
from support import *
from fault_model import prepare,read_cover,materialize

def build_model(columns,D,cap):
    m=cp_model.CpModel();variables=[m.new_bool_var(f'f{j}') for j in range(len(columns))]
    incidence=[[] for _ in range(D+1)]
    for j,col in enumerate(columns):
        for i in col:incidence[i].append(variables[j])
    for i,vs in enumerate(incidence):
        if not vs:
            if i==D:m.add_bool_or([])
            continue
        # Exact integer encoding of parity; no truncation of detector supports.
        parity=m.new_int_var(0,len(vs)//2,f'parity{i}')
        m.add(sum(vs)==2*parity+int(i==D))
    m.add(sum(variables)<=cap)
    return m,variables

def solve_cap(spec,cap,seconds=900,workers=2,seed=579):
    dst=result_dir(spec);info=read(dst/'model.json');nr,D,columns=read_cover(dst/'reduced.cover')
    assert D==nr-1
    folder=dst/'bounds'/f'cap_{cap:02d}';folder.mkdir(parents=True,exist_ok=True)
    prior=read(folder/'result.json') if (folder/'result.json').exists() else None
    if prior:
        assert prior['model_sha256']==digest(info)
        if prior['status'] in ('INFEASIBLE','PHYSICAL_WITNESS'):return prior
    attempt=1
    while (folder/f'attempt_{attempt:03d}.json').exists():attempt+=1
    prefix=folder/f'attempt_{attempt:03d}'
    if cap<=2:
        # Exhaustive linear-time meet-in-the-middle for one/two signatures.
        # Uses the full original signature set, including all hyperedges.
        t0=time.monotonic();_,_,original=read_cover(dst/'all_faults.cover')
        signatures=[sum(1<<v for v in col) for col in original]
        index={v:j for j,v in enumerate(signatures)};target=1<<D
        candidates=[]
        if target in index:candidates.append([index[target]])
        if cap==2:
            for j,v in enumerate(signatures):
                other=index.get(v^target)
                if other is not None and j<other:candidates.append([j,other])
        witness=None;attempted=0
        for selected in candidates:
            attempted+=1;witness=materialize(spec,selected,prefix.with_name(prefix.name+'_witness'))
            if witness:break
        result=dict(case=spec['id'],cap=cap,status='PHYSICAL_WITNESS' if witness else 'INFEASIBLE',
            method='Exhaustive full-signature hash enumeration plus exact physical-site assignment',
            signatures=len(signatures),candidate_sets=len(candidates),physical_assignments_tested=attempted,
            model_sha256=digest(info),circuit_sha256=info['circuit_sha256'],seconds=time.monotonic()-t0,
            physical_weight=witness['physical_weight'] if witness else None,
            witness_path=str(prefix.with_name(prefix.name+'_witness').relative_to(dst)) if witness else None,
            lower_bound=None if witness else cap+1)
        write(prefix.with_suffix('.json'),result);write(folder/'result.json',result)
        print('BOUND',spec['id'],cap,result['status'],'full hash enumeration',flush=True)
        return result
    cuts=read(folder/'site_exclusion_cuts.json') if (folder/'site_exclusion_cuts.json').exists() else []
    model,vs=build_model(columns,D,cap)
    for selected in cuts:model.add(sum(vs[j] for j in selected)<=len(selected)-1)
    initial_model=prefix.with_suffix('.model.pbtxt.gz')
    with gzip.open(initial_model,'wt',encoding='utf-8') as f:f.write(str(model.proto))
    t0=time.monotonic();iterations=[];status='UNKNOWN';witness=None
    with prefix.with_suffix('.solver.log').open('w',encoding='utf-8') as log:
        while True:
            remain=seconds-(time.monotonic()-t0) if seconds>0 else None
            if remain is not None and remain<=0:break
            solver=cp_model.CpSolver()
            if remain is not None:solver.parameters.max_time_in_seconds=remain
            solver.parameters.num_search_workers=workers;solver.parameters.random_seed=seed
            solver.parameters.log_search_progress=True;solver.parameters.log_to_stdout=False
            solver.log_callback=lambda s:(log.write(s+'\n'),log.flush())
            raw=solver.solve(model)
            it=dict(solver_status=solver.status_name(raw),solver_response=solver.response_stats(),seconds=time.monotonic()-t0)
            if raw==cp_model.INFEASIBLE:
                status='INFEASIBLE';iterations.append(it);break
            if raw not in (cp_model.OPTIMAL,cp_model.FEASIBLE):
                status='UNKNOWN' if raw==cp_model.UNKNOWN else 'MODEL_INVALID';iterations.append(it);break
            selected=[j for j,v in enumerate(vs) if solver.value(v)]
            it['selected_columns']=selected
            witness=materialize(spec,selected,prefix.with_name(prefix.name+'_witness'))
            it['physical_site_assignment_passed']=witness is not None;iterations.append(it)
            if witness is not None:
                assert witness['physical_weight']<=cap
                status='PHYSICAL_WITNESS';break
            # No valid physical solution can contain this unassignable subset.
            cuts.append(selected);write(folder/'site_exclusion_cuts.json',cuts)
            model.add(sum(vs[j] for j in selected)<=len(selected)-1)
    result=dict(case=spec['id'],cap=cap,status=status,seconds=time.monotonic()-t0,
        time_limit_seconds=seconds if seconds else None,solver_workers=workers,random_seed=seed,
        solver_version=ortools.__version__,model_sha256=digest(info),circuit_sha256=info['circuit_sha256'],
        exported_initial_model=initial_model.name,exported_initial_model_sha256=sha(initial_model),
        site_exclusion_cuts=len(cuts),iterations=iterations,
        physical_weight=witness['physical_weight'] if witness else None,
        witness_path=str(prefix.with_name(prefix.name+'_witness').relative_to(dst)) if witness else None,
        lower_bound=cap+1 if status=='INFEASIBLE' else None)
    write(prefix.with_suffix('.json'),result);write(folder/'result.json',result)
    print('BOUND',spec['id'],cap,status,round(result['seconds'],2),flush=True)
    return result

def summarize_case(spec,save=True):
    dst=result_dir(spec)
    if not (dst/'model.json').exists():
        p=case_dir(spec)/'seed_check.json'
        if p.exists():
            seed=read(p)
            assert seed['circuit_sha256']==sha(case_dir(spec)/'full_original.stim')
            if seed['upper_bound'] is not None:
                return dict(case=spec['id'],configuration=spec,status='UPPER_BOUND_ONLY',lower_bound=None,
                    upper_bound=seed['upper_bound'],exact_distance=None,circuit_sha256=seed['circuit_sha256'],
                    witness_path=str((case_dir(spec)/'seed_witness').relative_to(ROOT)),
                    scope='Physical counterexample only; complete-model distance search not started')
        return dict(case=spec['id'],status='NOT_SEARCHED',lower_bound=None,upper_bound=None,exact_distance=None)
    model=read(dst/'model.json');upper=model['seed_upper_bound'];lower=1;witness_path='upper_witness' if upper else None
    checked=[];incomplete=[]
    for p in sorted((dst/'bounds').glob('cap_*/result.json')):
        r=read(p)
        if r['model_sha256']!=digest(model):raise ValueError('Bound belongs to a different fault model')
        checked.append(dict(cap=r['cap'],status=r['status']))
        if r['status']=='INFEASIBLE':lower=max(lower,r['cap']+1)
        elif r['status']=='PHYSICAL_WITNESS':
            w=r['physical_weight']
            if upper is None or w<upper:upper=w;witness_path=r['witness_path']
        else:incomplete.append(r['cap'])
    if upper is not None and lower>upper:raise ValueError('Contradictory bounds; stop and investigate')
    exact=upper is not None and lower==upper
    status='EXACT' if exact else 'BOUNDED' if upper is not None else 'LOWER_BOUND_ONLY' if lower>1 else 'UNKNOWN'
    r=dict(case=spec['id'],configuration=spec,status=status,lower_bound=lower,upper_bound=upper,
        exact_distance=upper if exact else None,witness_path=witness_path,checked_caps=checked,incomplete_caps=incomplete,
        circuit_sha256=model['circuit_sha256'],model_sha256=digest(model),
        scope='Circuit-level undetectable logical fault distance of this XY or ZZ test with ideal terminal code/logical MPP; one physical site counts once')
    if save:write(dst/'distance.json',r)
    return r

def search(spec,seconds=900,workers=2,maximum=None,reduce_seconds=180):
    prepare(spec,reduce_seconds=reduce_seconds)
    maximum=spec['d'] if maximum is None else maximum
    for cap in range(1,maximum+1):
        state=summarize_case(spec)
        if state['status']=='EXACT':break
        if cap<state['lower_bound']:continue
        if state['upper_bound'] is not None and cap>=state['upper_bound']:break
        r=solve_cap(spec,cap,seconds,workers)
        if r['status'] not in ('INFEASIBLE','PHYSICAL_WITNESS'):break
    state=summarize_case(spec);print('DISTANCE',spec['id'],state['status'],state.get('lower_bound'),state.get('upper_bound'),flush=True)
    return state
