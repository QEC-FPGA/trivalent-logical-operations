"""All physical Pauli branches, full detector signatures, and physical-site assignment.

Duplicate nonzero signatures can be represented by one Boolean: any minimal
undetectable logical set contains no repeated signature (equal pairs cancel).
Every physical realization is kept. No multi-detector column is discarded.
"""
import sys,time,subprocess,random,shutil
import numpy as np
import stim
from support import *
import terminal_move
from physical import replay,fc

NOISE={'X_ERROR','Y_ERROR','Z_ERROR','DEPOLARIZE1','DEPOLARIZE2'}
PAULI='IXYZ'
def count_m(op):
    if op.name in ('M','MX','MY'):return len(op.targets_copy())
    return len(fc.mpp_products(op)) if op.name=='MPP' else 0
def load_circuit(spec):return stim.Circuit((case_dir(spec)/'full_original.stim').read_text(encoding='utf-8'))
def dem_signatures(c):
    vals=set();D=c.num_detectors
    for op in c.detector_error_model(decompose_errors=False,flatten_loops=True):
        if op.type=='error' and op.args_copy()[0]>0:vals.add(fc.signature(op,D))
    vals.discard(0);return sorted(vals)
def enumerate_branches(c,signatures):
    assert c.num_observables==1
    ops=list(c);D=c.num_detectors;records=[0]*c.num_measurements;m=0;di=0;tick=0;ticks=[]
    for op in ops:
        ticks.append(tick);tick+=op.name=='TICK'
        if op.name in ('DETECTOR','OBSERVABLE_INCLUDE'):
            if op.name=='OBSERVABLE_INCLUDE':assert op.gate_args_copy()==[0]
            v=1<<(di if op.name=='DETECTOR' else D)
            for r in op.targets_copy():
                assert r.is_measurement_record_target
                records[m+r.value]^=v
            di+=op.name=='DETECTOR'
        else:m+=count_m(op)
    assert m==len(records) and di==D
    lookup={v:j for j,v in enumerate(signatures)};seen=set();x=[0]*c.num_qubits;z=x.copy()
    offsets=[];q1=[];q2=[];branches=[];columns=[];sites=[];total=0;zero=0
    for i in range(len(ops)-1,-1,-1):
        op=ops[i];nm=op.name;ts=op.targets_copy();qs=[t.value for t in ts]
        if nm in NOISE:
            if not op.gate_args_copy()[0]>0:continue
            size=2 if nm=='DEPOLARIZE2' else 1
            for j in range(0,len(qs),size):
                a=qs[j];b=qs[j+1] if size==2 else -1;site=len(offsets)
                offsets.append(i);q1.append(a);q2.append(b)
                pa=(0,x[a],x[a]^z[a],z[a]);pb=(0,x[b],x[b]^z[b],z[b]) if b>=0 else (0,)
                opts=range(1,16) if size==2 else range(1,4) if nm=='DEPOLARIZE1' else (PAULI.index(nm[0]),)
                for branch in opts:
                    total+=1;sig=pa[branch//4]^pb[branch%4] if size==2 else pa[branch]
                    if not sig:zero+=1;continue
                    if sig not in lookup:raise ValueError('Physical branch missing from undecomposed Stim model')
                    seen.add(sig);sites.append(site);columns.append(lookup[sig]);branches.append(branch)
        elif nm in ('TICK','QUBIT_COORDS','DETECTOR','OBSERVABLE_INCLUDE'):continue
        elif nm=='R':
            for q in qs:x[q]=z[q]=0
        elif nm=='H':
            for q in qs:x[q],z[q]=z[q],x[q]
        elif nm in ('S','S_DAG'):
            for q in qs:x[q]^=z[q]
        elif nm=='CX':
            for a,b in zip(qs[::2],qs[1::2]):x[a]^=x[b];z[b]^=z[a]
        elif nm=='CZ':
            for a,b in zip(qs[::2],qs[1::2]):x[a]^=z[b];x[b]^=z[a]
        elif nm in ('M','MX','MY'):
            m-=len(qs)
            for j,q in enumerate(qs):
                if nm in ('M','MY'):x[q]^=records[m+j]
                if nm in ('MX','MY'):z[q]^=records[m+j]
        elif nm=='MPP':
            products=fc.mpp_products(op);m-=len(products)
            for j,product in enumerate(products):
                for t in product:
                    if t.is_z_target or t.is_y_target:x[t.value]^=records[m+j]
                    if t.is_x_target or t.is_y_target:z[t.value]^=records[m+j]
        else:raise ValueError(f'Unsupported physical instruction: {nm}')
    assert m==0 and seen==set(signatures)
    arrays=dict(site_offsets=np.asarray(offsets,dtype=np.int32),site_q1=np.asarray(q1,dtype=np.int32),
        site_q2=np.asarray(q2,dtype=np.int32),branch_sites=np.asarray(sites,dtype=np.int32),
        branch_columns=np.asarray(columns,dtype=np.int32),branch_paulis=np.asarray(branches,dtype=np.int8),
        instruction_ticks=np.asarray(ticks,dtype=np.int32))
    return arrays,dict(physical_sites=len(offsets),all_physical_branches=total,zero_response_branches=zero,
        nonzero_physical_branches=len(sites),unique_nonzero_signatures=len(signatures),
        all_undecomposed_Stim_signatures_equal=True)
def event(c,data,branch_index,meta=None):
    j=int(branch_index);s=int(data['branch_sites'][j]);i=int(data['site_offsets'][s]);op=c[i]
    a=int(data['site_q1'][s]);b=int(data['site_q2'][s]);p=int(data['branch_paulis'][j])
    qs=[a,b] if b>=0 else [a];ps=[PAULI[p//4],PAULI[p%4]] if b>=0 else [PAULI[p]]
    tick=int(data['instruction_ticks'][i])
    e=dict(instruction_offset=i,tick=tick,noise_kind=op.name,noise_targets=qs,
        noise_args=op.gate_args_copy(),paulis={str(q):ax for q,ax in zip(qs,ps) if ax!='I'})
    if meta:
        st=next((s for s in reversed(meta['stages']) if s['start_tick']<=tick),None)
        e.update(stage=st['stage'] if st else 'input_preparation',local_tick=tick-st['start_tick'] if st else tick)
    return e
def assign(selected,data):
    needed=set(map(int,selected));choices={j:{} for j in needed}
    for bi,col in enumerate(data['branch_columns']):
        j=int(col)
        if j in needed:choices[j].setdefault(int(data['branch_sites'][bi]),bi)
    match={}
    def visit(j,seen):
        for s in choices[j]:
            if s in seen:continue
            seen.add(s)
            if s not in match or visit(match[s],seen):match[s]=j;return True
        return False
    for j in sorted(needed,key=lambda j:len(choices[j])):
        if not visit(j,set()):return None
    return [choices[j][s] for s,j in sorted(match.items())]
def materialize(spec,selected,dest):
    data=np.load(result_dir(spec)/'physical_sites.npz',allow_pickle=False)
    assigned=assign(selected,data)
    if assigned is None:return None
    c=load_circuit(spec);meta=read(case_dir(spec)/'metadata.json')
    es=[event(c,data,j,meta) for j in assigned]
    r=replay(c,es,dest)
    if r['status']!='UNDETECTED_LOGICAL':raise ValueError('Selected solution failed physical Stim replay')
    r['selected_signature_columns']=list(map(int,selected));write(Path(dest)/'witness_events.json',r)
    return r
def read_cover(path):
    with Path(path).open() as f:
        nr,n,_=map(int,f.readline().split());target=int(f.readline())
        cols=[tuple(map(int,line.split()[1:])) for line in f]
    assert n==len(cols);return nr,target,cols
def verify_reduction(original,reduced,operations):
    nr,target,cols=read_cover(original);nr2,target2,red=read_cover(reduced)
    assert (nr,target,len(cols))==(nr2,target2,len(red))
    rows=[0]*nr;actual=[0]*nr;tv=1<<target
    for j,col in enumerate(cols):
        for r in col:rows[r]^=1<<j
    for line in Path(operations).read_text().splitlines():
        if not line.strip():continue
        a,b=map(int,line.split());assert 0<=a<nr and 0<=b<nr and a!=b
        rows[a]^=rows[b]
        if tv>>b&1:tv^=1<<a
    for j,col in enumerate(red):
        for r in col:actual[r]^=1<<j
    assert rows==actual and tv==1<<target
    return red
def prepare(spec,reduce_seconds=180):
    dst=result_dir(spec);dst.mkdir(parents=True,exist_ok=True);mp=dst/'model.json'
    cs=sha(case_dir(spec)/'full_original.stim')
    if mp.exists():
        m=read(mp)
        if m['circuit_sha256']!=cs or m['config_sha256']!=spec['config_sha256']:raise ValueError('Stale fault model')
        for p,h in m['artifacts_sha256'].items():
            if sha(dst/p)!=h:raise ValueError(f'Fault model artifact changed: {p}')
        return m
    t0=time.monotonic();c=load_circuit(spec);sig=dem_signatures(c);D=c.num_detectors
    data,counts=enumerate_branches(c,sig)
    np.savez_compressed(dst/'physical_sites.npz',**data)
    cover=dst/'all_faults.cover';reduced=dst/'reduced.cover';ops=dst/'reduced.cover.rowops'
    with cover.open('w') as f:
        f.write(f'{D+1} {len(sig)} 1\n{D}\n')
        for v in sig:
            col=list(bits(v));f.write(str(len(col))+' '+' '.join(map(str,col))+'\n')
    # Audit physical responses independently of the backward enumerator.
    rng=random.Random(579);chosen=rng.sample(range(len(data['branch_columns'])),min(64,len(data['branch_columns'])))
    es=[event(c,data,j) for j in chosen];actual=fc.propagate(c,es)
    assert actual==[sig[int(data['branch_columns'][j])] for j in chosen]
    for e,v in zip(es[:3],actual[:3]):assert fc.injected(c,[e])[0]==v
    reducer=ROOT/'engine/vendor/lib/frozen/native/lib/reduce_cover.py'
    try:
        p=subprocess.run([sys.executable,str(reducer),str(cover),str(reduced)],capture_output=True,text=True,
            timeout=reduce_seconds,check=True)
        red=verify_reduction(cover,reduced,ops);reduction='VERIFIED_INVERTIBLE'
        (dst/'reduction.log').write_text(p.stdout+p.stderr,encoding='utf-8')
    except subprocess.TimeoutExpired:
        # Identity is a complete equivalent model, never a truncated model.
        reduced.write_bytes(cover.read_bytes());ops.write_text('',encoding='utf-8')
        red=read_cover(cover)[2];reduction='IDENTITY_TIMEOUT'
    from upper_seeds import prepare_seeds
    seeds=prepare_seeds(spec);upper=seeds['upper_bound'];local=seeds['checks']
    graph_status='SKIPPED_LOCAL_PHYSICAL_WITNESS_AVAILABLE'
    if upper is not None:
        dst.joinpath('upper_witness').mkdir(exist_ok=True)
        for name in ('witness_events.json','witness_injected.stim'):
            shutil.copy2(case_dir(spec)/'seed_witness'/name,dst/'upper_witness'/name)
    else:
        # The optional graph heuristic must not block complete-model search.
        # Losing this heuristic on timeout never removes a physical column.
        try:
            child=subprocess.run([sys.executable,str(ROOT/'upper_seeds.py'),'--graph-case',spec['id'],
                '--dest',str(dst/'graph_seed')],capture_output=True,text=True,timeout=60,check=True)
            (dst/'graph_seed.log').write_text(child.stdout+child.stderr,encoding='utf-8')
            graph_status='FINISHED_WITHOUT_WITNESS'
            if (dst/'graph_seed'/'witness_events.json').exists():
                r=read(dst/'graph_seed'/'witness_events.json')
                assert r['status']=='UNDETECTED_LOGICAL'
                replay(c,r['events'],dst/'upper_witness');upper=r['physical_weight']
                graph_status='PHYSICAL_WITNESS'
        except subprocess.TimeoutExpired:graph_status='TIMEOUT_60_SECONDS_NO_DISTANCE_CONCLUSION'
    files=['physical_sites.npz','all_faults.cover','reduced.cover','reduced.cover.rowops']
    if upper is not None:files+=['upper_witness/witness_events.json','upper_witness/witness_injected.stim']
    m=dict(**counts,detectors=D,config_sha256=spec['config_sha256'],circuit_sha256=cs,
        fault_model='All positive-probability SI1000 Pauli branches at original physical sites',
        signature_compression='Equal nonzero responses deduplicated; all physical realizations retained',
        physical_assignment='One branch per original noise site; exact bipartite assignment for each feasible signature solution',
        reduction=reduction,seed_upper_bound=upper,local_upper_seeds=local,graph_seed_status=graph_status,
        seed_scope='Graph subset and local physical constructions only seed upper bounds; exact search uses every signature',
        independent_forward_checks=len(es),independent_Stim_checks=3,seconds=time.monotonic()-t0,
        artifacts_sha256={p:sha(dst/p) for p in files})
    write(mp,m);print('MODEL',spec['id'],counts['unique_nonzero_signatures'],'signatures','upper',upper,flush=True)
    return m
