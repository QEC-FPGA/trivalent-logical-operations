"""Cheap rejection checks only. PASS here never means a distance lower bound."""
from pathlib import Path
import argparse,json,subprocess,sys,time
import stim
import generate as gen

def signature(op,D):
    out=set()
    for t in op.targets_copy():
        assert not t.is_separator(),'Do not decompose hyperedges'
        out.symmetric_difference_update((t.val if t.is_relative_detector_id() else D,))
    return tuple(sorted(out))

def replay_known(folder,report):
    d=report['distance'];faultfile=gen.HERE/'regression_faults'/f'd{d}.json'
    if not faultfile.exists():return {'status':'NO_ARCHIVED_WITNESS'}
    source=folder/f'd{d}_YX_SI1000_body.stim'
    if not source.exists():return {'status':'YX_NOT_GENERATED'}
    w=report['resources']['wide_rounds']
    b=gen.build_builder(d,'X','custom',w,report['resources']['twist_rounds'])
    start=gen.engine.y_mpp_preparation(b)[0].num_ticks
    stages={}
    for name,c in b.segments.items():
        if name=='data_init':continue
        stages[name]=start;start+=c.num_ticks
    faults=json.loads(faultfile.read_text());mapped=[]
    for fault in faults:
        old_index=int(fault['stage'].split('_')[1]);old_k=old_index-d-1
        new_k=old_k-(d-w)
        if new_k<0:return {'status':'WITNESS_EARLIER_WIDE_ROUND_REMOVED'}
        phase='A' if new_k%2==0 else 'B'
        assert phase==fault['stage'].split('_')[-1]
        name=f'wide_{d+1+new_k}_{phase}'
        mapped.append(dict(fault,tick=stages[name]+fault['stage_local_tick'],new_stage=name))
    replay=stim.Circuit();tick=0;inserted=set()
    for op in stim.Circuit.from_file(str(source)).without_noise():
        replay.append(op)
        if op.name=='TICK':tick+=1
        for j,f in enumerate(mapped):
            gate={'R_err':'R','CX_depol':'CX','CZ_depol':'CZ'}[f['fault_label']]
            if tick!=f['tick'] or op.name!=gate:continue
            qs=[t.value for t in op.targets_copy()]
            if gate=='R':valid=f['qubit_ids'][0] in qs
            else:valid=tuple(f['qubit_ids']) in list(zip(qs[::2],qs[1::2]))
            assert valid and j not in inserted
            targets=[getattr(stim,'target_'+axis.lower())(q) for q,axis in zip(f['qubit_ids'],f['pauli_branch']) if axis!='I']
            replay.append('E',targets,1);inserted.add(j)
    assert len(inserted)==len(mapped)
    sig=set();dem=replay.detector_error_model(decompose_errors=False)
    for op in dem:
        if op.type=='error':
            assert op.args_copy()[0]==1
            sig.symmetric_difference_update(signature(op,replay.num_detectors))
    D=sorted(i for i in sig if i<replay.num_detectors);L=int(replay.num_detectors in sig)
    return {'status':'DETECTED' if D else ('COUNTEREXAMPLE' if L else 'NO_LOGICAL_EFFECT'),
            'physical_fault_count':len(mapped),'flipped_detectors':D,'logical_flip':L,
            'mapped_faults':mapped,'method':'Same physical fault type and local gate, align wide rounds from interface; deterministic E(1) replay against full original parities.'}

def graph_subset(folder,d,ch):
    path=folder/f'd{d}_{ch}_SI1000_body.stim'
    circuit=stim.Circuit.from_file(str(path));D=circuit.num_detectors
    dempath=path.with_suffix('.dem')
    dem=stim.DetectorErrorModel.from_file(str(dempath)) if dempath.exists() else circuit.detector_error_model(decompose_errors=False)
    cols=sorted({signature(op,D) for op in dem if op.type=='error' and op.args_copy()[0]>0})
    cols=[c for c in cols if c]
    cover=folder/f'{ch}_screen.cover';reduced=folder/f'{ch}_screen_reduced.cover'
    with cover.open('w') as out:
        out.write(f'{D+1} {len(cols)} 1\n{D}\n')
        for col in cols:out.write(str(len(col))+' '+' '.join(map(str,col))+'\n')
    subprocess.run([sys.executable,str(gen.HERE/'reduce_cover.py'),str(cover),str(reduced)],check=True,capture_output=True)
    with reduced.open() as f:
        m,n,k=map(int,f.readline().split());assert int(f.readline())==D
        red=[tuple(map(int,line.split()[1:])) for line in f]
    assert n==len(cols)==len(red) and m==D+1 and k==1
    # Independently replay reversible row operations and their target.
    rows=[0]*m
    for j,c in enumerate(cols):
        for b in c:rows[b]|=1<<j
    target=1<<D
    for line in Path(str(reduced)+'.rowops').read_text().splitlines():
        a,b=map(int,line.split());assert a!=b
        rows[a]^=rows[b]
        if (target>>b)&1:target^=1<<a
    assert target==1<<D
    rr=[0]*m
    for j,c in enumerate(red):
        for b in c:rr[b]|=1<<j
    assert rr==rows
    sparse=stim.DetectorErrorModel();index={}
    def targets(col):return [stim.target_logical_observable_id(0) if r==D else stim.target_relative_detector_id(r) for r in col]
    for j,col in enumerate(red):
        sparse.append('error',.001,targets(col));index[col]=j
    try:chain=sparse.shortest_graphlike_error(ignore_ungraphlike_errors=True)
    except ValueError as ex:return {'status':'NO_GRAPH_SUBSET_WITNESS','detail':str(ex),'lower_bound':None}
    selected=[index[signature(op,D)] for op in chain if op.type=='error']
    return verify_columns(circuit,cols,selected,d,folder,ch,
                          'Only graphlike subset AFTER reversible full-hypergraph row operations; no hyperedge decomposition.')

def verify_columns(circuit,cols,selected,d,folder,ch,method):
    D=circuit.num_detectors
    def targets(col):return [stim.target_logical_observable_id(0) if r==D else stim.target_relative_detector_id(r) for r in col]
    residual=set()
    original_filter=stim.DetectorErrorModel()
    for j in selected:
        residual.symmetric_difference_update(cols[j]);original_filter.append('error',1,targets(cols[j]))
    assert residual=={D}
    physical=circuit.explain_detector_error_model_errors(dem_filter=original_filter,reduce_to_one_representative_error=True)
    assert len(physical)==len(selected) and all(len(e.circuit_error_locations)==1 for e in physical)
    # Reinject the actual physical Pauli at its noise instruction. Every source
    # is an explicit SI1000 Pauli channel (no measurement-argument noise).
    locations={};details=[]
    for error in physical:
        loc=error.circuit_error_locations[0]
        assert len(loc.stack_frames)==1
        offset=loc.stack_frames[0].instruction_offset
        paulis=[p.gate_target for p in loc.flipped_pauli_product]
        assert paulis
        locations.setdefault(offset,[]).append(paulis)
        details.append({'tick':loc.tick_offset,'instruction_offset':offset,'description':str(loc)})
    replay=stim.Circuit()
    for i,op in enumerate(circuit):
        if op.name in gen.NOISE:
            for targets_ in locations.get(i,[]):replay.append('E',targets_,1)
        else:
            assert i not in locations
            replay.append(op)
    check=set()
    for op in replay.detector_error_model(decompose_errors=False):
        if op.type=='error':
            assert op.args_copy()[0]==1
            check.symmetric_difference_update(signature(op,D))
    assert check=={D}
    # No duplicate noise-site selection is accepted as a literal witness.
    seen=set()
    for error in physical:
        loc=error.circuit_error_locations[0]
        key=(loc.stack_frames[0].instruction_offset,str(loc.instruction_targets))
        assert key not in seen,'Selected multiple branches at one physical noise site'
        seen.add(key)
    result={'method':method,
            'physical_weight':len(selected),'status':'COUNTEREXAMPLE' if len(selected)<d else 'UPPER_BOUND_ONLY',
            'lower_bound':None,'original_parity_xor':'D=0,L=1','physical_replay':'PASS',
            'columns':selected,'physical_faults':details}
    (folder/f'{ch}_physical_upper_witness.txt').write_text('\n\n'.join(map(str,physical)))
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('folder',type=Path)
    ap.add_argument('--old-witness-only',action='store_true');args=ap.parse_args()
    report=json.loads((args.folder/'report.json').read_text());d=report['distance']
    result={'distance':d,'profile':report['profile'],'lower_bound_claimed':False,'archived_witness':replay_known(args.folder,report)}
    if not args.old_witness_only:
        result['channels']={}
        for ch in report['channels']:
            result['channels'][ch]=graph_subset(args.folder,d,ch)
            print(ch,result['channels'][ch].get('physical_weight'),result['channels'][ch]['status'],flush=True)
    (args.folder/'screen.json').write_text(json.dumps(result,indent=2))
    print('old witness:',result['archived_witness']['status'],flush=True)

if __name__=='__main__':main()
