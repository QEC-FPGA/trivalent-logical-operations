"""Reference-assisted instrument benchmark, separate ideal and physical regions."""
from pathlib import Path
import sys,json,hashlib,collections
import stim
MIDOUT_SRC = Path(__file__).resolve().parents[2] / 'third_party/midout/src'
if not (MIDOUT_SRC / 'midout').is_dir():
    raise ModuleNotFoundError(
        'The pinned midout dependency is missing. Run git submodule update --init --recursive '
        'from the repository root.'
    )
sys.path.insert(0, str(MIDOUT_SRC))
from midout.planar import _three_coupler_surface_codes as m

def sha(x):return hashlib.sha256(x if isinstance(x,bytes) else str(x).encode()).hexdigest()
def bits(indices):
    v=0
    for k in indices:v^=1<<k
    return v
def indices(v):
    out=[]
    while v:b=v&-v;out.append(b.bit_length()-1);v^=b
    return out
def add_basis(basis,v):
    while v:
        k=v.bit_length()-1
        if k not in basis:basis[k]=v;return True
        v^=basis[k]
    return False
def mpp(c,p):
    ts=[]
    for q in p.pauli_indices():
        if ts:ts.append(stim.target_combiner())
        ts.append({1:stim.target_x,2:stim.target_y,3:stim.target_z}[p[q]](q))
    if p.sign==-1:ts[0]=stim.target_inv(ts[0])
    c.append('MPP',ts)

def geometry(d,dm):
    raw=m.make_lattice_surgery_hex_planar_cx_mzz_code(basis='Z',distance=d,rounds=d,Dm=dm).circuit.flattened()
    from seam_repair import repair_initialization
    raw,_ = repair_initialization(raw,d)
    qmap={complex(v[0],v[1]):q for q,v in raw.get_final_qubit_coordinates().items()}
    n=raw.num_qubits;checks=[];logical=[];patches=[]
    for origin in [0,2j*(d+1)]:
        l=m.ThreeCouplerLayoutHelper(basis='Z',distance=d,width=d,height=d,origin=origin,gate='CX',rounds=d,wiggle=False,detector_basis='Z')
        patches.append({qmap[q] for q in l.patch.data_set})
        for tile in l.patch.tiles:
            pp=stim.PauliString(n+2)
            for q,b in zip(tile.ordered_data_qubits,tile.bases):
                if q is not None:pp[qmap[q]]=b
            checks.append(pp)
        axes={}
        for axis in 'XZ':
            pp=stim.PauliString(n+2)
            for q in l.patch.data_set:
                if (axis=='X' and q.real==origin.real) or (axis=='Z' and q.imag==origin.imag):pp[qmap[q]]=axis
            axes[axis]=pp
        logical.append(axes)
    return raw,n,checks,logical,patches

def local_output_checks(raw,d,n):
    """Local tile checks in the existing B-end frame, before destructive readout."""
    qmap={complex(v[0],v[1]):q for q,v in raw.get_final_qubit_coordinates().items()}
    checks=[]
    for origin in [0,2j*(d+1)]:
        layout=m.ThreeCouplerLayoutHelper(basis='Z',distance=d,width=d,height=d,origin=origin,
            gate='CX',rounds=d,wiggle=False,detector_basis=None)
        for tile in layout.patch.tiles:
            pp=stim.PauliString(n+2)
            for q in layout.final_readout_qubits_for_end_tile(tile,end_cycle='B'):pp[qmap[q]]=tile.basis
            checks.append(pp)
    return checks

def si1000(body,p):
    """Same explicit p/10, p, 2p, 5p convention as the archived H/S runner."""
    if p==0:return body.copy()
    if not 0<p<.1:raise ValueError('p must be 0 or between 0 and 0.1')
    active={t.value for op in body for t in op.targets_copy() if t.is_qubit_target and op.name!='QUBIT_COORDS'}
    out=stim.Circuit();moment=[]
    mf={'M':'X_ERROR','MX':'Z_ERROR','MY':'X_ERROR'};rf={'R':'X_ERROR','RX':'Z_ERROR','RY':'X_ERROR'}
    def flush():
        operated=set();collapsed=set()
        if not moment:return
        for op in moment:
            qs=[t.value for t in op.targets_copy() if t.is_qubit_target];name=op.name
            if name in mf:out.append(mf[name],qs,5*p)
            out.append(op);operated.update(qs)
            if name in mf or name in rf:collapsed.update(qs)
            if name in rf:out.append(rf[name],qs,2*p)
            elif name in ['H','X','Y','Z','S','S_DAG']:out.append('DEPOLARIZE1',qs,p/10)
            elif name in ['CX','CZ','XCX']:out.append('DEPOLARIZE2',qs,p)
            elif name not in mf:raise ValueError(f'Unsupported physical operation: {op}')
        idle=sorted(active-operated)
        if idle:out.append('DEPOLARIZE1',idle,p/10)
        waiting=sorted(active-collapsed)
        if collapsed and waiting:out.append('DEPOLARIZE1',waiting,2*p)
        moment.clear()
    for op in body:
        if op.name in ['TICK','QUBIT_COORDS']:
            flush();out.append(op)
        else:moment.append(op)
    flush();return out

def annotate(c,rows,obs):
    c=c.copy();total=c.num_measurements
    for v in rows:c.append('DETECTOR',[stim.target_rec(k-total) for k in indices(v)])
    for i,v in enumerate(obs):c.append('OBSERVABLE_INCLUDE',[stim.target_rec(k-total) for k in indices(v)],i)
    return c

def sparsify(rows,dem,passes=6):
    """Invertible detector-row XORs; never discard a detector or error."""
    supports=[set() for _ in rows];columns=[]
    for op in dem.flattened():
        if op.type!='error':continue
        col={t.val for t in op.targets_copy() if t.is_relative_detector_id()}
        for i in col:supports[i].add(len(columns))
        columns.append(col)
    before=sum(map(len,supports));operations=[];rows=list(rows)
    for _ in range(passes):
        changed=False;weights=list(map(len,supports))
        for i in sorted(range(len(rows)),key=lambda k:-weights[k]):
            overlap=collections.Counter(j for k in supports[i] for j in columns[k] if j!=i)
            choices=sorted(overlap,key=lambda j:(weights[j]-2*overlap[j],j))
            bestj=choices[0] if choices and weights[choices[0]]-2*overlap[choices[0]]<0 else None
            if bestj is not None:
                for k in supports[bestj]:
                    if k in supports[i]:supports[i].remove(k);columns[k].remove(i)
                    else:supports[i].add(k);columns[k].add(i)
                rows[i]^=rows[bestj];weights[i]=len(supports[i]);operations.append([i,bestj]);changed=True
        if not changed:break
    return rows,dict(row_xor_count=len(operations),operations=operations,before_detector_error_incidence=before,after_detector_error_incidence=sum(map(len,supports)))

def make(d,dm=None,p=.001,measurement='ZZ',detector_rows=None):
    if measurement not in ['ZZ','XX','XZ']:raise ValueError('Supported measurements: ZZ, XX, XZ')
    dm=d if dm is None else dm
    if d<3 or d%2==0 or dm<1:raise ValueError('Use odd d >= 3 and Dm >= 1')
    raw,n,checks,logs,patches=geometry(d,dm);memory=set.union(*patches)
    rotate=[k for k,a in enumerate(measurement) if a=='X']
    h_data=sorted({q for k in rotate for q in patches[k]})
    data_h=stim.Circuit()
    if h_data:data_h.append('H',h_data)
    both_h=stim.Circuit()
    if h_data:both_h.append('H',h_data+[n+k for k in rotate])
    # Two encoded Bell pairs with noiseless reference qubits n,n+1.
    state_checks=list(checks)
    for q in sorted(set(range(n))-memory):
        pp=stim.PauliString(n+2);pp[q]='Z';state_checks.append(pp)
    for k in range(2):
        for a in 'XZ':
            pp=logs[k][a].copy();pp[n+k]=a;state_checks.append(pp)
    if h_data:state_checks=[pp.after(both_h) for pp in state_checks]
    assert len(state_checks)==n+2
    prep=stim.Circuit();prep.append('R',range(n+2));prep+=stim.Tableau.from_stabilizers(state_checks).to_circuit('elimination');prep.append('TICK')
    last_m=max(k for k,op in enumerate(raw) if op.name=='M')
    assert raw[last_m-1].name=='MX', 'Expected the paired terminal MX/M readout'
    end_readout=last_m-1
    body=stim.Circuit();old_record=0;obs0=[];old_det=[];removed=False
    for k,op in enumerate(raw):
        if k==end_readout:break
        if op.name=='R' and set(t.value for t in op.targets_copy())==memory and not removed:removed=True;continue
        if op.name=='OBSERVABLE_INCLUDE' and op.gate_args_copy()==[0]:obs0.extend(old_record+t.value for t in op.targets_copy())
        if op.name=='DETECTOR':old_det.append([old_record+t.value for t in op.targets_copy()])
        if op.name not in ['DETECTOR','OBSERVABLE_INCLUDE','SHIFT_COORDS']:body.append(op)
        one=stim.Circuit();one.append(op);old_record+=one.num_measurements
    assert removed and obs0
    base_body_sha=sha(body)
    if h_data:
        wrapper=stim.Circuit();wrapper.append('H',h_data);wrapper.append('TICK');wrapper+=body
        body=wrapper
        # The chosen input block is H-conjugated. One physical H brings it to
        # the native ZZ surgery frame; output stays native, with logical axes
        # tracked explicitly instead of paying for a second physical H.
        for k in rotate:logs[k]={'X':logs[k]['Z'],'Z':logs[k]['X']}
    prefix=prep+body
    # End-code projections are diagnostic only, and must already be predictable.
    checks_out=local_output_checks(raw,d,n)
    assert len(checks_out)==len(memory)-2
    assert all(pp.commutes(logs[k][a]) for pp in checks_out for k in range(2) for a in 'XZ')
    recs=prefix.solve_flow_measurements([stim.Flow(output=check) for check in checks_out])
    assert all(r is not None for r in recs),'Output is not in the declared code space'
    end=stim.Circuit()
    for check in checks_out:mpp(end,check)
    logical_ops=[]
    zzref=stim.PauliString(n+2)
    for k,a in enumerate(measurement):zzref[n+k]=a
    logical_ops.append(zzref)
    for k in range(2):
        a=measurement[k];op=logs[k][a].copy();op[n+k]=a;logical_ops.append(op)
    complements=['X' if a=='Z' else 'Z' for a in measurement]
    xx=logs[0][complements[0]]*logs[1][complements[1]]
    for k,a in enumerate(complements):xx[n+k]=a
    logical_ops.append(xx)
    flows=prefix.solve_flow_measurements([stim.Flow(output=op) for op in logical_ops])
    assert all(r is not None for r in flows)
    assert prefix.has_flow(stim.Flow(output=zzref,measurements=obs0))
    flows[0]=obs0 # Preserve the actual joint-result definition.
    obs=[];signs=[];final_indices=[]
    for op,rec in zip(logical_ops,flows):
        sign=1 if prefix.has_flow(stim.Flow(output=op,measurements=rec)) else -1
        assert prefix.has_flow(stim.Flow(output=sign*op,measurements=rec))
        idx=prefix.num_measurements+end.num_measurements;final_indices.append(idx)
        obs.append(bits(rec+[idx]));signs.append(sign);mpp(end,sign*op)
    physical=prefix+end;ideal=physical.copy();basis={};rows=[]
    # Complete deterministic parity space, quotient only the four terminal observables.
    candidates=set()
    def collect(c):
        for flow in c.flow_generators():
            if any(flow.input_copy()) or any(flow.output_copy()):continue
            v=bits(flow.measurements_copy())
            for j,k in enumerate(final_indices):
                if (v>>k)&1:v^=obs[j]
            if v:candidates.add(v)
    if detector_rows is None:
        # Prefixes supply short time-local representatives of the same complete space.
        part=prep.copy()
        for op in body:
            part.append(op)
            if op.name in ['M','MX','MY']:collect(part)
        collect(physical)
        for rec in old_det:
            if prefix.has_flow(stim.Flow(measurements=rec)):candidates.add(bits(rec))
        causal_basis={}
        for v in sorted(candidates,key=lambda v:(v.bit_length(),v.bit_count(),v)):
            while v:
                k=(v&-v).bit_length()-1
                if k not in causal_basis:
                    causal_basis[k]=v;rows.append(v);break
                v^=causal_basis[k]
    else:
        # H changes Pauli bases, not measurement indices. Reuse only after the
        # independent-rank and deterministic-DEM checks below have passed.
        rows=list(detector_rows)
    for v in rows:assert add_basis(basis,v)
    rank=physical.count_determined_measurements()
    assert len(rows)+4==rank,(len(rows),rank)
    assert all(add_basis(basis,v) for v in obs)
    noisy=prep+si1000(body,p)+end
    physical_region=[len(prep),len(noisy)-len(end)]
    assert noisy.num_measurements==physical.num_measurements
    initial_dem=annotate(noisy,rows,obs).detector_error_model(decompose_errors=False,approximate_disjoint_errors=False)
    if detector_rows is None:rows,optimization=sparsify(rows,initial_dem)
    else:optimization=dict(reused_basis=True,row_xor_count=0,operations=[])
    ideal=annotate(ideal,rows,obs);noisy=annotate(noisy,rows,obs)
    ideal.detector_error_model()
    dem=noisy.detector_error_model(decompose_errors=False,approximate_disjoint_errors=False)
    edges=set()
    for op in body:
        if op.name in ['CX','CZ','XCX']:
            qs=[t.value for t in op.targets_copy()]
            edges.update(tuple(sorted(qs[k:k+2])) for k in range(0,len(qs),2))
    degree=max(sum(q in e for e in edges) for q in range(n));assert degree<=3
    meta=dict(d=d,Dm=dm,merge_extraction_rounds=dm,merge_end_phase="A",post_split_alignment_rounds=1,p=p,measurement=measurement,physical_qubits=n,ideal_reference_qubits=2,
        physical_body_sha256=sha(body),initial_state_sha256=sha(prep),ideal_sha256=sha(ideal),circuit_sha256=sha(noisy),dem_sha256=sha(dem),
        physical_body_instruction_range=physical_region,
        detectors=len(rows),determined_measurement_rank=rank,observables=4,observable_signs=signs,
        observable_definitions=[f'joint_{measurement}_vs_reference_{measurement}',f'output_{measurement[0]}1_vs_reference_{measurement[0]}1',f'output_{measurement[1]}2_vs_reference_{measurement[1]}2',f'output_{complements[0]}1{complements[1]}2_vs_same_reference_product'],
        input_code_frames=['H-conjugated' if k in rotate else 'native' for k in range(2)],
        output_code_frames=['native_B_end','native_B_end'],
        output_logical_axis_swap=[k in rotate for k in range(2)],
        transversal_H_layers=1 if rotate else 0,transversal_H_gate_count=len(h_data),base_ZZ_body_sha256=base_body_sha,
        physical_logical_axes=[{a:str(l[a]) for a in 'XZ'} for l in logs],
        max_degree=degree,body_ticks=body.num_ticks,body_two_qubit_layers=sum(op.name=='CX' for op in body),
        output_code_checks=[str(c) for c in checks_out],output_code_check_count=len(checks_out),
        initial_and_final_boundary='ideal encoded Bell pairs and ideal terminal MPP; only physical body has SI1000 noise',
        source_circuit_sha256=sha(raw),complete_detector_rank_verified=True)
    meta['detector_basis_optimization']=optimization
    return ideal,noisy,dem,meta

if __name__=='__main__':
    import time
    t=time.monotonic();ideal,c,dem,meta=make(3);print(json.dumps(meta,indent=2));print('seconds',time.monotonic()-t)
    Path('test_d3.stim').write_text(str(c));Path('test_d3.dem').write_text(str(dem))
