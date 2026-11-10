"""Search-only recovery. Frozen physical source and strict acceptance are unchanged."""
import hashlib,pathlib,time
PREFIX_SECONDS=1200
SPARSE_SECONDS=120
GENERAL_SECONDS=1800
def source_hashes():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(pathlib.Path(__file__).parent.glob('*.py'))}
def prefix_local_partial(circuit,seconds=PREFIX_SECONDS):
    import compile_matching as cm
    c=cm.compiler
    from reverse_faults import nmeas,NOISE
    import stim
    original,measurements=c.extract(circuit);original_basis=c.basis(original)
    if len(original_basis)!=circuit.num_detectors:raise ValueError('Dependent original detector rows')
    candidates=set(original);prefix=stim.Circuit();start=time.monotonic();cuts=0;exhausted=False
    for op in circuit:
        if op.name in NOISE or op.name in ('DETECTOR','OBSERVABLE_INCLUDE'):continue
        prefix.append(op)
        if not nmeas(op) or op.name=='MPP':continue
        if time.monotonic()-start>seconds:exhausted=True;break
        cuts+=1
        for flow in prefix.flow_generators():
            if any(flow.input_copy()) or any(flow.output_copy()):continue
            row=sum(1<<i for i in flow.measurements_copy())
            if row and c.solve(row,original_basis) is not None:candidates.add(row)
    # On timeout keep validated candidates plus the entire original basis.
    # Never substitute a truncated physical circuit or a truncated syndrome space.
    if not exhausted:
        for flow in prefix.flow_generators():
            if any(flow.input_copy()) or any(flow.output_copy()):continue
            row=sum(1<<i for i in flow.measurements_copy())
            if row and c.solve(row,original_basis) is not None:candidates.add(row)
    def score(row):
        ids=list(c.bits(row));ticks=[measurements[i]['tick'] for i in ids]
        xy=[measurements[i].get('coordinate') for i in ids];xy=[q for q in xy if q]
        spread=(max(q[0] for q in xy)-min(q[0] for q in xy)+max(q[1] for q in xy)-min(q[1] for q in xy)) if xy else 0
        return row.bit_count(),max(ticks)-min(ticks),spread,max(ticks),row
    selected=[];pivots={}
    for row in sorted(candidates,key=score):
        v=row
        while v:
            k=v.bit_length()-1
            if k not in pivots:pivots[k]=v;selected.append(row);break
            v^=pivots[k]
    if len(selected)!=len(original):raise ValueError('Incomplete detector span after bounded prefix search')
    if not all(c.solve(row,original_basis) is not None for row in selected):raise ValueError('Candidate outside original detector span')
    return c.annotate(circuit,selected),dict(cuts=cuts,candidates=len(candidates),budget_seconds=seconds,
        budget_exhausted=exhausted,elapsed_seconds=time.monotonic()-start,full_original_basis_retained=True)
def continue_search(candidate,original,dest,attempts,prefix_fn=prefix_local_partial,general_fn=None):
    """Stages have separate budgets; prefix timeout never consumes general search budget."""
    import compile_matching as cm
    if general_fn is None:
        from strict_repair.general_search import repair as general_fn
    dest.mkdir(parents=True,exist_ok=True)
    def save_candidate(cc,event):
        (dest/'recovery_candidate.stim').write_text(str(cc),encoding='utf-8')
        cm.write(dest/'recovery_progress.json',dict(event=event,attempts=attempts,recovery_sources=source_hashes()))
    save_candidate(candidate,'before_prefix')
    try:
        local,info=prefix_fn(original,seconds=PREFIX_SECONDS)
    except TimeoutError as exc:
        attempts.append(dict(basis='prefix_local_timeout_fallback',strict=False,error=str(exc),fallback='previous complete equivalent sparse basis'))
        save_candidate(candidate,'prefix_exception_fallback')
    else:
        print('PREFIX',info,flush=True)
        candidate,logs=cm.compiler.sparse_reduce(local,seconds=SPARSE_SECONDS)
        ok,err=cm.compiler.strict_dem(candidate)
        attempts.append(dict(basis='bounded_prefix_sparse',strict=ok,error=err,details=info,logs=logs))
        save_candidate(candidate,'after_prefix_sparse')
        if ok:return candidate
    def checkpoint(cc,operations,event):
        (dest/'repair_checkpoint.stim').write_text(str(cc),encoding='utf-8')
        cm.write(dest/'repair_checkpoint.json',dict(operations=operations,event=event,recovery_sources=source_hashes()))
    print('GENERAL_REPAIR independent_seconds=',GENERAL_SECONDS,flush=True)
    repaired,info=general_fn(candidate,extract=cm.compiler.extract,annotate=cm.compiler.annotate,
        strict_dem=cm.compiler.strict_dem,seconds=GENERAL_SECONDS,callback=checkpoint)
    attempts.append(dict(basis='general_pair_repair_recovery',strict=repaired is not None,details=info,budget_seconds=GENERAL_SECONDS))
    cm.write(dest/'recovery_progress.json',dict(event='general_finished',attempts=attempts,recovery_sources=source_hashes()))
    return repaired
def compile_case(folder,try_sparse=False,repair=False):
    import compile_matching as cm
    folder=pathlib.Path(folder);dest=cm.ROOT/'matching'/folder.name
    if (dest/'compile.json').exists():
        old=cm.read(dest/'compile.json')
        if old['status']=='ACCEPTED':
            if old['original_circuit_sha256']!=cm.sha(folder/'full_original.stim'):raise ValueError('Original changed')
            return old
    circuit=cm.circuit(folder/'full_original.stim');meta=cm.read(folder/'metadata.json')
    if meta['d']!=15 or meta['channel']!='XY' or meta['boundary']!='full':raise ValueError('Recovery scope is d15/full/XY only')
    obs=None;old=[];n=circuit.num_measurements
    for op in circuit:
        if op.name=='OBSERVABLE_INCLUDE':obs=[n+t.value for t in op.targets_copy()]
        elif op.name=='DETECTOR':old.append(sum(1<<(n+t.value) for t in op.targets_copy()))
    attempts=[];accepted=None;physical=cm.strip(circuit)
    dets,rank=cm.causal_detectors.complete(physical.without_noise(),obs,[n-1]);causal=cm.base.annotate(physical,dets,obs)
    old_basis=cm.memory.p.basis(old)
    assert len(old_basis)==len(old)==len(dets)==rank-1
    assert all(not cm.memory.p.residual(sum(1<<j for j in row),old_basis) for row in dets)
    assert cm.strip(causal)==physical
    for label,candidate in [('original',circuit),('causal',causal)]:
        ok,err=cm.compiler.strict_dem(candidate);attempts.append(dict(basis=label,strict=ok,error=err))
        if ok:accepted=candidate;break
    if accepted is None:
        candidate,logs=cm.compiler.sparse_reduce(causal,seconds=300)
        ok,err=cm.compiler.strict_dem(candidate);attempts.append(dict(basis='causal_sparse',strict=ok,error=err,logs=logs))
        if ok:accepted=candidate
    if accepted is None:accepted=continue_search(candidate,circuit,dest,attempts)
    report=dict(status='UNSUPPORTED' if accepted is None else 'ACCEPTED',configuration=meta,
        original_circuit_sha256=cm.sha(folder/'full_original.stim'),full_detector_span_preserved=True,
        physical_and_observable_unchanged=True,block_remnant_edges=True,basis_attempts=attempts,
        formal_sampling_started=False,repair_attempted=True,recovery_sources=source_hashes(),
        recovery_budgets=dict(prefix=PREFIX_SECONDS,sparse=SPARSE_SECONDS,general=GENERAL_SECONDS))
    dest.mkdir(parents=True,exist_ok=True)
    if accepted is not None:
        assert cm.strip(accepted)==physical
        assert [str(o) for o in accepted if o.name=='OBSERVABLE_INCLUDE']==[str(o) for o in circuit if o.name=='OBSERVABLE_INCLUDE']
        rows=[]
        for op in accepted:
            if op.name=='DETECTOR':rows.append(sum(1<<(n+t.value) for t in op.targets_copy()))
        assert len(cm.memory.p.basis(rows))==len(old_basis) and all(not cm.memory.p.residual(v,old_basis) for v in rows)
        cp=dest/'full_matching.stim';cp.write_text(str(accepted),encoding='utf-8');hashes={}
        for p in ('0.0005','0.001','0.002'):
            dem=cm.scaled(accepted,p).detector_error_model(decompose_errors=True,ignore_decomposition_failures=False,block_decomposition_from_introducing_remnant_edges=True)
            path=dest/f'p{p}.dem';path.write_text(str(dem),encoding='utf-8');hashes[p]=cm.sha(path)
        report.update(full_matching_sha256=cm.sha(cp),dem_sha256=hashes)
    cm.write(dest/'compile.json',report)
    if accepted is not None:report=cm.freeze_sampling_circuits(dest,report)
    print(folder.name,report['status'],flush=True);return report
