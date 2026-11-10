"""Strict decoder gate: complete equivalent detector space; no remnant edges."""
from physical import *
from numerical_io import scaled,probability_key
import causal_detectors,time,argparse

def freeze_sampling_circuits(dest,report):
    if report['status']!='ACCEPTED':return report
    if report.get('stim_version',stim.__version__)!=stim.__version__:
        raise ValueError('Use the Stim version which generated the frozen DEMs')
    base_path=dest/'full_matching.stim';assert sha(base_path)==report['full_matching_sha256']
    c=circuit(base_path);hashes={}
    for p in ('0.0005','0.001','0.002'):
        path=dest/f'p{p}.stim';path.write_text(str(scaled(c,p)),encoding='utf-8');hashes[p]=sha(path)
    report.update(sampling_circuit_sha256=hashes,stim_version=stim.__version__,pymatching_version='2.3.1',
                  actual_probability_circuits_frozen=True)
    write(dest/'compile.json',report);return report

def strip(c):
    out=stim.Circuit()
    for op in c:
        if op.name not in ('DETECTOR','OBSERVABLE_INCLUDE'):out.append(op)
    return out

def compile_case(folder,try_sparse=False,repair=False):
    metadata=read(Path(folder)/'metadata.json')
    if metadata['d']==15 and metadata['channel']=='XY' and repair:
        from recovery_compile import compile_case as recovered_compile
        return recovered_compile(folder,try_sparse=try_sparse,repair=repair)
    folder=Path(folder)
    dest=ROOT/'matching'/folder.name
    if (dest/'compile.json').exists():
        old=read(dest/'compile.json')
        if old['status']=='ACCEPTED':
            assert old['original_circuit_sha256']==sha(folder/'full_original.stim'), 'Source circuit changed'
            return freeze_sampling_circuits(dest,old)
        if repair and old.get('repair_attempted'):return old
        if not repair and (not try_sparse or any(a['basis']=='causal_sparse' for a in old['basis_attempts'])):return old
    c=circuit(folder/'full_original.stim');m=read(folder/'metadata.json');obs=None;old=[];n=c.num_measurements
    for op in c:
        if op.name=='OBSERVABLE_INCLUDE':obs=[n+t.value for t in op.targets_copy()]
        elif op.name=='DETECTOR':old.append(sum(1<<(n+t.value) for t in op.targets_copy()))
    attempts=[];accepted=None
    physical=strip(c)
    dets,rank=causal_detectors.complete(physical.without_noise(),obs,[n-1]);causal=base.annotate(physical,dets,obs)
    old_basis=memory.p.basis(old)
    assert len(old_basis)==len(old)==len(dets)==rank-1
    for row in dets:assert not memory.p.residual(sum(1<<j for j in row),old_basis)
    assert strip(causal)==physical
    for label,candidate in [('original',c),('causal',causal)]:
        ok,err=compiler.strict_dem(candidate);attempts.append(dict(basis=label,strict=ok,error=err))
        if ok:accepted=candidate;break
    if accepted is None and (try_sparse or repair):
        candidate,logs=compiler.sparse_reduce(causal,seconds=300)
        ok,err=compiler.strict_dem(candidate);attempts.append(dict(basis='causal_sparse',strict=ok,error=err,logs=logs))
        if ok:accepted=candidate
    if accepted is None and repair:
        started=time.monotonic();dest.mkdir(parents=True,exist_ok=True)
        local,info=compiler.prefix_local(c,seconds=180)
        candidate,logs=compiler.sparse_reduce(local,seconds=120)
        ok,err=compiler.strict_dem(candidate)
        attempts.append(dict(basis='prefix_local_sparse',strict=ok,error=err,details=info,logs=logs))
        if ok:accepted=candidate
        else:
            from strict_repair.general_search import repair as repair_basis
            def checkpoint(cc,operations,event):
                (dest/'repair_checkpoint.stim').write_text(str(cc),encoding='utf-8')
                write(dest/'repair_checkpoint.json',dict(operations=operations,event=event))
            candidate,info=repair_basis(candidate,extract=compiler.extract,annotate=compiler.annotate,
                strict_dem=compiler.strict_dem,seconds=max(1,900-(time.monotonic()-started)),callback=checkpoint)
            attempts.append(dict(basis='general_pair_repair',strict=candidate is not None,details=info))
            if candidate is not None:accepted=candidate
    report=dict(status='UNSUPPORTED' if accepted is None else 'ACCEPTED',configuration=m,
                original_circuit_sha256=sha(folder/'full_original.stim'),
                full_detector_span_preserved=True,physical_and_observable_unchanged=True,
                block_remnant_edges=True,basis_attempts=attempts,formal_sampling_started=False,repair_attempted=repair)
    if accepted is not None:
        assert strip(accepted)==physical
        newobs=[str(o) for o in accepted if o.name=='OBSERVABLE_INCLUDE']
        assert newobs==[str(o) for o in c if o.name=='OBSERVABLE_INCLUDE']
        newrows=[]
        for op in accepted:
            if op.name=='DETECTOR':newrows.append(sum(1<<(n+t.value) for t in op.targets_copy()))
        assert len(memory.p.basis(newrows))==len(old_basis) and all(not memory.p.residual(v,old_basis) for v in newrows)
        dest.mkdir(parents=True,exist_ok=True);cp=dest/'full_matching.stim';cp.write_text(str(accepted),encoding='utf-8')
        hashes={}
        for p in ('0.0005','0.001','0.002'):
            dem=scaled(accepted,p).detector_error_model(decompose_errors=True,ignore_decomposition_failures=False,block_decomposition_from_introducing_remnant_edges=True)
            path=dest/f'p{p}.dem';path.write_text(str(dem),encoding='utf-8');hashes[p]=sha(path)
        report.update(full_matching_sha256=sha(cp),dem_sha256=hashes)
    write(dest/'compile.json',report)
    if accepted is not None:report=freeze_sampling_circuits(dest,report)
    print(folder.name,report['status'],flush=True);return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);ap.add_argument('--try-sparse',action='store_true');ap.add_argument('--repair',action='store_true');a=ap.parse_args()
    compile_case(a.folder,a.try_sparse,a.repair)
