"""Explicitly invoked future sampling; this study does not launch sampling."""
from study import ROOT,read,write,sha
from pathlib import Path
from decimal import Decimal
import argparse,time,stim,numpy as np
ALLOWED={Decimal('0.0005'),Decimal('0.001'),Decimal('0.002')}
NOISE={'X_ERROR','Y_ERROR','Z_ERROR','DEPOLARIZE1','DEPOLARIZE2'}

def probability_key(p):
    v=Decimal(str(p))
    if v not in ALLOWED:raise ValueError('Allowed p: 0.0005, 0.001, 0.002')
    return format(v.normalize(),'f')

def scaled(c,p):
    key=probability_key(p);factor=float(Decimal(key)/Decimal('0.001'));out=stim.Circuit()
    for op in c:
        if isinstance(op,stim.CircuitRepeatBlock):raise ValueError('Supply the frozen flattened circuit')
        if op.name in NOISE:
            args=[a*factor for a in op.gate_args_copy()]
            if any(a<0 or a>1 for a in args):raise ValueError('Invalid scaled noise probability')
            out.append(op.name,op.targets_copy(),args)
        else:
            if op.name in ('M','MX','MY','MR','MRX','MRY','MPP') and any(op.gate_args_copy()):
                raise ValueError('Embedded measurement noise requires an explicitly reviewed scaling rule')
            out.append(op)
    assert out.without_noise()==c.without_noise()
    return out

def sample(folder,p,shots,seed,out,stage,provenance=None):
    import pymatching
    if pymatching.__version__!='2.3.1':raise ValueError('Pinned correlated PyMatching 2.3.1 required')
    if shots<1 or seed<0 or stage not in ('pilot','confirmation'):raise ValueError('Explicit positive fixed budget and stage required')
    folder=Path(folder);out=Path(out);key=probability_key(p);m=read(folder/'compile.json')
    if m['status']!='ACCEPTED' or not m['block_remnant_edges']:raise ValueError('Strict full model not accepted')
    if stim.__version__!=m['stim_version']:raise ValueError('Use the frozen Stim version')
    base_cp=folder/'full_matching.stim';cp=folder/f'p{key}.stim';dp=folder/f'p{key}.dem'
    assert sha(base_cp)==m['full_matching_sha256'] and sha(cp)==m['sampling_circuit_sha256'][key] and sha(dp)==m['dem_sha256'][key]
    identity=dict(circuit_sha256=sha(cp),base_circuit_sha256=sha(base_cp),dem_sha256=sha(dp),p=key,shots=shots,seed=seed,stage=stage)
    if provenance:
        if set(provenance)&set(identity):raise ValueError('Provenance must not replace sampling parameters')
        identity.update(provenance)
    if out.exists():
        old=read(out)
        if old.get('status')!='COMPLETE' or any(old.get(k)!=v for k,v in identity.items()):raise ValueError('Existing shard mismatch')
        return old
    c=stim.Circuit(cp.read_text(encoding='utf-8'))
    dem=stim.DetectorErrorModel(dp.read_text(encoding='utf-8'))
    dec=pymatching.Matching.from_detector_error_model(dem,enable_correlations=True)
    sampler=c.compile_detector_sampler(seed=seed);done=errors=0;t=time.monotonic()
    batch_size=max(1,min(4096,16000000//max(1,c.num_detectors)))
    while done<shots:
        ds,truth=sampler.sample(min(batch_size,shots-done),separate_observables=True)
        pred=dec.decode_batch(ds,enable_correlations=True);assert truth.shape==pred.shape
        errors+=int(np.any(pred!=truth,axis=1).sum());done+=len(ds)
    result=dict(identity,status='COMPLETE',mistakes=errors,decoder='pymatching 2.3.1 correlated',
                stim_version=stim.__version__,pymatching_version=pymatching.__version__,
                numpy_version=np.__version__,batch_size=batch_size,seconds=time.monotonic()-t,
                definition='Whole-circuit failures / shots; not divided by rounds',configuration=m['configuration'])
    write(out,result);return result

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--compiled',type=Path,required=True);a.add_argument('--p',required=True)
    a.add_argument('--shots',type=int,required=True);a.add_argument('--seed',type=int,required=True)
    a.add_argument('--stage',choices=['pilot','confirmation'],required=True);a.add_argument('--out',type=Path,required=True)
    v=a.parse_args();print(sample(v.compiled,v.p,v.shots,v.seed,v.out,v.stage))
