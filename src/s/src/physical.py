"""Physical-site lookup and exact independent fault injection."""
from study import *
import stim, importlib.util
spec=importlib.util.spec_from_file_location('study_fault_checks',Path(__file__).resolve().parent/'vendor/lib/fault_checks.py')
fc=importlib.util.module_from_spec(spec);spec.loader.exec_module(fc)
NOISE=fc.NOISE

def circuit(path):return stim.Circuit(Path(path).read_text(encoding='utf-8'))
def sites(c):return list(fc.noise_sites(c))
def find(c, tick, paulis, noise=None, targets=None, occurrence=0):
    opts=[]
    for i,t,op,qs in sites(c):
        if t!=tick or (noise and op.name!=noise) or (targets is not None and list(targets)!=qs):continue
        if not set(paulis)<=set(qs):continue
        if op.name.endswith('_ERROR') and any(p!=op.name[0] for p in paulis.values()):continue
        opts.append(dict(instruction_offset=i,tick=t,noise_kind=op.name,noise_targets=qs,
                         noise_args=op.gate_args_copy(),paulis={str(q):p for q,p in paulis.items()}))
    if not opts:raise ValueError(f'No physical site {tick} {paulis} {noise} {targets}')
    return opts[occurrence]
def normalized(e):
    v=dict(e)
    if isinstance(v['paulis'],list):v['paulis']={str(p['qubit']):p['axis'] for p in v['paulis']}
    return v
def validate_events(c,events):
    seen=set()
    for raw in events:
        e=normalized(raw);op=c[e['instruction_offset']]
        assert op.name in NOISE and op.name==e['noise_kind']
        qs=[t.value for t in op.targets_copy()];size=2 if op.name=='DEPOLARIZE2' else 1
        pair=tuple(e['noise_targets'])
        assert pair in {tuple(qs[i:i+size]) for i in range(0,len(qs),size)}
        assert e['paulis'] and {int(q) for q in e['paulis']}<=set(pair)
        assert all(p in 'XYZ' for p in e['paulis'].values())
        if op.name.endswith('_ERROR'):assert all(p==op.name[0] for p in e['paulis'].values())
        assert op.gate_args_copy()[0]>0
        site=(e['instruction_offset'],pair);assert site not in seen;seen.add(site)
    return True
def replay(c,events,dest=None):
    events=[normalized(e) for e in events]
    validate_events(c,events)
    cols=fc.propagate(c,events);v=0
    for col in cols:v^=col
    exact,inj=fc.injected(c,events);assert v==exact
    r=dict(physical_weight=len(events),detectors_fired=[j for j in range(c.num_detectors) if v>>j&1],
           logical_flip=bool(v>>c.num_detectors&1),events=events,independent_exact_Stim_replay=True)
    r['status']='UNDETECTED_LOGICAL' if exact==1<<c.num_detectors else 'DETECTED_OR_NOT_LOGICAL'
    if dest:
        dest=Path(dest);write(dest/'witness_events.json',r)
        (dest/'witness_injected.stim').write_text(str(inj),encoding='utf-8')
    return r
