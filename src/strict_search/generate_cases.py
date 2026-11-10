"""All routes retain final B,A,B; check actual output code and phase continuity."""
from functools import lru_cache
import terminal_move as tm
from support import *
from terminal_move import base,compiler,memory,stim,frozen

@lru_cache(maxsize=3)
def reference(d):
    b=frozen.build(d,2*((d+6)//6)+1,3,route(d-3,2),[2])
    checks,_=compiler.code_stabilizers(b,base.engine.body_after_init(b))
    return memory.p.basis(memory.p.mask(p) for p in checks)

@lru_cache(maxsize=32)
def audit(d,w,moves,holds,is_original=False):
    assert holds[-1]==3 and all(v==2 for v in holds[:-1])
    b=tm.build(d,w,moves,holds,'A')
    assert b.wall_first=='A' and b.wall_count==d
    assert [r['phase'] for r in b.terminal_details[-3:]]==['B','A','B']
    checks,_=compiler.code_stabilizers(b,base.engine.body_after_init(b));ref=reference(d)
    same=len(checks)==len(ref) and all(not memory.p.residual(memory.p.mask(p),ref) for p in checks)
    logic=tm.flow_check(b)
    if not same or not logic['passed']:raise ValueError('Output code or arbitrary-input logical map failed')
    regression=None
    if is_original:
        regression=b.circuit==compiler.moving.builder(d,w,3).circuit
        if not regression:raise ValueError('Original two-step circuit changed')
    return dict(output_stabilizer_rank=len(checks),same_output_code_as_original=same,
        logical_check=logic,original_physical_regression=regression,terminal_phases=['B','A','B'])

def generate(spec):
    dst=case_dir(spec);mp=dst/'metadata.json';ap=dst/'acceptance.json'
    if ap.exists():
        a=read(ap)
        if a['config_sha256']!=spec['config_sha256'] or sha(dst/'full_original.stim')!=a['circuit_sha256']:
            raise ValueError('Circuit/configuration hash mismatch; do not mix releases')
        return dst
    d,w=spec['d'],spec['W'];moves,holds=spec['moves'],spec['holds']
    a=dict(audit(d,w,tuple(moves),tuple(holds),spec['is_original']))
    if not mp.exists():
        b=tm.build(d,w,moves,holds,'A')
        old=frozen.legacy.build;frozen.legacy.build=lambda *args,**kw:b
        try:frozen.legacy.generate(d,w,3,1,2,moves[-1],spec['channel'],dst)
        finally:frozen.legacy.build=old
        m=read(mp);m.update(configuration=spec,all_moves=moves,intermediate_holds=holds,
            step=None,move_rounds=None,growth_widths=[sum(moves[:j+1]) for j in range(len(moves)-1)],
            fullwall_first_phase='A',fullwall_rounds=d,phase_sequence=b.terminal_details,
            total_extraction_rounds=spec['total_extraction_rounds'],wide_to_fullwall_extractions=spec['core_extraction_rounds'],
            circuit_sha256=sha(dst/'full_original.stim'))
        write(mp,m)
    else:
        m=read(mp)
        if m['configuration']!=spec:raise ValueError('Partial generation has different parameters')
    c=stim.Circuit((dst/'full_original.stim').read_text(encoding='utf-8'))
    c.detector_error_model(decompose_errors=False)
    assert not c.without_noise().compile_detector_sampler().sample(32,append_observables=True).any()
    a.update(config_sha256=spec['config_sha256'],circuit_sha256=sha(dst/'full_original.stim'),
        max_connectivity_union_degree=m['max_degree'],noiseless_samples=32,noiseless_errors=0,all_checks_passed=True)
    assert a['max_connectivity_union_degree']==3
    write(ap,a);print('ACCEPT',spec['id'],'moves',moves,'holds',holds,flush=True)
    return dst
