"""Isolated step-size comparison; generation is not a distance certificate."""
from pathlib import Path
import sys, json, argparse, time, hashlib
import stim

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'vendor'))
import compile_case as compiler
base = compiler.base


def build(d, wide, twist, step=3, move_rounds=2, stop_col=3):
    if d < 5 or d % 2 != 1:
        raise ValueError('d must be odd and >=5')
    if wide < 1 or wide % 2 != 1 or twist < 1 or twist % 2 != 1:
        raise ValueError('This comparison keeps W and T positive odd (final B -> wall A).')
    if step < 1 or move_rounds < 2 or move_rounds % 2:
        raise ValueError('step >=1; move_rounds must be positive even and >=2')
    end_width = d - stop_col
    if end_width < 1 or stop_col < 1:
        raise ValueError('Need 1 <= stop_col < d')
    b = base.Stage1(d)
    b.right_seam_width = end_width
    b.data_init('Z')
    for j in range(d):
        b.standard_round('B' if j % 2 == 0 else 'A', j+1)
    b.expansion()
    for j in range(wide):
        b.wide_round('A' if j % 2 == 0 else 'B', d+1+j)
    previous = 0
    while previous < end_width:
        width = min(previous+step, end_width)
        segment = stim.Circuit()
        segment.append('MY', [b.data(i,d+1) for i in range(d-width,d-previous)])
        segment.append('TICK')
        b.append_segment(f'grow_{previous}_to_{width}_MY', segment)
        mid = base.Stage1(d)
        mid.right_seam_width = width
        rounds = twist if width == end_width else move_rounds
        for j in range(rounds):
            phase = 'B' if j % 2 == 0 else 'A'
            if width == end_width:
                index = 2*d+5 if j == rounds-1 else 2000+j
                name = f'twist_final_{j+1}_{phase}'
            else:
                index = 1000+width*10+j
                name = f'grow_{width}_{phase}' if move_rounds == 2 else f'grow_{width}_{j+1}_{phase}'
            mid.twist_round(phase,index)
            b.append_segment(name,list(mid.segments.values())[-1])
        previous = width
    b.my_seam('left')
    for j in range(d):
        b.fullwall_round('A' if j % 2 == 0 else 'B',j+1)
    b.h_lower()
    b.shrink()
    return b


def generate(d=9, w=5, t=3, step=3, move_rounds=2, stop_col=3, channel='XY', out=None):
    start = time.monotonic()
    name=f'd{d}_step{step}_R{move_rounds}_W{w}_T{t}_{channel}'
    folder=Path(out) if out else ROOT/'cases'/name
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'metadata.json').exists():
        raise FileExistsError('Choose a new --out directory; completed cases are not overwritten.')
    b=build(d,w,t,step,move_rounds,stop_col)
    if step == 2 and move_rounds == 2 and stop_col == 3:
        assert b.circuit == compiler.moving.builder(d,w,t).circuit
    body=base.engine.body_after_init(b)
    prep=base.Stage1(d);prep.data_init('X' if channel=='XY' else 'Z')
    tick=prep.circuit.num_ticks
    stages=[];edges=set();extractions=0;two_qubit_layers=0
    for label,c in b.segments.items():
        if label=='data_init':continue
        stages.append(dict(stage=label,start_tick=tick,ticks=c.num_ticks))
        tick+=c.num_ticks
        if any(o.name in ('CX','CZ') for o in c):extractions+=1
        layer_targets=set()
        for op in c:
            if op.name=='TICK':
                two_qubit_layers += bool(layer_targets)
                layer_targets=set()
            if op.name in ('CX','CZ'):
                qs=[v.value for v in op.targets_copy()]
                assert len(qs)==len(set(qs)) and not(layer_targets & set(qs))
                layer_targets.update(qs)
                edges.update(tuple(sorted((x,y))) for x,y in zip(qs[::2],qs[1::2]))
        two_qubit_layers += bool(layer_targets)
    degrees={q:sum(q in e for e in edges) for q in {q for e in edges for q in e}}
    assert max(degrees.values())==3
    n=b.circuit.num_qubits;data=[b.data(i,j) for i in range(d) for j in range(1,d+1)]
    probe=stim.Circuit();probe.append('R',[q for q in range(n) if q not in data])
    for check in base.standard_stabilizers(d):
        pp=stim.PauliString(n)
        for q,v in zip(data,check):pp[q]=v
        compiler.mpp(probe,pp)
    probe+=body
    logic={axis:base.flow(probe,base.logicals(b,'input')[axis],base.logicals(b,'shrink')[axis]) for axis in 'XZ'}
    if not all(x['exists'] for x in logic.values()):
        (folder/'logic_failure.json').write_text(json.dumps(logic,indent=2),encoding='utf-8')
        raise ValueError('Arbitrary-input logical flow failed; not an accepted S circuit.')
    rec=sorted(set(logic['X']['measurements'])^set(logic['Z']['measurements']))
    sign=-logic['X']['sign']*logic['Z']['sign']
    assert probe.has_flow(stim.Flow(input=base.logicals(b,'input')['Y'],output=sign*base.logicals(b,'shrink')['Y'],measurements=rec))
    print('LOGIC PASS',name,round(time.monotonic()-start,1),flush=True)
    old=compiler.moving.builder
    compiler.moving.builder=lambda dd,ww,tt:build(dd,ww,t,step,move_rounds,stop_col)
    try:
        physical,c,meta=compiler.physical_circuit(d,w,channel)
    finally:
        compiler.moving.builder=old
    dem=c.detector_error_model(decompose_errors=False)
    meta.update(status='LOGIC_AND_CONNECTIVITY_VALIDATED_DISTANCE_NOT_CERTIFIED',d=d,W=w,T=t,
                step=step,move_rounds=move_rounds,stop_column=stop_col,stages=stages,
                max_degree=3,logical_input_unfixed=True,logical_XZ=logic,Y_sign=sign,
                extraction_rounds=extractions,two_qubit_layers=two_qubit_layers,
                growth_widths=[min(j,end) for end in [d-stop_col] for j in range(step,end+step,step)],
                total_ticks=c.num_ticks,seconds=time.monotonic()-start,
                decoder_note='Original complete detector basis; no matching decomposition acceptance claimed.')
    for name_,text in [('physical.stim',str(physical)),('full_ideal.stim',str(c.without_noise())),('full_original.stim',str(c))]:
        (folder/name_).write_text(text,encoding='utf-8')
    (folder/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print('GENERATED',folder.name,round(time.monotonic()-start,1),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--d',type=int,default=9);p.add_argument('--w',type=int,default=None)
    p.add_argument('--t',type=int,default=3);p.add_argument('--step',type=int,default=3)
    p.add_argument('--move-rounds',type=int,default=2);p.add_argument('--stop-col',type=int,default=3)
    p.add_argument('--channel',choices=['XY','ZZ'],default='XY');p.add_argument('--out',type=Path)
    a=p.parse_args();w=2*((a.d+6)//6)+1 if a.w is None else a.w
    generate(a.d,w,a.t,a.step,a.move_rounds,a.stop_col,a.channel,a.out)
