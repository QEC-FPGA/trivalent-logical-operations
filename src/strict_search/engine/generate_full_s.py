"""Explicit moves and uninterrupted extraction phase; frozen two-step regression."""
from study import *
import argparse, stim, importlib.util
spec=importlib.util.spec_from_file_location('comparison_frozen',Path(__file__).resolve().parent/'generate_step.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)

def route(total,step):
    return [min(step,total-j) for j in range(0,total,step)]

def build(d,w,t,moves,between):
    if d<5 or d%2!=1 or w<1 or w%2!=1 or t<1:raise ValueError('Need odd d>=5, positive odd W and positive T')
    moves=list(moves);between=list(between)
    if not moves or any(x<=0 for x in moves) or sum(moves)!=d-3:raise ValueError('Positive displacements must sum to d-3')
    if len(between)==1:between*=len(moves)-1
    if len(between)!=len(moves)-1 or any(x<0 for x in between):raise ValueError('One nonnegative count per intermediate position')
    if (sum(between)+t)%2!=1:
        raise ValueError('The current terminal interface requires last twist phase B. Choose an odd total intermediate+T count; no hidden alignment round is added.')
    b=base.Stage1(d);b.right_seam_width=d-3;b.data_init('Z')
    for j in range(d):b.standard_round('B' if j%2==0 else 'A',j+1)
    b.expansion()
    for j in range(w):b.wide_round('A' if j%2==0 else 'B',d+1+j)
    width=0;phase='B';details=[]
    for j,step in enumerate(moves):
        dest=width+step
        b.append_segment(f'grow_{width}_to_{dest}_MY',memory.p.old.my_growth(d,width,dest));width=dest
        count=t if j==len(moves)-1 else between[j]
        for z in range(count):
            c=memory.extraction(d,width,phase)
            # Keep the legacy joint tick at the final twist -> left MY boundary.
            if j==len(moves)-1 and z==count-1:
                assert c[-1].name=='TICK';c=c[:-1]
            name=f'twist_final_{z+1}_{phase}' if j==len(moves)-1 else f'grow_{width}_{z+1}_{phase}'
            b.append_segment(name,c);details.append(dict(k=width,phase=phase,terminal=j==len(moves)-1))
            phase='A' if phase=='B' else 'B'
    assert phase=='A'
    b.my_seam('left')
    for j in range(d):b.fullwall_round('A' if j%2==0 else 'B',j+1)
    b.h_lower();b.shrink()
    b.explicit_details=details;b.explicit_between=between
    return b

def generate(d,w,t,moves,between,channel,out=None):
    name=f'd{d}_m'+ '-'.join(map(str,moves))+'_r'+'-'.join(map(str,between))+f'_W{w}_T{t}_{channel}'
    folder=Path(out) if out else ROOT/'full_s'/name
    if (folder/'metadata.json').exists():
        m=read(folder/'metadata.json')
        expanded=list(between)*(len(moves)-1) if len(between)==1 else list(between)
        assert m['moves']==list(moves) and m['between_rounds']==expanded
        assert m['d']==d and m['W']==w and m['T']==t and m['channel']==channel
        assert sha(folder/'full_original.stim')==m['circuit_sha256']
        return folder
    b=build(d,w,t,moves,between)
    baseline=build(d,w,t,route(d-3,2),[2])
    if t==3:assert baseline.circuit==compiler.moving.builder(d,w,t).circuit,'Original two-step physical schedule changed'
    original=legacy.build;legacy.build=lambda *args,**kwargs:b
    try:legacy.generate(d,w,t,1,0,3,channel,folder)
    finally:legacy.build=original
    m=read(folder/'metadata.json');m.update(moves=list(moves),between_rounds=b.explicit_between,
        step=None,move_rounds=None,growth_widths=[sum(moves[:i+1]) for i in range(len(moves))],
        phase_sequence=b.explicit_details,zero_rounds_do_not_advance_phase=True,
        terminal_phase='B',final_twist_rounds=t,baseline_physical_regression=(t==3),
        corrected_odd_memory=True,formal_sampling_started=False,
        circuit_sha256=sha(folder/'full_original.stim'))
    write(folder/'metadata.json',m);return folder

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--d',type=int,default=9);ap.add_argument('--w',type=int)
    ap.add_argument('--t',type=int,default=3)
    group=ap.add_mutually_exclusive_group();group.add_argument('--moves',type=int,nargs='+');group.add_argument('--step',type=int,default=2)
    ap.add_argument('--between-rounds',type=int,nargs='+',default=[2]);ap.add_argument('--channel',choices=['XY','ZZ','both'],default='both')
    ap.add_argument('--out',type=Path);a=ap.parse_args()
    for ch in (['XY','ZZ'] if a.channel=='both' else [a.channel]):
        generate(a.d,2*((a.d+6)//6)+1 if a.w is None else a.w,a.t,
                 a.moves if a.moves else route(a.d-3,a.step),a.between_rounds,ch,a.out/ch if a.out and a.channel=='both' else a.out)
