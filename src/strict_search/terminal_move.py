"""Treat the final left-seam MY as an explicit terminal displacement.

Independent experimental builder. Frozen mainline is only read.
"""
from pathlib import Path
import sys,os,argparse,time
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
PACKAGE=HERE
os.environ['MOVE_W_OUTPUT']=str(HERE/'output')
sys.path.insert(0,str(PACKAGE/'engine'))
from study import base,compiler,memory,read,write,sha
import generate_full_s as frozen
import stim

def build(d,w,moves,holds,wall_phase='A',wall_rounds=None):
    if d<5 or d%2!=1 or w<1 or w%2!=1:raise ValueError('Odd d>=5 and positive odd W')
    if sum(moves)!=d or any(v<=0 for v in moves):raise ValueError('All displacements including final MY must sum to d')
    if len(holds)!=len(moves)-1 or any(v<1 for v in holds):raise ValueError('Positive extraction count at each internal stop')
    b=base.Stage1(d);b.right_seam_width=sum(moves[:-1]);b.data_init('Z')
    for j in range(d):b.standard_round('B' if j%2==0 else 'A',j+1)
    b.expansion()
    for j in range(w):b.wide_round('A' if j%2==0 else 'B',d+1+j)
    width=0;phase='B';details=[]
    for j,(step,count) in enumerate(zip(moves[:-1],holds)):
        dest=width+step
        b.append_segment(f'grow_{width}_to_{dest}_MY',memory.p.old.my_growth(d,width,dest));width=dest
        for z in range(count):
            c=memory.extraction(d,width,phase)
            terminal=j==len(holds)-1
            if terminal and z==count-1:
                assert c[-1].name=='TICK';c=c[:-1]
            name=f'twist_final_{z+1}_{phase}' if terminal else f'grow_{width}_{z+1}_{phase}'
            b.append_segment(name,c)
            details.append(dict(k=width,phase=phase,terminal=terminal))
            phase='A' if phase=='B' else 'B'
    b.my_seam('left')
    first=phase if wall_phase.startswith('continue') else wall_phase
    count=d if wall_rounds is None else wall_rounds
    if wall_phase=='continue-close-A' and wall_rounds is None and first=='B':
        count+=1
    for j in range(count):b.fullwall_round(first if j%2==0 else ('B' if first=='A' else 'A'),j+1)
    b.h_lower();b.shrink()
    b.terminal_details=details;b.wall_first=first;b.wall_count=count
    return b

def flow_check(b):
    d=b.d;n=b.circuit.num_qubits;data=[b.data(i,j) for i in range(d) for j in range(1,d+1)]
    probe=stim.Circuit();probe.append('R',[q for q in range(n) if q not in data])
    for check in base.standard_stabilizers(d):
        p=stim.PauliString(n)
        for q,v in zip(data,check):p[q]=v
        compiler.mpp(probe,p)
    probe+=base.engine.body_after_init(b)
    f={a:base.flow(probe,base.logicals(b,'input')[a],base.logicals(b,'shrink')[a]) for a in 'XZ'}
    if not all(x['exists'] for x in f.values()):return dict(passed=False,flows=f)
    sign=-f['X']['sign']*f['Z']['sign'];rec=sorted(set(f['X']['measurements'])^set(f['Z']['measurements']))
    yp=probe.has_flow(stim.Flow(input=base.logicals(b,'input')['Y'],output=sign*base.logicals(b,'shrink')['Y'],measurements=rec))
    return dict(passed=yp,flows=f,Y_sign=sign)

def name(d,w,moves,holds,wall_phase,channel):
    return f'd{d}_W{w}_m'+ '-'.join(map(str,moves))+'_h'+'-'.join(map(str,holds))+f'_wall{wall_phase}_{channel}'

def generate(d,w,moves,holds,wall_phase,channel):
    folder=HERE/'cases'/name(d,w,moves,holds,wall_phase,channel)
    if (folder/'metadata.json').exists():return folder
    b=build(d,w,moves,holds,wall_phase)
    old=frozen.legacy.build;frozen.legacy.build=lambda *args,**kwargs:b
    try:frozen.legacy.generate(d,w,holds[-1] if holds else 0,1,2,moves[-1],channel,folder)
    finally:frozen.legacy.build=old
    m=read(folder/'metadata.json')
    m.update(all_moves=list(moves),intermediate_holds=list(holds),final_M_Y_length=moves[-1],
        step=None,move_rounds=None,growth_widths=[sum(moves[:j+1]) for j in range(len(moves)-1)],
        fullwall_first_phase=b.wall_first,phase_sequence=b.terminal_details,
        fullwall_rounds=b.wall_count,total_extraction_rounds=d+w+sum(holds)+b.wall_count,
        circuit_sha256=sha(folder/'full_original.stim'),
        wide_to_fullwall_extractions=w+sum(holds),
        sampling_started=False,distance_certified=False)
    write(folder/'metadata.json',m)
    return folder

def probe():
    old=frozen.build(9,5,3,[2,2,2],[2])
    assert build(9,5,[2,2,2,3],[2,2,3],'A').circuit==old.circuit
    rows=[]
    for d,w,route in [(9,5,[4,5]),(9,5,[4,4,1]),(9,5,[6,3]),(13,9,[6,7])]:
        for tail in [1,2,3]:
            holds=[2]*(len(route)-2)+[tail]
            for wall in ['A','continue']:
                if tail%2 and wall=='continue':continue
                t=time.monotonic();b=build(d,w,route,holds,wall);r=flow_check(b)
                rows.append(dict(d=d,W=w,moves=route,holds=holds,wall=wall,wall_actual=b.wall_first,
                    passed=r['passed'],seconds=time.monotonic()-t))
                print('IDEAL',rows[-1],flush=True)
                write(HERE/'ideal_probe.json',rows)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--probe',action='store_true')
    ap.add_argument('--d',type=int,default=9);ap.add_argument('--W',type=int,default=5)
    ap.add_argument('--moves',type=int,nargs='+');ap.add_argument('--holds',type=int,nargs='+')
    ap.add_argument('--wall',choices=['A','B','continue','continue-close-A'],default='A')
    ap.add_argument('--channel',choices=['XY','ZZ','both'],default='XY')
    a=ap.parse_args()
    if a.probe:probe()
    else:
        for ch in ['XY','ZZ'] if a.channel=='both' else [a.channel]:
            print('CASE',generate(a.d,a.W,a.moves,a.holds,a.wall,ch),flush=True)
