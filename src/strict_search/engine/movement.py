"""Local encoded-memory tests of twist displacement; ideal MPP boundaries."""
from pathlib import Path
import argparse,json,sys,hashlib,time
import stim
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
import compile_case as compiler
base=compiler.base

def positive(p):
    p=p.copy();p.sign=1;return p

def split_ref(stabs,ref):
    """Separate one encoded Bell pair from commuting physical code checks."""
    rows=[p.copy() for p in stabs]
    xp=next((p for p in rows if p[ref] in (1,2)),None)
    if xp is None:raise ValueError('Reference X correlation lost')
    rows.remove(xp)
    rows=[p*xp if p[ref] in (1,2) else p for p in rows]
    zp=next((p for p in rows if p[ref]==3),None)
    if zp is None:raise ValueError('Reference Z correlation lost')
    rows.remove(zp)
    rows=[p*zp if p[ref]==3 else p for p in rows]
    if xp[ref]==2:xp*=zp
    assert xp[ref]==1 and zp[ref]==3 and all(p[ref]==0 for p in rows)
    checks=[positive(p[:ref]) for p in rows]
    x,z=positive(xp[:ref]),positive(zp[:ref])
    assert not x.commutes(z)
    assert all(p.commutes(x) and p.commutes(z) for p in checks)
    assert len(checks)==ref-1
    return checks,{'X':x,'Z':z}

def extraction(d,k,phase):
    b=base.Stage1(d);b.right_seam_width=k
    if k==0:b.wide_round(phase,10000)
    else:b.twist_round(phase,10000) # never use legacy index which omits a TICK
    return b.circuit

def my_growth(d,k,end):
    b=base.Stage1(d);c=stim.Circuit()
    c.append('MY',[b.data(i,d+1) for i in range(d-end,d-k)])
    c.append('TICK');return c

def initial_code(d,k,last_phase):
    """Derive static code from ideal canonical evolution, preserving a Bell reference."""
    b=base.Stage1(d);n=b.stride*d+b.h+1
    data=[b.data(i,j) for i in range(d) for j in range(1,d+1)]
    st=[]
    for q in range(n):
        if q not in data:
            p=stim.PauliString(n+1);p[q]='Z';st.append(p)
    for check in base.standard_stabilizers(d):
        p=stim.PauliString(n+1)
        for q,v in zip(data,check):p[q]=v
        st.append(p)
    for ax in 'XZ':
        p=base.logicals(b,'input')[ax];p+=stim.PauliString(n+1-len(p));p[n]=ax;st.append(p)
    sim=stim.TableauSimulator(seed=19)
    sim.set_inverse_tableau(stim.Tableau.from_stabilizers(st).inverse())
    for j in range(d):b.standard_round('B' if j%2==0 else 'A',j)
    b.expansion()
    wide_n=(3 if last_phase=='A' else 4) if k==0 else 3
    for j in range(wide_n):b.wide_round('A' if j%2==0 else 'B',100+j)
    if k:
        b.append_segment('form',my_growth(d,0,k))
        for j in range(3 if last_phase=='B' else 4):b.append_segment('tw'+str(j),extraction(d,k,'B' if j%2==0 else 'A'))
    sim.do(b.circuit)
    return split_ref(sim.canonical_stabilizers(),n)

def name_of(d,k,moves,pause,pre,post,phase,channel):
    seq='-'.join(map(str,moves)) or 'static'
    pauses='-'.join(map(str,pause)) or '0'
    timing='r'+'-'.join(str(s+1) for s in pause) if any(s<0 for s in pause) else 's'+pauses
    return f'd{d}_k{k}_m{seq}_{timing}_pre{pre}_post{post}_{phase}_{channel}'

def generate(d,k,moves,pause,pre,post,phase,channel,out=None,between_rounds=None):
    start=time.monotonic();moves=list(moves)
    if between_rounds is not None:
        if pause is not None:raise ValueError('pause and between_rounds are mutually exclusive')
        between_rounds=list(between_rounds)
        if not between_rounds or any(x<0 for x in between_rounds):raise ValueError('between_rounds must be nonnegative')
        pause=[x-1 for x in between_rounds]
    else:
        pause=list(pause if pause is not None else [0])
        if any(x<0 for x in pause):raise ValueError('pause is extra waiting and must be nonnegative')
    assert d>=5 and d%2 and 0<=k<=d-3 and all(x>0 for x in moves)
    assert k+sum(moves)<=d-3 and pre>=1 and post>=1
    if len(pause)==1:pause=pause*max(0,len(moves)-1)
    assert len(pause)==max(0,len(moves)-1) and all(s>=-1 for s in pause)
    name=name_of(d,k,moves,pause,pre,post,phase,channel)
    folder=Path(out) if out else ROOT/'cases'/name
    expected_hash=None
    if (folder/'metadata.json').exists():
        old=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        wanted=dict(d=d,k=k,moves=moves,pauses=pause,pre=pre,post=post,phase_before_move=phase,channel=channel)
        assert all(old[key]==value for key,value in wanted.items()),'Existing output has different parameters'
        if all((folder/file).exists() for file in ['full_original.stim','full_ideal.stim','body.stim']):return folder
        expected_hash=old['circuit_sha256']
    folder.mkdir(parents=True,exist_ok=True)
    first=phase if pre%2 else ('B' if phase=='A' else 'A')
    previous='B' if first=='A' else 'A'
    checks,lin=initial_code(d,k,previous)
    n=len(lin['X']);body=stim.Circuit();stages=[];active_phase=first;width=k
    def add(label,c):
        stages.append(dict(stage=label,start_tick=body.num_ticks,ticks=c.num_ticks,width=width))
        body.__iadd__(c)
    def rounds(label,count):
        nonlocal active_phase
        for j in range(count):
            add(f'{label}_{j+1}_{active_phase}',extraction(d,width,active_phase))
            active_phase='B' if active_phase=='A' else 'A'
    rounds('pre',pre)
    for j,step in enumerate(moves):
        add(f'move_{width}_to_{width+step}',my_growth(d,width,width+step));width+=step
        rounds('post' if j==len(moves)-1 else f'hold_k{width}',post if j==len(moves)-1 else 1+pause[j])
    if not moves:rounds('post',post)
    # Propagate an encoded Bell state through the local body to obtain output code.
    st=[]
    for p in checks:st.append(p+stim.PauliString(1))
    for ax in 'XZ':
        p=lin[ax]+stim.PauliString(1);p[n]=ax;st.append(p)
    sim=stim.TableauSimulator(seed=29);sim.set_inverse_tableau(stim.Tableau.from_stabilizers(st).inverse());sim.do(body)
    checks_out,lout=split_ref(sim.canonical_stabilizers(),n)
    codeprep=stim.Circuit()
    for p in checks:compiler.mpp(codeprep,p)
    probe=codeprep+body
    logic={ax:base.flow(probe,lin[ax],lout[ax]) for ax in 'XZ'}
    assert all(v['exists'] for v in logic.values()),'Logical flow lost'
    yyin=positive(lin['X']*lin['Z']);yyout=positive(lout['X']*lout['Z'])
    logic['Y']=base.flow(probe,yyin,yyout);assert logic['Y']['exists']
    outflows=probe.solve_flow_measurements([stim.Flow(output=p) for p in checks_out])
    assert all(x is not None for x in outflows),'Output check depends on unknown logical input'
    prep=stim.Circuit();prep.append('R',list(range(n)));prep+=codeprep;compiler.mpp(prep,lin[channel])
    terminal=stim.Circuit()
    for p in checks_out:compiler.mpp(terminal,p)
    compiler.mpp(terminal,lout[channel])
    prefix=prep+body;physical=prefix+terminal
    obs=base.flow(prefix,stim.PauliString(n),lout[channel]);assert obs['exists']
    obs_indices=sorted(set(obs['measurements'])^{physical.num_measurements-1})
    dets,rank=base.complete_detectors(physical,obs_indices,[physical.num_measurements-1],None)
    ideal=base.annotate(physical,dets,obs_indices)
    noisy=base.annotate(prep+base.engine.si1000_with_ideal_mpp(body,.001)+terminal,dets,obs_indices)
    b=base.Stage1(d)
    coordinates=stim.Circuit()
    for q,xy in b.coords().items():coordinates.append('QUBIT_COORDS',[q],xy)
    ideal=coordinates+ideal;noisy=coordinates+noisy
    assert noisy.without_noise()==ideal
    ds,ls=ideal.compile_detector_sampler(seed=42).sample(16,separate_observables=True)
    assert not ds.any() and not ls.any()
    noisy.detector_error_model(decompose_errors=False)
    edges=set();layer=set();layers=0
    for op in body:
        if op.name=='TICK':layers+=bool(layer);layer=set()
        elif op.name in ('CX','CZ'):
            q=[t.value for t in op.targets_copy()];assert not(layer&set(q)) and len(q)==len(set(q));layer.update(q)
            edges.update(tuple(sorted(p)) for p in zip(q[::2],q[1::2]))
    layers+=bool(layer)
    degree={q:sum(q in e for e in edges) for e in edges for q in e};assert max(degree.values())<=3
    m=dict(d=d,k=k,moves=moves,pauses=pause,pre=pre,post=post,phase_before_move=phase,channel=channel,
           between_rounds=[s+1 for s in pause],
           boundary='Ideal input/output code MPP; noise only on the entire local memory/movement body',
           max_degree=max(degree.values()),stages=stages,logical_flows=logic,
           input_logicals={a:str(p) for a,p in lin.items()},output_logicals={a:str(p) for a,p in lout.items()},
           input_checks=[str(p) for p in checks],output_checks=[str(p) for p in checks_out],
           detectors=len(dets),parity_rank=rank,ticks=noisy.num_ticks,two_qubit_layers=layers,
           circuit_sha256=hashlib.sha256(str(noisy).encode()).hexdigest(),seconds=time.monotonic()-start)
    if expected_hash is not None:assert expected_hash==m['circuit_sha256'],'Regeneration differs from saved circuit hash'
    for filename,c in [('full_original.stim',noisy),('full_ideal.stim',ideal),('body.stim',body)]:
        (folder/filename).write_text(str(c),encoding='utf-8')
    (folder/'metadata.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
    print('GENERATED',name,round(time.monotonic()-start,2),flush=True)
    return folder

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--d',type=int,default=9);ap.add_argument('--k',type=int,default=1)
    ap.add_argument('--moves',type=int,nargs='*',default=[3])
    timing=ap.add_mutually_exclusive_group()
    timing.add_argument('--pause',type=int,nargs='+')
    timing.add_argument('--between-rounds',type=int,nargs='+')
    ap.add_argument('--pre',type=int);ap.add_argument('--post',type=int)
    ap.add_argument('--phase',choices=['A','B'],default='A');ap.add_argument('--channel',choices=['X','Z'],default='X')
    ap.add_argument('--out',type=Path);a=ap.parse_args()
    generate(a.d,a.k,a.moves,a.pause,a.pre if a.pre is not None else a.d,a.post if a.post is not None else a.d,a.phase,a.channel,a.out,between_rounds=a.between_rounds)
