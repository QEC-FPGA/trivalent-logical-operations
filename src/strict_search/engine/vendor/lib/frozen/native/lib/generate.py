"""Portable partial-wall S-gate family. Generation is not a distance proof."""
from pathlib import Path
import argparse,hashlib,json
import stim
import engine

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'source_work_topcrop'
Stage1,_,annotate,_,complete_detectors,_,_,observable_indices=engine.load_source(SOURCE)
from d7_readout import standard_stabilizers
from d7_stage1_flow import logicals,flow
NOISE={'X_ERROR','Y_ERROR','Z_ERROR','DEPOLARIZE1','DEPOLARIZE2'}
KNOWN_FAILURES={(7,1,1):4,(7,5,1):6,(9,5,3):8,(9,7,1):7,(9,7,3):8,(9,7,5):8,(9,9,9):8}

def round_counts(d,profile='baseline',wide=None,twist=None):
    if d<5 or d%2!=1:raise ValueError('This geometry requires odd distance >=5 (5,7,9,11,...).')
    if profile=='baseline':w,t=d,d
    elif profile=='trim-twist':w,t=d,d-2
    elif profile=='compact':
        # For d>=7, w+t=d-1, both odd: total 2d+w+t+2=3d+1.
        # d=5 has no partial-wall interval and uses a separate 3+3 trial.
        if d==5:w,t=3,3
        else:
            w=(d-1)//2
            if w%2==0:w+=1
            t=d-1-w
    elif profile=='custom':
        if wide is None or twist is None:raise ValueError('custom requires --wide and --twist')
        w,t=wide,twist
    else:raise ValueError(profile)
    if profile!='custom' and (wide is not None or twist is not None):raise ValueError('Use --profile custom for explicit counts')
    if min(w,t)<1 or w%2!=1 or t%2!=1:raise ValueError('wide and twist counts must be positive odd numbers')
    return w,t

def build_builder(d,basis='Z',profile='baseline',wide=None,twist=None):
    w,t=round_counts(d,profile,wide,twist)
    b=Stage1(d);b.right_seam_width=d-3
    b.data_init(basis)
    for k in range(d):b.standard_round('B' if k%2==0 else 'A',k+1)
    b.expansion()
    for k in range(w):b.wide_round('A' if k%2==0 else 'B',d+1+k)
    if d>5:
        mid=Stage1(d);mid.right_seam_width=d-5
        mid.my_seam('right');mid.twist_round('B',100);mid.twist_round('A',101)
        for name,c in zip(('handoff_bulk_MY','handoff_twist_B','handoff_twist_A'),mid.segments.values()):
            b.append_segment(name,c)
        release=stim.Circuit();release.append('MY',[b.data(i,d+1) for i in (3,4)]);release.append('TICK')
        b.append_segment('handoff_endpoint_MY',release)
    else:b.my_seam('right')
    for k in range(t):
        # The legacy last twist omits its final TICK. Anchor numbering to that
        # LAST round, including after shortening, so the exit interface stays
        # byte-identical. With t=d this is exactly the archived numbering.
        b.twist_round('B' if k%2==0 else 'A',2*d+6-t+k)
    b.my_seam('left')
    for k in range(d):b.fullwall_round('A' if k%2==0 else 'B',k+1)
    b.h_lower();b.shrink()
    return b

def logical_audit(b):
    n=b.circuit.num_qubits;d=b.d
    input_data={b.data(i,j+1) for i in range(d) for j in range(d)}
    prep=stim.Circuit();prep.append('R',[q for q in range(n) if q not in input_data])
    for check in standard_stabilizers(d):
        ts=[]
        for i in range(d):
            for j in range(d):
                p=check[d*i+j]
                if p:ts.append('IXYZ'[p]+str(b.data(i,j+1)))
        prep+=stim.Circuit('MPP '+'*'.join(ts))
    c=prep+engine.body_after_init(b)
    pin,pout=logicals(b,'input'),logicals(b,'shrink')
    answer={axis:flow(c,pin[axis],pout[axis]) for axis in 'XZY'}
    assert all(x['exists'] for x in answer.values()),'Lost an encoded logical flow'
    return {'logical_input_unfixed':True,'physical_output_axes':{'X':'Y','Z':'Z','Y':'X'},
            'record_domain':'Ideal code-space flow probe: first d*d-1 records are stabilizer projections; subsequent records are body measurements. These are NOT indices in the YX/ZZ preparation circuits.',
            'probe_stabilizer_measurement_count':d*d-1,
            'note':'Signs and record-dependent Pauli frame are explicit in each signed flow; do not discard them.',
            'flows':answer}

def resources(b,w,t):
    edges=set();used=set();degrees={};twoq_ticks=set();tick=0;segments=[]
    for name,c in b.segments.items():
        segments.append({'name':name,'tick_start_in_builder':tick,'ticks':c.num_ticks,'measurements':c.num_measurements})
        for op in c:
            if op.name=='TICK':tick+=1;continue
            qs=[q.value for q in op.targets_copy() if q.is_qubit_target]
            used.update(qs)
            if op.name in ('CX','CZ'):
                twoq_ticks.add(tick)
                edges.update(tuple(sorted((a,z))) for a,z in zip(qs[::2],qs[1::2]))
    for a,z in edges:degrees[a]=degrees.get(a,0)+1;degrees[z]=degrees.get(z,0)+1
    assert max(degrees.values())<=3,'Union coupling graph exceeds degree three'
    return {'standard_rounds':b.d,'wide_rounds':w,'handoff_rounds':2 if b.d>5 else 0,
            'twist_rounds':t,'fullwall_rounds':b.d,'total_extraction_rounds':2*b.d+w+t+(2 if b.d>5 else 0),
            'round_definition':'One A or B extraction block, not a two-block A/B period',
            'body_ticks_excluding_input_prep':engine.body_after_init(b).num_ticks,
            'two_qubit_gate_layers':len(twoq_ticks),'used_qubits':len(used),'coupling_edges':len(edges),
            'maximum_union_coupling_degree':max(degrees.values()),'segments':segments}

def filter_reference(c,start):
    out=stim.Circuit();tick=0
    for op in c:
        if op.name=='TICK':tick+=1
        if op.name in NOISE and tick<start:continue
        out.append(op)
    assert out.without_noise()==c.without_noise()
    return out

def build_channel(d,ch,p,b,out,emit_dem=False):
    if ch=='YX':prep,mpp=engine.y_mpp_preparation(b)
    else:prep=b.segments['data_init'].copy();mpp=None
    body=engine.body_after_init(b);prefix=prep+body
    final=[b.data(i,j) for i in range(d) for j in range(1,d+1)]
    terminal=stim.Circuit()
    if ch=='YX':terminal.append('H',final);terminal.append('TICK')
    terminal.append('M',final);terminal.append('TICK')
    physical=engine.with_coordinates(prefix+terminal,b)
    obs,expected=observable_indices(physical,prefix,b,ch)
    first=prefix.num_measurements
    dual=([first+d*i for i in range(d)] if ch=='YX' else [first+j for j in range(d)])
    cumulative=engine.with_coordinates(prep,b);prefixes=[]
    for name,c in b.segments.items():
        if name!='data_init':cumulative+=c;prefixes.append(cumulative.copy())
    detectors,rank=complete_detectors(physical,obs,dual,prefixes)
    assert len(detectors)+1==rank
    ideal=annotate(physical,detectors,obs)
    ds,ls=ideal.compile_detector_sampler(seed=23).sample(16,separate_observables=True)
    assert not ds.any() and not ls.any()
    samples=physical.compile_sampler(seed=29).sample(16)
    assert all(sum(int(row[i]) for i in obs)%2==expected for row in samples)
    if ch=='YX':noisy=engine.with_coordinates(prep+engine.si1000_with_ideal_mpp(body+terminal,p),b)
    else:noisy=engine.si1000_with_ideal_mpp(physical,p)
    first_standard=next(c for name,c in b.segments.items() if name.startswith('standard_'))
    start=prep.num_ticks+first_standard.num_ticks
    assert start==(11 if ch=='YX' else 9)
    noisy=filter_reference(noisy,start)
    assert noisy.without_noise()==physical
    analysis=annotate(noisy,detectors,obs)
    # Full ideal-flow checks above establish detector determinism. DEM export
    # is optional because its size and computation grow with distance.
    files={}
    for suffix,c in [('physical',physical),('ideal_analysis',ideal),('SI1000_body',analysis)]:
        path=out/f'd{d}_{ch}_{suffix}.stim';path.write_text(str(c)+'\n',encoding='utf-8');files[suffix]=path.name
    if emit_dem:
        dem=analysis.detector_error_model(decompose_errors=False,allow_gauge_detectors=False)
        path=out/f'd{d}_{ch}_SI1000_body.dem';path.write_text(str(dem)+'\n');files['dem']=path.name
    return {'channel':ch,'faults_start_tick_inclusive':start,'ideal_preparation_and_first_standard_reference':True,
            'detectors':len(detectors),'measurements':physical.num_measurements,'complete_parity_rank':rank,
            'raw_observable_expected_bit':expected,'ideal_relative_detector_and_observable_flips':0,
            'physical_ticks':physical.num_ticks,'mpp':mpp,'files':files,'si1000_p':p}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--distance',type=int,required=True)
    ap.add_argument('--profile',choices=['baseline','trim-twist','compact','custom'],default='baseline')
    ap.add_argument('--wide',type=int);ap.add_argument('--twist',type=int)
    ap.add_argument('--channels',nargs='+',choices=['YX','ZZ'],default=['YX','ZZ'])
    ap.add_argument('--p',type=float,default=.001)
    ap.add_argument('--body-only',action='store_true',help='Export ideal body and signed logical flows; skip complete detector/noise export')
    ap.add_argument('--dem',action='store_true',help='Also export full undecomposed DEM')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();w,t=round_counts(args.distance,args.profile,args.wide,args.twist)
    if not 0<args.p<.1:ap.error('--p must be between 0 and 0.1')
    if args.body_only and args.dem:ap.error('--body-only and --dem are incompatible')
    args.out.mkdir(parents=True,exist_ok=True)
    report={'distance':args.distance,'profile':args.profile,'stim_version':stim.__version__,
            'status':'BASELINE CANDIDATE: d7 strict under reference boundary; d9 refuted by a new physical witness; no all-distance proof' if args.profile=='baseline' else 'EXPERIMENTAL COMPRESSION; NOT DISTANCE VERIFIED',
            'distance_search':'NOT RUN','channels':{}}
    if (args.distance,w,t) in KNOWN_FAILURES:
        report['status']='REJECTED: known physical logical counterexample'
        report['known_YX_distance_upper_bound']=KNOWN_FAILURES[args.distance,w,t]
    if args.profile=='compact':
        report['family_status']='Rejected as a general distance-preserving family: d9,w5,t3 has a physical weight-8 witness.'
    print(report['status'],flush=True)
    b=build_builder(args.distance,'Z',args.profile,args.wide,args.twist)
    report['resources']=resources(b,w,t);report['logical_audit']=logical_audit(b)
    (args.out/'body.stim').write_text(str(engine.with_coordinates(engine.body_after_init(b),b))+'\n')
    for ch in args.channels:
        if args.body_only:continue
        print('Building',args.distance,args.profile,ch,flush=True)
        bb=build_builder(args.distance,'X' if ch=='YX' else 'Z',args.profile,args.wide,args.twist)
        report['channels'][ch]=build_channel(args.distance,ch,args.p,bb,args.out,args.dem)
    report['source_sha256']={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(HERE.glob('*.py'))+sorted(SOURCE.glob('*.py'))}
    (args.out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:report['resources'][k] for k in ('wide_rounds','twist_rounds','total_extraction_rounds','body_ticks_excluding_input_prep','maximum_union_coupling_degree')}) )

if __name__=='__main__':main()
