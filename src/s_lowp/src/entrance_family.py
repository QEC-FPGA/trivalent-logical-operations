"""Finite test of the specified initialization entrance/turn/staircase family.

This is not an all-fault search and never establishes full circuit distance.
The first fault is fixed to the first wide extraction, not the moving interface.
"""
from study import *
from physical import circuit,fc,replay,sites
from generate_full_s import build,generate,route
import argparse,collections,time

def template(d,a,n,delta,form):
    es=[(j,2,[(1.5,d-.5+delta-j),(2,d-1+delta-j)],'XI','entrance') for j in range(a)]
    if form=='odd':
        es.extend([(a,4,[(1,d-a+delta),(1.5,d-a+.5+delta)],'XY','turn1'),
                   (a+1,4,[(.5,d-a-.5+delta),(1,d-a+delta)],'ZX','turn2L'),
                   (a+1,4,[(2,d-a+delta),(2.5,d-a+.5+delta)],'YZ','turn2R')])
        for j in range(1,n+1):
            x=2.5+.5*j;y=d-a+.5+delta+.5*j
            es.append((a+1+j,3,[(x,y),(x+.5,y+.5)],'YI','stair'))
    else:
        es.extend([(a,2,[(1,d-a-1+delta),(.5,d-a-.5+delta)],'IZ','turn1'),
                   (a,4,[(1.5,d-a-.5+delta),(2,d-a+delta)],'IY','turn2')])
        for j in range(1,n+1):
            x=2+.5*j;y=d-a+.5*j+delta
            es.append((a+j,3,[(x,y),(x+.5,y+.5)],'YI','stair'))
        x=2+n/2;y=d-a+n/2+delta
        es.append((a+n+1,4,[(x,y),(x+.5,y+.5)],'ZX','closure'))
    return es

def scan(d,step,w,maximum=None):
    start=time.monotonic();maximum=d if maximum is None else maximum
    f=generate(d,w,3,route(d-3,step),[2],'XY');c=circuit(f/'full_original.stim');meta=read(f/'metadata.json')
    b=build(d,w,3,route(d-3,step),[2]);stages={s['stage']:s for s in meta['stages']}
    rows=[stages[name] for name,seg in b.segments.items() if any(o.name in ('CX','CZ') for o in seg)]
    first=next(i for i,s in enumerate(rows) if s['stage'].startswith('wide_'));rows=rows[first:]
    inv={tuple(v):q for q,v in c.get_final_qubit_coordinates().items()}
    site_map={(t,tuple(qs)):(i,op,qs) for i,t,op,qs in sites(c) if op.name=='DEPOLARIZE2'}
    pool=[];lookup={};mapped=[];attempted=0
    for weight in range(4,maximum+1):
        for a in range(1,weight-2):
            n=weight-a-3
            for delta in range(0,d,2):
                for form in ('even','odd'):
                    attempted+=1;spec=template(d,a,n,delta,form);ids=[]
                    for block,lt,coords,paulis,part in spec:
                        if block>=len(rows):break
                        tick=rows[block]['start_tick']+lt;qs=tuple(inv.get(tuple(v)) for v in coords)
                        site=site_map.get((tick,qs))
                        if site is None:break
                        i,op,qs=site;key=(i,tuple(qs),paulis)
                        if key not in lookup:
                            lookup[key]=len(pool);pool.append(dict(instruction_offset=i,tick=tick,noise_kind=op.name,
                                noise_targets=qs,noise_args=op.gate_args_copy(),paulis={str(q):p for q,p in zip(qs,paulis) if p!='I'},
                                part=part,stage=rows[block]['stage'],local_tick=lt))
                        ids.append(lookup[key])
                    if len(ids)==weight:mapped.append(dict(weight=weight,a=a,n=n,delta=delta,form=form,ids=ids))
    print('ENTRANCE_POOL',d,step,w,attempted,len(mapped),len(pool),flush=True)
    cols=fc.propagate(c,pool);counts=collections.Counter();hits=[]
    for spec in mapped:
        sig=0
        for j in spec['ids']:sig^=cols[j]
        state='UNDETECTED_LOGICAL' if sig==1<<c.num_detectors else 'DETECTED' if sig&((1<<c.num_detectors)-1) else 'NO_LOGICAL_EFFECT'
        counts[state]+=1
        if state=='UNDETECTED_LOGICAL':hits.append(spec)
    out=ROOT/'entrance_family'/f'd{d}_s{step}_W{w}';out.mkdir(parents=True,exist_ok=True)
    for j,spec in enumerate(sorted(hits,key=lambda v:v['weight'])[:3]):
        spec['replay']=replay(c,[pool[i] for i in spec['ids']],out/f'witness_{j}')
        assert spec['replay']['status']=='UNDETECTED_LOGICAL'
    result=dict(d=d,step=step,W=w,maximum_weight=maximum,attempted=attempted,mapped=len(mapped),
        responses=dict(counts),minimum_found=min((s['weight'] for s in hits),default=None),
        hits=hits,circuit_sha256=sha(f/'full_original.stim'),seconds=time.monotonic()-start,
        scope='specified two turn forms; first-wide entrance X; even delta 0..d-1; fixed Pauli pattern',
        excludes_all_left_entrance_faults=False,full_distance_lower_bound=False)
    write(out/'result.json',result)
    print('ENTRANCE_RESULT',d,step,w,result['minimum_found'],dict(counts),round(result['seconds'],2),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--d',type=int,default=13);p.add_argument('--step',type=int,default=3)
    p.add_argument('--W',type=int,nargs='+',default=[1,7,9,11,13]);a=p.parse_args()
    for w in a.W:scan(a.d,a.step,w)
