"""Independent exhaustive oracle, durable checkpoints and multi-process tests."""
import itertools
import json
from pathlib import Path
import random
import tempfile
import time
import resumable_xor as sx


def valid_sites(selected, choices):
    return any(len(set(a))==len(a) for a in itertools.product(*(choices[j] for j in selected)))


def main():
    rng = random.Random(57920261020)
    scratch = Path(__file__).resolve().parent/'test_scratch'
    scratch.mkdir(exist_ok=True)
    partitions = resumed = exclusions = 0
    with tempfile.TemporaryDirectory(dir=scratch,prefix='dfs_') as td:
        td = Path(td)
        for trial in range(160):
            D = rng.randrange(3,7)
            n = rng.randrange(5,min(14,1<<(D+1)))
            vectors = rng.sample(range(1,1<<(D+1)),n)
            roots = [j for j,v in enumerate(vectors) if v>>D&1]
            if not roots:
                continue
            choices = [tuple(rng.sample(range(n),rng.randrange(1,3))) for _ in vectors]
            cap = 1+trial%8
            expected = []
            for k in range(1,min(cap,n)+1):
                for selected in itertools.combinations(range(n),k):
                    v=0
                    for j in selected:
                        v ^= vectors[j]
                    if v==1<<D and valid_sites(selected,choices):
                        expected.append(set(selected))
            # Invertible detector-row operation: same complete solution space.
            transformed = [v^(((v>>1)&1)<<(D if trial%3==2 else 0))
                           if trial%3 else v for v in vectors]
            cols = [tuple(i for i in range(D+1) if v>>i&1) for v in transformed]
            problem = sx.Problem(cols,D,roots,cap,choices)
            for start in range(0,len(roots),2):
                stop=min(start+2,len(roots))
                oracle=[s for s in expected if start<=roots.index(min(s&set(roots)))<stop]
                checkpoint=td/f'{trial}_{start}.json'
                for attempt in range(10000):
                    result=sx.run_lane(problem,checkpoint,'test',start,stop,1,0,
                        time.monotonic()+20,node_budget=1)
                    if result['status']!='CHECKPOINT':
                        break
                    resumed+=1
                else:
                    raise AssertionError('Resume made no progress')
                assert (result['status']=='PHYSICAL_CANDIDATE')==bool(oracle),(trial,start,result)
                if oracle:
                    assert set(result['selected']) in oracle
                else:
                    exclusions+=1
                before=result['nodes_total']
                again=sx.run_lane(problem,checkpoint,'test',start,stop,1,0,time.monotonic()+20)
                assert again['nodes_total']==before and again['status']==result['status']
                partitions+=1
                wrapped=json.loads(checkpoint.read_text())
                wrapped['payload']['identity']['cap_wrong']=True
                wrapped['sha256']=sx.digest(wrapped['payload'])
                checkpoint.write_text(json.dumps(wrapped))
                try:
                    sx.load_state(checkpoint,sx.identity(problem,'test',start,stop,1,0),problem)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Changed search identity accepted')
        D=4
        cols=[(0,),(0,1),(1,2),(2,3),(3,4)]
        choices=[(j,) for j in range(5)]
        roots=[4]
        problem=sx.Problem(cols,D,roots,4,choices)
        r=sx.run_lane(problem,td/'zero.json','timeout',0,1,1,0,time.monotonic()-1)
        assert r['status']=='CHECKPOINT' and r['nodes_total']==0
        r=sx.run_lane(problem,td/'zero.json','timeout',0,1,1,0,time.monotonic()+10)
        assert r['status']=='INFEASIBLE'
        for cap in (4,5):
            r=sx.solve(cols,D,roots,cap,0,1,choices,td/f'parallel{cap}','parallel',
                td/f'out{cap}',
                lambda selected:dict(status='UNDETECTED_LOGICAL',physical_weight=len(selected)),
                20,2)
            assert r['status']==('INFEASIBLE' if cap==4 else 'PHYSICAL_WITNESS')
        cols=[(0,2),(1,2),(0,1,2)]
        roots=[0,1,2]
        choices=[(0,),(1,),(2,)]
        r=sx.solve(cols,2,roots,2,0,3,choices,td/'multi','multi',td/'multi_out',
            lambda s:dict(status='UNDETECTED_LOGICAL',physical_weight=len(s)),20,2)
        assert r['status']=='INFEASIBLE' and r['lanes_complete']==3
        r=sx.solve(cols,2,roots,2,0,3,choices,td/'multi','multi',td/'multi_again',
            lambda s:dict(status='UNDETECTED_LOGICAL',physical_weight=len(s)),20,1)
        assert r['status']=='INFEASIBLE'
        assert sum(x['nodes_this_attempt'] for x in r['lane_results'])==0
        for distance in (7,9):
            D=distance-1
            cols=[(0,)]+[(j-1,j) for j in range(1,distance)]
            roots=[distance-1]
            choices=[(j,) for j in range(distance)]
            for cap in (distance-1,distance):
                problem=sx.Problem(cols,D,roots,cap,choices)
                checkpoint=td/f'chain_{distance}_{cap}.json'
                for attempt in range(100):
                    r=sx.run_lane(problem,checkpoint,'chain',0,1,1,0,
                                  time.monotonic()+10,node_budget=1)
                    if r['status']!='CHECKPOINT':
                        break
                assert r['status']==('INFEASIBLE' if cap<distance else 'PHYSICAL_CANDIDATE')
                if cap==distance:
                    assert len(r['selected'])==distance
    record=dict(status='PASS',random_models=160,partitions=partitions,
                excluded_partitions=exclusions,checkpoint_continuations=resumed,
                caps=list(range(1,9)),physical_site_oracle='independent exhaustive assignment',
                changed_identity_rejected=True,deadline_not_proof=True,
                multiprocess=True,worker_count_change=True,repeat_completed_no_work=True)
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
