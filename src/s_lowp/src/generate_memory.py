"""Parity-aware twist memory / movement candidate. No changes to frozen sources.

Odd k: move tip S in A from after coupling layer 3 to after layer 2;
move S_DAG in B from after layer 2 to after layer 3. Even k is unchanged.
TICK moments 2,3,4,5 contain coupling layers 1,2,3,4.
The reference is obtained from wide-code Y projections, not odd-k extraction.
"""
from pathlib import Path
from functools import lru_cache
import argparse, json, hashlib
import stim
import probe_candidates as p

ROOT=Path(__file__).resolve().parent


def extraction(d,k,phase):
    return p.candidate(d,k,phase,'S',True,False)


@lru_cache(maxsize=64)
def initial_code(d,k,last_phase):
    if k==0: return p.ORIGINAL_INITIAL_CODE(d,k,last_phase)
    checks, lin=p.projected_code(d,k,'A')
    sim,n=p.sim_from(checks,lin)
    cycle=extraction(d,k,'B')+extraction(d,k,'A')
    sim.do(cycle)
    out,lout=p.old.split_ref(sim.canonical_stabilizers(),n)
    b=p.old.base.Stage1(d)
    keep={b.data(i,j) for i in range(d) for j in range(1,b.h)}-{b.data(i,d+1) for i in range(d-k,d)}
    target=p.restricted_basis(checks,keep); actual=p.restricted_basis(out,keep)
    assert len(target)==len(actual) and all(not p.residual(v,target) for v in actual.values()), 'Wrong target code'
    if last_phase=='B':
        sim.do(extraction(d,k,'B'))
        out,lout=p.old.split_ref(sim.canonical_stabilizers(),n)
    return out,lout


def generate(d,k,moves,pre,post,phase,channel,between_rounds=None,out=None):
    p.old.extraction=extraction
    p.old.initial_code=initial_code
    between_rounds=[2] if between_rounds is None else between_rounds
    suffix='-'.join(map(str,moves)) or 'static'
    name=f'd{d}_k{k}_m{suffix}_r'+ '-'.join(map(str,between_rounds))+f'_pre{pre}_post{post}_{phase}_{channel}'
    folder=Path(out) if out else ROOT/'cases'/name
    from case_lock import locked
    with locked(folder,'generation'):
        folder=p.old.generate(d,k,moves,None,pre,post,phase,channel,out=folder,between_rounds=between_rounds)
        mp=folder/'metadata.json'; m=json.loads(mp.read_text(encoding='utf-8'))
        m.update(generator='parity-aware tip-S timing candidate 20261018',
                 independent_target='wide A-phase code followed by Y measurement of the k seam sites',
                 correction='odd k: A S after layer2, B S_DAG after layer3; even k frozen baseline',
                 generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        temp=mp.with_suffix('.new.json');temp.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(mp)
        return folder


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--d',type=int,default=9);a.add_argument('--k',type=int,default=1)
    a.add_argument('--moves',type=int,nargs='*',default=[]);a.add_argument('--pre',type=int);a.add_argument('--post',type=int)
    a.add_argument('--phase',choices=['A','B'],default='A');a.add_argument('--channel',choices=['X','Z'],default='X')
    a.add_argument('--between-rounds',type=int,nargs='+',default=[2]);a.add_argument('--out',type=Path)
    v=a.parse_args();generate(v.d,v.k,v.moves,v.pre or v.d,v.post or v.d,v.phase,v.channel,v.between_rounds,v.out)
