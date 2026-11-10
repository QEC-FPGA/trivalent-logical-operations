"""Coordinate-level candidate family, independent of detector row reduction."""
from physical import *

def predicted(d,k,length,phase):
    if k==0:return d-((length-1)//2 if phase=='A' else length//2)
    efficient=(k%2==1 and phase=='B') or (k%2==0 and phase=='A')
    return d-((length-1)//2 if efficient else (length-2)//2)

def build_events(c,d,k,length,phase,mt,variant=3):
    a=d-k;b=a-length;effective=a
    loc={}
    for i,t,op,qs in sites(c):
        if t-mt not in (-8,-5,-4,-1):continue
        for q in qs:loc.setdefault((t,q),[]).append((i,op,qs))
    ev=[]
    def add(tt,support,kind=None):
        pa={qubit(q,d):p for q,p in support};q0=next(iter(pa));candidates=[]
        for i,op,qs in loc.get((mt+tt,q0),[]):
            if kind and op.name!=kind:continue
            if not set(pa)<=set(qs):continue
            if op.name.endswith('_ERROR') and any(p!=op.name[0] for p in pa.values()):continue
            candidates.append(dict(instruction_offset=i,tick=mt+tt,noise_kind=op.name,noise_targets=qs,noise_args=op.gate_args_copy(),paulis={str(q):p for q,p in pa.items()}))
        if not candidates:raise ValueError((tt,support,kind))
        ev.append(candidates[0])
    efficient=(k%2==1 and phase=='B') or (k%2==0 and phase=='A')
    if k==0:
        if phase=='A':add(-1,[(('q',d-1,d+1),'X')],'DEPOLARIZE1')
        else:effective=d+1
    elif efficient:
        add(-5,[(('a',a,d+1),'Y')],'DEPOLARIZE2')
        for x in range(a,d-(1 if phase=='B' else 0)):
            add(-1,[(('q',x,d),'Z')],'DEPOLARIZE1')
        if phase=='B':add(-8,[(('a',d,d+2),'X')],'X_ERROR')
    elif phase=='A':
        effective=a-1
        add(-1,[(('q',a-2,d+1),'X')],'DEPOLARIZE1')
        for x in range(a-1,d-1):add(-1,[(('q',x,d+2),'X')],'DEPOLARIZE1')
        add(-8,[(('a',d,d+3),'X')],'X_ERROR')
    else:
        effective=a-1
        if variant==0:
            add(-5,[(('a',effective,d+1),'Y')],'DEPOLARIZE2')
            for x in range(effective,d-1):add(-1,[(('q',x,d),'Z')],'DEPOLARIZE1')
            add(-8,[(('a',d,d+2),'X')],'X_ERROR')
        else:
            yy=(variant-1)//6+1;rr=((variant-1)//2)%3+1;bb=2+(variant-1)%2
            add(-1,[(('q',a-2,d+yy),'X')],'DEPOLARIZE1')
            for x in range(a-1,d-1):add(-1,[(('q',x,d+rr),'X')],'DEPOLARIZE1')
            add(-8,[(('a',d,d+bb),'X')],'X_ERROR')
    ell=effective-b
    for x in range(b+1+(ell%2==0),effective-1,2):
        add(-4,[(('a',x,d+1),'X'),(('q',x,d+1),'Z')],'DEPOLARIZE2')
    for x in range(b):add(-1,[(('q',x,d+1),'Z')],'DEPOLARIZE1')
    if ell%2==0:add(-1,[(('q',b,d+1),'X')],'DEPOLARIZE1')
    return ev

if __name__=='__main__':
    results=[]
    for d in (9,11,13):
        for k in (1,2,3):
            for step in (3,4,5):
                for ph in 'AB':
                    f=PREVIOUS/'cases'/f'd{d}_k{k}_m{step}_r2_pre{d}_post{d}_{ph}_X'
                    if not f.exists():continue
                    c=circuit(f/'full_original.stim')
                    try:r=replay(c,build_events(c,d,k,step,ph,8*d))
                    except ValueError as e:r=dict(status='NOT_CONSTRUCTED',error=str(e))
                    results.append(dict(case=f.name,**r));write(ROOT/'explicit_family_probe.json',results)
                    print(f.name,r['status'],r.get('physical_weight'),len(r.get('detectors_fired',[])),flush=True)
