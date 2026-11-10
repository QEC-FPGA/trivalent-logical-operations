"""Exact unsigned code-space ranks for the partial-wall candidate.

Use an ideal encoded Bell pair with a reference qubit to exclude the logical
state from the stabilizer group. No static or circuit distance claim.
"""
import argparse,json,sys
from pathlib import Path
import stim


def rank(rows):
    piv={}
    for v in rows:
        while v:
            p=v.bit_length()-1
            if p in piv:v^=piv[p]
            else:piv[p]=v;break
    return len(piv)


def physical_code(sim,n):
    # Eliminate the two reference coordinates from the full stabilizer span.
    piv={};physical=[]
    for pauli in sim.canonical_stabilizers():
        x=z=0
        for q in range(n+1):
            if pauli[q] in (1,2):x|=1<<q
            if pauli[q] in (2,3):z|=1<<q
        ref=((x>>n)&1)+2*((z>>n)&1)
        v=(x&((1<<n)-1))|((z&((1<<n)-1))<<n)
        while ref:
            p=ref.bit_length()-1
            if p in piv:
                a,b=piv[p];ref^=a;v^=b
            else:piv[p]=(ref,v);break
        if not ref:physical.append(v)
    return physical


def gauge_summary(a,b,n):
    piv={}
    for v in a+b:
        while v:
            p=v.bit_length()-1
            if p in piv:v^=piv[p]
            else:piv[p]=v;break
    gens=list(piv.values());mask=(1<<n)-1
    gram=[]
    for p in gens:
        x,z=p&mask,p>>n
        row=0
        for j,q in enumerate(gens):
            if ((x&(q>>n)).bit_count()+(z&(q&mask)).bit_count())%2:row|=1<<j
        gram.append(row)
    anti_rank=rank(gram);g=len(gens);s=g-anti_rank
    return {'gauge_rank':g,'center_rank':s,'gauge_qubits':anti_rank//2,
            'protected_k':n-(g+s)//2,'common_stabilizer_rank':len(a)+len(b)-g}

