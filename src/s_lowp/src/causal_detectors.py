"""Complete causal detector basis from symbolic Pauli frames.

Random measurement results are symbolic record bits. Kickbacks update the
conditional Pauli frame; deterministic measurements emit exact parity rows.
All resets must act on a disentangled qubit (checked), as in these circuits.
This changes only the detector basis, never the physical circuit or observable.
"""
from study import *
import stim,time

def indices(v):
    out=[]
    while v:
        b=v&-v;out.append(b.bit_length()-1);v^=b
    return out

def rows(c):
    s=stim.TableauSimulator(seed=719);s.set_num_qubits(c.num_qubits)
    x=[0]*c.num_qubits;z=x.copy();result=[];count=0;randoms=0
    def h(q):s.h(q);x[q],z[q]=z[q],x[q]
    def sg(q,inverse=False):
        if inverse:s.s_dag(q)
        else:s.s(q)
        z[q]^=x[q]
    def cx(a,b):s.cx(a,b);x[b]^=x[a];z[a]^=z[b]
    def meas(q):
        nonlocal count,randoms
        out,kick=s.measure_kickback(q);flip=(1<<count)^x[q]
        if kick is None:result.append(flip)
        else:
            randoms+=1
            for j,p in enumerate(kick):
                if p in (1,2):x[j]^=flip
                if p in (2,3):z[j]^=flip
        count+=1
    for op in c:
        nm=op.name;ts=op.targets_copy();qs=[t.value for t in ts]
        if nm in ('TICK','QUBIT_COORDS','DETECTOR','OBSERVABLE_INCLUDE'):continue
        if nm=='R':
            for q in qs:
                assert s.peek_bloch(q).weight==1,('Entangled reset requires a latent variable',q)
                s.reset(q);x[q]=z[q]=0
        elif nm=='H':
            s.do(op)
            for q in qs:x[q],z[q]=z[q],x[q]
        elif nm in ('S','S_DAG'):
            s.do(op)
            for q in qs:z[q]^=x[q]
        elif nm=='CX':
            s.do(op)
            for a,b in zip(qs[::2],qs[1::2]):x[b]^=x[a];z[a]^=z[b]
        elif nm=='CZ':
            s.do(op)
            for a,b in zip(qs[::2],qs[1::2]):z[b]^=x[a];z[a]^=x[b]
        elif nm in ('M','MX','MY'):
            for q in qs:
                if nm=='MY':sg(q,True)
                if nm!='M':h(q)
                meas(q)
                if nm!='M':h(q)
                if nm=='MY':sg(q)
        elif nm=='MPP':
            products=[];join=False
            for t in ts:
                if t.is_combiner:join=True;continue
                if not join:products.append([])
                products[-1].append(t);join=False
            for product in products:
                # Most ideal code-boundary products are already deterministic.
                # Measure these directly instead of expanding a long canonical
                # Pauli into thousands of single CNOT simulator calls.
                pauli=stim.PauliString(c.num_qubits);flip=1<<count
                for t in product:
                    q=t.value
                    pauli[q]='Y' if t.is_y_target else 'X' if t.is_x_target else 'Z'
                    if t.is_x_target or t.is_y_target:flip^=z[q]
                    if t.is_z_target or t.is_y_target:flip^=x[q]
                if s.peek_observable_expectation(pauli):
                    s.measure_observable(pauli);result.append(flip);count+=1
                    continue
                for t in product:
                    if t.is_y_target:sg(t.value,True)
                    if t.is_x_target or t.is_y_target:h(t.value)
                q=product[-1].value
                for t in product[:-1]:cx(t.value,q)
                meas(q)
                for t in reversed(product[:-1]):cx(t.value,q)
                for t in reversed(product):
                    if t.is_x_target or t.is_y_target:h(t.value)
                    if t.is_y_target:sg(t.value)
        else:raise ValueError(f'Unsupported operation {nm}')
    assert count==c.num_measurements and len(result)+randoms==count
    return result

def complete(physical,obs,terminal_logical_indices,prefixes=None):
    assert terminal_logical_indices==[physical.num_measurements-1]
    rr=rows(physical);rank=physical.count_determined_measurements()
    assert len(rr)==rank
    logical=1<<terminal_logical_indices[0];ob=sum(1<<j for j in obs)
    assert rr[-1]&logical and ob&logical
    # Triangular rows ending before the last logical measurement form the full
    # detector subspace. The specified observable must be in their complement.
    piv={v.bit_length()-1:v for v in rr};rem=ob
    while rem:
        j=rem.bit_length()-1
        assert j in piv,'Specified observable not deterministic'
        rem^=piv[j]
    return [indices(v) for v in rr[:-1]],rank

def verify(folder):
    c=stim.Circuit((folder/'full_ideal.stim').read_text(encoding='utf-8'));physical=stim.Circuit()
    old=[];obs=None;n=c.num_measurements
    for op in c:
        if op.name=='DETECTOR':old.append(sum(1<<(n+t.value) for t in op.targets_copy()))
        elif op.name=='OBSERVABLE_INCLUDE':obs=[n+t.value for t in op.targets_copy()]
        else:physical.append(op)
    t=time.monotonic();dets,rank=complete(physical,obs,[n-1]);new=[sum(1<<j for j in row) for row in dets]
    def basis(rr):
        b={}
        for v in rr:
            while v:
                j=v.bit_length()-1
                if j in b:v^=b[j]
                else:b[j]=v;break
        return b
    b=basis(old)
    assert len(b)==len(new)==len(old)==rank-1
    for v in new:
        while v:
            j=v.bit_length()-1;assert j in b;v^=b[j]
    annotated=base.annotate(physical,dets,obs)
    dd,ll=annotated.compile_detector_sampler(seed=389).sample(16,separate_observables=True)
    assert not dd.any() and not ll.any()
    return dict(case=folder.name,full_parity_span_equal=True,physical_and_observable_unchanged=True,
                rank=rank,seconds=time.monotonic()-t,reset_disentanglement_checked=True)

if __name__=='__main__':
    names=['d5_k1_mstatic_r2_pre5_post5_A_X','d9_k1_m3_r2_pre9_post9_B_X',
           'd9_k1_m4_r2_pre9_post9_A_Z','d13_k1_m5_r2_pre13_post13_B_X']
    rs=[]
    for nm in names:
        r=verify(PREVIOUS/'cases'/nm);rs.append(r);write(ROOT/'causal_basis_validation.json',rs);print(r,flush=True)
