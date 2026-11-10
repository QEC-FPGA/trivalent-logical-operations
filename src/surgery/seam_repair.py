"""Coordinate-based staggered seam preparation and entry measurement relations."""
import stim

def repair_initialization(c,d):
    c=c.flattened();coords=c.get_final_qubit_coordinates();ops=list(c)
    seam={q for q,v in coords.items() if len(v)>=2 and v[1]==2*d and 0<=v[0]<2*d-1 and int(v[0])%2==0}
    if len(seam)!=d:raise ValueError('Unexpected seam data coordinates')
    pairs=[]
    for i in range(len(ops)-1):
        if ops[i].name=='R' and ops[i+1].name=='RX':
            rz={t.value for t in ops[i].targets_copy()};rx={t.value for t in ops[i+1].targets_copy()}
            if seam<=rz|rx:pairs.append(i)
    if len(pairs)!=1:raise ValueError('Cannot uniquely identify seam initialization')
    entry=pairs[0]
    previous=next((i for i in range(entry-1,-1,-1) if ops[i].name=='R' and ops[i+1].name=='RX'),None)
    if previous is None:raise ValueError('Missing preceding memory reset')
    changes=[]
    for i,upper in [(previous,True),(entry,False)]:
        rz={t.value for t in ops[i].targets_copy()};rx={t.value for t in ops[i+1].targets_copy()}
        for q in sorted(rz|rx):
            x,y=coords[q][:2]
            if not (0<x<2*d-1 and int(x)%2==1):continue
            if upper:
                if y!=2*d-1 or int(x)%4!=1:continue
                desired='R'
            else:
                if y!=2*d+1:continue
                desired='RX' if int(x)%4==1 else 'R'
            old='R' if q in rz else 'RX'
            if old!=desired:changes.append(dict(q=q,coords=[x,y],old=old,new=desired))
            rz.discard(q);rx.discard(q);(rz if desired=='R' else rx).add(q)
        ops[i]=stim.CircuitInstruction('R',sorted(rz))
        ops[i+1]=stim.CircuitInstruction('RX',sorted(rx))
    out=stim.Circuit()
    for op in ops:out.append(op)
    return out,dict(d=d,reset_changes=changes,entry_reset_instruction=entry)

def entry_detectors(c,d):
    """Resolve the added checks from actual measured qubits, never fixed rec offsets."""
    _,info=repair_initialization(c,d)
    coords=c.get_final_qubit_coordinates();record=0;tick=0;groups={}
    for i,op in enumerate(c):
        if op.name=='TICK':tick+=1
        if op.name in ['M','MX','MY']:
            for t in op.targets_copy():
                if i>info['entry_reset_instruction']:
                    groups.setdefault(tick,{})[(op.name,tuple(coords[t.value][:2]))]=record
                record+=1
        else:
            one=stim.Circuit();one.append(op);record+=one.num_measurements
    ts=sorted(groups)
    if len(ts)<2:raise ValueError('Two merge extraction records required')
    first,second=groups[ts[0]],groups[ts[1]];out=stim.Circuit();meta=[]
    def add(label,where,records):
        out.append('DETECTOR',[stim.target_rec(j-record) for j in records],list(where)+[ts[1]])
        meta.append(dict(label=label,coordinates=list(where),measurement_indices=records))
    for x in range(1,2*d-3,2):
        y=2*d+1 if x%4==1 else 2*d-1
        add('entry_A_B_pair',(x+1,y),[first[('M',(x,y))],second[('M',(x+2,y))]])
    add('entry_boundary_Z',(1,2*d-1),[second[('M',(1,2*d-1))]])
    for x in range(1,2*d-1,4):
        add('entry_prepared_X',(x,2*d+1),[second[('MX',(x,2*d+1))]])
    return out,meta
