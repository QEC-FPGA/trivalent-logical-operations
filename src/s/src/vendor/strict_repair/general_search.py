"""Size-independent, bounded, invertible detector-basis repair.

No detector IDs, distances, channels, or stored successful bases are hard coded.
Only a native strict Stim success is an acceptance. Search failure is UNKNOWN.
"""
from __future__ import annotations
import collections, re, time
import stim

FLAGS=dict(decompose_errors=True,ignore_decomposition_failures=False,
           block_decomposition_from_introducing_remnant_edges=True)

def repair(circuit, *, extract, annotate, strict_dem, seconds=900,
           pair_limit=96, sweeps=100, callback=None):
    start=time.monotonic();deadline=start+seconds
    rows,_=extract(circuit);operations=[];logs=[]
    ok,err=strict_dem(circuit)
    if ok:return circuit,dict(status='ALREADY_GRAPHLIKE',operations=[],sweeps=[],seconds=time.monotonic()-start)
    raw=circuit.detector_error_model(decompose_errors=False)
    columns=[set(t.val for t in op.targets_copy() if t.is_relative_detector_id())
             for op in raw.flattened() if op.type=='error']
    incidence=[set() for _ in rows]
    for k,col in enumerate(columns):
        for i in col:incidence[i].add(k)
    # Cache only queried bit rows, instead of allocating an enormous dense
    # integer for every detector. The source of truth remains sparse incidence.
    cache=collections.OrderedDict();max_cache=512
    def bitrow(i):
        if i in cache:
            cache.move_to_end(i);return cache[i]
        v=sum(1<<k for k in incidence[i]);cache[i]=v
        if len(cache)>max_cache:cache.popitem(last=False)
        return v
    def add(i,j):
        assert i!=j
        rows[i]^=rows[j]
        for k in incidence[j]:
            if k in incidence[i]:incidence[i].remove(k);columns[k].remove(i)
            else:incidence[i].add(k);columns[k].add(i)
        cache.pop(i,None);operations.append([i,j])
    def diagnose(candidate):
        active=set();hist=collections.Counter()
        try:
            dem=candidate.detector_error_model(**(FLAGS|dict(ignore_decomposition_failures=True)))
            for op in dem.flattened():
                if op.type!='error':continue
                group=[]
                for t in [*op.targets_copy(),stim.target_separator()]:
                    if t.is_separator():
                        if len(group)>2:active.update(group);hist[len(group)]+=1
                        group=[]
                    elif t.is_relative_detector_id():group.append(t.val)
        except ValueError:
            pass
        return active,dict(hist)
    active,hist=diagnose(circuit)
    stages=[(pair_limit,False),(max(pair_limit,192),False),(max(pair_limit,384),True)]
    for stage,(width,triples) in enumerate(stages):
        for sweep in range(sweeps):
            if time.monotonic()>deadline:break
            before=len(operations);max_moves=0
            for i in sorted(range(len(rows)),key=lambda k:len(incidence[k]),reverse=True):
                if time.monotonic()>deadline:break
                moves=0
                # Finish a troublesome row before rescanning the whole circuit.
                # The old algorithm made just one move per full sweep.
                while time.monotonic()<deadline:
                    ov=collections.Counter(j for k in incidence[i] for j in columns[k] if j!=i)
                    if not ov:break
                    choices=sorted(ov,key=lambda j:(len(incidence[j])-2*ov[j],(rows[i]^rows[j]).bit_count(),j))
                    j=choices[0];gain=2*ov[j]-len(incidence[j])
                    if gain>0 or (gain==0 and (rows[i]^rows[j]).bit_count()<rows[i].bit_count()):
                        add(i,j);moves+=1;continue
                    # No absolute 130-incidence cutoff: even a short residual
                    # row is searched, as are nonlocal measurement parities.
                    if i not in active and rows[i].bit_count()<=4 and len(incidence[i])<130:break
                    cand=choices[:width];v=bitrow(i);initial=len(incidence[i]);wc=rows[i].bit_count()
                    best=(0,0,None)
                    for off,j in enumerate(cand):
                        w=v^bitrow(j)
                        seconds_set=set(cand[off+1:])
                        if stage>=1:
                            # A useful second source may have zero overlap with
                            # the original target, and instead cancel support
                            # introduced by the first source. The old search
                            # excluded these neighbors completely.
                            seconds_set.update(k for col in incidence[j] for k in columns[col] if k not in (i,j))
                        for k in sorted(seconds_set):
                            gain=initial-(w^bitrow(k)).bit_count()
                            if gain<best[0]:continue
                            mg=wc-(rows[i]^rows[j]^rows[k]).bit_count()
                            if (gain,mg)>best[:2]:best=(gain,mg,(j,k))
                        if time.monotonic()>deadline:break
                    if best[2] is None and triples and i in active:
                        cand=choices[:32]
                        for a,j in enumerate(cand):
                            w=v^bitrow(j)
                            for b in range(a+1,len(cand)):
                                k=cand[b];z=w^bitrow(k)
                                for l in cand[b+1:]:
                                    gain=initial-(z^bitrow(l)).bit_count()
                                    if gain<best[0]:continue
                                    mg=wc-(rows[i]^rows[j]^rows[k]^rows[l]).bit_count()
                                    if (gain,mg)>best[:2]:best=(gain,mg,(j,k,l))
                            if time.monotonic()>deadline:break
                    if best[2] is None:break
                    for j in best[2]:add(i,j)
                    moves+=len(best[2])
                max_moves=max(max_moves,moves)
            candidate=annotate(circuit,rows);ok,err=strict_dem(candidate)
            if ok or len(operations)==before:active,hist=diagnose(candidate)
            event=dict(stage=stage,sweep=sweep,pair_width=width,triples=triples,two_hop_sources=stage>=1,
                operations=len(operations),new_operations=len(operations)-before,
                max_row_operations_in_sweep=max_moves,strict=ok,
                seconds=time.monotonic()-start,total_incidence=sum(map(len,incidence)),
                residual_histogram_at_last_diagnosis=hist,first_error=err[:700])
            logs.append(event)
            if callback:callback(candidate,operations,event)
            if ok:return candidate,dict(status='ACCEPTED',operations=operations,sweeps=logs,seconds=time.monotonic()-start)
            if len(operations)==before:break
        if time.monotonic()>deadline:break
    return None,dict(status='NOT_FOUND_WITHIN_BUDGET',operations=operations,sweeps=logs,
                     seconds=time.monotonic()-start,last_error=err)
