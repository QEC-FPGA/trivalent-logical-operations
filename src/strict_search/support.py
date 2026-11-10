"""Versioned routes, including explicit small-size clipping and deduplication."""
from pathlib import Path
import hashlib,json,os,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
VERSION='20261020-T3-variable-tail-v2'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+f'.{os.getpid()}.tmp')
    t.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def bits(v):
    while v:
        b=v&-v;yield b.bit_length()-1;v^=b
def route(total,step):return [min(step,total-k) for k in range(0,total,step)]
def routes(d):
    unique={}
    requests=[('original',2,'original',3)]
    requests += [(f'step{s}',s,'third',3) for s in (3,4,5)]
    requests += [(f'step{s}',s,'optimized',5 if s in (3,4) else 7) for s in (3,4,5)]
    for family,step,policy,requested_q in requests:
        q=min(requested_q,d-2)
        moves=route(d-q,step)+[q]
        holds=[2]*(len(moves)-2)+[3]
        key=(tuple(moves),tuple(holds))
        entry=unique.setdefault(key,dict(d=d,moves=moves,holds=holds,terminal_q=q,aliases=[],is_original=False))
        entry['is_original'] |= family=='original'
        entry['aliases'].append(dict(family=family,nominal_step=step,tail_policy=policy,
            requested_q=requested_q,actual_q=q,small_size_clipped=q!=requested_q,
            actual_internal_moves=moves[:-1],has_nominal_internal_step=step in moves[:-1]))
    return list(unique.values())
def plan():
    out=[]
    for d in (5,7,9):
        wc=2*((d+6)//6)+1
        for rr in routes(d):
            for w in sorted({wc,d}):
                for channel in ('XY','ZZ'):
                    label='-'.join(map(str,rr['moves']))
                    v=dict(**rr,id=f'd{d}_m{label}_W{w}_{channel}',W=w,channel=channel,
                        family='original' if rr['is_original'] else 'multi',wall='A',fullwall_rounds=d,
                        final_twist_rounds=3,waiting_policy='Wc_original_reference' if w==wc else 'Wd',
                        core_extraction_rounds=w+sum(rr['holds']),total_extraction_rounds=2*d+w+sum(rr['holds']),
                        package_version=VERSION)
                    v['config_sha256']=digest(v);out.append(v)
    # Keep the original per-case identities, but expose ONLY the three baseline
    # geometries in this release. Multistep routes and W=d are not scheduled.
    return [v for v in out if v['is_original'] and
            v['W']==2*((v['d']+6)//6)+1]
def case_dir(spec):return ROOT/'cases'/spec['id']
def result_dir(spec):return ROOT/'results'/spec['id']
def export_plan():
    rows=plan();write(ROOT/'matrix.json',dict(version=VERSION,conditions=rows,count=len(rows)))
    lines=['Three-round tails: deduplicated movement routes',
           'The final number is the span of the transition to fullwall; the preceding position retains three B,A,B rounds.',
           'Other intermediate positions use two B,A rounds; one round is one A or B extraction.','']
    for d in (5,7,9):
        lines.append(f'd={d}: W={2*((d+6)//6)+1}, {d}; each route generates XY and ZZ.')
        for r in routes(d):
            aliases='; '.join(f"{a['family']}/{a['tail_policy']}:q{a['requested_q']}->q{a['actual_q']}" for a in r['aliases'])
            lines.append(f"  {'+'.join(map(str,r['moves']))}; extraction rounds per position={r['holds']}; {aliases}")
        lines.append('')
    lines += ['All requests at d=5 reduce to 2+3, leaving one actual route; they cannot establish the effect of larger steps.',
        'The 2+5 route at d=7 and 2+7 route at d=9 move only two cells before the tail; they test the tail transition, not internal three-, four- or five-cell moves.',
        'Duplicate routes are searched once; aliases in matrix.json retain all request mappings.']
    (ROOT/'route_summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
