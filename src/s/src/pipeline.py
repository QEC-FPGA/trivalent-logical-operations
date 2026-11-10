"""Version-bound preparation, independent pilots, fixed-budget confirmations."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib,json,math,os,sys,traceback,platform,time
from study import ROOT,PACKAGE,read,write,sha
from case_lock import locked

def protocol():return read(PACKAGE/'protocol.json')
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def code_digest():
    files=sorted((PACKAGE/'src').rglob('*.py'))+[PACKAGE/'run.py',PACKAGE/'protocol.json']
    return digest({p.relative_to(PACKAGE).as_posix():sha(p) for p in files})
def selected(group,ids=None):
    ps=[p for p in protocol()['points'] if group=='all' or p['group']==group]
    if ids:
        unknown=set(ids)-{p['id'] for p in ps}
        if unknown:raise ValueError('Unknown point IDs: '+str(sorted(unknown)))
        ps=[p for p in ps if p['id'] in ids]
    if not ps:raise ValueError('No selected conditions')
    return ps
def folder_name(p):return f"d{p['d']}_m"+'-'.join(map(str,p['moves']))+f"_r2_W{p['W']}_T3_{p['channel']}"
def case_folder(p):return ROOT/'full_s'/folder_name(p)
def compiled_folder(p):return ROOT/'matching'/folder_name(p)

def runtime(require_decoder=False):
    import stim,numpy,scipy,importlib.metadata
    v=dict(python=platform.python_version(),platform=platform.platform(),stim=stim.__version__,
           numpy=numpy.__version__,scipy=scipy.__version__)
    if v['stim']!='1.16.0':raise ValueError('Stim 1.16.0 is required')
    if sys.version_info[:2]!=(3,12):raise ValueError('Use Python 3.12 for this frozen experiment')
    if require_decoder:
        import pymatching
        v['pymatching']=pymatching.__version__
        if v['pymatching']!='2.3.1':raise ValueError('PyMatching 2.3.1 is required; no version substitution')
    return v

def binding(p):
    m=read(compiled_folder(p)/'compile.json');cp=case_folder(p)/'full_original.stim'
    if m['status']!='ACCEPTED' or not m['block_remnant_edges']:raise ValueError('Strict decoder not accepted: '+p['id'])
    if m['original_circuit_sha256']!=sha(cp):raise ValueError('Original circuit changed')
    cfg=m['configuration']
    for key in ('d','W','T','channel','moves'):
        if cfg[key]!=p[key]:raise ValueError('Condition mismatch: '+key)
    if cfg['between_rounds']!=[2]*(len(p['moves'])-1):raise ValueError('Intermediate rounds changed')
    for path,h in [(compiled_folder(p)/'full_matching.stim',m['full_matching_sha256']),
                   (compiled_folder(p)/f"p{p['p']}.stim",m['sampling_circuit_sha256'][p['p']]),
                   (compiled_folder(p)/f"p{p['p']}.dem",m['dem_sha256'][p['p']])]:
        if sha(path)!=h:raise ValueError('Frozen input hash mismatch: '+str(path))
    return dict(original_circuit_sha256=m['original_circuit_sha256'],
                circuit_sha256=m['sampling_circuit_sha256'][p['p']],dem_sha256=m['dem_sha256'][p['p']],
                base_circuit_sha256=m['full_matching_sha256'],stim=m['stim_version'],pymatching='2.3.1')

def prepare_one(p):
    from generate_full_s import generate
    from compile_matching import compile_case
    from physical import circuit
    try:
        with locked(case_folder(p),'prepare',timeout=3600):
            f=generate(p['d'],p['W'],3,p['moves'],[2],p['channel'])
            m=compile_case(f,repair=True)
            if m['status']!='ACCEPTED':raise ValueError('Strict decomposition unsupported; nothing may be sampled')
            edges=set()
            for op in circuit(f/'full_original.stim'):
                if op.name in ('CX','CZ'):
                    q=[t.value for t in op.targets_copy()]
                    edges.update(tuple(sorted(pair)) for pair in zip(q[::2],q[1::2]))
            a=dict(case=p['case'],d=p['d'],status='READY',binding=binding(p),
                   code_sha256=code_digest(),edges=sorted(edges))
            write(f/'package_acceptance.json',a)
            return a
    except Exception as exc:
        r=dict(case=p['case'],d=p['d'],status='BLOCKED',error=str(exc),traceback=traceback.format_exc())
        write(ROOT/'blocked'/f"{p['case']}.json",r);return r

def parallel(fn,jobs,workers):
    if workers<1:raise ValueError('workers must be positive')
    results=[]
    if workers==1:
        for job in jobs:
            r=fn(job);results.append(r);print(r.get('case',r.get('id','job')),r.get('status','DONE'),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            fs={pool.submit(fn,job):job for job in jobs}
            for f in as_completed(fs):
                r=f.result();results.append(r);print(r.get('case',r.get('id','job')),r.get('status','DONE'),flush=True)
    return results

def prepare(group,workers,ids=None):
    runtime();points=selected(group,ids);unique={p['case']:p for p in points}
    rs=parallel(prepare_one,list(unique.values()),workers)
    networks=[]
    for d in sorted({p['d'] for p in points}):
        edges=set()
        for r in rs:
            if r['status']=='READY' and r['d']==d:edges.update(map(tuple,r['edges']))
        degree={q:sum(q in e for e in edges) for q in {v for e in edges for v in e}}
        maximum=max(degree.values(),default=0)
        if maximum>3:raise ValueError('Cross-condition physical edge union exceeds degree 3')
        networks.append(dict(d=d,max_degree=maximum,edges=sorted(edges),scope='accepted selected circuits'))
    write(ROOT/'preparation'/f'{group}.json',dict(results=rs,networks=networks))
    if any(r['status']!='READY' for r in rs):raise RuntimeError('Some circuits are blocked; see output/blocked. No decoder settings were relaxed.')
    return rs

def seed_for(identity):return int(hashlib.sha256(identity.encode()).hexdigest()[:16],16)%(2**63)
def make_shards(p,stage,plan_id,shots):
    size=protocol()['settings']['shard_shots']
    for index,start in enumerate(range(0,shots,size)):
        yield dict(point=p,stage=stage,plan_id=plan_id,index=index,shots=min(size,shots-start),
                   seed=seed_for(f"{protocol()['experiment']}/{p['id']}/{stage}/{plan_id}/{index}"))

def shard_worker(job):
    from numerical_io import sample
    p=job['point'];stage=job['stage'];ident=job['plan_id'];index=job['index']
    out=ROOT/'raw'/stage/ident/p['id']/f'{index:06d}.json'
    with locked(out.parent,f'{index:06d}',timeout=3600):
        expected=dict(point_id=p['id'],plan_id=ident,shard_index=index,code_sha256=code_digest())
        r=sample(compiled_folder(p),p['p'],job['shots'],job['seed'],out,stage,provenance=expected)
        for key,val in expected.items():
            if key in r and r[key]!=val:raise ValueError('Shard identity mismatch')
    return dict(id=p['id']+f'/{index:06d}',status='COMPLETE',shots=r['shots'],mistakes=r['mistakes'])

def pilot(group,workers,batch=0,shots=None,ids=None):
    env=runtime(True);ps=selected(group,ids)
    if batch<0:raise ValueError('Nonnegative batch required')
    entries=[dict(point=p,shots=shots or p['pilot_shots'],binding=binding(p)) for p in ps]
    if any(e['shots']<1 for e in entries):raise ValueError('Positive fixed pilot budget required')
    ident=f'{group}_b{batch:03d}';path=ROOT/'plans'/f'pilot_{ident}.json'
    plan=dict(stage='pilot',plan_id=ident,code_sha256=code_digest(),protocol_sha256=sha(PACKAGE/'protocol.json'),entries=entries)
    with locked(path.parent,'pilot_plan_'+ident):
        if path.exists() and read(path)!=plan:raise ValueError('This pilot batch is already frozen differently; use a new --batch')
        if not path.exists():write(path,plan)
    jobs=[job for e in entries for job in make_shards(e['point'],'pilot',ident,e['shots'])]
    return parallel(shard_worker,jobs,workers)

def collect_entry(point,stage,plan_id,shots):
    expected=list(make_shards(point,stage,plan_id,shots));rows=[];sources={}
    bound=binding(point)
    for job in expected:
        path=ROOT/'raw'/stage/plan_id/point['id']/f"{job['index']:06d}.json"
        if not path.exists():return None
        r=read(path)
        if r.get('status')!='COMPLETE' or r.get('point_id')!=point['id']:raise ValueError('Uncommitted or wrong shard')
        for key in ('stage','plan_id','seed','shots'):
            if r[key]!=job[key]:raise ValueError('Wrong shard '+key)
        if r['code_sha256']!=code_digest():raise ValueError('Code changed since sampling')
        for key in ('circuit_sha256','dem_sha256','base_circuit_sha256'):
            if r[key]!=bound[key]:raise ValueError('Shard source changed')
        if r['stim_version']!='1.16.0' or r['pymatching_version']!='2.3.1':raise ValueError('Decoder/runtime mismatch')
        if not 0<=r['mistakes']<=r['shots']:raise ValueError('Invalid counts')
        rows.append(r);sources[path.relative_to(ROOT).as_posix()]=sha(path)
    return dict(shots=sum(r['shots'] for r in rows),mistakes=sum(r['mistakes'] for r in rows),sources=sources,
                seconds=sum(r['seconds'] for r in rows))

def budget(group,max_total=None):
    runtime(True);ps=selected(group);entries=[];all_sources={}
    settings=protocol()['settings'];target=settings['target_expected_errors'];minimum=settings['min_pilot_errors']
    for p in ps:
        shots=errors=0;seconds=0.;sources={}
        for pp in sorted((ROOT/'plans').glob('pilot_*.json')):
            plan=read(pp)
            if plan['code_sha256']!=code_digest():raise ValueError('Pilot source code mismatch')
            for e in plan['entries']:
                if e['point']['id']!=p['id']:continue
                got=collect_entry(p,'pilot',plan['plan_id'],e['shots'])
                if got is None:raise ValueError('Finish the already started pilot batch first: '+plan['plan_id'])
                shots+=got['shots'];errors+=got['mistakes'];seconds+=got['seconds'];sources.update(got['sources'])
        row=dict(point=p,pilot_shots=shots,pilot_errors=errors,pilot_sources=sources,binding=binding(p))
        if errors<minimum:row.update(status='MORE_PILOT_REQUIRED',confirmation_shots=None)
        else:
            required=math.ceil(target*shots/errors/settings['shard_shots'])*settings['shard_shots']
            row.update(status='READY',confirmation_shots=required,estimated_worker_hours=seconds/shots*required/3600)
        entries.append(row);all_sources.update(sources)
    proposal=dict(group=group,stage='confirmation',code_sha256=code_digest(),protocol_sha256=sha(PACKAGE/'protocol.json'),
        target_expected_errors=target,entries=entries,total_shots=sum(e['confirmation_shots'] or 0 for e in entries),
        all_ready=all(e['status']=='READY' for e in entries),pilot_sources=all_sources)
    proposal['plan_id']=group+'_'+digest(proposal)[:16]
    write(ROOT/'budgets'/f'{group}_proposal.json',proposal)
    print('BUDGET',proposal['total_shots'],'shots; all_ready=',proposal['all_ready'],flush=True)
    if max_total is None:return proposal
    if not proposal['all_ready']:raise ValueError('Insufficient pilot failures; no confirmation budget frozen')
    if proposal['total_shots']>max_total:raise ValueError('Required shots exceed --max-total-shots; proposal saved, no confirmation frozen')
    path=ROOT/'budgets'/f'{group}.json'
    with locked(path.parent,'budget_'+group):
        if path.exists() and read(path)!=proposal:raise ValueError('Confirmation budget already frozen; do not change it using confirmation outcomes')
        write(path,proposal)
    return proposal

def frozen_budget(group):
    b=read(ROOT/'budgets'/f'{group}.json')
    if not b['all_ready'] or b['code_sha256']!=code_digest() or b['protocol_sha256']!=sha(PACKAGE/'protocol.json'):
        raise ValueError('Invalid or changed frozen budget')
    for path,h in b['pilot_sources'].items():
        if sha(ROOT/path)!=h:raise ValueError('Pilot evidence changed')
    for e in b['entries']:
        if binding(e['point'])!=e['binding']:raise ValueError('Compiled input changed since budget freeze')
    return b

def confirm(group,workers):
    runtime(True);b=frozen_budget(group)
    jobs=[job for e in b['entries'] for job in make_shards(e['point'],'confirmation',b['plan_id'],e['confirmation_shots'])]
    return parallel(shard_worker,jobs,workers)

def final_counts(group):
    if group=='all':
        groups=[g for g in protocol()['groups'] if (ROOT/'budgets'/f'{g}.json').exists()]
        if not groups:raise ValueError('No frozen confirmation budgets')
        rows=[r for g in groups for r in final_counts(g)]
        if len({r['id'] for r in rows})!=len(rows):raise ValueError('Repeated confirmation condition')
        return rows
    b=frozen_budget(group);rows=[];missing=[]
    for e in b['entries']:
        p=e['point'];r=collect_entry(p,'confirmation',b['plan_id'],e['confirmation_shots'])
        if r is None:missing.append(p['id']);continue
        meta=read(case_folder(p)/'metadata.json')
        rows.append(dict(**p,shots=r['shots'],mistakes=r['mistakes'],rate=r['mistakes']/r['shots'],
                         extraction_rounds=meta['extraction_rounds'],two_qubit_layers=meta['two_qubit_layers'],ticks=meta['total_ticks'],
                         plan_id=b['plan_id'],sources=r['sources']))
    if missing:raise ValueError('Confirmation incomplete; missing conditions: '+', '.join(missing))
    return rows
