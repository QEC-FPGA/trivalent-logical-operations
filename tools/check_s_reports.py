"""Check normalized S shard statistics, exact report membership, and cohort totals."""
import argparse,csv,hashlib,io,json,tarfile
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);a=p.parse_args()
    reports={}
    for folder in ['w3','s_lowp']:
        for file in sorted((ROOT/'data'/folder).glob('*_counts.json')):
            for r in json.loads(file.read_text()):
                assert r['report_id'] not in reports;reports[r['report_id']]=r
    with tarfile.open(a.archive,'r:gz') as t:
        prefix='bundles/fig10-11-records/'
        contexts=json.load(t.extractfile(prefix+'contexts.json'))
        for h,r in contexts.items():
            assert hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()==h
        totals=defaultdict(lambda:[0,0,0]);cohorts=defaultdict(lambda:[0,0,0]);membership=defaultdict(list);seen=set()
        for r in csv.DictReader(io.TextIOWrapper(t.extractfile(prefix+'stats.csv'))):
            h=r['record_sha256'];assert len(h)==64 and h not in seen;seen.add(h)
            n,e=int(r['shots']),int(r['errors']);assert 0<=e<=n and n>0
            c=contexts[r['context_id']];task=c.get('task',{})
            assert c['circuit_sha256']==r['circuit_sha256'] and c['dem_sha256']==r['dem_sha256']
            assert c.get('plan_id',task.get('release'))==r['cohort']
            assert c.get('point_id',task.get('key'))==r['point_id']
            assert c.get('stage',task.get('phase')) in ['confirmation','production']
            for rid in filter(None,r['report_ids'].split(';')):
                assert rid in reports;membership[rid].append(h)
                v=totals[rid];v[0]+=n;v[1]+=e;v[2]+=1
            v=cohorts[(r['cohort'],r['point_id'])];v[0]+=n;v[1]+=e;v[2]+=1
    assert len(seen)==189792
    for rid,r in reports.items():
        assert totals[rid]==[r['shots'],r['mistakes'],r['source_shards']],rid
        assert hashlib.sha256(('\n'.join(sorted(membership[rid]))+'\n').encode()).hexdigest()==r['membership_sha256'],rid
    expected=list(csv.DictReader((ROOT/'data/s_cohort_counts.csv').open()))
    assert len(expected)==len(cohorts)
    for r in expected:
        assert cohorts[(r['cohort'],r['point_id'])]==[int(r[k]) for k in ['shots','errors','shards']]
    for r in json.loads((ROOT/'data/fig10_d13_cohorts.json').read_text()):
        for c in r['cohorts']:
            assert cohorts[(c['cohort'],c['point_id'])]==[c['shots'],c['errors'],c['shards']]
    print(f'PASS: {len(reports)} S reports, exact membership, all {len(cohorts)} cohorts, 3 d13 rows; {len(seen)} shards')
if __name__=='__main__':main()
