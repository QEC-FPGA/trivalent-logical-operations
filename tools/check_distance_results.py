"""Check archived partition coverage and artifact bindings; not a fresh exhaustive search."""
import argparse,hashlib,json
from pathlib import Path
def read(p):return json.loads(p.read_text())
def digest(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parents[1]/'bundles/distance-evidence');a=ap.parse_args()
    plan=read(a.bundle/'search-plan.json')
    for c in plan['conditions']:
        model=read(a.bundle/'results'/c['case']/'model.json');assert digest(model)==c['model_sha256']
        for n,h in model['artifacts_sha256'].items():assert hashlib.sha256((a.bundle/'results'/c['case']/n).read_bytes()).hexdigest()==h
        cp=a.bundle/'cases'/c['case']/'full_original.stim';assert hashlib.sha256(cp.read_bytes()).hexdigest()==model['circuit_sha256']
        tasks=sorted([t for t in plan['tasks'] if t['case']==c['case']],key=lambda t:t['start']);end=0
        for t in tasks:
            assert t['start']==end and t['stop']>end and t['cap']==c['cap'] and t['model_sha256']==c['model_sha256']
            folder=a.bundle/'partitions'/t['case']/f"{t['start']:06d}-{t['stop']:06d}"
            r=read(folder/'result.json');assert r['status']=='INFEASIBLE' and r['task']==t
            for n,h in r['artifacts'].items():assert hashlib.sha256((folder/n).read_bytes()).hexdigest()==h
            if 'enumerated_roots' in r:assert sorted(r['enumerated_roots'])==list(range(t['start'],t['stop']))
            end=t['stop']
        assert end==c['roots'];assert model['seed_upper_bound']==c['cap']+1==c['d']
        print(c['case'],'PASS: saved partition coverage, model bindings, and upper-bound metadata')
if __name__=='__main__':main()
