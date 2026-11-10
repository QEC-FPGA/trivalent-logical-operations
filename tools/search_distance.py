"""Run a new capped CP-SAT/full-signature search from an archived fault model."""
import argparse,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,default=ROOT/'bundles/distance-evidence')
    p.add_argument('--case',required=True,choices=['d5_m2-3_W3_XY','d5_m2-3_W3_ZZ','d7_m2-2-3_W5_XY','d7_m2-2-3_W5_ZZ'])
    p.add_argument('--cap',type=int,required=True);p.add_argument('--seconds',type=float,default=900);p.add_argument('--workers',type=int,default=2);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('choose a new output directory')
    if a.cap<1 or a.seconds<=0 or a.workers<1:p.error('cap, seconds and workers must be positive')
    for folder in ['cases','results']:
        shutil.copytree(a.bundle/folder/a.case,a.out/folder/a.case,ignore=shutil.ignore_patterns('pair_index.npy'))
    sys.path.insert(0,str(ROOT/'src/strict_search'))
    import support
    support.case_dir=lambda spec:a.out/'cases'/spec['id']
    support.result_dir=lambda spec:a.out/'results'/spec['id']
    from exact_search import solve_cap
    spec=next(s for s in support.plan() if s['id']==a.case)
    r=solve_cap(spec,a.cap,seconds=a.seconds,workers=a.workers)
    print(r['status']);print(a.out.resolve())
if __name__=='__main__':main()
