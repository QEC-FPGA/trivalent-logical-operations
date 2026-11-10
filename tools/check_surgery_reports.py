"""Reconstruct all Figure 12 counts from archived four-bit residual histograms."""
import argparse,csv,io,json,tarfile
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);a=p.parse_args()
    hist=defaultdict(lambda:[0]*16);seen=set()
    with tarfile.open(a.archive,'r:gz') as t:
        prefix='bundles/fig12-confirmation/'
        contexts=json.load(t.extractfile(prefix+'contexts.json'))
        for r in csv.DictReader(io.TextIOWrapper(t.extractfile(prefix+'stats.csv'))):
            h=r['record_sha256'];assert h not in seen;seen.add(h)
            assert contexts[r['context_id']]['decoder']=='pymatching_correlated'
            counts=[int(r[f'residual_{i}']) for i in range(16)]
            assert all(x>=0 for x in counts) and sum(counts)==int(r['shots'])
            v=hist[(r['circuit_sha256'],r['dem_sha256'])]
            for i,n in enumerate(counts):v[i]+=n
    table=list(csv.DictReader((ROOT/'data/counts/fig12_surgery_counts.csv').open()))
    assert len(hist)==len(table)==81
    for r in table:
        h=hist[(r['circuit_sha256'],r['dem_sha256'])]
        got=[sum(h),sum(h[1:]),sum(v for i,v in enumerate(h) if i&1),sum(v for i,v in enumerate(h) if i&14),sum(v for i,v in enumerate(h) if i&1 and i&14)]
        assert got==[int(r[k]) for k in ['shots','errors','joint','remaining','both']],r
    print(f'PASS: all 81 Figure 12 conditions and failure categories from {len(seen)} shards')
if __name__=='__main__':main()
